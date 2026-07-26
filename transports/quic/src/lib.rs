//! QUIC Transport for tensor data transfer.
//!
//! Provides high-performance, low-latency data plane communication using QUIC protocol.

use cluster_types::{NodeId, TensorId};
use observability::{LogContext, MetricsCollector};
use bytes::Bytes;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use quinn::{Endpoint, ServerConfig, ClientConfig, Connection, RecvStream, SendStream};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// QUIC transport server.
pub struct QuicServer {
    endpoint: Endpoint,
    metrics: MetricsCollector,
    active_connections: Arc<RwLock<HashMap<NodeId, Connection>>>,
}

impl QuicServer {
    pub async fn new(bind_addr: SocketAddr, cert: CertificateDer<'static>, key: PrivateKeyDer<'static>) -> Result<Self, QuicError> {
        let ctx = LogContext::new("quic_server");
        info!("Starting QUIC server on {}", bind_addr);

        // Configure server TLS
        let server_config = ServerConfig::with_single_cert(vec![cert.clone()], key.clone())
            .map_err(|e| QuicError::ConfigError(e.to_string()))?;

        let endpoint = Endpoint::server(server_config, bind_addr.into())
            .map_err(|e| QuicError::BindError(e.to_string()))?;

        let metrics = MetricsCollector::new("quic-transport".to_string());

        ctx.info(&format!("QUIC server listening on {}", bind_addr));

        Ok(Self {
            endpoint,
            metrics,
            active_connections: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Accept incoming connections.
    pub async fn accept(&self) -> Result<(NodeId, Connection, RecvStream, SendStream), QuicError> {
        let accepting = self.endpoint.accept()
            .await
            .ok_or_else(|| QuicError::AcceptFailed("No incoming connection".to_string()))?;

        let conn = accepting.await
            .map_err(|e| QuicError::ConnectionError(e.to_string()))?;

        let node_id = self.extract_node_id(&conn).await?;

        // Open streams for bidirectional communication
        let (send, recv) = conn.open_bi().await
            .map_err(|e| QuicError::StreamError(e.to_string()))?;

        // Store connection
        let mut connections = self.active_connections.write().await;
        connections.insert(node_id, conn.clone());

        self.metrics.increment_counter("connections_accepted", 1, &[]);

        Ok((node_id, conn, recv, send))
    }

    /// Extract node ID from connection (simplified - in real implementation would use TLS certificates).
    async fn extract_node_id(&self, _conn: &Connection) -> Result<NodeId, QuicError> {
        // In a real implementation, this would extract the node ID from the client's certificate
        // For now, generate a random node ID
        Ok(uuid::Uuid::new_v4())
    }

    /// Send tensor data to a node.
    pub async fn send_tensor(&self, node_id: NodeId, tensor_id: TensorId, data: Bytes) -> Result<(), QuicError> {
        let connections = self.active_connections.read().await;
        
        let conn = connections.get(&node_id)
            .ok_or_else(|| QuicError::NodeNotFound(node_id))?;

        let mut send = conn.open_bi().await
            .map_err(|e| QuicError::StreamError(e.to_string()))?.0;

        // Send tensor metadata
        let metadata = serde_json::json!({
            "tensor_id": tensor_id,
            "size": data.len(),
        });
        
        let metadata_bytes = metadata.to_string().into_bytes();
        let metadata_len = (metadata_bytes.len() as u32).to_be_bytes();
        
        send.write_all(&metadata_len).await
            .map_err(|e| QuicError::SendError(e.to_string()))?;
        
        send.write_all(&metadata_bytes).await
            .map_err(|e| QuicError::SendError(e.to_string()))?;
        
        // Send tensor data
        send.write_all(&data).await
            .map_err(|e| QuicError::SendError(e.to_string()))?;

        send.finish().await
            .map_err(|e| QuicError::SendError(e.to_string()))?;

        self.metrics.increment_counter("tensors_sent", 1, &[]);
        self.metrics.increment_counter("bytes_sent", data.len() as u64, &[]);

        Ok(())
    }

    /// Get server statistics.
    pub async fn get_stats(&self) -> ServerStats {
        let connections = self.active_connections.read().await;
        
        ServerStats {
            active_connections: connections.len(),
        }
    }
}

/// QUIC transport client.
pub struct QuicClient {
    endpoint: Endpoint,
    metrics: MetricsCollector,
    server_addr: SocketAddr,
    server_cert: CertificateDer<'static>,
}

impl QuicClient {
    pub async fn new(server_addr: SocketAddr, server_cert: CertificateDer<'static>) -> Result<Self, QuicError> {
        let ctx = LogContext::new("quic_client");
        info!("Creating QUIC client for {}", server_addr);

        // Configure client TLS
        let mut roots = rustls::RootCertStore::empty();
        roots.add(server_cert.clone())
            .map_err(|e| QuicError::ConfigError(e.to_string()))?;

        let client_config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth()
            .map_err(|e| QuicError::ConfigError(e.to_string()))?;

        let endpoint = Endpoint::new(client_config)
            .map_err(|e| QuicError::ConfigError(e.to_string()))?;

        let metrics = MetricsCollector::new("quic-transport".to_string());

        ctx.info("QUIC client created successfully");

        Ok(Self {
            endpoint,
            metrics,
            server_addr,
            server_cert,
        })
    }

    /// Connect to server.
    pub async fn connect(&self) -> Result<Connection, QuicError> {
        info!("Connecting to QUIC server at {}", self.server_addr);

        let conn = self.endpoint.connect(self.server_addr, "localhost")
            .map_err(|e| QuicError::ConnectionError(e.to_string()))?
            .await
            .map_err(|e| QuicError::ConnectionError(e.to_string()))?;

        self.metrics.increment_counter("connections_established", 1, &[]);

        Ok(conn)
    }

    /// Receive tensor data from server.
    pub async fn receive_tensor(&self, conn: &Connection) -> Result<(TensorId, Bytes), QuicError> {
        let (mut recv, _send) = conn.accept_bi().await
            .map_err(|e| QuicError::StreamError(e.to_string()))?;

        // Read metadata length
        let mut len_bytes = [0u8; 4];
        recv.read_exact(&mut len_bytes).await
            .map_err(|e| QuicError::ReceiveError(e.to_string()))?;
        
        let metadata_len = u32::from_be_bytes(len_bytes) as usize;

        // Read metadata
        let mut metadata_bytes = vec![0u8; metadata_len];
        recv.read_exact(&mut metadata_bytes).await
            .map_err(|e| QuicError::ReceiveError(e.to_string()))?;
        
        let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| QuicError::ParseError(e.to_string()))?;
        
        let tensor_id = metadata["tensor_id"].as_str()
            .ok_or_else(|| QuicError::ParseError("Missing tensor_id".to_string()))?
            .to_string();
        
        let size = metadata["size"].as_u64()
            .ok_or_else(|| QuicError::ParseError("Missing size".to_string()))? as usize;

        // Read tensor data
        let mut data = vec![0u8; size];
        recv.read_exact(&mut data).await
            .map_err(|e| QuicError::ReceiveError(e.to_string()))?;

        self.metrics.increment_counter("tensors_received", 1, &[]);
        self.metrics.increment_counter("bytes_received", size as u64, &[]);

        Ok((tensor_id, Bytes::from(data)))
    }

    /// Send tensor data to server.
    pub async fn send_tensor(&self, conn: &Connection, tensor_id: TensorId, data: Bytes) -> Result<(), QuicError> {
        let mut send = conn.open_bi().await
            .map_err(|e| QuicError::StreamError(e.to_string()))?.0;

        // Send tensor metadata
        let metadata = serde_json::json!({
            "tensor_id": tensor_id,
            "size": data.len(),
        });
        
        let metadata_bytes = metadata.to_string().into_bytes();
        let metadata_len = (metadata_bytes.len() as u32).to_be_bytes();
        
        send.write_all(&metadata_len).await
            .map_err(|e| QuicError::SendError(e.to_string()))?;
        
        send.write_all(&metadata_bytes).await
            .map_err(|e| QuicError::SendError(e.to_string()))?;
        
        // Send tensor data
        send.write_all(&data).await
            .map_err(|e| QuicError::SendError(e.to_string()))?;

        send.finish().await
            .map_err(|e| QuicError::SendError(e.to_string()))?;

        self.metrics.increment_counter("tensors_sent", 1, &[]);
        self.metrics.increment_counter("bytes_sent", data.len() as u64, &[]);

        Ok(())
    }
}

/// Server statistics.
#[derive(Debug, Clone)]
pub struct ServerStats {
    pub active_connections: usize,
}

/// QUIC transport errors.
#[derive(Debug, thiserror::Error)]
pub enum QuicError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Bind error: {0}")]
    BindError(String),
    
    #[error("Accept failed: {0}")]
    AcceptFailed(String),
    
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    #[error("Stream error: {0}")]
    StreamError(String),
    
    #[error("Send error: {0}")]
    SendError(String),
    
    #[error("Receive error: {0}")]
    ReceiveError(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quic_config() {
        // Test that we can create a QUIC configuration
        let cert = CertificateDer::from(vec![1, 2, 3]);
        let key = PrivateKeyDer::Pkcs8(vec![4, 5, 6]);
        
        let result = QuicServer::new("127.0.0.1:0".parse().unwrap(), cert, key).await;
        // This will fail with invalid cert, but tests the configuration path
        assert!(result.is_err());
    }
}
