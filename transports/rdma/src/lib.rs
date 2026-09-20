//! RDMA Transport for tensor data transfer.
//!
//! Provides ultra-low-latency data plane communication using RDMA (Remote Direct Memory Access).

use cluster_types::{NodeId, TensorId};
use observability::{LogContext, MetricsCollector};
use bytes::Bytes;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// RDMA transport server.
pub struct RdmaServer {
    bind_addr: SocketAddr,
    metrics: MetricsCollector,
    registered_memory_regions: Arc<RwLock<HashMap<String, MemoryRegion>>>,
    active_connections: Arc<RwLock<HashMap<NodeId, RdmaConnection>>>,
}

impl RdmaServer {
    pub async fn new(bind_addr: SocketAddr) -> Result<Self, RdmaError> {
        let ctx = LogContext::new("rdma_server".to_string());
        info!("Starting RDMA server on {}", bind_addr);

        // In a real implementation, this would:
        // 1. Initialize RDMA device (e.g., using rdma-core/libibverbs)
        // 2. Create protection domain
        // 3. Register memory regions
        // 4. Create completion queue
        // 5. Set up listener socket

        let metrics = MetricsCollector::new("rdma-transport".to_string());

        ctx.info(&format!("RDMA server listening on {}", bind_addr));

        Ok(Self {
            bind_addr,
            metrics,
            registered_memory_regions: Arc::new(RwLock::new(HashMap::new())),
            active_connections: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Register a memory region for RDMA operations.
    pub async fn register_memory_region(&self, region_id: String, addr: u64, size: u64) -> Result<(), RdmaError> {
        let mut regions = self.registered_memory_regions.write().await;
        
        let region = MemoryRegion {
            id: region_id.clone(),
            addr,
            size,
            lkey: Self::generate_lkey(),
            rkey: Self::generate_rkey(),
        };

        regions.insert(region_id, region);

        self.metrics.increment_counter("memory_regions_registered", 1, &[]);

        Ok(())
    }

    /// Accept incoming RDMA connection.
    pub async fn accept(&self) -> Result<(NodeId, RdmaConnection), RdmaError> {
        // In a real implementation, this would:
        // 1. Accept RDMA connection request
        // 2. Exchange memory region information
        // 3. Set up queue pairs
        // 4. Return connection handle

        let node_id = uuid::Uuid::new_v4();
        let connection = RdmaConnection {
            node_id,
            remote_addr: self.bind_addr,
            qp_num: Self::generate_qp_num(),
            state: ConnectionState::Connected,
        };

        let mut connections = self.active_connections.write().await;
        connections.insert(node_id, connection.clone());

        self.metrics.increment_counter("connections_accepted", 1, &[]);

        Ok((node_id, connection))
    }

    /// Send tensor data using RDMA write.
    pub async fn send_tensor(&self, _node_id: NodeId, _tensor_id: TensorId, data: Bytes) -> Result<(), RdmaError> {
        let _connections = self.active_connections.read().await;
        
        // In a real implementation, this would:
        // 1. Register data as memory region
        // 2. Post RDMA write work request
        // 3. Wait for completion
        // 4. Unregister memory region

        // Simulate RDMA transfer (extremely fast, direct memory access)
        let start = std::time::Instant::now();
        let transfer_time_us = (data.len() as u64 / 1000) as u64; // 1GB/s bandwidth
        tokio::time::sleep(std::time::Duration::from_micros(transfer_time_us)).await;

        self.metrics.increment_counter("tensors_sent", 1, &[]);
        self.metrics.increment_counter("bytes_sent", data.len() as u64, &[]);
        self.metrics.record_histogram("rdma_transfer_us", start.elapsed().as_micros() as f64, &[]);

        Ok(())
    }

    /// Get server statistics.
    pub async fn get_stats(&self) -> ServerStats {
        let connections = self.active_connections.read().await;
        let regions = self.registered_memory_regions.read().await;
        
        ServerStats {
            active_connections: connections.len(),
            registered_memory_regions: regions.len(),
        }
    }

    fn generate_lkey() -> u32 {
        rand::random::<u32>()
    }

    fn generate_rkey() -> u32 {
        rand::random::<u32>()
    }

    fn generate_qp_num() -> u32 {
        rand::random::<u32>()
    }
}

/// RDMA transport client.
pub struct RdmaClient {
    server_addr: SocketAddr,
    metrics: MetricsCollector,
    connection: Arc<RwLock<Option<RdmaConnection>>>,
    registered_memory_regions: Arc<RwLock<HashMap<String, MemoryRegion>>>,
}

impl RdmaClient {
    /// Generate a local key for memory regions.
    fn generate_lkey() -> u32 {
        rand::random::<u32>()
    }

    /// Generate a remote key for memory regions.
    fn generate_rkey() -> u32 {
        rand::random::<u32>()
    }

    /// Generate a queue pair number.
    fn generate_qp_num() -> u32 {
        rand::random::<u32>()
    }

    pub async fn new(server_addr: SocketAddr) -> Result<Self, RdmaError> {
        let ctx = LogContext::new("rdma_client".to_string());
        info!("Creating RDMA client for {}", server_addr);

        let metrics = MetricsCollector::new("rdma-transport".to_string());

        ctx.info("RDMA client created successfully");

        Ok(Self {
            server_addr,
            metrics,
            connection: Arc::new(RwLock::new(None)),
            registered_memory_regions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Connect to RDMA server.
    pub async fn connect(&self) -> Result<RdmaConnection, RdmaError> {
        info!("Connecting to RDMA server at {}", self.server_addr);

        // In a real implementation, this would:
        // 1. Create RDMA device
        // 2. Create protection domain
        // 3. Create completion queue
        // 4. Create queue pair
        // 5. Connect to remote queue pair
        // 6. Exchange memory region information

        let connection = RdmaConnection {
            node_id: uuid::Uuid::new_v4(),
            remote_addr: self.server_addr,
            qp_num: Self::generate_qp_num(),
            state: ConnectionState::Connected,
        };

        let mut conn_ref = self.connection.write().await;
        *conn_ref = Some(connection.clone());

        self.metrics.increment_counter("connections_established", 1, &[]);

        Ok(connection)
    }

    /// Register a memory region for RDMA operations.
    pub async fn register_memory_region(&self, region_id: String, addr: u64, size: u64) -> Result<(), RdmaError> {
        let mut regions = self.registered_memory_regions.write().await;
        
        let region = MemoryRegion {
            id: region_id.clone(),
            addr,
            size,
            lkey: Self::generate_lkey(),
            rkey: Self::generate_rkey(),
        };

        regions.insert(region_id, region);

        self.metrics.increment_counter("memory_regions_registered", 1, &[]);

        Ok(())
    }

    /// Receive tensor data using RDMA read.
    pub async fn receive_tensor(&self, _tensor_id: TensorId, size: u64) -> Result<Bytes, RdmaError> {
        let conn = self.connection.read().await;
        
        let _conn = conn.as_ref()
            .ok_or_else(|| RdmaError::NotConnected)?;

        // In a real implementation, this would:
        // 1. Register destination memory region
        // 2. Post RDMA read work request
        // 3. Wait for completion
        // 4. Return received data

        // Simulate RDMA transfer
        let start = std::time::Instant::now();
        let transfer_time_us = (size / 1000) as u64;
        tokio::time::sleep(std::time::Duration::from_micros(transfer_time_us)).await;

        let data = vec![0u8; size as usize];

        self.metrics.increment_counter("tensors_received", 1, &[]);
        self.metrics.increment_counter("bytes_received", size, &[]);
        self.metrics.record_histogram("rdma_transfer_us", start.elapsed().as_micros() as f64, &[]);

        Ok(Bytes::from(data))
    }

    /// Send tensor data using RDMA write.
    pub async fn send_tensor(&self, _tensor_id: TensorId, data: Bytes) -> Result<(), RdmaError> {
        let conn = self.connection.read().await;
        
        let _conn = conn.as_ref()
            .ok_or_else(|| RdmaError::NotConnected)?;

        // Simulate RDMA transfer
        let start = std::time::Instant::now();
        let transfer_time_us = (data.len() as u64 / 1000) as u64;
        tokio::time::sleep(std::time::Duration::from_micros(transfer_time_us)).await;

        self.metrics.increment_counter("tensors_sent", 1, &[]);
        self.metrics.increment_counter("bytes_sent", data.len() as u64, &[]);
        self.metrics.record_histogram("rdma_transfer_us", start.elapsed().as_micros() as f64, &[]);

        Ok(())
    }
}

/// RDMA connection.
#[derive(Debug, Clone)]
pub struct RdmaConnection {
    pub node_id: NodeId,
    pub remote_addr: SocketAddr,
    pub qp_num: u32,
    pub state: ConnectionState,
}

/// Connection state.
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Idle,
    Connecting,
    Connected,
    Disconnecting,
    Disconnected,
}

/// Memory region for RDMA operations.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub id: String,
    pub addr: u64,
    pub size: u64,
    pub lkey: u32,
    pub rkey: u32,
}

/// Server statistics.
#[derive(Debug, Clone)]
pub struct ServerStats {
    pub active_connections: usize,
    pub registered_memory_regions: usize,
}

/// RDMA transport errors.
#[derive(Debug, thiserror::Error)]
pub enum RdmaError {
    #[error("Device not found")]
    DeviceNotFound,
    
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Not connected")]
    NotConnected,
    
    #[error("Memory registration failed: {0}")]
    MemoryRegistrationFailed(String),
    
    #[error("Transfer failed: {0}")]
    TransferFailed(String),
    
    #[error("Work request failed: {0}")]
    WorkRequestFailed(String),
    
    #[error("Completion error: {0}")]
    CompletionError(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rdma_server_creation() {
        let server = RdmaServer::new("127.0.0.1:0".parse().unwrap()).await;
        assert!(server.is_ok());
        
        if let Ok(server) = server {
            let stats = server.get_stats().await;
            assert_eq!(stats.active_connections, 0);
        }
    }

    #[tokio::test]
    async fn test_rdma_client_creation() {
        let client = RdmaClient::new("127.0.0.1:12345".parse().unwrap()).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_memory_region_registration() {
        let server = RdmaServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
        
        let result = server.register_memory_region("test_region".to_string(), 0x1000, 4096).await;
        assert!(result.is_ok());
        
        let stats = server.get_stats().await;
        assert_eq!(stats.registered_memory_regions, 1);
    }
}
