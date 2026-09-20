//! TCP Transport for tensor data transfer.
//!
//! Provides reliable TCP-based data plane communication for tensor transfers
//! between nodes in the cluster.

use bytes::Bytes;
use cluster_types::{NodeId, TensorId};
use observability::{LogContext, MetricsCollector};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use uuid::Uuid;

/// TCP transport configuration.
#[derive(Debug, Clone)]
pub struct TcpConfig {
    pub bind_address: String,
    pub max_connections: usize,
    pub buffer_size: usize,
    pub timeout_secs: u64,
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:0".to_string(),
            max_connections: 100,
            buffer_size: 64 * 1024, // 64 KB
            timeout_secs: 30,
        }
    }
}

/// TCP transport service.
pub struct TcpTransport {
    config: TcpConfig,
    metrics: MetricsCollector,
    active_connections: Arc<RwLock<usize>>,
}

impl TcpTransport {
    pub fn new(config: TcpConfig) -> Self {
        Self {
            config,
            metrics: MetricsCollector::new("tcp-transport".to_string()),
            active_connections: Arc::new(RwLock::new(0)),
        }
    }

    /// Start the TCP transport server.
    pub async fn serve(&self) -> Result<(), TcpError> {
        let listener = TcpListener::bind(&self.config.bind_address)
            .await
            .map_err(|e| TcpError::BindFailed(e.to_string()))?;

        info!("TCP transport listening on {}", self.config.bind_address);

        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    info!("Accepted connection from {}", addr);
                    
                    let transport = self.clone_for_handler();
                    tokio::spawn(async move {
                        if let Err(e) = transport.handle_connection(stream).await {
                            error!("Connection handler error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                }
            }
        }
    }

    /// Handle an incoming connection.
    async fn handle_connection(&self, mut stream: TcpStream) -> Result<(), TcpError> {
        // Increment connection count
        {
            let mut count = self.active_connections.write().await;
            *count += 1;
        }

        // Read message header
        let header = self.read_header(&mut stream).await?;

        match header.message_type {
            MessageType::TensorTransfer => {
                self.handle_tensor_transfer(&mut stream, &header).await?;
            }
            MessageType::Ping => {
                self.handle_ping(&mut stream).await?;
            }
            MessageType::Ack => {
                // Acknowledgment - no action needed
            }
        }

        // Decrement connection count
        {
            let mut count = self.active_connections.write().await;
            *count = count.saturating_sub(1);
        }

        Ok(())
    }

    /// Read message header from stream.
    async fn read_header(&self, stream: &mut TcpStream) -> Result<MessageHeader, TcpError> {
        let mut header_buf = vec![0u8; 24]; // Fixed header size
        stream.read_exact(&mut header_buf).await?;

        let message_type = MessageType::from_u8(header_buf[0])?;
        let tensor_id_bytes = &header_buf[1..17];
        let tensor_id = String::from_utf8_lossy(tensor_id_bytes).trim_end_matches('\0').to_string();
        
        let payload_length = u64::from_be_bytes([
            header_buf[17], header_buf[18], header_buf[19], header_buf[20],
            header_buf[21], header_buf[22], header_buf[23], 0,
        ]);

        Ok(MessageHeader {
            message_type,
            tensor_id,
            payload_length,
        })
    }

    /// Handle tensor transfer.
    async fn handle_tensor_transfer(
        &self,
        stream: &mut TcpStream,
        header: &MessageHeader,
    ) -> Result<(), TcpError> {
        let ctx = LogContext::new("handle_tensor_transfer".to_string())
            .with_session_id(header.tensor_id.clone());

        info!("Receiving tensor: {} ({} bytes)", header.tensor_id, header.payload_length);

        let mut buffer = vec![0u8; self.config.buffer_size];
        let mut total_received = 0u64;

        while total_received < header.payload_length {
            let remaining = (header.payload_length - total_received) as usize;
            let to_read = buffer.len().min(remaining);
            
            let n = stream.read(&mut buffer[..to_read]).await?;
            if n == 0 {
                return Err(TcpError::ConnectionClosed);
            }

            total_received += n as u64;

            // In a real implementation, we would:
            // 1. Write the data to the appropriate tensor location
            // 2. Verify checksums
            // 3. Update the tensor directory
        }

        self.metrics.increment_counter("tensor_bytes_received", total_received, &[]);
        ctx.info(&format!("Tensor {} received successfully", header.tensor_id));

        // Send acknowledgment
        self.send_ack(stream, true).await?;

        Ok(())
    }

    /// Handle ping message.
    async fn handle_ping(&self, stream: &mut TcpStream) -> Result<(), TcpError> {
        // Send pong response
        let header = MessageHeader {
            message_type: MessageType::Ack,
            tensor_id: String::new(),
            payload_length: 0,
        };
        
        self.write_header(stream, &header).await?;
        Ok(())
    }

    /// Send acknowledgment.
    async fn send_ack(&self, stream: &mut TcpStream, success: bool) -> Result<(), TcpError> {
        let ack_data = if success { [1u8] } else { [0u8] };
        stream.write_all(&ack_data).await?;
        Ok(())
    }

    /// Write message header to stream.
    async fn write_header(&self, stream: &mut TcpStream, header: &MessageHeader) -> Result<(), TcpError> {
        let mut header_buf = [0u8; 24];
        header_buf[0] = header.message_type.as_u8();
        
        let tensor_id_bytes = header.tensor_id.as_bytes();
        header_buf[1..1+tensor_id_bytes.len().min(16)].copy_from_slice(
            &tensor_id_bytes[..tensor_id_bytes.len().min(16)]
        );

        let length_bytes = header.payload_length.to_be_bytes();
        header_buf[17..25].copy_from_slice(&length_bytes[..7]);

        stream.write_all(&header_buf).await?;
        Ok(())
    }

    /// Connect to a remote endpoint.
    pub async fn connect(&self, addr: &str) -> Result<TcpClient, TcpError> {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|e| TcpError::ConnectFailed(e.to_string()))?;

        info!("Connected to {}", addr);

        Ok(TcpClient {
            stream,
            buffer_size: self.config.buffer_size,
        })
    }

    /// Clone for handler (shares metrics and config).
    fn clone_for_handler(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics: self.metrics.clone(),
            active_connections: self.active_connections.clone(),
        }
    }

    /// Get transport statistics.
    pub async fn get_stats(&self) -> TransportStats {
        let active_connections = *self.active_connections.read().await;
        
        TransportStats {
            active_connections,
            max_connections: self.config.max_connections,
        }
    }
}

/// TCP client for sending data.
pub struct TcpClient {
    stream: TcpStream,
    buffer_size: usize,
}

impl TcpClient {
    /// Send a tensor to the remote endpoint.
    pub async fn send_tensor(&mut self, tensor_id: &str, data: Bytes) -> Result<(), TcpError> {
        let header = MessageHeader {
            message_type: MessageType::TensorTransfer,
            tensor_id: tensor_id.to_string(),
            payload_length: data.len() as u64,
        };

        // Write header
        self.write_header(&header).await?;

        // Write data
        self.stream.write_all(&data).await?;

        // Wait for acknowledgment
        let mut ack_buf = [0u8; 1];
        self.stream.read_exact(&mut ack_buf).await?;

        if ack_buf[0] != 1 {
            return Err(TcpError::TransferFailed("Remote acknowledged failure".to_string()));
        }

        Ok(())
    }

    /// Send a ping message.
    pub async fn ping(&mut self) -> Result<(), TcpError> {
        let header = MessageHeader {
            message_type: MessageType::Ping,
            tensor_id: String::new(),
            payload_length: 0,
        };

        self.write_header(&header).await?;

        // Wait for pong
        let mut header_buf = vec![0u8; 24];
        self.stream.read_exact(&mut header_buf).await?;

        Ok(())
    }

    /// Write message header.
    async fn write_header(&mut self, header: &MessageHeader) -> Result<(), TcpError> {
        let mut header_buf = [0u8; 24];
        header_buf[0] = header.message_type.as_u8();
        
        let tensor_id_bytes = header.tensor_id.as_bytes();
        header_buf[1..1+tensor_id_bytes.len().min(16)].copy_from_slice(
            &tensor_id_bytes[..tensor_id_bytes.len().min(16)]
        );

        let length_bytes = header.payload_length.to_be_bytes();
        header_buf[17..25].copy_from_slice(&length_bytes[..7]);

        self.stream.write_all(&header_buf).await?;
        Ok(())
    }
}

/// Message header.
#[derive(Debug, Clone)]
struct MessageHeader {
    message_type: MessageType,
    tensor_id: String,
    payload_length: u64,
}

/// Message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageType {
    TensorTransfer,
    Ping,
    Ack,
}

impl MessageType {
    fn from_u8(value: u8) -> Result<Self, TcpError> {
        match value {
            0 => Ok(MessageType::TensorTransfer),
            1 => Ok(MessageType::Ping),
            2 => Ok(MessageType::Ack),
            _ => Err(TcpError::InvalidMessage(value)),
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            MessageType::TensorTransfer => 0,
            MessageType::Ping => 1,
            MessageType::Ack => 2,
        }
    }
}

/// Transport statistics.
#[derive(Debug, Clone)]
pub struct TransportStats {
    pub active_connections: usize,
    pub max_connections: usize,
}

/// TCP transport errors.
#[derive(Debug, thiserror::Error)]
pub enum TcpError {
    #[error("Bind failed: {0}")]
    BindFailed(String),
    
    #[error("Connect failed: {0}")]
    ConnectFailed(String),
    
    #[error("Connection closed")]
    ConnectionClosed,
    
    #[error("Transfer failed: {0}")]
    TransferFailed(String),
    
    #[error("Invalid message type: {0}")]
    InvalidMessage(u8),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tcp_config_default() {
        let config = TcpConfig::default();
        assert_eq!(config.bind_address, "0.0.0.0:0");
        assert_eq!(config.max_connections, 100);
    }

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(MessageType::from_u8(0).unwrap(), MessageType::TensorTransfer);
        assert_eq!(MessageType::from_u8(1).unwrap(), MessageType::Ping);
        assert_eq!(MessageType::from_u8(2).unwrap(), MessageType::Ack);
        assert!(MessageType::from_u8(99).is_err());
        
        assert_eq!(MessageType::TensorTransfer.as_u8(), 0);
        assert_eq!(MessageType::Ping.as_u8(), 1);
        assert_eq!(MessageType::Ack.as_u8(), 2);
    }

    #[test]
    fn test_message_header() {
        let header = MessageHeader {
            message_type: MessageType::TensorTransfer,
            tensor_id: "test_tensor".to_string(),
            payload_length: 1024,
        };

        assert_eq!(header.message_type, MessageType::TensorTransfer);
        assert_eq!(header.tensor_id, "test_tensor");
        assert_eq!(header.payload_length, 1024);
    }
}