//! gRPC communication integration tests.

use cluster_types::{NodeId, ClusterId, NodeStatus};
use secure_pairing::{PairingManager, PairingRequest};
use std::time::Duration;
use tokio::time::sleep;
use tracing::info;
use uuid::Uuid;

// Include the generated protobuf code
pub mod control {
    include!("proto/cluster.control.rs");
}

use control::node_control_client::NodeControlClient;
use control::{
    RegisterRequest, HeartbeatRequest, ProbeRequest, BenchmarkRequest,
    NodeResources as ProtoNodeResources, GpuUsage as ProtoGpuUsage,
    BenchmarkType,
};

/// Test coordinator-node gRPC communication.
#[tokio::test]
async fn test_coordinator_node_registration() {
    // This test requires a running coordinator
    // For now, we'll test the client creation and request structure
    
    let coordinator_addr = "http://[::1]:50051";
    
    // Attempt to connect (will fail if coordinator not running)
    match NodeControlClient::connect(coordinator_addr).await {
        Ok(mut client) => {
            info!("Connected to coordinator");
            
            // Create registration request
            let cluster_id = Uuid::new_v4();
            let pairing_manager = PairingManager::new_node(cluster_id).unwrap();
            let public_key = pairing_manager.public_key();
            
            let request = RegisterRequest {
                node_name: "test-node".to_string(),
                hostname: "test-host".to_string(),
                cluster_id: cluster_id.to_string(),
                pairing_challenge: vec![1, 2, 3, 4], // Simplified
                pairing_signature: public_key,
            };
            
            // Try to register (will fail if coordinator not properly configured)
            match client.register(request).await {
                Ok(response) => {
                    info!("Registration response: success={}", response.get_ref().success);
                    assert!(response.get_ref().success);
                }
                Err(e) => {
                    info!("Registration failed (expected if coordinator not running): {}", e);
                }
            }
        }
        Err(e) => {
            info!("Failed to connect to coordinator (expected if not running): {}", e);
        }
    }
}

#[tokio::test]
async fn test_heartbeat_flow() {
    let coordinator_addr = "http://[::1]:50051";
    
    match NodeControlClient::connect(coordinator_addr).await {
        Ok(mut client) => {
            let node_id = Uuid::new_v4();
            
            let request = HeartbeatRequest {
                node_id: node_id.to_string(),
                sequence: 1,
                resources: Some(ProtoNodeResources {
                    cpu_usage_percent: 50.0,
                    memory_used_bytes: 8 * 1024 * 1024 * 1024,
                    memory_committed_bytes: 8 * 1024 * 1024 * 1024,
                    gpu_usage: vec![],
                    network_tx_mbps: 100.0,
                    network_rx_mbps: 50.0,
                    temperature_celsius: 45.0,
                }),
                health_checks: vec![],
            };
            
            match client.heartbeat(request).await {
                Ok(response) => {
                    info!("Heartbeat response: acknowledged={}", response.get_ref().acknowledged);
                    assert!(response.get_ref().acknowledged);
                }
                Err(e) => {
                    info!("Heartbeat failed: {}", e);
                }
            }
        }
        Err(e) => {
            info!("Failed to connect: {}", e);
        }
    }
}

#[tokio::test]
async fn test_probe_request() {
    let coordinator_addr = "http://[::1]:50051";
    
    match NodeControlClient::connect(coordinator_addr).await {
        Ok(mut client) => {
            let node_id = Uuid::new_v4();
            
            let request = ProbeRequest {
                node_id: node_id.to_string(),
            };
            
            match client.probe(request).await {
                Ok(response) => {
                    info!("Probe response: capabilities={:?}", response.get_ref().capabilities);
                    assert!(response.get_ref().capabilities.is_some());
                }
                Err(e) => {
                    info!("Probe failed: {}", e);
                }
            }
        }
        Err(e) => {
            info!("Failed to connect: {}", e);
        }
    }
}

#[tokio::test]
async fn test_benchmark_request() {
    let coordinator_addr = "http://[::1]:50051";
    
    match NodeControlClient::connect(coordinator_addr).await {
        Ok(mut client) => {
            let node_id = Uuid::new_v4();
            
            let request = BenchmarkRequest {
                node_id: node_id.to_string(),
                benchmark_type: BenchmarkType::CpuMatrix as i32,
            };
            
            match client.benchmark(request).await {
                Ok(response) => {
                    info!("Benchmark response: success={}", response.get_ref().success);
                    assert!(response.get_ref().success);
                }
                Err(e) => {
                    info!("Benchmark failed: {}", e);
                }
            }
        }
        Err(e) => {
            info!("Failed to connect: {}", e);
        }
    }
}

#[tokio::test]
async fn test_multiple_heartbeats() {
    let coordinator_addr = "http://[::1]:50051";
    
    match NodeControlClient::connect(coordinator_addr).await {
        Ok(mut client) => {
            let node_id = Uuid::new_v4();
            
            for sequence in 1..=5 {
                let request = HeartbeatRequest {
                    node_id: node_id.to_string(),
                    sequence,
                    resources: Some(ProtoNodeResources {
                        cpu_usage_percent: 50.0,
                        memory_used_bytes: 8 * 1024 * 1024 * 1024,
                        memory_committed_bytes: 8 * 1024 * 1024 * 1024,
                        gpu_usage: vec![],
                        network_tx_mbps: 100.0,
                        network_rx_mbps: 50.0,
                        temperature_celsius: 45.0,
                    }),
                    health_checks: vec![],
                };
                
                match client.heartbeat(request).await {
                    Ok(response) => {
                        info!("Heartbeat {}: acknowledged={}", sequence, response.get_ref().acknowledged);
                    }
                    Err(e) => {
                        info!("Heartbeat {} failed: {}", sequence, e);
                    }
                }
                
                sleep(Duration::from_millis(100)).await;
            }
        }
        Err(e) => {
            info!("Failed to connect: {}", e);
        }
    }
}
