//! Cluster Coordinator - main control plane service.
//!
//! The coordinator manages the authoritative cluster state, node registration,
//! model planning, and exposes the API/UI endpoints.

use cluster_types::{ClusterId, NodeId, PhysicalNode, NodeStatus, LogicalCluster};
use secure_pairing::{PairingManager, PairingRequest, PairingChallenge, PairingResponse};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::Server;
use tracing::{info, error, warn};
use uuid::Uuid;

mod node_control;
mod api;
mod planner;

use node_control::NodeControlService;
use api::ApiServer;

/// Cluster state managed by the coordinator.
#[derive(Clone)]
struct ClusterState {
    cluster_id: ClusterId,
    nodes: Arc<RwLock<HashMap<NodeId, PhysicalNode>>>,
    pairing_manager: Arc<RwLock<PairingManager>>,
}

impl ClusterState {
    fn new() -> Self {
        let cluster_id = Uuid::new_v4();
        let pairing_manager = PairingManager::new_coordinator(cluster_id)
            .expect("Failed to create pairing manager");

        Self {
            cluster_id,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            pairing_manager: Arc::new(RwLock::new(pairing_manager)),
        }
    }

    async fn get_logical_cluster(&self) -> LogicalCluster {
        let nodes = self.nodes.read().await;
        let node_map = nodes.clone();
        
        // Calculate logical resources from all nodes
        let mut cpu_compute_units = 0.0;
        let mut host_memory_bytes = 0u64;
        let mut device_memory_bytes = 0u64;
        let mut storage_bytes = 0u64;

        for node in node_map.values() {
            cpu_compute_units += node.capabilities.cpu.cores as f32;
            host_memory_bytes += node.capabilities.memory.available_bytes;
            for gpu in &node.capabilities.gpus {
                device_memory_bytes += gpu.vram_bytes;
            }
            storage_bytes += node.capabilities.storage.available_bytes;
        }

        // Apply safety margins (20% reserve)
        host_memory_bytes = (host_memory_bytes as f64 * 0.8) as u64;
        device_memory_bytes = (device_memory_bytes as f64 * 0.85) as u64;

        LogicalCluster {
            id: self.cluster_id,
            name: "local-cluster".to_string(),
            logical_resources: cluster_types::LogicalResources {
                cpu_compute_units,
                host_memory_bytes,
                device_memory_bytes,
                storage_bytes,
                preferred_parallelism: cluster_types::ParallelismStrategy::PipelineHybrid,
            },
            nodes: node_map,
            topology: cluster_types::ClusterTopology {
                nodes: HashMap::new(),
                links: vec![],
            },
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();

    info!("Starting Cluster Coordinator");

    let state = ClusterState::new();
    let node_control = NodeControlService::new(state.clone());
    let api_server = ApiServer::new(state.clone());

    // Start heartbeat monitoring for node failure detection
    let state_for_monitor = state.clone();
    tokio::spawn(async move {
        heartbeat_monitor(state_for_monitor).await;
    });

    // Start gRPC server for node control
    let addr = "[::1]:50051".parse()?;
    let node_control_server = Server::builder()
        .add_service(node_control.into_server())
        .serve(addr);

    info!("Node control server listening on {}", addr);

    // Start API server (HTTP/REST)
    let api_addr = "[::1]:8080".parse()?;
    let api_handle = tokio::spawn(async move {
        if let Err(e) = api_server.serve(api_addr).await {
            error!("API server error: {}", e);
        }
    });

    // Run gRPC server
    if let Err(e) = node_control_server.await {
        error!("gRPC server error: {}", e);
    }

    api_handle.await??;

    Ok(())
}

/// Heartbeat monitor for detecting node failures.
async fn heartbeat_monitor(state: ClusterState) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    
    loop {
        interval.tick().await;
        
        let now = chrono::Utc::now();
        let mut nodes = state.nodes.write().await;
        let mut nodes_to_remove = Vec::new();
        
        for (node_id, node) in nodes.iter_mut() {
            let time_since_heartbeat = now.signed_duration_since(node.last_heartbeat);
            
            // Mark node as suspect if no heartbeat for 10 seconds
            if time_since_heartbeat.num_seconds() > 10 && node.status == NodeStatus::Healthy {
                warn!("Node {} has not sent heartbeat for {}s, marking as suspect", 
                    node_id, time_since_heartbeat.num_seconds());
                node.status = NodeStatus::Suspect;
            }
            
            // Mark node as offline if no heartbeat for 30 seconds
            if time_since_heartbeat.num_seconds() > 30 {
                warn!("Node {} has not sent heartbeat for {}s, marking as offline", 
                    node_id, time_since_heartbeat.num_seconds());
                node.status = NodeStatus::Offline;
                nodes_to_remove.push(*node_id);
            }
        }
        
        // Remove offline nodes from active cluster
        for node_id in nodes_to_remove {
            info!("Removing offline node {} from cluster", node_id);
            nodes.remove(&node_id);
        }
    }
}
