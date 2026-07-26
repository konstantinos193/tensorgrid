//! gRPC client for communicating with the coordinator.

use cluster_types::{NodeId, NodeResources};
use std::collections::HashMap;
use tonic::transport::Channel;
use tracing::{info, error};

// Include the generated protobuf code
pub mod control {
    include!("../proto/cluster.control.rs");
}

use control::node_control_client::NodeControlClient;
use control::{
    RegisterRequest, RegisterResponse,
    HeartbeatRequest, HeartbeatResponse,
    NodeResources as ProtoNodeResources,
    GpuUsage as ProtoGpuUsage,
    HealthCheck,
};

/// Client for coordinator communication.
pub struct CoordinatorClient {
    client: NodeControlClient<Channel>,
}

impl CoordinatorClient {
    /// Connect to the coordinator.
    pub async fn connect(addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Connecting to coordinator at {}", addr);
        
        let channel = Channel::from_static(addr).connect().await?;
        let client = NodeControlClient::new(channel);
        
        Ok(Self { client })
    }

    /// Register with the coordinator.
    pub async fn register(
        &mut self,
        request: RegisterRequest,
    ) -> Result<RegisterResponse, Box<dyn std::error::Error>> {
        let response = self.client.register(request).await?;
        Ok(response.into_inner())
    }

    /// Send heartbeat to coordinator.
    pub async fn send_heartbeat(
        &mut self,
        node_id: NodeId,
        sequence: u64,
        resources: NodeResources,
    ) -> Result<HeartbeatResponse, Box<dyn std::error::Error>> {
        let request = HeartbeatRequest {
            node_id: node_id.to_string(),
            sequence,
            resources: Some(convert_node_resources(&resources)),
            health_checks: vec![],
        };

        let response = self.client.heartbeat(request).await?;
        Ok(response.into_inner())
    }
}

fn convert_node_resources(resources: &NodeResources) -> ProtoNodeResources {
    ProtoNodeResources {
        cpu_usage_percent: resources.cpu_usage_percent,
        memory_used_bytes: resources.memory_used_bytes,
        memory_committed_bytes: resources.memory_committed_bytes,
        gpu_usage: resources.gpu_usage.iter().map(|gpu| ProtoGpuUsage {
            device_id: gpu.device_id.clone(),
            utilization_percent: gpu.utilization_percent,
            vram_used_bytes: gpu.vram_used_bytes,
            temperature_celsius: gpu.temperature_celsius,
            power_draw_watts: gpu.power_draw_watts.unwrap_or(0.0),
        }).collect(),
        network_tx_mbps: resources.network_tx_mbps,
        network_rx_mbps: resources.network_rx_mbps,
        temperature_celsius: resources.temperature_celsius.unwrap_or(0.0),
    }
}
