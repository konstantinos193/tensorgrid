//! HTTP client for communicating with the coordinator.

use cluster_types::{NodeId, NodeResources, NodeCapabilities};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Serialize)]
pub struct RegisterRequest {
    pub node_name: String,
    pub hostname: String,
    pub cluster_id: String,
    pub capabilities: NodeCapabilities,
    pub resources: NodeResources,
}

#[derive(Deserialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub node_id: String,
    pub error_message: Option<String>,
}

#[derive(Serialize)]
struct HeartbeatRequest {
    sequence: u64,
    resources: NodeResources,
}

#[derive(Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub error_message: Option<String>,
}

/// Client for coordinator communication.
pub struct CoordinatorClient {
    base_url: String,
    http_client: reqwest::Client,
}

impl CoordinatorClient {
    /// Connect to the coordinator.
    pub async fn connect(addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Connecting to coordinator at {}", addr);
        
        // Convert gRPC address to HTTP if needed
        let base_url = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.to_string()
        } else {
            format!("http://{}", addr)
        };

        Ok(Self {
            base_url,
            http_client: reqwest::Client::new(),
        })
    }

    /// Register with the coordinator.
    pub async fn register(
        &mut self,
        request: RegisterRequest,
    ) -> Result<RegisterResponse, Box<dyn std::error::Error>> {
        let url = format!("{}/api/nodes/register", self.base_url);
        
        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Parse failed: {}", e))?;
        
        Ok(response)
    }

    /// Send heartbeat to coordinator.
    pub async fn send_heartbeat(
        &mut self,
        node_id: NodeId,
        sequence: u64,
        resources: NodeResources,
    ) -> Result<HeartbeatResponse, Box<dyn std::error::Error>> {
        let url = format!("{}/api/nodes/{}/heartbeat", self.base_url, node_id);
        
        let request = HeartbeatRequest {
            sequence,
            resources,
        };
        
        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Parse failed: {}", e))?;
        
        Ok(response)
    }
}
