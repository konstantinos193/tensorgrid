//! HTTP client for communicating with the coordinator API.

use serde::{Deserialize, Serialize};

/// Cluster information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub id: String,
    pub name: String,
    pub logical_resources: LogicalResources,
    pub node_count: usize,
}

/// Logical resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalResources {
    pub cpu_compute_units: f32,
    pub host_memory_bytes: u64,
    pub device_memory_bytes: u64,
    pub storage_bytes: u64,
    pub preferred_parallelism: String,
}

/// Node information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub name: String,
    pub hostname: String,
    pub status: String,
    pub capabilities: NodeCapabilities,
    pub resources: NodeResources,
}

/// Node capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub cpu_cores: u32,
    pub gpu_count: usize,
    pub memory_bytes: u64,
}

/// Node resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResources {
    pub cpu_usage_percent: f32,
    pub memory_used_bytes: u64,
}

/// API client for the coordinator.
pub struct ClusterClient {
    base_url: String,
    http_client: reqwest::Client,
}

impl ClusterClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            http_client: reqwest::Client::new(),
        }
    }

    /// Get cluster information.
    pub async fn get_cluster(&self) -> Result<ClusterInfo, String> {
        let url = format!("{}/api/cluster", self.base_url);
        
        self.http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse failed: {}", e))
    }

    /// Get all nodes.
    pub async fn get_nodes(&self) -> Result<Vec<NodeInfo>, String> {
        let url = format!("{}/api/cluster/nodes", self.base_url);
        
        let response: serde_json::Value = self.http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse failed: {}", e))?;

        serde_json::from_value(response["nodes"].clone())
            .map_err(|e| format!("Parse failed: {}", e))
    }

    /// Get list of models.
    pub async fn get_models(&self) -> Result<Vec<ModelInfo>, String> {
        let url = format!("{}/api/v1/models", self.base_url);
        
        let response: serde_json::Value = self.http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse failed: {}", e))?;

        serde_json::from_value(response["data"].clone())
            .map_err(|e| format!("Parse failed: {}", e))
    }
}

/// Model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}
