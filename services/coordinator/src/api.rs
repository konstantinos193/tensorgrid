//! HTTP/REST API server for cluster management.

use crate::ClusterState;
use axum::{
    routing::{get, post},
    Router,
    Json,
    response::IntoResponse,
};
use cluster_types::{PhysicalNode, NodeCapabilities, NodeResources, NodeStatus};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;
use chrono::Utc;

pub struct ApiServer {
    state: ClusterState,
}

impl ApiServer {
    pub fn new(state: ClusterState) -> Self {
        Self { state }
    }

    pub async fn serve(self, addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
        let state = Arc::new(self.state);

        let app = Router::new()
            .route("/api/cluster", get(get_cluster))
            .route("/api/cluster/nodes", get(get_nodes))
            .route("/api/v1/models", get(list_models))
            .route("/api/nodes/register", post(register_node))
            .route("/api/nodes/:node_id/heartbeat", post(send_heartbeat))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!("API server listening on http://{}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn get_cluster(
    axum::extract::State(state): axum::extract::State<Arc<ClusterState>>,
) -> impl IntoResponse {
    let cluster = state.get_logical_cluster().await;
    Json(cluster)
}

async fn get_nodes(
    axum::extract::State(state): axum::extract::State<Arc<ClusterState>>,
) -> impl IntoResponse {
    let nodes = state.nodes.read().await;
    Json((*nodes).clone())
}

async fn list_models() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": []
    }))
}

#[derive(Deserialize)]
struct RegisterNodeRequest {
    node_name: String,
    hostname: String,
    cluster_id: String,
    capabilities: NodeCapabilities,
    resources: NodeResources,
}

#[derive(Serialize)]
struct RegisterNodeResponse {
    success: bool,
    node_id: String,
    error_message: Option<String>,
}

async fn register_node(
    axum::extract::State(state): axum::extract::State<Arc<ClusterState>>,
    axum::extract::Path(_): axum::extract::Path<()>,
    Json(req): Json<RegisterNodeRequest>,
) -> impl IntoResponse {
    let node_id = Uuid::new_v4();
    let cluster_id = Uuid::parse_str(&req.cluster_id).unwrap_or_else(|_| state.cluster_id);
    
    let node = PhysicalNode {
        id: node_id,
        name: req.node_name,
        hostname: req.hostname,
        cluster_id,
        capabilities: req.capabilities,
        resources: req.resources,
        permissions: cluster_types::NodePermissions {
            max_cpu_percent: 80,
            max_cpu_threads: None,
            max_ram_bytes: 16 * 1024 * 1024 * 1024,
            gpu_enabled: true,
            max_vram_bytes: 8 * 1024 * 1024 * 1024,
            max_storage_bytes: 100 * 1024 * 1024 * 1024,
            allow_when_on_battery: false,
            pause_when_user_active: true,
            allowed_schedule: None,
        },
        status: NodeStatus::Healthy,
        last_heartbeat: Utc::now(),
        certificate_fingerprint: None,
    };
    
    let mut nodes = state.nodes.write().await;
    nodes.insert(node_id, node.clone());
    
    info!("Node registered: {} ({})", node.name, node_id);
    
    Json(RegisterNodeResponse {
        success: true,
        node_id: node_id.to_string(),
        error_message: None,
    })
}

#[derive(Deserialize)]
struct HeartbeatRequest {
    sequence: u64,
    resources: NodeResources,
}

#[derive(Serialize)]
struct HeartbeatResponse {
    success: bool,
    error_message: Option<String>,
}

async fn send_heartbeat(
    axum::extract::State(state): axum::extract::State<Arc<ClusterState>>,
    axum::extract::Path(node_id): axum::extract::Path<String>,
    Json(req): Json<HeartbeatRequest>,
) -> impl IntoResponse {
    let node_uuid = match Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => {
            return Json(HeartbeatResponse {
                success: false,
                error_message: Some("Invalid node ID".to_string()),
            });
        }
    };
    
    let mut nodes = state.nodes.write().await;
    if let Some(node) = nodes.get_mut(&node_uuid) {
        node.last_heartbeat = Utc::now();
        node.resources = req.resources;
        node.status = NodeStatus::Healthy;
        
        Json(HeartbeatResponse {
            success: true,
            error_message: None,
        })
    } else {
        Json(HeartbeatResponse {
            success: false,
            error_message: Some("Node not found".to_string()),
        })
    }
}