//! Tauri commands for the desktop UI.

use crate::cluster_client::{ClusterClient, ClusterInfo, NodeInfo, ModelInfo};
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

/// Application state.
pub struct AppState {
    client: Arc<RwLock<ClusterClient>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            client: Arc::new(RwLock::new(ClusterClient::new("http://[::1]:8080".to_string()))),
        }
    }
}

/// Get cluster information.
#[tauri::command]
async fn get_cluster(state: State<'_, AppState>) -> Result<ClusterInfo, String> {
    let client = state.client.read().await;
    client.get_cluster().await
}

/// Get all nodes.
#[tauri::command]
async fn get_nodes(state: State<'_, AppState>) -> Result<Vec<NodeInfo>, String> {
    let client = state.client.read().await;
    client.get_nodes().await
}

/// Get list of models.
#[tauri::command]
async fn get_models(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, String> {
    let client = state.client.read().await;
    client.get_models().await
}

/// Set coordinator URL.
#[tauri::command]
async fn set_coordinator_url(state: State<'_, AppState>, url: String) -> Result<(), String> {
    let mut client = state.client.write().await;
    *client = ClusterClient::new(url);
    Ok(())
}
