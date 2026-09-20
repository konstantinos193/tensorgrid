//! Node Agent - runs on each participating computer.
//!
//! The node agent handles hardware detection, resource management,
//! tensor allocation, and communicates with the coordinator.

use cluster_types::{NodeId, ClusterId, NodePermissions};
use secure_pairing::PairingManager;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, error};
use uuid::Uuid;
use coordinator_client::{CoordinatorClient, RegisterRequest};

mod coordinator_client;
mod resource_manager;
mod runtime_adapter;

use resource_manager::ResourceManager;
use runtime_adapter::RuntimeAdapter;

/// Node agent state.
struct NodeAgent {
    node_id: Option<NodeId>,
    cluster_id: ClusterId,
    coordinator_addr: String,
    coordinator_client: Option<CoordinatorClient>,
    pairing_manager: Arc<RwLock<PairingManager>>,
    resource_manager: ResourceManager,
    runtime_adapter: RuntimeAdapter,
    permissions: NodePermissions,
}

impl NodeAgent {
    fn new(cluster_id: ClusterId, coordinator_addr: String) -> Self {
        let pairing_manager = PairingManager::new_node(cluster_id)
            .expect("Failed to create pairing manager");

        Self {
            node_id: None,
            cluster_id,
            coordinator_addr,
            coordinator_client: None,
            pairing_manager: Arc::new(RwLock::new(pairing_manager)),
            resource_manager: ResourceManager::new(),
            runtime_adapter: RuntimeAdapter::new(),
            permissions: NodePermissions {
                max_cpu_percent: 80,
                max_cpu_threads: None,
                max_ram_bytes: 16 * 1024 * 1024 * 1024, // 16 GB default
                gpu_enabled: true,
                max_vram_bytes: 8 * 1024 * 1024 * 1024, // 8 GB default
                max_storage_bytes: 100 * 1024 * 1024 * 1024, // 100 GB default
                allow_when_on_battery: false,
                pause_when_user_active: true,
                allowed_schedule: None,
            },
        }
    }

    async fn register(&mut self, coordinator_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        info!("Registering with coordinator at {}", coordinator_addr);

        let mut client = CoordinatorClient::connect(coordinator_addr).await?;
        
        let hostname = gethostname::gethostname()
            .to_string_lossy()
            .to_string();

        // Probe hardware capabilities
        let capabilities = hardware_probe::HardwareProbe::probe_all()?;
        let resources = self.resource_manager.get_current_resources().await;

        let request = RegisterRequest {
            node_name: hostname.clone(),
            hostname,
            cluster_id: self.cluster_id.to_string(),
            capabilities,
            resources,
        };

        let response = client.register(request).await?;

        if !response.success {
            return Err(format!("Registration failed: {}", response.error_message.unwrap_or_else(|| "Unknown error".to_string())).into());
        }

        let node_id = Uuid::parse_str(&response.node_id)?;
        self.node_id = Some(node_id);
        self.coordinator_client = Some(client);

        info!("Registered successfully with node ID: {}", node_id);
        Ok(())
    }

    async fn start_heartbeat_loop(&mut self) {
        let node_id = self.node_id.expect("Node ID not set");
        let mut sequence = 0u64;
        let mut reconnect_attempts = 0u32;
        let max_reconnect_attempts = 5;

        loop {
            // Ensure we have a client
            if self.coordinator_client.is_none() {
                let coordinator_addr = self.coordinator_addr.clone();
                info!("No coordinator client, attempting to register...");
                if let Err(e) = self.register_with_retry(&coordinator_addr, reconnect_attempts).await {
                    error!("Registration failed: {}", e);
                    reconnect_attempts += 1;
                    if reconnect_attempts >= max_reconnect_attempts {
                        error!("Max reconnection attempts reached, giving up");
                        break;
                    }
                    drop(e); // Drop error before await to fix Send trait
                    tokio::time::sleep(Duration::from_secs(5 * reconnect_attempts as u64)).await;
                    continue;
                }
                reconnect_attempts = 0;
            }

            // Send heartbeat
            if let Some(ref mut client) = self.coordinator_client {
                let resources = self.resource_manager.get_current_resources().await;
                
                match client.send_heartbeat(node_id, sequence, resources).await {
                    Ok(_) => {
                        sequence += 1;
                        reconnect_attempts = 0; // Reset on success
                    }
                    Err(e) => {
                        error!("Heartbeat failed: {}", e);
                        // Clear client to force reconnection
                        self.coordinator_client = None;
                        reconnect_attempts += 1;
                        if reconnect_attempts >= max_reconnect_attempts {
                            error!("Max reconnection attempts reached, giving up");
                            break;
                        }
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    async fn register_with_retry(&mut self, coordinator_addr: &str, attempt: u32) -> Result<(), Box<dyn std::error::Error>> {
        info!("Attempting registration (attempt {})", attempt + 1);
        self.register(coordinator_addr).await
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();

    info!("Starting Node Agent");

    let cluster_id = Uuid::new_v4();
    
    // Get coordinator address from environment or use default
    let coordinator_addr = std::env::var("COORDINATOR_ADDR")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    
    let mut agent = NodeAgent::new(cluster_id, coordinator_addr.clone());

    // Register with coordinator
    agent.register(&coordinator_addr).await?;

    // Start heartbeat loop
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            agent.start_heartbeat_loop().await;
        });
    });

    // Keep the agent running
    tokio::signal::ctrl_c().await?;
    info!("Shutting down");

    Ok(())
}
