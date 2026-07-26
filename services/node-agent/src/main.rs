//! Node Agent - runs on each participating computer.
//!
//! The node agent handles hardware detection, resource management,
//! tensor allocation, and communicates with the coordinator.

use cluster_types::{NodeId, ClusterId, NodeCapabilities, NodeResources, NodePermissions};
use hardware_probe::HardwareProbe;
use secure_pairing::{PairingManager, PairingRequest, PairingChallenge, PairingResponse};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, error, warn};
use uuid::Uuid;

mod coordinator_client;
mod resource_manager;
mod runtime_adapter;

use coordinator_client::CoordinatorClient;
use resource_manager::ResourceManager;
use runtime_adapter::RuntimeAdapter;

/// Node agent state.
struct NodeAgent {
    node_id: Option<NodeId>,
    cluster_id: ClusterId,
    coordinator_client: Option<CoordinatorClient>,
    pairing_manager: Arc<RwLock<PairingManager>>,
    resource_manager: ResourceManager,
    runtime_adapter: RuntimeAdapter,
    permissions: NodePermissions,
}

impl NodeAgent {
    fn new(cluster_id: ClusterId) -> Self {
        let pairing_manager = PairingManager::new_node(cluster_id)
            .expect("Failed to create pairing manager");

        Self {
            node_id: None,
            cluster_id,
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
        
        let pairing_manager = self.pairing_manager.read().await;
        let public_key = pairing_manager.public_key();
        drop(pairing_manager);

        let hostname = gethostname::gethostname()
            .to_string_lossy()
            .to_string();

        let request = secure_pairing::PairingRequest {
            node_name: hostname.clone(),
            hostname,
            cluster_id: self.cluster_id,
            public_key: public_key.clone(),
            fingerprint: secure_pairing::generate_fingerprint(&public_key),
        };

        // Generate pairing challenge
        let challenge = pairing_manager.generate_challenge(&request)?;

        // Send registration request to coordinator
        let register_request = coordinator_client::control::RegisterRequest {
            node_name: request.node_name.clone(),
            hostname: request.hostname,
            cluster_id: self.cluster_id.to_string(),
            pairing_challenge: challenge.challenge.clone(),
            pairing_signature: public_key,
        };

        let response = client.register(register_request).await?;

        if !response.success {
            return Err(anyhow::anyhow!("Registration failed: {}", response.error_message));
        }

        let node_id = Uuid::parse_str(&response.node_id)?;
        self.node_id = Some(node_id);
        self.coordinator_client = Some(client);

        info!("Registered successfully with node ID: {}", node_id);
        Ok(())
    }

    async fn start_heartbeat_loop(&self) {
        let node_id = self.node_id.expect("Node ID not set");
        let cluster_id = self.cluster_id;
        let mut sequence = 0u64;
        let mut reconnect_attempts = 0u32;
        let max_reconnect_attempts = 5;

        loop {
            // Ensure we have a client
            if self.coordinator_client.is_none() {
                info!("No coordinator client, attempting to register...");
                if let Err(e) = self.register_with_retry(&self.coordinator_addr, cluster_id, reconnect_attempts).await {
                    error!("Registration failed: {}", e);
                    reconnect_attempts += 1;
                    if reconnect_attempts >= max_reconnect_attempts {
                        error!("Max reconnection attempts reached, giving up");
                        break;
                    }
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

    async fn register_with_retry(&mut self, coordinator_addr: &str, cluster_id: ClusterId, attempt: u32) -> Result<(), Box<dyn std::error::Error>> {
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
    let mut agent = NodeAgent::new(cluster_id);

    // Get coordinator address from environment or use default
    let coordinator_addr = std::env::var("COORDINATOR_ADDR")
        .unwrap_or_else(|_| "http://[::1]:50051".to_string());

    // Register with coordinator
    agent.register(&coordinator_addr).await?;

    // Start heartbeat loop
    let agent_arc = Arc::new(agent);
    let heartbeat_agent = agent_arc.clone();
    tokio::spawn(async move {
        heartbeat_agent.start_heartbeat_loop().await;
    });

    // Keep the agent running
    tokio::signal::ctrl_c().await?;
    info!("Shutting down");

    Ok(())
}
