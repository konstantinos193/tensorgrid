//! Resource management for the node agent.

use cluster_types::{NodeCapabilities, NodeResources};
use hardware_probe::HardwareProbe;
use std::sync::Arc;
use tokio::sync::RwLock;
use sysinfo::System;

/// Manager for node resources.
pub struct ResourceManager {
    capabilities: NodeCapabilities,
    system: Arc<RwLock<System>>,
}

impl ResourceManager {
    pub fn new() -> Self {
        let capabilities = HardwareProbe::probe_all()
            .unwrap_or_else(|_| Self::default_capabilities());
        
        let mut system = System::new_all();
        system.refresh_all();
        
        Self {
            capabilities,
            system: Arc::new(RwLock::new(system)),
        }
    }

    fn default_capabilities() -> NodeCapabilities {
        NodeCapabilities {
            cpu: cluster_types::CpuCapabilities {
                architecture: "unknown".to_string(),
                cores: 4,
                threads: 8,
                frequency_mhz: 3000,
                features: vec![],
                numa_nodes: 1,
            },
            gpus: vec![],
            memory: cluster_types::MemoryCapabilities {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 16 * 1024 * 1024 * 1024,
                bandwidth_mbps: 25000,
            },
            storage: cluster_types::StorageCapabilities {
                total_bytes: 1024 * 1024 * 1024 * 1024,
                available_bytes: 1024 * 1024 * 1024 * 1024,
                read_throughput_mbps: 3500,
                write_throughput_mbps: 3000,
            },
            network: cluster_types::NetworkCapabilities {
                interfaces: vec![],
            },
            supported_runtimes: vec![cluster_types::RuntimeBackend::GGML, cluster_types::RuntimeBackend::CPU],
        }
    }

    /// Get current resource usage.
    pub async fn get_current_resources(&self) -> NodeResources {
        let mut sys = self.system.write().await;
        sys.refresh_all();

        let total_memory = sys.total_memory();
        let used_memory = sys.used_memory();
        let available_memory = sys.available_memory();

        // Calculate CPU usage
        let cpu_usage = sys.global_cpu_info().cpu_usage();

        // Get GPU usage (placeholder - in real implementation would query GPU)
        let gpu_usage = vec![];

        // Get network stats (placeholder)
        let network_tx_mbps = 0.0;
        let network_rx_mbps = 0.0;

        // Get temperature (placeholder)
        let temperature_celsius = None;

        NodeResources {
            cpu_usage_percent: cpu_usage as f32,
            memory_used_bytes: used_memory,
            memory_committed_bytes: used_memory,
            gpu_usage,
            network_tx_mbps,
            network_rx_mbps,
            temperature_celsius,
        }
    }

    /// Get node capabilities.
    pub fn capabilities(&self) -> &NodeCapabilities {
        &self.capabilities
    }
}
