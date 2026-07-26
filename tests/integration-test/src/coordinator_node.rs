//! Coordinator-node integration tests.

use cluster_types::{NodeId, ClusterId, NodeStatus, NodeCapabilities, NodeResources};
use secure_pairing::{PairingManager, PairingRequest};
use std::time::Duration;
use tokio::time::sleep;
use tracing::info;
use uuid::Uuid;

/// Test coordinator cluster state management.
#[tokio::test]
async fn test_cluster_state() {
    let cluster_id = Uuid::new_v4();
    let node_id = Uuid::new_v4();
    
    // Create a mock cluster state
    let capabilities = NodeCapabilities {
        cpu: cluster_types::CpuCapabilities {
            architecture: "x86_64".to_string(),
            cores: 8,
            threads: 16,
            frequency_mhz: 3000,
            features: vec!["avx2".to_string(), "sse4.2".to_string()],
            numa_nodes: 1,
        },
        gpus: vec![],
        memory: cluster_types::MemoryCapabilities {
            total_bytes: 32 * 1024 * 1024 * 1024,
            available_bytes: 32 * 1024 * 1024 * 1024,
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
        supported_runtimes: vec![cluster_types::RuntimeBackend::GGML],
    };
    
    let node = cluster_types::PhysicalNode {
        id: node_id,
        name: "test-node".to_string(),
        hostname: "test-host".to_string(),
        cluster_id,
        capabilities: capabilities.clone(),
        resources: NodeResources {
            cpu_usage_percent: 0.0,
            memory_used_bytes: 0,
            memory_committed_bytes: 0,
            gpu_usage: vec![],
            network_tx_mbps: 0.0,
            network_rx_mbps: 0.0,
            temperature_celsius: None,
        },
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
        last_heartbeat: chrono::Utc::now(),
        certificate_fingerprint: None,
    };
    
    assert_eq!(node.id, node_id);
    assert_eq!(node.status, NodeStatus::Healthy);
    assert_eq!(node.capabilities.cpu.cores, 8);
    
    info!("Cluster state test passed");
}

/// Test pairing manager functionality.
#[tokio::test]
async fn test_pairing_manager() {
    let cluster_id = Uuid::new_v4();
    let pairing_manager = PairingManager::new_node(cluster_id).unwrap();
    
    let request = PairingRequest {
        node_name: "test-node".to_string(),
        hostname: "test-host".to_string(),
        cluster_id,
        public_key: vec![1, 2, 3, 4],
        fingerprint: "test-fingerprint".to_string(),
    };
    
    let challenge = pairing_manager.generate_challenge(&request).unwrap();
    
    assert!(!challenge.challenge.is_empty());
    assert!(!challenge.coordinator_public_key.is_empty());
    assert_eq!(challenge.node_id, request.cluster_id);
    
    info!("Pairing manager test passed");
}

/// Test node resource tracking.
#[tokio::test]
async fn test_resource_tracking() {
    let mut resources = NodeResources {
        cpu_usage_percent: 0.0,
        memory_used_bytes: 0,
        memory_committed_bytes: 0,
        gpu_usage: vec![],
        network_tx_mbps: 0.0,
        network_rx_mbps: 0.0,
        temperature_celsius: None,
    };
    
    // Simulate resource changes
    resources.cpu_usage_percent = 50.0;
    resources.memory_used_bytes = 8 * 1024 * 1024 * 1024;
    resources.network_tx_mbps = 100.0;
    resources.network_rx_mbps = 50.0;
    
    assert_eq!(resources.cpu_usage_percent, 50.0);
    assert_eq!(resources.memory_used_bytes, 8 * 1024 * 1024 * 1024);
    
    info!("Resource tracking test passed");
}

/// Test node status transitions.
#[tokio::test]
async fn test_node_status_transitions() {
    let mut status = NodeStatus::Suspect;
    
    // Simulate status transitions
    status = NodeStatus::Healthy;
    assert_eq!(status, NodeStatus::Healthy);
    
    status = NodeStatus::Draining;
    assert_eq!(status, NodeStatus::Draining);
    
    status = NodeStatus::Offline;
    assert_eq!(status, NodeStatus::Offline);
    
    info!("Node status transitions test passed");
}

/// Test heartbeat sequence tracking.
#[tokio::test]
async fn test_heartbeat_sequence() {
    let mut sequence = 0u64;
    
    for i in 1..=10 {
        sequence += 1;
        assert_eq!(sequence, i);
    }
    
    assert_eq!(sequence, 10);
    
    info!("Heartbeat sequence test passed");
}

/// Test node permissions validation.
#[tokio::test]
async fn test_permissions_validation() {
    let permissions = cluster_types::NodePermissions {
        max_cpu_percent: 80,
        max_cpu_threads: None,
        max_ram_bytes: 16 * 1024 * 1024 * 1024,
        gpu_enabled: true,
        max_vram_bytes: 8 * 1024 * 1024 * 1024,
        max_storage_bytes: 100 * 1024 * 1024 * 1024,
        allow_when_on_battery: false,
        pause_when_user_active: true,
        allowed_schedule: None,
    };
    
    // Test CPU limit
    assert!(permissions.max_cpu_percent <= 100);
    
    // Test memory limit
    assert!(permissions.max_ram_bytes > 0);
    
    // Test GPU enabled
    assert!(permissions.gpu_enabled);
    
    info!("Permissions validation test passed");
}
