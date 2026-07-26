//! gRPC service for node control communication.

use cluster_types::{NodeId, ClusterId, NodeStatus, NodeCapabilities, NodeResources};
use crate::ClusterState;
use std::collections::HashMap;
use std::time::Duration;
use tonic::{Request, Response, Status};
use tracing::{info, warn, error};
use tokio::time::interval;

// Include the generated protobuf code
pub mod control {
    include!("../proto/cluster.control.rs");
}

use control::{
    node_control_server::{NodeControl, NodeControlServer},
    RegisterRequest, RegisterResponse,
    HeartbeatRequest, HeartbeatResponse,
    ProbeRequest, ProbeResponse,
    BenchmarkRequest, BenchmarkResponse,
    PreparePlanRequest, PreparePlanResponse,
    CommitPlanRequest, CommitPlanResponse,
    CancelPlanRequest, CancelPlanResponse,
    DrainRequest, DrainResponse,
    SetPermissionsRequest, SetPermissionsResponse,
};

/// Node control service implementation.
pub struct NodeControlService {
    state: ClusterState,
}

impl NodeControlService {
    pub fn new(state: ClusterState) -> Self {
        Self { state }
    }

    pub fn into_server(self) -> NodeControlServer<Self> {
        NodeControlServer::new(self)
    }
}

#[tonic::async_trait]
impl NodeControl for NodeControlService {
    async fn register(&self, request: Request<RegisterRequest>) -> Result<Response<RegisterResponse>, Status> {
        let req = request.into_inner();
        
        info!("Registration request from: {}", req.node_name);

        // Validate cluster ID
        let cluster_id = Uuid::parse_str(&req.cluster_id)
            .map_err(|_| Status::invalid_argument("Invalid cluster ID"))?;

        if cluster_id != self.state.cluster_id {
            return Ok(Response::new(RegisterResponse {
                success: false,
                node_id: String::new(),
                device_certificate: Vec::new(),
                error_message: "Cluster ID mismatch".to_string(),
            }));
        }

        // Generate pairing challenge
        let mut pairing_manager = self.state.pairing_manager.write().await;
        let pairing_req = secure_pairing::PairingRequest {
            node_name: req.node_name.clone(),
            hostname: req.hostname.clone(),
            cluster_id,
            public_key: req.pairing_signature.clone(),
            fingerprint: hex::encode(&req.pairing_signature[..8]),
        };

        let challenge = pairing_manager.generate_challenge(&pairing_req)
            .map_err(|e| Status::internal(format!("Failed to generate challenge: {}", e)))?;

        // Create node entry with pending status
        let node_id = challenge.node_id;
        let node = cluster_types::PhysicalNode {
            id: node_id,
            name: req.node_name,
            hostname: req.hostname,
            cluster_id,
            capabilities: cluster_types::NodeCapabilities {
                cpu: cluster_types::CpuCapabilities {
                    architecture: "unknown".to_string(),
                    cores: 0,
                    threads: 0,
                    frequency_mhz: 0,
                    features: vec![],
                    numa_nodes: 1,
                },
                gpus: vec![],
                memory: cluster_types::MemoryCapabilities {
                    total_bytes: 0,
                    available_bytes: 0,
                    bandwidth_mbps: 0,
                },
                storage: cluster_types::StorageCapabilities {
                    total_bytes: 0,
                    available_bytes: 0,
                    read_throughput_mbps: 0,
                    write_throughput_mbps: 0,
                },
                network: cluster_types::NetworkCapabilities {
                    interfaces: vec![],
                },
                supported_runtimes: vec![],
            },
            resources: cluster_types::NodeResources {
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
            status: cluster_types::NodeStatus::Suspect,
            last_heartbeat: chrono::Utc::now(),
            certificate_fingerprint: Some(hex::encode(&req.pairing_signature[..8])),
        };

        // Add node to cluster state
        let mut nodes = self.state.nodes.write().await;
        nodes.insert(node_id, node);

        Ok(Response::new(RegisterResponse {
            success: true,
            node_id: node_id.to_string(),
            device_certificate: challenge.coordinator_public_key,
            error_message: String::new(),
        }))
    }

    async fn heartbeat(&self, request: Request<HeartbeatRequest>) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        
        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|_| Status::invalid_argument("Invalid node ID"))?;

        // Update node resources
        let mut nodes = self.state.nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            node.resources = convert_node_resources(&req.resources);
            node.last_heartbeat = chrono::Utc::now();
            node.status = NodeStatus::Healthy;
        } else {
            warn!("Heartbeat from unregistered node: {}", node_id);
        }

        Ok(Response::new(HeartbeatResponse {
            acknowledged: true,
            expected_sequence: req.sequence + 1,
            commands: vec![],
        }))
    }

    async fn probe(&self, request: Request<ProbeRequest>) -> Result<Response<ProbeResponse>, Status> {
        let req = request.into_inner();
        
        info!("Probe request from node: {}", req.node_id);

        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|_| Status::invalid_argument("Invalid node ID"))?;

        // Update node capabilities with actual probe data
        let capabilities = hardware_probe::HardwareProbe::probe_all()
            .map_err(|e| Status::internal(format!("Hardware probe failed: {}", e)))?;

        // Update node in cluster state
        let mut nodes = self.state.nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            node.capabilities = capabilities.clone();
            node.status = cluster_types::NodeStatus::Healthy;
        }

        // Convert to protobuf format
        let proto_capabilities = control::NodeCapabilities {
            cpu: Some(control::CpuCapabilities {
                architecture: capabilities.cpu.architecture,
                cores: capabilities.cpu.cores,
                threads: capabilities.cpu.threads,
                frequency_mhz: capabilities.cpu.frequency_mhz,
                features: capabilities.cpu.features,
                numa_nodes: capabilities.cpu.numa_nodes,
            }),
            gpus: capabilities.gpus.into_iter().map(|gpu| control::GpuCapabilities {
                id: gpu.id,
                name: gpu.name,
                vendor: format!("{:?}", gpu.vendor),
                vram_bytes: gpu.vram_bytes,
                compute_capability: gpu.compute_capability,
                supports_peer_to_peer: gpu.supports_peer_to_peer,
                supports_nvml: gpu.supports_nvml,
                supported_precisions: gpu.supported_precisions.iter().map(|p| format!("{:?}", p)).collect(),
            }).collect(),
            memory: Some(control::MemoryCapabilities {
                total_bytes: capabilities.memory.total_bytes,
                available_bytes: capabilities.memory.available_bytes,
                bandwidth_mbps: capabilities.memory.bandwidth_mbps,
            }),
            storage: Some(control::StorageCapabilities {
                total_bytes: capabilities.storage.total_bytes,
                available_bytes: capabilities.storage.available_bytes,
                read_throughput_mbps: capabilities.storage.read_throughput_mbps,
                write_throughput_mbps: capabilities.storage.write_throughput_mbps,
            }),
            supported_runtimes: capabilities.supported_runtimes.iter().map(|r| format!("{:?}", r)).collect(),
        };

        let proto_interfaces = capabilities.network.interfaces.into_iter().map(|iface| {
            control::NetworkInterface {
                name: iface.name,
                ip_address: iface.ip_address,
                mac_address: iface.mac_address,
                bandwidth_mbps: iface.bandwidth_mbps,
                supports_rdma: iface.supports_rdma,
            }
        }).collect();

        Ok(Response::new(ProbeResponse {
            capabilities: Some(proto_capabilities),
            interfaces: proto_interfaces,
        }))
    }

    async fn benchmark(&self, request: Request<BenchmarkRequest>) -> Result<Response<BenchmarkResponse>, Status> {
        let req = request.into_inner();
        
        info!("Benchmark request from node: {}", req.node_id);

        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|_| Status::invalid_argument("Invalid node ID"))?;

        // Execute benchmark based on type
        let mut metrics = HashMap::new();
        let duration_ms = match req.benchmark_type() {
            control::BenchmarkType::CpuMatrix => {
                // Simulate CPU matrix benchmark
                metrics.insert("gflops".to_string(), 50.0);
                100
            }
            control::BenchmarkType::CpuMemory => {
                // Simulate memory bandwidth benchmark
                metrics.insert("bandwidth_mbps".to_string(), 25000.0);
                50
            }
            control::BenchmarkType::GpuMatrix => {
                // Simulate GPU matrix benchmark
                metrics.insert("tflops".to_string(), 10.0);
                200
            }
            control::BenchmarkType::NetworkLatency => {
                // Simulate network latency
                metrics.insert("latency_us".to_string(), 150.0);
                10
            }
            control::BenchmarkType::NetworkBandwidth => {
                // Simulate network bandwidth
                metrics.insert("bandwidth_mbps".to_string(), 8500.0);
                1000
            }
            _ => {
                return Ok(Response::new(BenchmarkResponse {
                    success: false,
                    result: None,
                    error_message: "Unsupported benchmark type".to_string(),
                }));
            }
        };

        Ok(Response::new(BenchmarkResponse {
            success: true,
            result: Some(control::BenchmarkResult {
                metrics,
                duration_ms,
            }),
            error_message: String::new(),
        }))
    }

    async fn prepare_plan(&self, request: Request<PreparePlanRequest>) -> Result<Response<PreparePlanResponse>, Status> {
        let req = request.into_inner();
        
        info!("Prepare plan request from node: {} for model: {}", req.node_id, req.model_id);

        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|_| Status::invalid_argument("Invalid node ID"))?;

        // Check if node exists and is healthy
        let nodes = self.state.nodes.read().await;
        let node = nodes.get(&node_id)
            .ok_or_else(|| Status::not_found("Node not found"))?;

        if node.status != cluster_types::NodeStatus::Healthy {
            return Ok(Response::new(PreparePlanResponse {
                accepted: false,
                rejection_reason: format!("Node status: {:?}", node.status),
                estimated_load_time_ms: 0,
            }));
        }

        // Check if node has enough resources for the requested shards
        let total_weight_bytes: u64 = req.shards.iter().map(|s| s.bytes).sum();
        let available_vram = node.capabilities.gpus.iter().map(|g| g.vram_bytes).sum();
        let available_ram = node.capabilities.memory.available_bytes;

        if total_weight_bytes > available_vram + available_ram {
            return Ok(Response::new(PreparePlanResponse {
                accepted: false,
                rejection_reason: "Insufficient memory for requested shards".to_string(),
                estimated_load_time_ms: 0,
            }));
        }

        // Estimate load time based on network bandwidth
        let estimated_load_time_ms = (total_weight_bytes / (100 * 1024 * 1024)) as u32 * 1000; // Assume 100 MB/s

        Ok(Response::new(PreparePlanResponse {
            accepted: true,
            rejection_reason: String::new(),
            estimated_load_time_ms,
        }))
    }

    async fn commit_plan(&self, request: Request<CommitPlanRequest>) -> Result<Response<CommitPlanResponse>, Status> {
        let req = request.into_inner();
        
        info!("Commit plan request for session: {}", req.session_id);

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|_| Status::invalid_argument("Invalid session ID"))?;

        // In a real implementation, this would:
        // 1. Validate the execution graph
        // 2. Reserve resources on nodes
        // 3. Update tensor directory
        // 4. Notify scheduler of new session

        // For now, just acknowledge
        Ok(Response::new(CommitPlanResponse {
            committed: true,
            error_message: String::new(),
        }))
    }

    async fn cancel_plan(&self, request: Request<CancelPlanRequest>) -> Result<Response<CancelPlanResponse>, Status> {
        let req = request.into_inner();
        
        info!("Cancel plan request for session: {} - reason: {}", req.session_id, req.reason);

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|_| Status::invalid_argument("Invalid session ID"))?;

        // In a real implementation, this would:
        // 1. Cancel the execution graph
        // 2. Release reserved resources
        // 3. Update tensor directory
        // 4. Notify scheduler

        Ok(Response::new(CancelPlanResponse {
            cancelled: true,
        }))
    }

    async fn drain(&self, request: Request<DrainRequest>) -> Result<Response<DrainResponse>, Status> {
        let req = request.into_inner();
        
        info!("Drain request for node: {} (emergency: {})", req.node_id, req.emergency);

        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|_| Status::invalid_argument("Invalid node ID"))?;

        let mut nodes = self.state.nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            node.status = cluster_types::NodeStatus::Draining;
            
            // In a real implementation, this would:
            // 1. Stop assigning new work to this node
            // 2. Migrate tensors to other nodes
            // 3. Wait for active requests to complete
            // 4. Release resources
        }

        let estimated_seconds = if req.emergency { 5 } else { 30 };

        Ok(Response::new(DrainResponse {
            draining: true,
            estimated_seconds,
        }))
    }

    async fn set_permissions(&self, request: Request<SetPermissionsRequest>) -> Result<Response<SetPermissionsResponse>, Status> {
        let req = request.into_inner();
        
        info!("Set permissions request for node: {}", req.node_id);

        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|_| Status::invalid_argument("Invalid node ID"))?;

        let mut nodes = self.state.nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            node.permissions = cluster_types::NodePermissions {
                max_cpu_percent: req.max_cpu_percent as u8,
                max_cpu_threads: if req.max_cpu_threads > 0 { Some(req.max_cpu_threads as u32) } else { None },
                max_ram_bytes: req.max_ram_bytes,
                gpu_enabled: req.gpu_enabled,
                max_vram_bytes: req.max_vram_bytes,
                max_storage_bytes: req.max_storage_bytes,
                allow_when_on_battery: req.allow_when_on_battery,
                pause_when_user_active: req.pause_when_user_active,
                allowed_schedule: if req.allowed_schedule.is_empty() { None } else { Some(req.allowed_schedule) },
            };
        }

        Ok(Response::new(SetPermissionsResponse {
            updated: true,
        }))
    }
}

fn convert_node_resources(resources: &control::NodeResources) -> NodeResources {
    NodeResources {
        cpu_usage_percent: resources.cpu_usage_percent,
        memory_used_bytes: resources.memory_used_bytes,
        memory_committed_bytes: resources.memory_committed_bytes,
        gpu_usage: resources.gpu_usage.iter().map(|gpu| cluster_types::GpuUsage {
            device_id: gpu.device_id.clone(),
            utilization_percent: gpu.utilization_percent,
            vram_used_bytes: gpu.vram_used_bytes,
            temperature_celsius: gpu.temperature_celsius,
            power_draw_watts: if gpu.power_draw_watts > 0.0 { Some(gpu.power_draw_watts) } else { None },
        }).collect(),
        network_tx_mbps: resources.network_tx_mbps,
        network_rx_mbps: resources.network_rx_mbps,
        temperature_celsius: if resources.temperature_celsius > 0.0 { Some(resources.temperature_celsius) } else { None },
    }
}