//! Core data structures for the distributed AI cluster.
//!
//! This crate defines the common types used across the coordinator,
//! node agents, scheduler, and planner.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier for a node in the cluster.
pub type NodeId = Uuid;

/// Unique identifier for a cluster.
pub type ClusterId = Uuid;

/// Unique identifier for a model.
pub type ModelId = String;

/// Unique identifier for a session.
pub type SessionId = Uuid;

/// Unique identifier for a tensor.
pub type TensorId = String;

/// Represents a physical node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalNode {
    pub id: NodeId,
    pub name: String,
    pub hostname: String,
    pub cluster_id: ClusterId,
    pub capabilities: NodeCapabilities,
    pub resources: NodeResources,
    pub permissions: NodePermissions,
    pub status: NodeStatus,
    pub last_heartbeat: DateTime<Utc>,
    pub certificate_fingerprint: Option<String>,
}

/// Hardware and software capabilities of a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub cpu: CpuCapabilities,
    pub gpus: Vec<GpuCapabilities>,
    pub memory: MemoryCapabilities,
    pub storage: StorageCapabilities,
    pub network: NetworkCapabilities,
    pub supported_runtimes: Vec<RuntimeBackend>,
}

/// CPU capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapabilities {
    pub architecture: String,
    pub cores: u32,
    pub threads: u32,
    pub frequency_mhz: u32,
    pub features: Vec<String>,
    pub numa_nodes: u32,
}

/// GPU capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    pub id: String,
    pub name: String,
    pub vendor: GpuVendor,
    pub vram_bytes: u64,
    pub compute_capability: Option<String>,
    pub supports_peer_to_peer: bool,
    pub supports_nvml: bool,
    pub supported_precisions: Vec<Precision>,
}

/// GPU vendor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

/// Supported precision formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
    Int8,
    Int4,
    Fp8,
}

/// Memory capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCapabilities {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub bandwidth_mbps: u64,
}

/// Storage capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageCapabilities {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub read_throughput_mbps: u32,
    pub write_throughput_mbps: u32,
}

/// Network capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    pub interfaces: Vec<NetworkInterface>,
}

/// Network interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub ip_address: String,
    pub mac_address: String,
    pub bandwidth_mbps: u32,
    pub supports_rdma: bool,
}

/// Runtime backend types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeBackend {
    GGML,
    CUDA,
    ROCm,
    MLX,
    Vulkan,
    CPU,
}

/// Current resource usage of a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResources {
    pub cpu_usage_percent: f32,
    pub memory_used_bytes: u64,
    pub memory_committed_bytes: u64,
    pub gpu_usage: Vec<GpuUsage>,
    pub network_tx_mbps: f32,
    pub network_rx_mbps: f32,
    pub temperature_celsius: Option<f32>,
}

/// GPU usage metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUsage {
    pub device_id: String,
    pub utilization_percent: f32,
    pub vram_used_bytes: u64,
    pub temperature_celsius: f32,
    pub power_draw_watts: Option<f32>,
}

/// User-defined permissions for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePermissions {
    pub max_cpu_percent: u8,
    pub max_cpu_threads: Option<u32>,
    pub max_ram_bytes: u64,
    pub gpu_enabled: bool,
    pub max_vram_bytes: u64,
    pub max_storage_bytes: u64,
    pub allow_when_on_battery: bool,
    pub pause_when_user_active: bool,
    pub allowed_schedule: Option<Schedule>,
}

/// Time schedule for resource availability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub allowed_hours: Vec<String>, // e.g., ["22:00-07:00"]
}

/// Current status of a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Healthy,
    Suspect,
    Unavailable,
    Draining,
    Maintenance,
}

/// Represents a logical cluster resource pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalCluster {
    pub id: ClusterId,
    pub name: String,
    pub logical_resources: LogicalResources,
    pub nodes: HashMap<NodeId, PhysicalNode>,
    pub topology: ClusterTopology,
}

/// Logical (schedulable) resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalResources {
    pub cpu_compute_units: f32,
    pub host_memory_bytes: u64,
    pub device_memory_bytes: u64,
    pub storage_bytes: u64,
    pub preferred_parallelism: ParallelismStrategy,
}

/// Parallelism strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelismStrategy {
    Pipeline,
    Tensor,
    PipelineHybrid,
    Expert,
    Data,
}

/// Cluster topology including link measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTopology {
    pub nodes: HashMap<NodeId, NodeTopology>,
    pub links: Vec<LinkMeasurement>,
}

/// Topology information for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTopology {
    pub cpu_score: f32,
    pub ram_usable_bytes: u64,
    pub devices: Vec<String>,
}

/// Measured link performance between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkMeasurement {
    pub from: NodeId,
    pub to: NodeId,
    pub transport: String,
    pub latency_us: u32,
    pub bandwidth_mbps: u32,
    pub jitter_us: u32,
    pub packet_loss_percent: f32,
}

/// Tensor placement in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorPlacement {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub placements: Vec<PhysicalPlacement>,
}

/// Physical location of a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalPlacement {
    pub node_id: NodeId,
    pub device: String,
    pub offset: u64,
    pub length: u64,
    pub role: PlacementRole,
}

/// Role of a tensor placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlacementRole {
    Primary,
    Replica,
    WarmCopy,
}

/// Memory tier for placement decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryTier {
    OnDeviceVram,
    LocalPinnedRam,
    LocalPageableRam,
    RemoteVram,
    RemotePinnedRam,
    RemotePageableRam,
    LocalNvme,
    RemoteNvme,
}

/// Execution plan for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub model_id: ModelId,
    pub strategy: ParallelismStrategy,
    pub estimated: PerformanceEstimate,
    pub stages: Vec<StagePlan>,
    pub kv_cache: KvCachePlan,
}

/// Estimated performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEstimate {
    pub load_seconds: f32,
    pub prompt_tokens_per_second: f32,
    pub decode_tokens_per_second: f32,
}

/// Plan for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePlan {
    pub id: u32,
    pub node_id: NodeId,
    pub device: String,
    pub layers: Vec<u32>,
    pub weight_bytes: u64,
}

/// KV cache plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCachePlan {
    pub policy: KvCachePolicy,
    pub reserved_bytes: u64,
}

/// KV cache policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KvCachePolicy {
    StageLocal,
    StageLocalWithHostSpill,
    Distributed,
}

/// Model shard specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelShardSpec {
    pub model_id: ModelId,
    pub shard_id: u32,
    pub layers: Vec<u32>,
    pub format: String,
    pub quantization: String,
    pub bytes: u64,
    pub checksum: String,
}

/// Tensor specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub bytes: u64,
}

/// Error types for cluster operations.
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),

    #[error("Node unavailable: {0}")]
    NodeUnavailable(NodeId),

    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    #[error("Invalid plan: {0}")]
    InvalidPlan(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(TensorId),

    #[error("Transport error: {0}")]
    TransportError(String),

    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}
