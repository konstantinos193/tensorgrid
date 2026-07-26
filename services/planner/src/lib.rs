//! Execution Planner Service.
//!
//! Creates distributed execution plans for models based on cluster topology
//! and hardware capabilities.

use cluster_types::{
    ModelId, NodeId, ExecutionPlan, ParallelismStrategy, PerformanceEstimate,
    StagePlan, KvCachePlan, KvCachePolicy, LogicalCluster, ClusterTopology,
    LinkMeasurement, MemoryTier,
};
use model_format::GgufModel;
use observability::{LogContext, MetricsCollector};
use topology::{TopologyProfiler, LinkQuality};
use std::collections::HashMap;
use uuid::Uuid;
use tracing::{info, warn, error};

/// Planning request.
#[derive(Debug, Clone)]
pub struct PlanningRequest {
    pub model_id: ModelId,
    pub model_metadata: ModelMetadata,
    pub context_length: u32,
    pub batch_size: u32,
    pub preferred_strategy: Option<ParallelismStrategy>,
}

/// Model metadata for planning.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub parameter_count: u64,
    pub layer_count: u32,
    pub architecture: String,
    pub quantization: String,
    pub estimated_weight_bytes: u64,
}

/// Planner service.
pub struct Planner {
    metrics: MetricsCollector,
}

impl Planner {
    pub fn new() -> Self {
        Self {
            metrics: MetricsCollector::new("planner".to_string()),
        }
    }

    /// Create an execution plan for a model.
    pub fn create_plan(
        &self,
        request: PlanningRequest,
        cluster: &LogicalCluster,
    ) -> Result<ExecutionPlan, PlanningError> {
        let ctx = LogContext::new("create_plan")
            .with_model_id(request.model_id.clone());

        info!("Creating execution plan for model: {}", request.model_id);

        // Select parallelism strategy
        let strategy = self.select_strategy(&request, cluster)?;

        // Partition model into stages
        let stages = self.partition_model(&request, cluster, &strategy)?;

        // Calculate KV cache requirements
        let kv_cache = self.plan_kv_cache(&request, cluster, &stages)?;

        // Estimate performance
        let estimated = self.estimate_performance(&request, cluster, &stages, &kv_cache)?;

        let plan = ExecutionPlan {
            model_id: request.model_id,
            strategy,
            estimated,
            stages,
            kv_cache,
        };

        self.metrics.increment_counter("plans_created", 1, &[
            ("strategy", format!("{:?}", strategy).as_str())
        ]);

        ctx.info(&format!("Plan created with {} stages", plan.stages.len()));

        Ok(plan)
    }

    /// Select the best parallelism strategy.
    fn select_strategy(
        &self,
        request: &PlanningRequest,
        cluster: &LogicalCluster,
    ) -> Result<ParallelismStrategy, PlanningError> {
        // Use user preference if specified
        if let Some(strategy) = request.preferred_strategy {
            return Ok(strategy);
        }

        // Analyze cluster topology
        let avg_bandwidth = self.calculate_average_bandwidth(&cluster.topology);
        let avg_latency = self.calculate_average_latency(&cluster.topology);

        // Strategy selection based on network quality
        if avg_bandwidth >= 10000 && avg_latency < 100 {
            // Excellent network - can use tensor parallelism
            Ok(ParallelismStrategy::PipelineHybrid)
        } else if avg_bandwidth >= 2500 && avg_latency < 500 {
            // Good network - pipeline parallelism
            Ok(ParallelismStrategy::Pipeline)
        } else if avg_bandwidth >= 1000 {
            // Acceptable network - pipeline with CPU offload
            Ok(ParallelismStrategy::PipelineHybrid)
        } else {
            // Poor network - warn but still attempt pipeline
            warn!("Network quality poor, may impact performance");
            Ok(ParallelismStrategy::Pipeline)
        }
    }

    /// Partition model into stages across nodes.
    fn partition_model(
        &self,
        request: &PlanningRequest,
        cluster: &LogicalCluster,
        strategy: &ParallelismStrategy,
    ) -> Result<Vec<StagePlan>, PlanningError> {
        let available_nodes: Vec<_> = cluster
            .nodes
            .iter()
            .filter(|(_, node)| node.status == cluster_types::NodeStatus::Healthy)
            .collect();

        if available_nodes.is_empty() {
            return Err(PlanningError::NoAvailableNodes);
        }

        let layer_count = request.model_metadata.layer_count;
        let node_count = available_nodes.len();

        match strategy {
            ParallelismStrategy::Pipeline => {
                // Divide layers evenly across nodes
                let layers_per_node = (layer_count as f32 / node_count as f32).ceil() as u32;
                let mut stages = Vec::new();

                let mut current_layer = 0u32;
                for (i, (node_id, node)) in available_nodes.iter().enumerate() {
                    let end_layer = (current_layer + layers_per_node).min(layer_count);
                    let layers: Vec<u32> = (current_layer..end_layer).collect();

                    if layers.is_empty() {
                        continue;
                    }

                    // Calculate weight bytes for this stage
                    let weight_bytes = self.estimate_stage_weight(
                        &layers,
                        request.model_metadata.estimated_weight_bytes,
                        layer_count,
                    );

                    // Select best device on this node
                    let device = self.select_device(node);

                    stages.push(StagePlan {
                        id: i as u32,
                        node_id: **node_id,
                        device,
                        layers,
                        weight_bytes,
                    });

                    current_layer = end_layer;
                }

                Ok(stages)
            }
            ParallelismStrategy::PipelineHybrid => {
                // Pipeline for major stages, with CPU offload for remaining
                let gpu_nodes: Vec<_> = available_nodes
                    .iter()
                    .filter(|(_, node)| !node.capabilities.gpus.is_empty())
                    .collect();

                if gpu_nodes.is_empty() {
                    // Fallback to CPU-only pipeline
                    return self.partition_model(request, cluster, &ParallelismStrategy::Pipeline);
                }

                let layers_per_gpu = (layer_count as f32 / gpu_nodes.len() as f32).ceil() as u32;
                let mut stages = Vec::new();

                let mut current_layer = 0u32;
                for (i, (node_id, node)) in gpu_nodes.iter().enumerate() {
                    let end_layer = (current_layer + layers_per_gpu).min(layer_count);
                    let layers: Vec<u32> = (current_layer..end_layer).collect();

                    if layers.is_empty() {
                        continue;
                    }

                    let weight_bytes = self.estimate_stage_weight(
                        &layers,
                        request.model_metadata.estimated_weight_bytes,
                        layer_count,
                    );

                    let device = self.select_device(node);

                    stages.push(StagePlan {
                        id: i as u32,
                        node_id: **node_id,
                        device,
                        layers,
                        weight_bytes,
                    });

                    current_layer = end_layer;
                }

                // Add CPU stage for remaining layers if any
                if current_layer < layer_count {
                    let cpu_nodes: Vec<_> = available_nodes
                        .iter()
                        .filter(|(_, node)| node.capabilities.gpus.is_empty())
                        .collect();

                    if let Some((node_id, node)) = cpu_nodes.first() {
                        let layers: Vec<u32> = (current_layer..layer_count).collect();
                        let weight_bytes = self.estimate_stage_weight(
                            &layers,
                            request.model_metadata.estimated_weight_bytes,
                            layer_count,
                        );

                        stages.push(StagePlan {
                            id: stages.len() as u32,
                            node_id: **node_id,
                            device: "cpu:0".to_string(),
                            layers,
                            weight_bytes,
                        });
                    }
                }

                Ok(stages)
            }
            _ => {
                // For other strategies, default to pipeline for now
                self.partition_model(request, cluster, &ParallelismStrategy::Pipeline)
            }
        }
    }

    /// Plan KV cache allocation.
    fn plan_kv_cache(
        &self,
        request: &PlanningRequest,
        cluster: &LogicalCluster,
        stages: &[StagePlan],
    ) -> Result<KvCachePlan, PlanningError> {
        // Estimate KV cache size
        let kv_bytes_per_token_per_layer = 2 * 4096 * 2; // Simplified: 2 KV heads * 4096 dim * 2 bytes
        let total_kv_bytes = kv_bytes_per_token_per_layer as u64
            * request.context_length as u64
            * request.model_metadata.layer_count as u64;

        // Check if it fits in VRAM
        let total_vram = cluster.logical_resources.device_memory_bytes;
        let policy = if total_kv_bytes > total_vram {
            KvCachePolicy::StageLocalWithHostSpill
        } else {
            KvCachePolicy::StageLocal
        };

        Ok(KvCachePlan {
            policy,
            reserved_bytes: total_kv_bytes,
        })
    }

    /// Estimate performance metrics.
    fn estimate_performance(
        &self,
        request: &PlanningRequest,
        cluster: &LogicalCluster,
        stages: &[StagePlan],
        kv_cache: &KvCachePlan,
    ) -> Result<PerformanceEstimate, PlanningError> {
        // Estimate load time based on model size and network
        let load_seconds = (request.model_metadata.estimated_weight_bytes as f32
            / (100.0 * 1024.0 * 1024.0)) // Assume 100 MB/s effective
            + 5.0; // Base overhead

        // Estimate throughput based on slowest stage
        let stage_scores: Vec<f32> = stages
            .iter()
            .map(|stage| {
                let node = cluster.nodes.get(&stage.node_id)?;
                let score = if stage.device.starts_with("cuda") || stage.device.starts_with("gpu") {
                    // GPU stage
                    node.capabilities.gpus.first()
                        .map(|gpu| gpu.vram_bytes as f32 / (1024.0 * 1024.0 * 1024.0) * 10.0)
                        .unwrap_or(5.0)
                } else {
                    // CPU stage
                    node.capabilities.cpu.cores as f32 * 0.5
                };
                Some(score)
            })
            .filter_map(|s| s)
            .collect();

        let min_score = stage_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let decode_tokens_per_second = if min_score.is_finite() {
            min_score
        } else {
            1.0 // Fallback
        };

        // Prompt processing is typically faster
        let prompt_tokens_per_second = decode_tokens_per_second * 3.0;

        // Apply penalty for KV cache spill
        let decode_tokens_per_second = match kv_cache.policy {
            KvCachePolicy::StageLocalWithHostSpill => decode_tokens_per_second * 0.7,
            _ => decode_tokens_per_second,
        };

        Ok(PerformanceEstimate {
            load_seconds,
            prompt_tokens_per_second,
            decode_tokens_per_second,
        })
    }

    /// Calculate average bandwidth across cluster links.
    fn calculate_average_bandwidth(&self, topology: &ClusterTopology) -> u32 {
        if topology.links.is_empty() {
            return 1000; // Default 1 Gbps
        }

        let total: u32 = topology.links.iter().map(|l| l.bandwidth_mbps).sum();
        total / topology.links.len() as u32
    }

    /// Calculate average latency across cluster links.
    fn calculate_average_latency(&self, topology: &ClusterTopology) -> u32 {
        if topology.links.is_empty() {
            return 1000; // Default 1 ms
        }

        let total: u32 = topology.links.iter().map(|l| l.latency_us).sum();
        total / topology.links.len() as u32
    }

    /// Estimate weight bytes for a stage.
    fn estimate_stage_weight(
        &self,
        layers: &[u32],
        total_weight_bytes: u64,
        total_layers: u32,
    ) -> u64 {
        let layer_fraction = layers.len() as f32 / total_layers as f32;
        (total_weight_bytes as f32 * layer_fraction) as u64
    }

    /// Select the best device on a node.
    fn select_device(&self, node: &cluster_types::PhysicalNode) -> String {
        if let Some(gpu) = node.capabilities.gpus.first() {
            format!("cuda:{}", gpu.id)
        } else {
            "cpu:0".to_string()
        }
    }
}

/// Planning errors.
#[derive(Debug, thiserror::Error)]
pub enum PlanningError {
    #[error("No available nodes")]
    NoAvailableNodes,
    
    #[error("Model too large for cluster")]
    ModelTooLarge,
    
    #[error("Invalid plan: {0}")]
    InvalidPlan(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use cluster_types::{NodeCapabilities, CpuCapabilities, MemoryCapabilities, NodeStatus, GpuCapabilities, GpuVendor};

    #[test]
    fn test_planner_creation() {
        let planner = Planner::new();
        assert_eq!(planner.metrics.service_name, "planner");
    }

    #[test]
    fn test_average_bandwidth() {
        let planner = Planner::new();
        let topology = ClusterTopology {
            nodes: HashMap::new(),
            links: vec![
                LinkMeasurement {
                    from: Uuid::new_v4(),
                    to: Uuid::new_v4(),
                    transport: "tcp".to_string(),
                    latency_us: 100,
                    bandwidth_mbps: 10000,
                    jitter_us: 10,
                    packet_loss_percent: 0.0,
                },
                LinkMeasurement {
                    from: Uuid::new_v4(),
                    to: Uuid::new_v4(),
                    transport: "tcp".to_string(),
                    latency_us: 150,
                    bandwidth_mbps: 5000,
                    jitter_us: 15,
                    packet_loss_percent: 0.1,
                },
            ],
        };

        let avg = planner.calculate_average_bandwidth(&topology);
        assert_eq!(avg, 7500);
    }
}
