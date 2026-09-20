//! Distributed Scheduler Service.
//!
//! Manages resource allocation, session scheduling, and admission control.

use cluster_types::{NodeId, SessionId, ModelId, LogicalCluster, NodeStatus};
use observability::{LogContext, MetricsCollector};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Session scheduling request.
#[derive(Debug, Clone)]
pub struct ScheduleRequest {
    pub session_id: SessionId,
    pub model_id: ModelId,
    pub context_length: u32,
    pub batch_size: u32,
    pub priority: RequestPriority,
}

/// Request priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Interactive = 3,
}

/// Scheduled session.
#[derive(Debug, Clone)]
pub struct ScheduledSession {
    pub session_id: SessionId,
    pub model_id: ModelId,
    pub assigned_nodes: Vec<NodeId>,
    pub allocated_resources: HashMap<NodeId, ResourceAllocation>,
    pub priority: RequestPriority,
    pub scheduled_at: chrono::DateTime<chrono::Utc>,
    pub estimated_tokens_per_second: f32,
}

/// Resource allocation for a node.
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub node_id: NodeId,
    pub cpu_percent: f32,
    pub ram_bytes: u64,
    pub vram_bytes: u64,
}

/// Scheduler service.
pub struct Scheduler {
    cluster: Arc<RwLock<LogicalCluster>>,
    sessions: Arc<RwLock<HashMap<SessionId, ScheduledSession>>>,
    node_allocations: Arc<RwLock<HashMap<NodeId, NodeAllocation>>>,
    metrics: MetricsCollector,
}

/// Current allocation state for a node.
#[derive(Debug, Clone)]
struct NodeAllocation {
    allocated_cpu_percent: f32,
    allocated_ram_bytes: u64,
    allocated_vram_bytes: u64,
    active_sessions: Vec<SessionId>,
}

impl Scheduler {
    pub fn new(cluster: LogicalCluster) -> Self {
        Self {
            cluster: Arc::new(RwLock::new(cluster)),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            node_allocations: Arc::new(RwLock::new(HashMap::new())),
            metrics: MetricsCollector::new("scheduler".to_string()),
        }
    }

    /// Schedule a new session.
    pub async fn schedule(&self, request: ScheduleRequest) -> Result<ScheduledSession, ScheduleError> {
        let ctx = LogContext::new("schedule".to_string())
            .with_session_id(request.session_id.to_string())
            .with_model_id(request.model_id.clone());

        let cluster = self.cluster.read().await;
        
        // Check admission control
        if !self.check_admission(&request, &cluster).await {
            self.metrics.increment_counter("schedule_rejections", 1, &[("reason", "admission")]);
            return Err(ScheduleError::AdmissionRejected);
        }

        // Select nodes for the session
        let selected_nodes = self.select_nodes(&request, &cluster).await?;
        
        if selected_nodes.is_empty() {
            self.metrics.increment_counter("schedule_rejections", 1, &[("reason", "no_nodes")]);
            return Err(ScheduleError::NoAvailableNodes);
        }

        // Calculate resource allocations
        let allocations = self.calculate_allocations(&request, &selected_nodes).await?;

        // Reserve resources
        self.reserve_resources(&request.session_id, &allocations).await?;

        // Create scheduled session
        let session = ScheduledSession {
            session_id: request.session_id,
            model_id: request.model_id,
            assigned_nodes: selected_nodes.clone(),
            allocated_resources: allocations.clone(),
            priority: request.priority,
            scheduled_at: chrono::Utc::now(),
            estimated_tokens_per_second: self.estimate_throughput(&selected_nodes, &cluster).await,
        };

        // Store session
        let mut sessions = self.sessions.write().await;
        sessions.insert(request.session_id, session.clone());

        self.metrics.increment_counter("sessions_scheduled", 1, &[]);
        ctx.info(&format!("Session scheduled on {} nodes", selected_nodes.len()));

        Ok(session)
    }

    /// Release resources for a completed session.
    pub async fn release_session(&self, session_id: SessionId) -> Result<(), ScheduleError> {
        let ctx = LogContext::new("release_session".to_string())
            .with_session_id(session_id.to_string());

        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.remove(&session_id) {
            // Release node allocations
            let mut node_allocations = self.node_allocations.write().await;
            
            for (node_id, allocation) in session.allocated_resources {
                if let Some(node_alloc) = node_allocations.get_mut(&node_id) {
                    node_alloc.allocated_cpu_percent -= allocation.cpu_percent;
                    node_alloc.allocated_ram_bytes -= allocation.ram_bytes;
                    node_alloc.allocated_vram_bytes -= allocation.vram_bytes;
                    node_alloc.active_sessions.retain(|s| s != &session_id);
                }
            }

            self.metrics.increment_counter("sessions_released", 1, &[]);
            ctx.info("Session resources released");
            Ok(())
        } else {
            Err(ScheduleError::SessionNotFound)
        }
    }

    /// Check if a request can be admitted.
    async fn check_admission(&self, request: &ScheduleRequest, cluster: &LogicalCluster) -> bool {
        // Check if cluster has enough total resources
        let total_ram = cluster.logical_resources.host_memory_bytes;
        let total_vram = cluster.logical_resources.device_memory_bytes;
        
        // Simplified admission check
        // In a real implementation, this would consider:
        // - KV cache budget
        // - Stage queue lengths
        // - Memory pressure
        // - Estimated latency
        
        let required_ram = (request.context_length as u64) * 1024; // Simplified
        let required_vram = (request.context_length as u64) * 512; // Simplified
        
        total_ram > required_ram && total_vram > required_vram
    }

    /// Select nodes for a session.
    async fn select_nodes(
        &self,
        _request: &ScheduleRequest,
        cluster: &LogicalCluster,
    ) -> Result<Vec<NodeId>, ScheduleError> {
        let mut node_scores: Vec<(NodeId, f32)> = cluster
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.status, NodeStatus::Healthy))
            .map(|(id, node)| {
                let score = self.calculate_node_score(node);
                (*id, score)
            })
            .collect();

        // Sort by score (descending)
        node_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top nodes (simplified - would use actual requirements)
        let selected: Vec<_> = node_scores
            .into_iter()
            .take(3) // Max 3 nodes for now
            .map(|(id, _)| id)
            .collect();

        Ok(selected)
    }

    /// Calculate a score for a node (higher is better).
    fn calculate_node_score(&self, node: &cluster_types::PhysicalNode) -> f32 {
        let mut score = 0.0;

        // CPU capacity
        score += node.capabilities.cpu.cores as f32 * 10.0;

        // Available RAM
        let available_ram = node.capabilities.memory.available_bytes as f32;
        score += (available_ram / (1024.0 * 1024.0 * 1024.0)) * 5.0; // Per GB

        // GPU VRAM
        for gpu in &node.capabilities.gpus {
            score += (gpu.vram_bytes as f32 / (1024.0 * 1024.0 * 1024.0)) * 20.0; // Per GB
        }

        // Penalty for high current usage
        if node.resources.cpu_usage_percent > 80.0 {
            score *= 0.5;
        }

        score
    }

    /// Calculate resource allocations for selected nodes.
    async fn calculate_allocations(
        &self,
        _request: &ScheduleRequest,
        nodes: &[NodeId],
    ) -> Result<HashMap<NodeId, ResourceAllocation>, ScheduleError> {
        let mut allocations = HashMap::new();

        // Simplified allocation - equal distribution
        let cpu_per_node = 30.0; // 30% CPU per node
        let ram_per_node = 4 * 1024 * 1024 * 1024; // 4 GB per node
        let vram_per_node = 2 * 1024 * 1024 * 1024; // 2 GB per node

        for node_id in nodes {
            allocations.insert(
                *node_id,
                ResourceAllocation {
                    node_id: *node_id,
                    cpu_percent: cpu_per_node,
                    ram_bytes: ram_per_node,
                    vram_bytes: vram_per_node,
                },
            );
        }

        Ok(allocations)
    }

    /// Reserve resources on nodes.
    async fn reserve_resources(
        &self,
        session_id: &SessionId,
        allocations: &HashMap<NodeId, ResourceAllocation>,
    ) -> Result<(), ScheduleError> {
        let mut node_allocations = self.node_allocations.write().await;

        for (node_id, allocation) in allocations {
            let node_alloc = node_allocations
                .entry(*node_id)
                .or_insert_with(|| NodeAllocation {
                    allocated_cpu_percent: 0.0,
                    allocated_ram_bytes: 0,
                    allocated_vram_bytes: 0,
                    active_sessions: Vec::new(),
                });

            // Check if allocation would exceed limits
            if node_alloc.allocated_cpu_percent + allocation.cpu_percent > 100.0 {
                return Err(ScheduleError::InsufficientResources(*node_id));
            }

            node_alloc.allocated_cpu_percent += allocation.cpu_percent;
            node_alloc.allocated_ram_bytes += allocation.ram_bytes;
            node_alloc.allocated_vram_bytes += allocation.vram_bytes;
            node_alloc.active_sessions.push(*session_id);
        }

        Ok(())
    }

    /// Estimate throughput for a session.
    async fn estimate_throughput(&self, nodes: &[NodeId], cluster: &LogicalCluster) -> f32 {
        // Simplified throughput estimation
        // In a real implementation, this would use measured performance data
        
        let mut total_score = 0.0;
        for node_id in nodes {
            if let Some(node) = cluster.nodes.get(node_id) {
                total_score += node.capabilities.cpu.cores as f32;
                for gpu in &node.capabilities.gpus {
                    total_score += (gpu.vram_bytes as f32 / (1024.0 * 1024.0 * 1024.0)) * 2.0;
                }
            }
        }

        // Base throughput estimate
        (total_score * 0.5).min(50.0) // Cap at 50 tokens/sec
    }

    /// Get current scheduler statistics.
    pub async fn get_stats(&self) -> SchedulerStats {
        let sessions = self.sessions.read().await;
        let node_allocations = self.node_allocations.read().await;

        let active_sessions = sessions.len();
        let total_allocated_cpu: f32 = node_allocations
            .values()
            .map(|n| n.allocated_cpu_percent)
            .sum();
        let total_allocated_ram: u64 = node_allocations
            .values()
            .map(|n| n.allocated_ram_bytes)
            .sum();
        let total_allocated_vram: u64 = node_allocations
            .values()
            .map(|n| n.allocated_vram_bytes)
            .sum();

        SchedulerStats {
            active_sessions,
            total_allocated_cpu,
            total_allocated_ram,
            total_allocated_vram,
        }
    }
}

/// Scheduler statistics.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub active_sessions: usize,
    pub total_allocated_cpu: f32,
    pub total_allocated_ram: u64,
    pub total_allocated_vram: u64,
}

/// Scheduling errors.
#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error("Admission rejected")]
    AdmissionRejected,
    
    #[error("No available nodes")]
    NoAvailableNodes,
    
    #[error("Insufficient resources on node {0}")]
    InsufficientResources(NodeId),
    
    #[error("Session not found")]
    SessionNotFound,
    
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use cluster_types::{NodeCapabilities, CpuCapabilities, MemoryCapabilities, NodeResources, NodeStatus};

    #[tokio::test]
    async fn test_scheduler() {
        let cluster = LogicalCluster {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            logical_resources: cluster_types::LogicalResources {
                cpu_compute_units: 16.0,
                host_memory_bytes: 32 * 1024 * 1024 * 1024,
                device_memory_bytes: 8 * 1024 * 1024 * 1024,
                storage_bytes: 1024 * 1024 * 1024 * 1024,
                preferred_parallelism: cluster_types::ParallelismStrategy::Pipeline,
            },
            nodes: HashMap::new(),
            topology: cluster_types::ClusterTopology {
                nodes: HashMap::new(),
                links: vec![],
            },
        };

        let scheduler = Scheduler::new(cluster);
        
        let request = ScheduleRequest {
            session_id: Uuid::new_v4(),
            model_id: "test-model".to_string(),
            context_length: 2048,
            batch_size: 1,
            priority: RequestPriority::Normal,
        };

        // This will fail without nodes, but tests the structure
        let result = scheduler.schedule(request).await;
        assert!(result.is_err());
    }
}
