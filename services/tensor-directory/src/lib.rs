//! Tensor Directory Service.
//!
//! Tracks the location and ownership of tensors across the cluster.

use cluster_types::{NodeId, TensorId, TensorPlacement, PhysicalPlacement, PlacementRole, MemoryTier};
use observability::{LogContext, MetricsCollector};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tracing::{info, error, warn};

// Include the generated protobuf code
pub mod memory {
    include!("../proto/cluster.memory.rs");
}

/// Tensor directory entry.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub placements: Vec<PhysicalPlacement>,
    pub version: u64,
    pub mutable: bool,
    pub owner_session: Option<Uuid>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

/// Tensor directory service.
pub struct TensorDirectory {
    tensors: Arc<RwLock<HashMap<TensorId, TensorEntry>>>,
    node_tensors: Arc<RwLock<HashMap<NodeId, Vec<TensorId>>>>,
    metrics: MetricsCollector,
}

impl TensorDirectory {
    pub fn new() -> Self {
        Self {
            tensors: Arc::new(RwLock::new(HashMap::new())),
            node_tensors: Arc::new(RwLock::new(HashMap::new())),
            metrics: MetricsCollector::new("tensor-directory".to_string()),
        }
    }

    /// Register a tensor in the directory.
    pub async fn register_tensor(
        &self,
        tensor_id: TensorId,
        shape: Vec<usize>,
        dtype: String,
        placement: PhysicalPlacement,
        mutable: bool,
        owner_session: Option<Uuid>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ctx = LogContext::new("register_tensor")
            .with_session_id(tensor_id.clone());

        let entry = TensorEntry {
            tensor_id: tensor_id.clone(),
            shape,
            dtype,
            placements: vec![placement],
            version: 1,
            mutable,
            owner_session,
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };

        let mut tensors = self.tensors.write().await;
        tensors.insert(tensor_id.clone(), entry);

        // Update node index
        let mut node_tensors = self.node_tensors.write().await;
        node_tensors
            .entry(placement.node_id)
            .or_insert_with(Vec::new)
            .push(tensor_id.clone());

        self.metrics.increment_counter("tensors_registered", 1, &[]);
        ctx.info("Tensor registered successfully");

        Ok(())
    }

    /// Get tensor information.
    pub async fn get_tensor(&self, tensor_id: &TensorId) -> Option<TensorEntry> {
        let mut tensors = self.tensors.write().await;
        
        if let Some(entry) = tensors.get_mut(tensor_id) {
            entry.last_accessed = chrono::Utc::now();
            Some(entry.clone())
        } else {
            None
        }
    }

    /// Add a placement for an existing tensor.
    pub async fn add_placement(
        &self,
        tensor_id: &TensorId,
        placement: PhysicalPlacement,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tensors = self.tensors.write().await;
        
        if let Some(entry) = tensors.get_mut(tensor_id) {
            entry.placements.push(placement);
            entry.version += 1;

            // Update node index
            let mut node_tensors = self.node_tensors.write().await;
            node_tensors
                .entry(placement.node_id)
                .or_insert_with(Vec::new)
                .push(tensor_id.clone());

            self.metrics.increment_counter("tensor_placements_added", 1, &[]);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Tensor not found: {}", tensor_id))
        }
    }

    /// Remove a placement from a tensor.
    pub async fn remove_placement(
        &self,
        tensor_id: &TensorId,
        node_id: NodeId,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tensors = self.tensors.write().await;
        
        if let Some(entry) = tensors.get_mut(tensor_id) {
            let original_len = entry.placements.len();
            entry.placements.retain(|p| p.node_id != node_id);
            
            if entry.placements.is_empty() {
                // No placements left, remove the tensor
                tensors.remove(tensor_id);
            } else {
                entry.version += 1;
            }

            // Update node index
            let mut node_tensors = self.node_tensors.write().await;
            if let Some(tensor_list) = node_tensors.get_mut(&node_id) {
                tensor_list.retain(|t| t != tensor_id);
                if tensor_list.is_empty() {
                    node_tensors.remove(&node_id);
                }
            }

            self.metrics.increment_counter("tensor_placements_removed", 1, &[]);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Tensor not found: {}", tensor_id))
        }
    }

    /// Get all tensors on a specific node.
    pub async fn get_tensors_on_node(&self, node_id: NodeId) -> Vec<TensorEntry> {
        let node_tensors = self.node_tensors.read().await;
        let tensors = self.tensors.read().await;
        
        if let Some(tensor_ids) = node_tensors.get(&node_id) {
            tensor_ids
                .iter()
                .filter_map(|tid| tensors.get(tid).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find tensors by session owner.
    pub async fn get_tensors_by_session(&self, session_id: Uuid) -> Vec<TensorEntry> {
        let tensors = self.tensors.read().await;
        
        tensors
            .values()
            .filter(|entry| entry.owner_session == Some(session_id))
            .cloned()
            .collect()
    }

    /// Evict tensors from a node (for node drain or memory pressure).
    pub async fn evict_from_node(
        &self,
        node_id: NodeId,
        reason: &str,
    ) -> Result<Vec<TensorId>, Box<dyn std::error::Error>> {
        let ctx = LogContext::new("evict_from_node")
            .with_node_id(node_id.to_string());

        let mut node_tensors = self.node_tensors.write().await;
        let mut tensors = self.tensors.write().await;
        
        let tensor_ids = node_tensors.remove(&node_id).unwrap_or_default();
        let mut evicted = Vec::new();

        for tensor_id in &tensor_ids {
            if let Some(entry) = tensors.get_mut(tensor_id) {
                // Remove this node's placement
                entry.placements.retain(|p| p.node_id != node_id);
                entry.version += 1;

                // If no placements left, remove the tensor
                if entry.placements.is_empty() {
                    tensors.remove(tensor_id);
                    evicted.push(tensor_id.clone());
                }
            }
        }

        self.metrics.increment_counter("tensors_evicted", evicted.len() as u64, &[]);
        ctx.info(&format!("Evicted {} tensors from node: {}", evicted.len(), reason));

        Ok(evicted)
    }

    /// Get directory statistics.
    pub async fn get_stats(&self) -> DirectoryStats {
        let tensors = self.tensors.read().await;
        let node_tensors = self.node_tensors.read().await;
        
        let total_tensors = tensors.len();
        let total_placements: usize = tensors.values().map(|e| e.placements.len()).sum();
        let mutable_count = tensors.values().filter(|e| e.mutable).count();
        let node_count = node_tensors.len();

        DirectoryStats {
            total_tensors,
            total_placements,
            mutable_count,
            node_count,
        }
    }

    /// Transfer a tensor from one node to another.
    pub async fn transfer_tensor(
        &self,
        tensor_id: &TensorId,
        from_node: NodeId,
        to_node: NodeId,
        data: Bytes,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ctx = LogContext::new("transfer_tensor".to_string())
            .with_session_id(tensor_id.clone());

        info!("Transferring tensor {} from {} to {} ({} bytes)", 
            tensor_id, from_node, to_node, data.len());

        // Verify tensor exists on source node
        let tensors = self.tensors.read().await;
        let entry = tensors.get(tensor_id)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", tensor_id))?;

        let source_placement = entry.placements.iter()
            .find(|p| p.node_id == from_node)
            .ok_or_else(|| anyhow::anyhow!("Tensor not on source node"))?;

        // Create new placement on destination
        let new_placement = PhysicalPlacement {
            node_id: to_node,
            device: source_placement.device.clone(),
            offset: 0,
            length: data.len() as u64,
            role: PlacementRole::Replica,
        };

        drop(tensors);

        // Add new placement
        self.add_placement(tensor_id, new_placement).await?;

        self.metrics.increment_counter("tensor_transfers", 1, &[]);
        ctx.info(&format!("Tensor {} transferred successfully", tensor_id));

        Ok(())
    }

    /// Get all tensors that need to be replicated for a session.
    pub async fn get_tensors_for_replication(&self, session_id: Uuid) -> Vec<TensorEntry> {
        let tensors = self.tensors.read().await;
        
        tensors
            .values()
            .filter(|entry| {
                // Return tensors that are on multiple nodes or need replication
                entry.placements.len() > 1 || entry.mutable
            })
            .cloned()
            .collect()
    }
}

/// Directory statistics.
#[derive(Debug, Clone)]
pub struct DirectoryStats {
    pub total_tensors: usize,
    pub total_placements: usize,
    pub mutable_count: usize,
    pub node_count: usize,
}

impl LogContext {
    fn with_tensor_id(mut self, tensor_id: String) -> Self {
        // Add tensor_id to context (simplified)
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cluster_types::MemoryTier;

    #[tokio::test]
    async fn test_register_tensor() {
        let dir = TensorDirectory::new();
        
        let placement = PhysicalPlacement {
            node_id: Uuid::new_v4(),
            device: "cpu:0".to_string(),
            offset: 0,
            length: 1024,
            role: PlacementRole::Primary,
        };

        dir.register_tensor(
            "test_tensor".to_string(),
            vec![128, 128],
            "f32".to_string(),
            placement,
            false,
            None,
        ).await.unwrap();

        let tensor = dir.get_tensor("test_tensor").await;
        assert!(tensor.is_some());
    }

    #[tokio::test]
    async fn test_add_placement() {
        let dir = TensorDirectory::new();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let placement1 = PhysicalPlacement {
            node_id: node1,
            device: "cpu:0".to_string(),
            offset: 0,
            length: 1024,
            role: PlacementRole::Primary,
        };

        dir.register_tensor(
            "test_tensor".to_string(),
            vec![128, 128],
            "f32".to_string(),
            placement1,
            false,
            None,
        ).await.unwrap();

        let placement2 = PhysicalPlacement {
            node_id: node2,
            device: "cpu:0".to_string(),
            offset: 0,
            length: 1024,
            role: PlacementRole::Replica,
        };

        dir.add_placement("test_tensor", placement2).await.unwrap();

        let tensor = dir.get_tensor("test_tensor").await;
        assert!(tensor.is_some());
        assert_eq!(tensor.unwrap().placements.len(), 2);
    }

    #[tokio::test]
    async fn test_evict_from_node() {
        let dir = TensorDirectory::new();
        let node = Uuid::new_v4();
        
        let placement = PhysicalPlacement {
            node_id: node,
            device: "cpu:0".to_string(),
            offset: 0,
            length: 1024,
            role: PlacementRole::Primary,
        };

        dir.register_tensor(
            "test_tensor".to_string(),
            vec![128, 128],
            "f32".to_string(),
            placement,
            false,
            None,
        ).await.unwrap();

        let evicted = dir.evict_from_node(node, "test").await.unwrap();
        assert_eq!(evicted.len(), 1);
        
        let tensor = dir.get_tensor("test_tensor").await;
        assert!(tensor.is_none());
    }
}
