//! Model Registry Service.
//!
//! Manages model metadata, caching, and distribution across the cluster.

use cluster_types::{ModelId, NodeId};
use observability::{LogContext, MetricsCollector};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Model metadata stored in the registry.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_id: ModelId,
    pub name: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub context_length: u64,
    pub quantization: String,
    pub file_size_bytes: u64,
    pub checksum: String,
    pub file_path: PathBuf,
    pub cached_nodes: Vec<NodeId>,
    pub registered_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

/// Model registry service.
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<ModelId, ModelMetadata>>>,
    cache_dir: PathBuf,
    metrics: MetricsCollector,
}

impl ModelRegistry {
    /// Create a new model registry.
    pub fn new(cache_dir: PathBuf) -> Self {
        // Ensure cache directory exists
        std::fs::create_dir_all(&cache_dir).ok();

        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            cache_dir,
            metrics: MetricsCollector::new("model-registry".to_string()),
        }
    }

    /// Register a model from a file.
    pub async fn register_model<P: AsRef<Path>>(
        &self,
        file_path: P,
        name: String,
    ) -> Result<ModelMetadata, Box<dyn std::error::Error>> {
        let file_path = file_path.as_ref();
        let ctx = LogContext::new("register_model".to_string());

        // Parse the model file
        let gguf_model = model_format::GgufParser::parse(file_path)?;
        
        // Extract metadata
        let architecture = model_format::GgufParser::get_architecture(&gguf_model)
            .unwrap_or_else(|| "unknown".to_string());
        
        let parameter_count = model_format::GgufParser::get_parameter_count(&gguf_model)
            .unwrap_or(0);
        
        let context_length = model_format::GgufParser::get_context_length(&gguf_model)
            .unwrap_or(2048);
        
        let file_size_bytes = std::fs::metadata(file_path)?.len();
        
        // Calculate checksum
        let checksum = self.calculate_checksum(file_path)?;
        
        // Generate model ID from checksum
        let model_id = format!("{}-{}", architecture, &checksum[..16]);
        
        // Copy to cache if not already there
        let cached_path = self.cache_path(&checksum);
        if !cached_path.exists() {
            std::fs::copy(file_path, &cached_path)?;
        }

        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            name,
            architecture,
            parameter_count,
            context_length,
            quantization: "q4_k_m".to_string(), // Simplified - would parse from file
            file_size_bytes,
            checksum,
            file_path: cached_path,
            cached_nodes: Vec::new(),
            registered_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };

        let mut models = self.models.write().await;
        models.insert(model_id.clone(), metadata.clone());

        self.metrics.increment_counter("models_registered", 1, &[]);
        ctx.info(&format!("Model registered: {}", model_id));

        Ok(metadata)
    }

    /// Get model metadata.
    pub async fn get_model(&self, model_id: &ModelId) -> Option<ModelMetadata> {
        let mut models = self.models.write().await;
        
        if let Some(metadata) = models.get_mut(model_id) {
            metadata.last_accessed = chrono::Utc::now();
            Some(metadata.clone())
        } else {
            None
        }
    }

    /// List all registered models.
    pub async fn list_models(&self) -> Vec<ModelMetadata> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }

    /// Mark a model as cached on a node.
    pub async fn mark_cached(&self, model_id: &ModelId, node_id: NodeId) -> Result<(), Box<dyn std::error::Error>> {
        let mut models = self.models.write().await;
        
        if let Some(metadata) = models.get_mut(model_id) {
            if !metadata.cached_nodes.contains(&node_id) {
                metadata.cached_nodes.push(node_id);
                self.metrics.increment_counter("model_cache_nodes", 1, &[]);
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("Model not found: {}", model_id).into())
        }
    }

    /// Remove a node's cache entry for a model.
    pub async fn unmark_cached(&self, model_id: &ModelId, node_id: NodeId) -> Result<(), Box<dyn std::error::Error>> {
        let mut models = self.models.write().await;
        
        if let Some(metadata) = models.get_mut(model_id) {
            metadata.cached_nodes.retain(|n| n != &node_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Model not found: {}", model_id).into())
        }
    }

    /// Get the cached file path for a model.
    pub fn get_cached_path(&self, model_id: &ModelId) -> Option<PathBuf> {
        let checksum = model_id.split('-').last()?;
        Some(self.cache_path(checksum))
    }

    /// Check if a model is cached on a specific node.
    pub async fn is_cached_on_node(&self, model_id: &ModelId, node_id: NodeId) -> bool {
        let models = self.models.read().await;
        
        models
            .get(model_id)
            .map(|m| m.cached_nodes.contains(&node_id))
            .unwrap_or(false)
    }

    /// Get models cached on a specific node.
    pub async fn get_models_on_node(&self, node_id: NodeId) -> Vec<ModelMetadata> {
        let models = self.models.read().await;
        
        models
            .values()
            .filter(|m| m.cached_nodes.contains(&node_id))
            .cloned()
            .collect()
    }

    /// Calculate SHA-256 checksum of a file.
    fn calculate_checksum<P: AsRef<Path>>(&self, path: P) -> Result<String, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(hex::encode(hasher.finalize()))
    }

    /// Get cache path for a checksum.
    fn cache_path(&self, checksum: &str) -> PathBuf {
        // Use content-addressed storage: cache/checksum[0:2]/checksum
        let prefix = &checksum[..2];
        self.cache_dir.join(prefix).join(checksum)
    }

    /// Remove a model from the registry.
    pub async fn unregister_model(&self, model_id: &ModelId) -> Result<(), Box<dyn std::error::Error>> {
        let mut models = self.models.write().await;
        
        if let Some(metadata) = models.remove(model_id) {
            // Optionally remove cached file
            if let Err(e) = std::fs::remove_file(&metadata.file_path) {
                warn!("Failed to remove cached file: {}", e);
            }
            
            self.metrics.increment_counter("models_unregistered", 1, &[]);
            info!("Model unregistered: {}", model_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Model not found: {}", model_id).into())
        }
    }

    /// Get registry statistics.
    pub async fn get_stats(&self) -> RegistryStats {
        let models = self.models.read().await;
        
        let total_models = models.len();
        let total_size_bytes: u64 = models.values().map(|m| m.file_size_bytes).sum();
        let total_cache_nodes: usize = models.values().map(|m| m.cached_nodes.len()).sum();

        RegistryStats {
            total_models,
            total_size_bytes,
            total_cache_nodes,
        }
    }
}

/// Registry statistics.
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_models: usize,
    pub total_size_bytes: u64,
    pub total_cache_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_model_registry() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(temp_dir.path().to_path_buf());
        
        // This test would require a real GGUF file
        // For now, we just test the structure
        let models = registry.list_models().await;
        assert_eq!(models.len(), 0);
        
        let stats = registry.get_stats().await;
        assert_eq!(stats.total_models, 0);
    }

    #[test]
    fn test_cache_path() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(temp_dir.path().to_path_buf());
        
        let checksum = "abcdef1234567890";
        let path = registry.cache_path(checksum);
        
        assert!(path.ends_with("ab/abcdef1234567890"));
    }
}
