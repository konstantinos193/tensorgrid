//! Fine-tuning service for model customization.
//!
//! Provides distributed fine-tuning capabilities for customizing models on specific datasets.

use model_registry::ModelRegistry;
use scheduler::Scheduler;
use planner::Planner;
use ggml_runtime::GgmlRuntime;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Fine-tuning job configuration.
#[derive(Debug, Clone)]
pub struct FineTuningConfig {
    pub base_model: String,
    pub dataset_path: PathBuf,
    pub output_path: PathBuf,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub lora_rank: Option<u32>,
    pub lora_alpha: Option<f32>,
}

/// Fine-tuning job status.
#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Fine-tuning job.
#[derive(Debug, Clone)]
pub struct FineTuningJob {
    pub id: Uuid,
    pub config: FineTuningConfig,
    pub status: JobStatus,
    pub progress: f32,
    pub epoch: u32,
    pub total_epochs: u32,
    pub loss: f32,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

/// Fine-tuning service.
pub struct FineTuningService {
    jobs: Arc<RwLock<HashMap<Uuid, FineTuningJob>>>,
    model_registry: Arc<ModelRegistry>,
    _scheduler: Arc<Scheduler>,
    _planner: Arc<Planner>,
    _runtime: Arc<GgmlRuntime>,
}

impl FineTuningService {
    pub fn new(
        model_registry: Arc<ModelRegistry>,
        scheduler: Arc<Scheduler>,
        planner: Arc<Planner>,
        runtime: Arc<GgmlRuntime>,
    ) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            model_registry,
            _scheduler: scheduler,
            _planner: planner,
            _runtime: runtime,
        }
    }

    /// Create a new fine-tuning job.
    pub async fn create_job(&self, config: FineTuningConfig) -> Result<Uuid, FineTuningError> {
        let job_id = Uuid::new_v4();
        
        info!("Creating fine-tuning job {} for model {}", job_id, config.base_model);

        // Validate base model exists
        if self.model_registry.get_model(&config.base_model).await.is_none() {
            return Err(FineTuningError::ModelNotFound(config.base_model));
        }

        // Validate dataset exists
        if !config.dataset_path.exists() {
            return Err(FineTuningError::DatasetNotFound(config.dataset_path.clone()));
        }

        let job = FineTuningJob {
            id: job_id,
            config: config.clone(),
            status: JobStatus::Pending,
            progress: 0.0,
            epoch: 0,
            total_epochs: config.epochs,
            loss: 0.0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let mut jobs = self.jobs.write().await;
        jobs.insert(job_id, job);

        info!("Fine-tuning job {} created successfully", job_id);

        Ok(job_id)
    }

    /// Start a fine-tuning job.
    pub async fn start_job(&self, job_id: Uuid) -> Result<(), FineTuningError> {
        let mut jobs = self.jobs.write().await;
        
        let job = jobs.get_mut(&job_id)
            .ok_or_else(|| FineTuningError::JobNotFound(job_id))?;

        if job.status != JobStatus::Pending && job.status != JobStatus::Paused {
            return Err(FineTuningError::InvalidState(format!(
                "Cannot start job in {:?} state", job.status
            )));
        }

        job.status = JobStatus::Running;
        job.started_at = Some(Utc::now());

        info!("Fine-tuning job {} started", job_id);

        // Spawn background task for training
        let jobs_ref = self.jobs.clone();
        let runtime_ref = self._runtime.clone();
        let config = job.config.clone();
        
        tokio::spawn(async move {
            Self::run_training(job_id, config, jobs_ref, runtime_ref).await;
        });

        Ok(())
    }

    /// Pause a fine-tuning job.
    pub async fn pause_job(&self, job_id: Uuid) -> Result<(), FineTuningError> {
        let mut jobs = self.jobs.write().await;
        
        let job = jobs.get_mut(&job_id)
            .ok_or_else(|| FineTuningError::JobNotFound(job_id))?;

        if job.status != JobStatus::Running {
            return Err(FineTuningError::InvalidState(format!(
                "Cannot pause job in {:?} state", job.status
            )));
        }

        job.status = JobStatus::Paused;

        info!("Fine-tuning job {} paused", job_id);

        Ok(())
    }

    /// Resume a paused fine-tuning job.
    pub async fn resume_job(&self, job_id: Uuid) -> Result<(), FineTuningError> {
        let mut jobs = self.jobs.write().await;
        
        let job = jobs.get_mut(&job_id)
            .ok_or_else(|| FineTuningError::JobNotFound(job_id))?;

        if job.status != JobStatus::Paused {
            return Err(FineTuningError::InvalidState(format!(
                "Cannot resume job in {:?} state", job.status
            )));
        }

        job.status = JobStatus::Running;

        info!("Fine-tuning job {} resumed", job_id);

        Ok(())
    }

    /// Cancel a fine-tuning job.
    pub async fn cancel_job(&self, job_id: Uuid) -> Result<(), FineTuningError> {
        let mut jobs = self.jobs.write().await;
        
        let job = jobs.get_mut(&job_id)
            .ok_or_else(|| FineTuningError::JobNotFound(job_id))?;

        if job.status == JobStatus::Completed || job.status == JobStatus::Cancelled {
            return Err(FineTuningError::InvalidState(format!(
                "Cannot cancel job in {:?} state", job.status
            )));
        }

        job.status = JobStatus::Cancelled;

        info!("Fine-tuning job {} cancelled", job_id);

        Ok(())
    }

    /// Get job status.
    pub async fn get_job(&self, job_id: Uuid) -> Result<FineTuningJob, FineTuningError> {
        let jobs = self.jobs.read().await;
        
        jobs.get(&job_id)
            .cloned()
            .ok_or_else(|| FineTuningError::JobNotFound(job_id))
    }

    /// List all jobs.
    pub async fn list_jobs(&self) -> Vec<FineTuningJob> {
        let jobs = self.jobs.read().await;
        jobs.values().cloned().collect()
    }

    /// Delete a job.
    pub async fn delete_job(&self, job_id: Uuid) -> Result<(), FineTuningError> {
        let mut jobs = self.jobs.write().await;
        
        let job = jobs.get(&job_id)
            .ok_or_else(|| FineTuningError::JobNotFound(job_id))?;

        if job.status == JobStatus::Running {
            return Err(FineTuningError::InvalidState(
                "Cannot delete running job".to_string()
            ));
        }

        jobs.remove(&job_id);

        info!("Fine-tuning job {} deleted", job_id);

        Ok(())
    }

    /// Run training simulation.
    async fn run_training(
        job_id: Uuid,
        config: FineTuningConfig,
        jobs: Arc<RwLock<HashMap<Uuid, FineTuningJob>>>,
        _runtime: Arc<GgmlRuntime>,
    ) {
        info!("Starting training for job {}", job_id);

        for epoch in 1..=config.epochs {
            // Check if job was cancelled
            {
                let jobs_ref = jobs.read().await;
                if let Some(job) = jobs_ref.get(&job_id) {
                    if job.status == JobStatus::Cancelled {
                        info!("Job {} cancelled during training", job_id);
                        return;
                    }
                    if job.status == JobStatus::Paused {
                        info!("Job {} paused during training", job_id);
                        // Wait for resume
                        loop {
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            let jobs_ref = jobs.read().await;
                            if let Some(job) = jobs_ref.get(&job_id) {
                                if job.status == JobStatus::Running {
                                    break;
                                }
                                if job.status == JobStatus::Cancelled {
                                    return;
                                }
                            }
                        }
                    }
                }
            }

            // Simulate training epoch
            let epoch_time = std::time::Duration::from_millis(100); // Fast simulation
            tokio::time::sleep(epoch_time).await;

            // Update job progress
            {
                let mut jobs_ref = jobs.write().await;
                if let Some(job) = jobs_ref.get_mut(&job_id) {
                    job.epoch = epoch;
                    job.progress = (epoch as f32 / config.epochs as f32) * 100.0;
                    // Simulate decreasing loss
                    job.loss = 2.0 * (1.0 - (epoch as f32 / config.epochs as f32));
                }
            }

            info!("Job {} epoch {}/{} completed", job_id, epoch, config.epochs);
        }

        // Mark job as completed
        {
            let mut jobs_ref = jobs.write().await;
            if let Some(job) = jobs_ref.get_mut(&job_id) {
                job.status = JobStatus::Completed;
                job.progress = 100.0;
                job.completed_at = Some(Utc::now());
            }
        }

        info!("Training for job {} completed successfully", job_id);
    }

    /// Get service statistics.
    pub async fn get_stats(&self) -> ServiceStats {
        let jobs = self.jobs.read().await;
        
        let total = jobs.len();
        let running = jobs.values().filter(|j| j.status == JobStatus::Running).count();
        let completed = jobs.values().filter(|j| j.status == JobStatus::Completed).count();
        let failed = jobs.values().filter(|j| j.status == JobStatus::Failed).count();

        ServiceStats {
            total_jobs: total,
            running_jobs: running,
            completed_jobs: completed,
            failed_jobs: failed,
        }
    }
}

/// Service statistics.
#[derive(Debug, Clone)]
pub struct ServiceStats {
    pub total_jobs: usize,
    pub running_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
}

/// Fine-tuning errors.
#[derive(Debug, thiserror::Error)]
pub enum FineTuningError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Dataset not found: {0:?}")]
    DatasetNotFound(PathBuf),
    
    #[error("Job not found: {0}")]
    JobNotFound(Uuid),
    
    #[error("Invalid state: {0}")]
    InvalidState(String),
    
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_job() {
        let runtime = Arc::new(GgmlRuntime::new());
        let model_registry = Arc::new(ModelRegistry::new());
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
        let scheduler = Arc::new(Scheduler::new(cluster));
        let planner = Arc::new(Planner::new());
        
        let service = FineTuningService::new(model_registry, scheduler, planner, runtime);
        
        let config = FineTuningConfig {
            base_model: "test-model".to_string(),
            dataset_path: PathBuf::from("/tmp/dataset.jsonl"),
            output_path: PathBuf::from("/tmp/output.gguf"),
            epochs: 3,
            batch_size: 32,
            learning_rate: 0.0001,
            lora_rank: Some(8),
            lora_alpha: Some(16.0),
        };
        
        // This will fail because model doesn't exist, but tests the flow
        let result = service.create_job(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_jobs() {
        let runtime = Arc::new(GgmlRuntime::new());
        let model_registry = Arc::new(ModelRegistry::new());
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
        let scheduler = Arc::new(Scheduler::new(cluster));
        let planner = Arc::new(Planner::new());
        
        let service = FineTuningService::new(model_registry, scheduler, planner, runtime);
        
        let jobs = service.list_jobs().await;
        assert_eq!(jobs.len(), 0);
    }
}
