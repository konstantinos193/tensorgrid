//! Pipeline Parallelism Executor.
//!
//! Coordinates distributed execution of models across multiple nodes using
//! pipeline parallelism.

use cluster_types::{NodeId, SessionId, ExecutionPlan, StagePlan, TensorHandle};
use observability::{LogContext, MetricsCollector};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;
use tracing::{info, warn, error};

/// Pipeline executor state.
pub struct PipelineExecutor {
    active_sessions: Arc<RwLock<HashMap<SessionId, SessionState>>>,
    metrics: MetricsCollector,
}

/// State for an active session.
struct SessionState {
    session_id: SessionId,
    plan: ExecutionPlan,
    stage_states: Vec<StageState>,
    tensor_buffer: HashMap<String, Vec<u8>>,
    status: SessionStatus,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// State for a single stage.
struct StageState {
    stage_id: u32,
    node_id: NodeId,
    status: StageStatus,
    input_tensors: Vec<String>,
    output_tensors: Vec<String>,
    last_execution: Option<chrono::DateTime<chrono::Utc>>,
}

/// Session status.
#[derive(Debug, Clone, PartialEq)]
enum SessionStatus {
    Initializing,
    Running,
    Paused,
    Completed,
    Failed(String),
}

/// Stage status.
#[derive(Debug, Clone, PartialEq)]
enum StageStatus {
    Idle,
    Ready,
    Executing,
    Completed,
    Failed(String),
}

/// Execution request for a session.
#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    pub session_id: SessionId,
    pub input_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
}

/// Execution result.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub session_id: SessionId,
    pub tokens: Vec<u32>,
    pub completion_time_ms: u32,
    pub tokens_per_second: f32,
}

impl PipelineExecutor {
    pub fn new() -> Self {
        Self {
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            metrics: MetricsCollector::new("pipeline-executor".to_string()),
        }
    }

    /// Initialize a new session with an execution plan.
    pub async fn initialize_session(
        &self,
        session_id: SessionId,
        plan: ExecutionPlan,
    ) -> Result<(), PipelineError> {
        let ctx = LogContext::new("initialize_session")
            .with_session_id(session_id.to_string());

        info!("Initializing session {} with {} stages", session_id, plan.stages.len());

        let stage_states = plan.stages.iter().map(|stage| {
            StageState {
                stage_id: stage.id,
                node_id: stage.node_id,
                status: StageStatus::Idle,
                input_tensors: vec![],
                output_tensors: vec![],
                last_execution: None,
            }
        }).collect();

        let session_state = SessionState {
            session_id,
            plan,
            stage_states,
            tensor_buffer: HashMap::new(),
            status: SessionStatus::Initializing,
            created_at: chrono::Utc::now(),
        };

        let mut sessions = self.active_sessions.write().await;
        sessions.insert(session_id, session_state);

        self.metrics.increment_counter("sessions_initialized", 1, &[]);
        ctx.info("Session initialized successfully");

        Ok(())
    }

    /// Execute a request on a session.
    pub async fn execute(&self, request: ExecutionRequest) -> Result<ExecutionResult, PipelineError> {
        let ctx = LogContext::new("execute")
            .with_session_id(request.session_id.to_string());

        info!("Executing request on session {} ({} input tokens)", 
            request.session_id, request.input_tokens.len());

        let start = std::time::Instant::now();

        // Get session state
        let mut sessions = self.active_sessions.write().await;
        let session = sessions.get_mut(&request.session_id)
            .ok_or_else(|| PipelineError::SessionNotFound(request.session_id))?;

        if session.status != SessionStatus::Initializing && session.status != SessionStatus::Running {
            return Err(PipelineError::SessionNotReady(format!(
                "Session status: {:?}", session.status
            )));
        }

        session.status = SessionStatus::Running;

        // Execute pipeline stages
        let mut tokens = request.input_tokens.clone();
        let mut generated_tokens = Vec::new();

        // Prefill phase
        let prefill_result = self.execute_prefill(session, &tokens).await?;
        tokens.extend(prefill_result);

        // Decode phase (generate tokens)
        for _ in 0..request.max_tokens {
            let decode_result = self.execute_decode(session, &tokens).await?;
            tokens.push(decode_result);
            generated_tokens.push(decode_result);

            // Check for EOS token (simplified - token 2 is typically EOS)
            if decode_result == 2 {
                break;
            }
        }

        let duration = start.elapsed();
        let tokens_per_second = generated_tokens.len() as f32 / duration.as_secs_f32();

        let result = ExecutionResult {
            session_id: request.session_id,
            tokens: generated_tokens,
            completion_time_ms: duration.as_millis() as u32,
            tokens_per_second,
        };

        self.metrics.record_histogram("execution_time_ms", duration.as_secs_f64(), &[]);
        self.metrics.record_gauge("tokens_per_second", tokens_per_second, &[]);
        ctx.info(&format!("Execution completed in {}ms ({} tokens/s)", 
            duration.as_millis(), tokens_per_second));

        Ok(result)
    }

    /// Execute prefill phase (process input tokens).
    async fn execute_prefill(&self, session: &mut SessionState, tokens: &[u32]) -> Result<Vec<u32>, PipelineError> {
        info!("Executing prefill phase with {} tokens", tokens.len());

        // In a real implementation, this would:
        // 1. Distribute input tokens to the first stage
        // 2. Execute pipeline stages in sequence
        // 3. Return processed tokens

        // Simulate prefill execution
        for stage_state in &mut session.stage_states {
            stage_state.status = StageStatus::Executing;
            
            // Simulate stage execution time
            let execution_time_ms = (tokens.len() as u32 * 2).min(100); // 2ms per token, max 100ms
            tokio::time::sleep(std::time::Duration::from_millis(execution_time_ms as u64)).await;
            
            stage_state.status = StageStatus::Completed;
            stage_state.last_execution = Some(chrono::Utc::now());
        }

        // Return processed tokens (simplified - just echo for now)
        Ok(tokens.to_vec())
    }

    /// Execute decode phase (generate one token).
    async fn execute_decode(&self, session: &mut SessionState, context: &[u32]) -> Result<u32, PipelineError> {
        // In a real implementation, this would:
        // 1. Pass context through pipeline stages
        // 2. Generate next token from final stage
        // 3. Return the generated token

        // Simulate decode execution
        for stage_state in &mut session.stage_states {
            stage_state.status = StageStatus::Executing;
            
            // Simulate stage execution time
            let execution_time_ms = 10; // Fixed 10ms per decode step
            tokio::time::sleep(std::time::Duration::from_millis(execution_time_ms as u64)).await;
            
            stage_state.status = StageStatus::Ready;
            stage_state.last_execution = Some(chrono::Utc::now());
        }

        // Simulate token generation (random for now)
        let token = (context.last().unwrap_or(&0) + 1) % 1000;
        Ok(token)
    }

    /// Pause a session.
    pub async fn pause_session(&self, session_id: SessionId) -> Result<(), PipelineError> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            session.status = SessionStatus::Paused;
            info!("Session {} paused", session_id);
            Ok(())
        } else {
            Err(PipelineError::SessionNotFound(session_id))
        }
    }

    /// Resume a paused session.
    pub async fn resume_session(&self, session_id: SessionId) -> Result<(), PipelineError> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            if session.status == SessionStatus::Paused {
                session.status = SessionStatus::Running;
                info!("Session {} resumed", session_id);
                Ok(())
            } else {
                Err(PipelineError::SessionNotReady(format!(
                    "Cannot resume session in status: {:?}", session.status
                )))
            }
        } else {
            Err(PipelineError::SessionNotFound(session_id))
        }
    }

    /// Cancel a session.
    pub async fn cancel_session(&self, session_id: SessionId) -> Result<(), PipelineError> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.remove(&session_id) {
            info!("Session {} cancelled", session_id);
            self.metrics.increment_counter("sessions_cancelled", 1, &[]);
            Ok(())
        } else {
            Err(PipelineError::SessionNotFound(session_id))
        }
    }

    /// Get session status.
    pub async fn get_session_status(&self, session_id: SessionId) -> Result<SessionStatus, PipelineError> {
        let sessions = self.active_sessions.read().await;
        
        sessions.get(&session_id)
            .map(|s| s.status.clone())
            .ok_or_else(|| PipelineError::SessionNotFound(session_id))
    }

    /// Get executor statistics.
    pub async fn get_stats(&self) -> ExecutorStats {
        let sessions = self.active_sessions.read().await;
        
        let total_sessions = sessions.len();
        let running_sessions = sessions.values()
            .filter(|s| s.status == SessionStatus::Running)
            .count();
        let paused_sessions = sessions.values()
            .filter(|s| s.status == SessionStatus::Paused)
            .count();

        ExecutorStats {
            total_sessions,
            running_sessions,
            paused_sessions,
        }
    }
}

/// Executor statistics.
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub total_sessions: usize,
    pub running_sessions: usize,
    pub paused_sessions: usize,
}

/// Pipeline execution errors.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Session not found: {0}")]
    SessionNotFound(SessionId),
    
    #[error("Session not ready: {0}")]
    SessionNotReady(String),
    
    #[error("Stage execution failed: {0}")]
    StageExecutionFailed(String),
    
    #[error("Tensor transfer failed: {0}")]
    TensorTransferFailed(String),
    
    #[error("Invalid execution plan: {0}")]
    InvalidPlan(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use cluster_types::{ParallelismStrategy, PerformanceEstimate, KvCachePlan, KvCachePolicy};

    #[tokio::test]
    async fn test_pipeline_executor() {
        let executor = PipelineExecutor::new();
        
        let session_id = Uuid::new_v4();
        let plan = ExecutionPlan {
            model_id: "test-model".to_string(),
            strategy: ParallelismStrategy::Pipeline,
            estimated: PerformanceEstimate {
                load_seconds: 5.0,
                prompt_tokens_per_second: 50.0,
                decode_tokens_per_second: 15.0,
            },
            stages: vec![
                StagePlan {
                    id: 0,
                    node_id: Uuid::new_v4(),
                    device: "cpu:0".to_string(),
                    layers: vec![0, 1, 2],
                    weight_bytes: 1024 * 1024,
                },
                StagePlan {
                    id: 1,
                    node_id: Uuid::new_v4(),
                    device: "cpu:0".to_string(),
                    layers: vec![3, 4, 5],
                    weight_bytes: 1024 * 1024,
                },
            ],
            kv_cache: KvCachePlan {
                policy: KvCachePolicy::StageLocal,
                reserved_bytes: 64 * 1024 * 1024,
            },
        };

        executor.initialize_session(session_id, plan).await.unwrap();

        let request = ExecutionRequest {
            session_id,
            input_tokens: vec![1, 2, 3],
            max_tokens: 10,
            temperature: 0.7,
        };

        let result = executor.execute(request).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.session_id, session_id);
        assert!(!result.tokens.is_empty());
    }

    #[tokio::test]
    async fn test_session_lifecycle() {
        let executor = PipelineExecutor::new();
        
        let session_id = Uuid::new_v4();
        let plan = ExecutionPlan {
            model_id: "test-model".to_string(),
            strategy: ParallelismStrategy::Pipeline,
            estimated: PerformanceEstimate {
                load_seconds: 5.0,
                prompt_tokens_per_second: 50.0,
                decode_tokens_per_second: 15.0,
            },
            stages: vec![],
            kv_cache: KvCachePlan {
                policy: KvCachePolicy::StageLocal,
                reserved_bytes: 64 * 1024 * 1024,
            },
        };

        executor.initialize_session(session_id, plan).await.unwrap();
        
        executor.pause_session(session_id).await.unwrap();
        executor.resume_session(session_id).await.unwrap();
        executor.cancel_session(session_id).await.unwrap();

        let status = executor.get_session_status(session_id).await;
        assert!(status.is_err());
    }
}