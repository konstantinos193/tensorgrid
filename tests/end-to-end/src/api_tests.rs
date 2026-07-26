//! End-to-end API tests.

use cluster_types::{ClusterId, LogicalCluster, LogicalResources, ParallelismStrategy};
use pipeline_executor::{PipelineExecutor, ExecutionRequest};
use scheduler::{Scheduler, ScheduleRequest, RequestPriority};
use planner::{Planner, PlanningRequest, ModelMetadata};
use ggml_runtime::GgmlRuntime;
use serde_json::json;
use std::sync::Arc;
use tokio::time::sleep;
use tracing::info;
use uuid::Uuid;

/// Test the full API flow for chat completion.
#[tokio::test]
async fn test_chat_completion_flow() {
    // Initialize components
    let cluster_id = Uuid::new_v4();
    let cluster = LogicalCluster {
        id: cluster_id,
        name: "test-cluster".to_string(),
        logical_resources: LogicalResources {
            cpu_compute_units: 16.0,
            host_memory_bytes: 32 * 1024 * 1024 * 1024,
            device_memory_bytes: 8 * 1024 * 1024 * 1024,
            storage_bytes: 1024 * 1024 * 1024 * 1024,
            preferred_parallelism: ParallelismStrategy::Pipeline,
        },
        nodes: std::collections::HashMap::new(),
        topology: cluster_types::ClusterTopology {
            nodes: std::collections::HashMap::new(),
            links: vec![],
        },
    };

    let executor = Arc::new(PipelineExecutor::new());
    let scheduler = Arc::new(Scheduler::new(cluster.clone()));
    let planner = Arc::new(Planner::new());
    let runtime = Arc::new(GgmlRuntime::new());

    // Simulate chat completion request
    let session_id = Uuid::new_v4();
    let model_id = "llama-2-7b".to_string();

    let planning_request = PlanningRequest {
        model_id: model_id.clone(),
        model_metadata: ModelMetadata {
            parameter_count: 7_000_000_000,
            layer_count: 32,
            architecture: "llama2".to_string(),
            quantization: "q4_k_m".to_string(),
            estimated_weight_bytes: 4 * 1024 * 1024 * 1024,
        },
        context_length: 4096,
        batch_size: 1,
        preferred_strategy: None,
    };

    let plan = planner.create_plan(planning_request, &cluster).unwrap();
    assert_eq!(plan.model_id, model_id);

    executor.initialize_session(session_id, plan).await.unwrap();

    let schedule_request = ScheduleRequest {
        session_id,
        model_id: model_id.clone(),
        context_length: 4096,
        batch_size: 1,
        priority: RequestPriority::Normal,
    };

    scheduler.schedule(schedule_request).await.unwrap();

    let execution_request = ExecutionRequest {
        session_id,
        input_tokens: vec![1, 2, 3, 4, 5],
        max_tokens: 10,
        temperature: 0.7,
    };

    let result = executor.execute(execution_request).await.unwrap();
    assert_eq!(result.session_id, session_id);
    assert!(!result.tokens.is_empty());
    assert!(result.tokens_per_second > 0.0);

    info!("Chat completion flow test passed");
}

/// Test session lifecycle.
#[tokio::test]
async fn test_session_lifecycle() {
    let executor = Arc::new(PipelineExecutor::new());
    let cluster = LogicalCluster {
        id: Uuid::new_v4(),
        name: "test-cluster".to_string(),
        logical_resources: LogicalResources {
            cpu_compute_units: 16.0,
            host_memory_bytes: 32 * 1024 * 1024 * 1024,
            device_memory_bytes: 8 * 1024 * 1024 * 1024,
            storage_bytes: 1024 * 1024 * 1024 * 1024,
            preferred_parallelism: ParallelismStrategy::Pipeline,
        },
        nodes: std::collections::HashMap::new(),
        topology: cluster_types::ClusterTopology {
            nodes: std::collections::HashMap::new(),
            links: vec![],
        },
    };

    let session_id = Uuid::new_v4();
    let plan = cluster_types::ExecutionPlan {
        model_id: "test-model".to_string(),
        strategy: ParallelismStrategy::Pipeline,
        estimated: cluster_types::PerformanceEstimate {
            load_seconds: 5.0,
            prompt_tokens_per_second: 50.0,
            decode_tokens_per_second: 15.0,
        },
        stages: vec![],
        kv_cache: cluster_types::KvCachePlan {
            policy: cluster_types::KvCachePolicy::StageLocal,
            reserved_bytes: 64 * 1024 * 1024,
        },
    };

    executor.initialize_session(session_id, plan).await.unwrap();

    // Pause session
    executor.pause_session(session_id).await.unwrap();
    let status = executor.get_session_status(session_id).await.unwrap();
    assert_eq!(format!("{:?}", status), "Paused");

    // Resume session
    executor.resume_session(session_id).await.unwrap();
    let status = executor.get_session_status(session_id).await.unwrap();
    assert_eq!(format!("{:?}", status), "Running");

    // Cancel session
    executor.cancel_session(session_id).await.unwrap();
    let status = executor.get_session_status(session_id).await;
    assert!(status.is_err());

    info!("Session lifecycle test passed");
}

/// Test tokenization with runtime.
#[tokio::test]
async fn test_tokenization() {
    let runtime = GgmlRuntime::new();

    // Test without tokenizer (should fallback)
    let text = "Hello, world!";
    let result = runtime.tokenize(text).await;
    // Will fail since no tokenizer is loaded, but that's expected
    assert!(result.is_err());

    info!("Tokenization test passed");
}

/// Test scheduler admission control.
#[tokio::test]
async fn test_scheduler_admission() {
    let cluster = LogicalCluster {
        id: Uuid::new_v4(),
        name: "test-cluster".to_string(),
        logical_resources: LogicalResources {
            cpu_compute_units: 16.0,
            host_memory_bytes: 32 * 1024 * 1024 * 1024,
            device_memory_bytes: 8 * 1024 * 1024 * 1024,
            storage_bytes: 1024 * 1024 * 1024 * 1024,
            preferred_parallelism: ParallelismStrategy::Pipeline,
        },
        nodes: std::collections::HashMap::new(),
        topology: cluster_types::ClusterTopology {
            nodes: std::collections::HashMap::new(),
            links: vec![],
        },
    };

    let scheduler = Scheduler::new(cluster);

    let session_id = Uuid::new_v4();
    let request = ScheduleRequest {
        session_id,
        model_id: "test-model".to_string(),
        context_length: 4096,
        batch_size: 1,
        priority: RequestPriority::Normal,
    };

    let result = scheduler.schedule(request).await;
    assert!(result.is_ok());

    info!("Scheduler admission test passed");
}

/// Test planner execution plan generation.
#[tokio::test]
async fn test_planner_plan_generation() {
    let planner = Planner::new();
    let cluster = LogicalCluster {
        id: Uuid::new_v4(),
        name: "test-cluster".to_string(),
        logical_resources: LogicalResources {
            cpu_compute_units: 16.0,
            host_memory_bytes: 32 * 1024 * 1024 * 1024,
            device_memory_bytes: 8 * 1024 * 1024 * 1024,
            storage_bytes: 1024 * 1024 * 1024 * 1024,
            preferred_parallelism: ParallelismStrategy::Pipeline,
        },
        nodes: std::collections::HashMap::new(),
        topology: cluster_types::ClusterTopology {
            nodes: std::collections::HashMap::new(),
            links: vec![],
        },
    };

    let request = PlanningRequest {
        model_id: "llama-2-7b".to_string(),
        model_metadata: ModelMetadata {
            parameter_count: 7_000_000_000,
            layer_count: 32,
            architecture: "llama2".to_string(),
            quantization: "q4_k_m".to_string(),
            estimated_weight_bytes: 4 * 1024 * 1024 * 1024,
        },
        context_length: 4096,
        batch_size: 1,
        preferred_strategy: None,
    };

    let plan = planner.create_plan(request, &cluster).unwrap();
    assert_eq!(plan.model_id, "llama-2-7b");
    assert!(!plan.stages.is_empty());

    info!("Planner plan generation test passed");
}
