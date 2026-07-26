//! End-to-end execution tests.

use cluster_types::{ClusterId, LogicalCluster, LogicalResources, ParallelismStrategy, StagePlan, KvCachePlan, KvCachePolicy};
use pipeline_executor::{PipelineExecutor, ExecutionRequest};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

/// Test pipeline execution with multiple stages.
#[tokio::test]
async fn test_pipeline_execution() {
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
        stages: vec![
            StagePlan {
                id: 0,
                node_id: Uuid::new_v4(),
                device: "cpu:0".to_string(),
                layers: vec![0, 1, 2, 3],
                weight_bytes: 1024 * 1024,
            },
            StagePlan {
                id: 1,
                node_id: Uuid::new_v4(),
                device: "cpu:0".to_string(),
                layers: vec![4, 5, 6, 7],
                weight_bytes: 1024 * 1024,
            },
        ],
        kv_cache: KvCachePlan {
            policy: KvCachePolicy::StageLocal,
            reserved_bytes: 64 * 1024 * 1024,
        },
    };

    executor.initialize_session(session_id, plan).await.unwrap();

    let execution_request = ExecutionRequest {
        session_id,
        input_tokens: vec![1, 2, 3, 4, 5],
        max_tokens: 20,
        temperature: 0.7,
    };

    let result = executor.execute(execution_request).await.unwrap();
    assert_eq!(result.session_id, session_id);
    assert!(!result.tokens.is_empty());
    assert!(result.tokens_per_second > 0.0);

    info!("Pipeline execution test passed");
}

/// Test executor statistics.
#[tokio::test]
async fn test_executor_stats() {
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

    let stats = executor.get_stats().await;
    assert_eq!(stats.total_sessions, 0);
    assert_eq!(stats.running_sessions, 0);

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
        kv_cache: KvCachePlan {
            policy: KvCachePolicy::StageLocal,
            reserved_bytes: 64 * 1024 * 1024,
        },
    };

    executor.initialize_session(session_id, plan).await.unwrap();

    let stats = executor.get_stats().await;
    assert_eq!(stats.total_sessions, 1);

    info!("Executor stats test passed");
}

/// Test concurrent session execution.
#[tokio::test]
async fn test_concurrent_sessions() {
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

    let plan = cluster_types::ExecutionPlan {
        model_id: "test-model".to_string(),
        strategy: ParallelismStrategy::Pipeline,
        estimated: cluster_types::PerformanceEstimate {
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

    // Create multiple sessions
    let session_ids: Vec<_> = (0..3).map(|_| Uuid::new_v4()).collect();
    for session_id in &session_ids {
        executor.initialize_session(*session_id, plan.clone()).await.unwrap();
    }

    let stats = executor.get_stats().await;
    assert_eq!(stats.total_sessions, 3);

    info!("Concurrent sessions test passed");
}
