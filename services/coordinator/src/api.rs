//! HTTP/REST API server for cluster management and OpenAI-compatible endpoints.

use crate::ClusterState;
use hyper::{Body, Request, Response, Server, Method, StatusCode};
use hyper::service::{make_service_fn, service_fn};
use std::convert::Infallible;
use std::sync::Arc;
use tracing::{info, error};
use serde_json::json;
use uuid::Uuid;
use futures::stream::{self, StreamExt};

use pipeline_executor::{PipelineExecutor, ExecutionRequest};
use scheduler::{Scheduler, ScheduleRequest, RequestPriority};
use planner::{Planner, PlanningRequest, ModelMetadata};
use ggml_runtime::GgmlRuntime;

pub struct ApiServer {
    state: ClusterState,
    executor: Arc<PipelineExecutor>,
    scheduler: Arc<Scheduler>,
    planner: Arc<Planner>,
    runtime: Arc<GgmlRuntime>,
}

impl ApiServer {
    pub fn new(state: ClusterState) -> Self {
        let cluster = state.get_logical_cluster().await;
        let executor = Arc::new(PipelineExecutor::new());
        let scheduler = Arc::new(Scheduler::new(cluster.clone()));
        let planner = Arc::new(Planner::new());
        let runtime = Arc::new(GgmlRuntime::new());
        
        Self { state, executor, scheduler, planner, runtime }
    }

    pub async fn serve(self, addr: std::net::SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
        let state = Arc::new(self.state);
        let executor = self.executor.clone();
        let scheduler = self.scheduler.clone();
        let planner = self.planner.clone();
        let runtime = self.runtime.clone();

        let make_svc = make_service_fn(move |_conn| {
            let state = state.clone();
            let executor = executor.clone();
            let scheduler = scheduler.clone();
            let planner = planner.clone();
            let runtime = runtime.clone();
            async move {
                Ok::<_, Infallible>(service_fn(move |req| {
                    handle_request(req, state.clone(), executor.clone(), scheduler.clone(), planner.clone(), runtime.clone())
                }))
            }
        });

        let server = Server::bind(&addr).serve(make_svc);

        info!("API server listening on http://{}", addr);

        server.await?;

        Ok(())
    }
}

async fn handle_request(
    req: Request<Body>,
    state: Arc<ClusterState>,
    executor: Arc<PipelineExecutor>,
    scheduler: Arc<Scheduler>,
    planner: Arc<Planner>,
    runtime: Arc<GgmlRuntime>,
) -> Result<Response<Body>, Infallible> {
    let path = req.uri().path();
    let method = req.method();

    info!("{} {}", method, path);

    let response = match (method, path) {
        (&Method::GET, "/api/cluster") => handle_get_cluster(state).await,
        (&Method::GET, "/api/cluster/nodes") => handle_get_nodes(state).await,
        (&Method::GET, "/api/v1/models") => handle_list_models().await,
        (&Method::POST, "/api/v1/chat/completions") => {
            handle_chat_completion(req, state, executor, scheduler, planner, runtime).await
        }
        (&Method::POST, "/api/v1/chat/completions/stream") => {
            handle_chat_completion_stream(req, state, executor, scheduler, planner, runtime).await
        }
        _ => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap(),
    };

    Ok(response)
}

async fn handle_get_cluster(state: Arc<ClusterState>) -> Response<Body> {
    let cluster = state.get_logical_cluster().await;
    
    match serde_json::to_string_pretty(&cluster) {
        Ok(json) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(json))
            .unwrap(),
        Err(e) => {
            error!("Failed to serialize cluster: {}", e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("Internal Server Error"))
                .unwrap()
        }
    }
}

async fn handle_get_nodes(state: Arc<ClusterState>) -> Response<Body> {
    let nodes = state.nodes.read().await;
    
    match serde_json::to_string_pretty(&*nodes) {
        Ok(json) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(json))
            .unwrap(),
        Err(e) => {
            error!("Failed to serialize nodes: {}", e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("Internal Server Error"))
                .unwrap()
        }
    }
}

async fn handle_list_models() -> Response<Body> {
    let models = serde_json::json!({
        "object": "list",
        "data": []
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(models.to_string()))
        .unwrap()
}

async fn handle_chat_completion(
    req: Request<Body>,
    state: Arc<ClusterState>,
    executor: Arc<PipelineExecutor>,
    scheduler: Arc<Scheduler>,
    planner: Arc<Planner>,
    runtime: Arc<GgmlRuntime>,
) -> Response<Body> {
    // Parse request body
    let body = match hyper::body::to_bytes(req.into_body()).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return error_response("Failed to read request body");
        }
    };

    let request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to parse request: {}", e);
            return error_response("Invalid request format");
        }
    };

    info!("Chat completion request: model={}, messages={}", 
        request.model, request.messages.len());

    // Create session ID
    let session_id = Uuid::new_v4();

    // Get cluster state
    let cluster = state.get_logical_cluster().await;

    // Create execution plan
    let planning_request = PlanningRequest {
        model_id: request.model.clone(),
        model_metadata: ModelMetadata {
            parameter_count: 7_000_000_000, // 7B parameters
            layer_count: 32,
            architecture: "llama2".to_string(),
            quantization: "q4_k_m".to_string(),
            estimated_weight_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
        },
        context_length: 4096,
        batch_size: 1,
        preferred_strategy: None,
    };

    let plan = match planner.create_plan(planning_request, &cluster) {
        Ok(plan) => plan,
        Err(e) => {
            error!("Failed to create execution plan: {}", e);
            return error_response(&format!("Failed to create execution plan: {}", e));
        }
    };

    // Initialize session
    if let Err(e) = executor.initialize_session(session_id, plan).await {
        error!("Failed to initialize session: {}", e);
        return error_response(&format!("Failed to initialize session: {}", e));
    }

    // Schedule session
    let schedule_request = ScheduleRequest {
        session_id,
        model_id: request.model.clone(),
        context_length: 4096,
        batch_size: 1,
        priority: RequestPriority::Normal,
    };

    if let Err(e) = scheduler.schedule(schedule_request).await {
        error!("Failed to schedule session: {}", e);
        return error_response(&format!("Failed to schedule session: {}", e));
    }

    // Execute request
    let input_tokens = match tokenize_messages_with_runtime(&request.messages, &runtime).await {
        Ok(tokens) => tokens,
        Err(e) => {
            error!("Tokenization failed: {}", e);
            return error_response(&format!("Tokenization failed: {}", e));
        }
    };
    
    let execution_request = ExecutionRequest {
        session_id,
        input_tokens,
        max_tokens: request.max_tokens.unwrap_or(100),
        temperature: request.temperature.unwrap_or(0.7),
    };

    let result = match executor.execute(execution_request).await {
        Ok(result) => result,
        Err(e) => {
            error!("Execution failed: {}", e);
            return error_response(&format!("Execution failed: {}", e));
        }
    };

    // Convert tokens back to text using runtime
    let content = match runtime.detokenize(&result.tokens).await {
        Ok(text) => text,
        Err(e) => {
            error!("Detokenization failed: {}", e);
            return error_response(&format!("Detokenization failed: {}", e));
        }
    };

    let response = json!({
        "id": format!("chatcmpl-{}", session_id),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": request.messages.len() * 10, // Simplified
            "completion_tokens": result.tokens.len(),
            "total_tokens": request.messages.len() * 10 + result.tokens.len(),
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(response.to_string()))
        .unwrap()
}

fn error_response(message: &str) -> Response<Body> {
    let response = json!({
        "error": {
            "message": message,
            "type": "api_error",
        }
    });

    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header("content-type", "application/json")
        .body(Body::from(response.to_string()))
        .unwrap()
}

/// Chat completion request (OpenAI-compatible).
#[derive(serde::Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

#[derive(serde::Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Simplified tokenization using runtime (placeholder until tokenizer is loaded).
async fn tokenize_messages_with_runtime(messages: &[ChatMessage], runtime: &Arc<GgmlRuntime>) -> Result<Vec<u32>, String> {
    // Try to use the runtime's tokenizer if available
    let mut all_tokens = Vec::new();
    
    for msg in messages {
        let text = format!("{}: {}", msg.role, msg.content);
        match runtime.tokenize(&text).await {
            Ok(mut tokens) => {
                all_tokens.append(&mut tokens);
            }
            Err(_) => {
                // Fallback to simple byte-based tokenization if tokenizer not loaded
                for byte in msg.content.bytes() {
                    all_tokens.push(byte as u32);
                }
            }
        }
    }
    
    Ok(all_tokens)
}

/// Handle streaming chat completion (SSE).
async fn handle_chat_completion_stream(
    req: Request<Body>,
    state: Arc<ClusterState>,
    executor: Arc<PipelineExecutor>,
    scheduler: Arc<Scheduler>,
    planner: Arc<Planner>,
    runtime: Arc<GgmlRuntime>,
) -> Response<Body> {
    // Parse request body
    let body = match hyper::body::to_bytes(req.into_body()).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return error_response("Failed to read request body");
        }
    };

    let request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to parse request: {}", e);
            return error_response("Invalid request format");
        }
    };

    info!("Streaming chat completion request: model={}, messages={}", 
        request.model, request.messages.len());

    let session_id = Uuid::new_v4();
    let cluster = state.get_logical_cluster().await;

    // Create execution plan
    let planning_request = PlanningRequest {
        model_id: request.model.clone(),
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

    let plan = match planner.create_plan(planning_request, &cluster) {
        Ok(plan) => plan,
        Err(e) => {
            error!("Failed to create execution plan: {}", e);
            return error_response(&format!("Failed to create execution plan: {}", e));
        }
    };

    if let Err(e) = executor.initialize_session(session_id, plan).await {
        error!("Failed to initialize session: {}", e);
        return error_response(&format!("Failed to initialize session: {}", e));
    }

    let schedule_request = ScheduleRequest {
        session_id,
        model_id: request.model.clone(),
        context_length: 4096,
        batch_size: 1,
        priority: RequestPriority::Normal,
    };

    if let Err(e) = scheduler.schedule(schedule_request).await {
        error!("Failed to schedule session: {}", e);
        return error_response(&format!("Failed to schedule session: {}", e));
    }

    // Create SSE stream
    let stream = stream::unfold(
        (session_id, request, executor, runtime, 0),
        |(session_id, request, executor, runtime, token_index)| async move {
            if token_index >= request.max_tokens.unwrap_or(100) {
                return None;
            }

            // Generate one token at a time
            let input_tokens = vec![token_index as u32];
            let execution_request = ExecutionRequest {
                session_id,
                input_tokens,
                max_tokens: 1,
                temperature: request.temperature.unwrap_or(0.7),
            };

            match executor.execute(execution_request).await {
                Ok(result) => {
                    if let Some(&token) = result.tokens.first() {
                        match runtime.detokenize(&[token]).await {
                            Ok(text) => {
                                let chunk = json!({
                                    "id": format!("chatcmpl-{}", session_id),
                                    "object": "chat.completion.chunk",
                                    "created": chrono::Utc::now().timestamp(),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": text,
                                        },
                                        "finish_reason": if token == 2 { "stop" } else { null },
                                    }]
                                });

                                let data = format!("data: {}\n\n", chunk.to_string());
                                Some((Ok::<_, String>(data), (session_id, request, executor, runtime, token_index + 1)))
                            }
                            Err(e) => {
                                error!("Detokenization failed: {}", e);
                                None
                            }
                        }
                    } else {
                        None
                    }
                }
                Err(e) => {
                    error!("Execution failed: {}", e);
                    None
                }
            }
        }
    );

    let body = Body::wrap_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}