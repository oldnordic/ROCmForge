//! InferenceServer main implementation and server lifecycle
//!
//! This module contains the core InferenceServer struct that manages
//! text generation requests and coordinates with the inference engine.
//! It also provides the server initialization and startup logic.

use crate::engine::InferenceEngine;
use crate::logging::init_logging_default;
use crate::metrics::{MetricRegistry, Metrics};
use crate::models::discover_models_with_cache;
use crate::otel_traces::{init_trace_store, TraceConfig};
use crate::scheduler::{GenerationRequest as SchedulerRequest, RequestState};
use crate::tokenizer::{
    embedded_tokenizer_from_gguf, infer_tokenizer_path, tokenizer_cache_counters,
    TokenizerAdapter,
};
use futures::stream::{self, Stream};
use std::collections::HashMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info};

// Re-export types from submodules
pub use crate::http::types::{
    GenerateRequest, GenerateResponse, GenerationState, ServerError, ServerResult,
    ServerState, TokenStream,
};

/// SSE event stream type for token streaming
type EventStreamResult = Pin<Box<dyn Stream<Item = Result<axum::response::sse::Event, Infallible>> + Send>>;

// InferenceServer - note: context feature requires separate routing
// due to HNSW index not being Send+Sync
#[derive(Clone)]
pub struct InferenceServer {
    pub state: ServerState,
    pub engine: Option<Arc<InferenceEngine>>,
    pub tokenizer: TokenizerAdapter,
    pub metrics_registry: MetricRegistry,
}

impl InferenceServer {
    pub fn new(engine: Option<Arc<InferenceEngine>>, tokenizer: TokenizerAdapter) -> Self {
        InferenceServer {
            state: Arc::new(RwLock::new(HashMap::new())),
            engine,
            tokenizer,
            metrics_registry: MetricRegistry::new(),
        }
    }

    fn engine(&self) -> Result<&Arc<InferenceEngine>, ServerError> {
        self.engine.as_ref().ok_or_else(|| {
            ServerError::InternalError(
                "Inference engine not initialized; cannot service request".to_string(),
            )
        })
    }

    fn require_engine(&self) -> Result<(), ServerError> {
        if self.engine.is_none() {
            return Err(ServerError::InternalError(
                "Inference engine not initialized; start the server with --gguf or ROCMFORGE_GGUF"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn tokenize_prompt(&self, prompt: &str) -> Vec<u32> {
        self.tokenizer.encode(prompt)
    }

    #[allow(dead_code)] // Reserved for future token-to-text conversion in API responses
    fn token_to_text(&self, token: u32) -> String {
        self.tokenizer.decode_token(token)
    }

    fn tokens_to_text(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }

    async fn update_generation_state_from_request(
        &self,
        request: &SchedulerRequest,
    ) -> Result<(), ServerError> {
        let mut state_map = self.state.write().await;
        let state = state_map
            .get_mut(&request.request_id)
            .ok_or(ServerError::RequestNotFound(request.request_id))?;

        state.generated_tokens = request.generated_tokens.clone();
        state.generated_text = self.tokens_to_text(&request.generated_tokens);
        if request.is_complete() {
            let reason = request
                .finish_reason
                .clone()
                .or_else(|| match request.state {
                    RequestState::Cancelled => Some("cancelled".to_string()),
                    RequestState::Failed => Some("failed".to_string()),
                    RequestState::Completed => Some("completed".to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| "completed".to_string());
            state.finish(reason);
        }

        Ok(())
    }

    async fn get_engine_request(&self, request_id: u32) -> Result<SchedulerRequest, ServerError> {
        let engine = self.engine()?;
        let status = engine
            .get_request_status(request_id)
            .await
            .map_err(|e| ServerError::GenerationFailed(e.to_string()))?;
        status.ok_or(ServerError::RequestNotFound(request_id))
    }

    async fn wait_for_completion(&self, request_id: u32) -> Result<(), ServerError> {
        loop {
            let status = self.get_engine_request(request_id).await?;
            self.update_generation_state_from_request(&status).await?;
            if status.is_complete() {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn build_response(&self, request_id: u32) -> Result<GenerateResponse, ServerError> {
        let state_map = self.state.read().await;
        let state = state_map
            .get(&request_id)
            .ok_or(ServerError::RequestNotFound(request_id))?;

        Ok(GenerateResponse {
            request_id: state.request_id,
            text: state.generated_text.clone(),
            tokens: state.generated_tokens.clone(),
            finished: state.finished,
            finish_reason: state.finish_reason.clone(),
        })
    }

    async fn start_engine_request(
        &self,
        request: &GenerateRequest,
        max_tokens: usize,
    ) -> Result<u32, ServerError> {
        let engine = self.engine()?;
        let prompt_tokens = self.tokenize_prompt(&request.prompt);

        let request_id = engine
            .submit_request(
                prompt_tokens,
                max_tokens,
                request.temperature.unwrap_or(1.0),
                request.top_k.unwrap_or(50),
                request.top_p.unwrap_or(0.9),
            )
            .await
            .map_err(|e| ServerError::GenerationFailed(e.to_string()))?;

        {
            let mut state_map = self.state.write().await;
            state_map.insert(
                request_id,
                GenerationState::new(request_id, request.prompt.clone(), max_tokens),
            );
        }

        Ok(request_id)
    }

    pub async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, ServerError> {
        self.require_engine()?;
        self.generate_with_engine(request).await
    }

    pub async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<EventStreamResult, ServerError> {
        self.require_engine()?;
        self.generate_stream_with_engine(request).await
    }

    pub async fn cancel_generation(
        &self,
        request_id: u32,
    ) -> Result<GenerateResponse, ServerError> {
        if let Some(engine) = &self.engine {
            engine
                .cancel_request(request_id)
                .await
                .map_err(|e| ServerError::GenerationFailed(e.to_string()))?;
        }

        {
            let mut state_map = self.state.write().await;
            let state = state_map
                .get_mut(&request_id)
                .ok_or(ServerError::RequestNotFound(request_id))?;
            state.finish("cancelled".to_string());
        }

        self.build_response(request_id).await
    }

    pub async fn get_request_status(
        &self,
        request_id: u32,
    ) -> Result<GenerateResponse, ServerError> {
        self.require_engine()?;
        self.build_response(request_id).await
    }

    async fn generate_with_engine(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, ServerError> {
        let max_tokens = request.max_tokens.unwrap_or(100);
        let request_id = self.start_engine_request(&request, max_tokens).await?;

        self.wait_for_completion(request_id).await?;
        self.build_response(request_id).await
    }

    async fn generate_stream_with_engine(
        &self,
        request: GenerateRequest,
    ) -> Result<EventStreamResult, ServerError> {
        use axum::response::sse::Event;

        let max_tokens = request.max_tokens.unwrap_or(100);
        let request_id = self.start_engine_request(&request, max_tokens).await?;

        let state_clone = self.state.clone();
        let engine = self.engine().cloned()?;
        let notifier = engine.subscribe_request(request_id).await;
        let tokenizer = self.tokenizer.clone();

        Ok(Box::pin(stream::unfold(
            (
                engine,
                state_clone,
                tokenizer.clone(),
                request_id,
                0usize,
                max_tokens,
                notifier,
            ),
            move |(engine, state, tokenizer, req_id, last_idx, max_tokens, notifier)| async move {
                loop {
                    match engine.get_request_status(req_id).await {
                        Ok(Some(status)) => {
                            let finished = status.is_complete();
                            let finish_reason =
                                status.finish_reason.clone().or_else(|| match status.state {
                                    RequestState::Cancelled => Some("cancelled".to_string()),
                                    RequestState::Failed => Some("failed".to_string()),
                                    RequestState::Completed => Some("completed".to_string()),
                                    _ => None,
                                });
                            let current_len = status.generated_tokens.len();

                            if current_len > last_idx {
                                let token = status.generated_tokens[last_idx];
                                let token_text = tokenizer.decode_token(token);

                                {
                                    let mut state_map = state.write().await;
                                    if let Some(gen_state) = state_map.get_mut(&req_id) {
                                        gen_state.add_token(token, token_text.clone());
                                        if finished {
                                            let reason = finish_reason
                                                .clone()
                                                .unwrap_or_else(|| "completed".to_string());
                                            gen_state.finish(reason);
                                        }
                                    }
                                }

                                let event = Event::default()
                                    .json_data(TokenStream {
                                        request_id: req_id,
                                        token,
                                        text: token_text,
                                        finished: finished && current_len == last_idx + 1,
                                        finish_reason: finish_reason.clone().filter(|_| finished),
                                    })
                                    // UNWRAP: TokenStream only contains simple serializable types
                                    .unwrap();

                                let next_state = (
                                    engine.clone(),
                                    state.clone(),
                                    tokenizer.clone(),
                                    req_id,
                                    last_idx + 1,
                                    max_tokens,
                                    notifier.clone(),
                                );

                                return Some((Ok(event), next_state));
                            }

                            if finished {
                                return None;
                            }
                        }
                        Ok(None) => return None,
                        Err(e) => {
                            error!("Failed to fetch request {} status: {}", req_id, e);
                            return None;
                        }
                    }

                    if let Some(notify) = &notifier {
                        notify.notified().await;
                    } else {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
            },
        )))
    }
}

/// Run the HTTP inference server
///
/// Initializes logging, telemetry, the inference engine, and starts
/// the Axum HTTP server on the specified address.
pub async fn run_server(
    addr: &str,
    gguf_path: Option<&str>,
    tokenizer_path: Option<&str>,
) -> ServerResult<()> {
    // Initialize tracing for structured logging (idempotent)
    init_logging_default();

    // Initialize OpenTelemetry trace store (idempotent)
    init_trace_store(TraceConfig::default());

    let model_path = gguf_path
        .map(|s| s.to_string())
        .or_else(|| std::env::var("ROCMFORGE_GGUF").ok());
    let tokenizer_path = tokenizer_path
        .map(|s| s.to_string())
        .or_else(|| std::env::var("ROCMFORGE_TOKENIZER").ok())
        .or_else(|| model_path.as_deref().and_then(infer_tokenizer_path));
    if let Some(path) = &tokenizer_path {
        info!("Using tokenizer definition {}", path);
    }
    let embedded_info = if tokenizer_path.is_none() {
        model_path.as_deref().and_then(|path| {
            embedded_tokenizer_from_gguf(path).map(|info| (info, path.to_string()))
        })
    } else {
        None
    };
    if tokenizer_path.is_none() {
        if let Some((ref info, ref source)) = embedded_info {
            if info.cached {
                info!("Using cached tokenizer embedded in {}", source);
            } else if info.refreshed {
                info!("Loaded tokenizer embedded in {}", source);
            }
        } else {
            info!("Tokenizer path not provided; falling back to hash-based tokenizer");
        }
    }
    let tokenizer = TokenizerAdapter::from_spec(
        tokenizer_path.as_deref(),
        embedded_info.as_ref().map(|(info, _)| info.json.as_str()),
    );
    let (hits, misses) = tokenizer_cache_counters();
    info!("Tokenizer cache stats: hits={}, misses={}", hits, misses);

    let model_path = model_path.ok_or_else(|| {
        anyhow::anyhow!(
            "ROCMFORGE_GGUF not set; pass --gguf or set ROCMFORGE_GGUF before starting the server"
        )
    })?;

    info!("Loading GGUF model from {}", model_path);
    // PHASE 24 FIX: Use from_gguf() instead of new(EngineConfig::default()) + load_gguf_model()
    // This creates the paged KV cache with correct model dimensions instead of wrong defaults
    // (32 heads, 128 head_dim, 1000 pages) which wastes memory and may cause dimension mismatches.
    // Matches the fix applied to rocmforge_cli.rs:create_engine()
    let engine = InferenceEngine::from_gguf(&model_path).await?;
    let engine = Arc::new(engine);
    engine.start().await?;

    // Start inference loop in background - don't block on it!
    // This follows the same pattern as rocmforge_cli.rs:474-479
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        // Ignore errors on shutdown
        let _ = engine_clone.run_inference_loop().await;
    });

    // Create and initialize metrics
    let metrics = Arc::new(Metrics::new());
    let server = InferenceServer::new(Some(engine), tokenizer.clone());
    server.metrics_registry.init(metrics.clone()).await;

    // Import create_router from routes module
    use crate::http::routes::create_router;
    let app = create_router(server);

    info!("Starting ROCmForge server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenizerAdapter;
    use std::sync::Arc;

    #[test]
    fn test_generation_state_creation() {
        let state = GenerationState::new(1, "Hello".to_string(), 10);

        assert_eq!(state.request_id, 1);
        assert_eq!(state.prompt, "Hello");
        assert_eq!(state.max_tokens, 10);
        assert!(!state.finished);
        assert!(state.generated_tokens.is_empty());
        assert!(state.generated_text.is_empty());
    }

    #[test]
    fn test_generation_state_add_token() {
        let mut state = GenerationState::new(1, "Hello".to_string(), 2);

        state.add_token(42, " world".to_string());
        assert_eq!(state.generated_tokens, vec![42]);
        assert_eq!(state.generated_text, " world");
        assert!(!state.finished);

        state.add_token(43, "!".to_string());
        assert_eq!(state.generated_tokens, vec![42, 43]);
        assert_eq!(state.generated_text, " world!");
        assert!(state.finished);
        assert_eq!(state.finish_reason, Some("length".to_string()));
    }

    #[test]
    fn test_generation_state_finish() {
        let mut state = GenerationState::new(1, "Hello".to_string(), 10);

        state.finish("stop".to_string());
        assert!(state.finished);
        assert_eq!(state.finish_reason, Some("stop".to_string()));
    }

    #[tokio::test]
    async fn test_generate_request() {
        let server = InferenceServer::new(None, TokenizerAdapter::default());

        let request = GenerateRequest {
            prompt: "Hello world".to_string(),
            max_tokens: Some(5),
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.9),
            stream: Some(false),
        };

        let response = server.generate(request).await;
        // Should fail because no engine is loaded
        assert!(response.is_err());

        let err = response.unwrap_err();
        assert!(matches!(err, ServerError::InternalError(_)));
        assert!(err.to_string().contains("Inference engine not initialized"));
    }

    #[tokio::test]
    async fn test_get_request_status() {
        let server = InferenceServer::new(None, TokenizerAdapter::default());

        // Test error path when model is not initialized
        let request = GenerateRequest {
            prompt: "Test".to_string(),
            max_tokens: Some(3),
            temperature: None,
            top_k: None,
            top_p: None,
            stream: None,
        };

        let gen_response = server.generate(request).await;
        assert!(gen_response.is_err()); // Should fail without model
    }

    #[tokio::test]
    async fn test_get_nonexistent_request_status() {
        let server = InferenceServer::new(None, TokenizerAdapter::default());

        let response = server.get_request_status(999).await;
        // Should return an error (RequestNotFound or other error is acceptable)
        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_server_creation_does_not_require_engine() {
        // Server can be created without an engine (for testing purposes)
        let tokenizer = TokenizerAdapter::default();
        let server = InferenceServer::new(None, tokenizer);

        // Server should reject requests without engine
        let request = GenerateRequest {
            prompt: "Test".to_string(),
            max_tokens: Some(5),
            temperature: None,
            top_k: None,
            top_p: None,
            stream: None,
        };

        let response = server.generate(request).await;
        assert!(response.is_err());
    }
}
