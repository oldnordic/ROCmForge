//! OpenAI-compatible HTTP server for rocmforge inference.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, sse::Sse, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde_json::json;
use tokio::sync::Mutex;

use crate::api::types::*;
use crate::config::ChatTemplate;
use crate::config::ModelConfig;
use crate::cpu::weights::CpuModelWeights;
use crate::loader::ModelFile;
use crate::tokenizer::BpeTokenizer;

// ── Per-model state ────────────────────────────────────────────────────────────

pub struct ModelEntry {
    pub model_path: String,
    pub config: ModelConfig,
    pub tokenizer: BpeTokenizer,
    pub chat_template: ChatTemplate,
    pub created_at: i64,
    /// One permit per model: serializes inference so concurrent requests
    /// do not race on weights or KV cache.  Arc so acquire_owned() yields
    /// an OwnedSemaphorePermit that is Send for spawn_blocking.
    pub inference_sem: Arc<tokio::sync::Semaphore>,
    /// Cached CPU weights — loaded once at model load time and shared across
    /// all inference requests via Arc so cloning into spawn_blocking is cheap.
    pub cpu_weights: Arc<CpuModelWeights>,
    #[cfg(feature = "gpu")]
    pub gpu_weights: Option<Arc<crate::gpu::GpuModelWeights>>,
    #[cfg(feature = "gpu")]
    pub speculative_engine: Option<Arc<Mutex<crate::gpu::SpeculativeEngine>>>,
}

impl ModelEntry {
    pub fn load(model_path: &str, draft_path: Option<&str>) -> Result<Self, String> {
        let file = ModelFile::open(model_path).map_err(|e| format!("model open: {}", e))?;
        let config = file.config().map_err(|e| format!("config: {}", e))?;
        let tokenizer = file.tokenizer();
        let chat_template = file.chat_template(&config, false); // enable template by default
        let cpu_weights = Arc::new(
            file.load_cpu_weights(&config)
                .map_err(|e| format!("weight load: {}", e))?,
        );

        #[cfg(feature = "gpu")]
        let gpu_weights = {
            let gpu_caps = crate::gpu::detect();
            if let Some(caps) = gpu_caps {
                crate::gpu::GpuDevice::get_or_init(caps.device_id)
                    .map_err(|e| format!("gpu init: {}", e))?;
                let w = file
                    .load_gpu_weights(&config, caps.device_id)
                    .map_err(|e| format!("gpu weight load: {}", e))?;
                Some(Arc::new(w))
            } else {
                None
            }
        };

        #[cfg(feature = "gpu")]
        let speculative_engine = if let Some(dp) = draft_path {
            let gpu_caps = crate::gpu::detect()
                .ok_or("GPU requested for speculative decoding but no AMD GPU detected")?;
            let device = crate::gpu::GpuDevice::get_or_init(gpu_caps.device_id)
                .map_err(|e| format!("gpu init: {}", e))?;
            let engine = crate::gpu::SpeculativeEngine::new(
                &device,
                model_path,
                dp,
                config.max_seq_len.min(2048),
                256,
            )
            .map_err(|e| format!("failed to instantiate speculative engine: {:?}", e))?;
            Some(Arc::new(Mutex::new(engine)))
        } else {
            None
        };

        Ok(ModelEntry {
            model_path: model_path.to_string(),
            config,
            tokenizer,
            chat_template,
            created_at: chrono::Utc::now().timestamp(),
            inference_sem: Arc::new(tokio::sync::Semaphore::new(1)),
            cpu_weights,
            #[cfg(feature = "gpu")]
            gpu_weights,
            #[cfg(feature = "gpu")]
            speculative_engine,
        })
    }
}

// ── Model manager (multi-model registry) ─────────────────────────────────────

pub struct ModelManager {
    models: HashMap<String, Arc<Mutex<ModelEntry>>>,
}

impl ModelManager {
    pub fn new(initial: ModelEntry) -> Self {
        let path = initial.model_path.clone();
        let mut map = HashMap::new();
        map.insert(path, Arc::new(Mutex::new(initial)));
        Self { models: map }
    }

    pub fn try_load(&mut self, path: &str, draft_path: Option<&str>) -> Result<(), String> {
        let entry = ModelEntry::load(path, draft_path)?;
        self.models
            .insert(path.to_string(), Arc::new(Mutex::new(entry)));
        Ok(())
    }

    pub fn try_load_entry(&mut self, entry: ModelEntry) {
        let path = entry.model_path.clone();
        self.models.insert(path, Arc::new(Mutex::new(entry)));
    }

    pub fn unload(&mut self, path: &str) -> bool {
        self.models.remove(path).is_some()
    }

    pub fn get(&self, path: &str) -> Option<Arc<Mutex<ModelEntry>>> {
        self.models.get(path).cloned()
    }

    pub fn keys(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

// ── Router construction ─────────────────────────────────────────────────────────

pub fn create_router(state: Arc<Mutex<ModelManager>>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(create_completion))
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/v1/messages", post(create_messages))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/unload", post(unload_model))
        .route("/v1/models/estimate", post(estimate_vram))
        .route("/v1/vram", get(list_vram))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .with_state(state)
}

// ── Handlers ───────────────────────────────────────────────────────────────────

async fn list_models(State(state): State<Arc<Mutex<ModelManager>>>) -> impl IntoResponse {
    let guard = state.lock().await;
    let now = chrono::Utc::now().timestamp();
    let data: Vec<_> = guard
        .keys()
        .into_iter()
        .map(|id| ModelInfo {
            id,
            object: "model",
            created: now,
            owned_by: "rocmforge",
        })
        .collect();
    (
        StatusCode::OK,
        Json(ModelList {
            object: "list",
            data,
        }),
    )
}

async fn load_model(
    State(state): State<Arc<Mutex<ModelManager>>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let path = req.model;
    if !std::path::Path::new(&path).exists() {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("Model file not found: {}", path),
            "invalid_model",
        );
    }
    if let Some(ref dp) = req.draft_model {
        if !std::path::Path::new(dp).exists() {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Draft model file not found: {}", dp),
                "invalid_draft_model",
            );
        }
    }

    let guard = state.lock().await;
    if guard.get(&path).is_some() {
        return (
            StatusCode::OK,
            Json(LoadModelResponse {
                id: path,
                object: "model",
                status: "already_loaded",
            }),
        )
            .into_response();
    }
    drop(guard);

    let path2 = path.clone();
    let draft_path2 = req.draft_model.clone();
    let entry =
        match tokio::task::spawn_blocking(move || ModelEntry::load(&path2, draft_path2.as_deref()))
            .await
        {
            Ok(Ok(e)) => e,
            Ok(Err(e)) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to load model: {}", e),
                    "load_error",
                );
            }
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Task panic: {}", e),
                    "load_error",
                );
            }
        };

    let mut guard = state.lock().await;
    if guard.get(&path).is_some() {
        return (
            StatusCode::OK,
            Json(LoadModelResponse {
                id: path,
                object: "model",
                status: "already_loaded",
            }),
        )
            .into_response();
    }

    guard.try_load_entry(entry); // inserts pre-built entry directly

    (
        StatusCode::OK,
        Json(LoadModelResponse {
            id: path,
            object: "model",
            status: "loaded",
        }),
    )
        .into_response()
}

async fn unload_model(
    State(state): State<Arc<Mutex<ModelManager>>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let mut guard = state.lock().await;
    let removed = guard.unload(&req.model);
    if !removed {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("Model '{}' not loaded", req.model),
            "invalid_model",
        );
    }
    (
        StatusCode::OK,
        Json(LoadModelResponse {
            id: req.model,
            object: "model",
            status: "unloaded",
        }),
    )
        .into_response()
}

async fn estimate_vram(
    State(state): State<Arc<Mutex<ModelManager>>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let guard = state.lock().await;
    let entry = match guard.get(&req.model) {
        Some(e) => e,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Model '{}' not found", req.model),
                "invalid_model",
            );
        }
    };
    let entry_guard = entry.lock().await;
    let config = &entry_guard.config;

    let model_weights = std::fs::metadata(&entry_guard.model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // KV cache: [num_layers, 2, max_seq, head_dim, num_heads] as f32.
    // simplified: hidden_size * num_layers * max_seq * 2 * sizeof(f32)
    let max_seq = config.max_seq_len.min(4096);
    let kv_estimate =
        (config.hidden_size * config.num_layers * max_seq * 2 * std::mem::size_of::<f32>()) as u64;

    // Scratch: prefill + decode scratch buffers (upper bound).
    let scratch_estimate =
        (config.hidden_size * config.intermediate_size * 4 * std::mem::size_of::<f32>()) as u64;

    let total = model_weights + kv_estimate + scratch_estimate;

    let (fits, budget_gb) = vram_budget_check(total);

    let response = VramEstimate {
        model_id: req.model.clone(),
        model_weights_gb: bytes_to_gb(model_weights),
        kv_cache_gb: bytes_to_gb(kv_estimate),
        scratch_gb: bytes_to_gb(scratch_estimate),
        total_gb: bytes_to_gb(total),
        fits,
        inference_budget_gb: budget_gb,
    };

    (StatusCode::OK, Json(response)).into_response()
}

async fn list_vram() -> Response {
    let info = collect_vram_info();
    (StatusCode::OK, Json(info)).into_response()
}

fn collect_vram_info() -> Vec<VramInfo> {
    #[cfg(feature = "gpu")]
    {
        let mut result = Vec::new();
        // Probe device 0 only (single-GPU system).
        if let Some(caps) = crate::gpu::GpuCapabilities::detect() {
            match crate::gpu::VramSession::new(caps.device_id) {
                Ok(sess) => {
                    result.push(VramInfo {
                        device_id: sess.device_id,
                        total_gb: bytes_to_gb(sess.total as u64),
                        free_gb: bytes_to_gb(sess.startup_free as u64),
                        used_gb: bytes_to_gb(sess.already_used as u64),
                        desktop_reserved_gb: bytes_to_gb(sess.desktop_reserved as u64),
                        inference_budget_gb: bytes_to_gb(sess.inference_budget as u64),
                    });
                }
                Err(_) => {}
            }
        }
        result
    }
    #[cfg(not(feature = "gpu"))]
    {
        Vec::new()
    }
}

fn vram_budget_check(total_bytes: u64) -> (bool, f64) {
    #[cfg(feature = "gpu")]
    {
        if let Some(caps) = crate::gpu::GpuCapabilities::detect() {
            if let Ok(sess) = crate::gpu::VramSession::new(caps.device_id) {
                let fits = total_bytes <= sess.inference_budget as u64;
                return (fits, bytes_to_gb(sess.inference_budget as u64));
            }
        }
        (false, 0.0)
    }
    #[cfg(not(feature = "gpu"))]
    {
        (true, 0.0)
    }
}

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

async fn create_completion(
    State(state): State<Arc<Mutex<ModelManager>>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let guard = state.lock().await;
    let entry = match guard.get(&req.model) {
        Some(e) => e,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Model '{}' not found", req.model),
                "invalid_model",
            );
        }
    };
    let e = entry.lock().await;

    if req.stream {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "Streaming not yet implemented for /v1/completions. Use stream: false.",
            "not_implemented",
        );
    }

    let prompt = req.prompt.clone();
    let model_path = e.model_path.clone();
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_p = req.top_p;
    let template = e.chat_template;
    let config = e.config.clone();
    let tok = e.tokenizer.clone();
    let cpu_weights2 = Arc::clone(&e.cpu_weights);
    #[cfg(feature = "gpu")]
    let gpu_weights2 = e.gpu_weights.clone();
    #[cfg(feature = "gpu")]
    let speculative_engine2 = e.speculative_engine.clone();
    let sem = Arc::clone(&e.inference_sem);
    drop(e);
    drop(guard);

    let _permit = match sem.acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Model inference is unavailable (semaphore closed)",
                "service_unavailable",
            );
        }
    };

    let applied = template.apply(&prompt);
    let prompt_tokens = tok.encode(&applied, false);
    let prompt_tokens_len = prompt_tokens.len();
    let model_path2 = model_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = _permit; // hold semaphore permit during inference
        run_sync_inference(
            &cpu_weights2,
            #[cfg(feature = "gpu")]
            &gpu_weights2,
            #[cfg(feature = "gpu")]
            &speculative_engine2,
            &model_path2,
            &config,
            &tok,
            &prompt_tokens,
            max_tokens,
            temperature,
            top_p,
        )
    })
    .await;

    let (generated, completion_tokens) = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {}", e),
                "inference_error",
            );
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task panic: {}", e),
                "inference_error",
            );
        }
    };

    let response = CompletionResponse {
        id: format!(
            "cmpl-{}-{}",
            model_path.split('/').next_back().unwrap_or("default"),
            chrono::Utc::now().timestamp()
        ),
        object: "text_completion",
        created: chrono::Utc::now().timestamp(),
        model: req.model.clone(),
        choices: vec![CompletionChoice {
            text: generated,
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens_len,
            completion_tokens,
            total_tokens: prompt_tokens_len + completion_tokens,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

async fn create_chat_completion(
    State(state): State<Arc<Mutex<ModelManager>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let guard = state.lock().await;
    let entry = match guard.get(&req.model) {
        Some(e) => e,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Model '{}' not found", req.model),
                "invalid_model",
            );
        }
    };
    let e = entry.lock().await;

    let prompt = format_chat_messages(&req.messages, e.chat_template);
    let model_path = e.model_path.clone();
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_p = req.top_p;
    let config = e.config.clone();
    let tok = e.tokenizer.clone();
    let template = e.chat_template;
    let stream = req.stream;
    let model_name = req.model.clone();
    let cpu_weights2 = Arc::clone(&e.cpu_weights);
    #[cfg(feature = "gpu")]
    let gpu_weights2 = e.gpu_weights.clone();
    #[cfg(feature = "gpu")]
    let speculative_engine2 = e.speculative_engine.clone();
    let sem = Arc::clone(&e.inference_sem);
    drop(e);
    drop(guard);

    let _permit = match sem.acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Model inference is unavailable (semaphore closed)",
                "service_unavailable",
            );
        }
    };

    let applied = template.apply(&prompt);
    let prompt_tokens = tok.encode(&applied, false);
    let prompt_tokens_len = prompt_tokens.len();

    if !stream {
        let model_path2 = model_path.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _permit = _permit;
            run_sync_inference(
                &cpu_weights2,
                #[cfg(feature = "gpu")]
                &gpu_weights2,
                #[cfg(feature = "gpu")]
                &speculative_engine2,
                &model_path2,
                &config,
                &tok,
                &prompt_tokens,
                max_tokens,
                temperature,
                top_p,
            )
        })
        .await;

        let (generated, completion_tokens) = match result {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Inference error: {}", e),
                    "inference_error",
                );
            }
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Task panic: {}", e),
                    "inference_error",
                );
            }
        };

        let response = ChatCompletionResponse {
            id: format!(
                "chatcmpl-{}-{}",
                model_path.split('/').next_back().unwrap_or("default"),
                chrono::Utc::now().timestamp()
            ),
            object: "chat.completion",
            created: chrono::Utc::now().timestamp(),
            model: model_name,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_tokens_len,
                completion_tokens,
                total_tokens: prompt_tokens_len + completion_tokens,
            },
        };

        return (StatusCode::OK, Json(response)).into_response();
    }

    // ── Streaming path ────────────────────────────────────────────────────────
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let model_name2 = model_name.clone();

    tokio::task::spawn_blocking(move || {
        let _permit = _permit;
        let _ = run_stream_inference(
            &cpu_weights2,
            #[cfg(feature = "gpu")]
            &gpu_weights2,
            #[cfg(feature = "gpu")]
            &speculative_engine2,
            &config,
            &tok,
            &prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            tx,
        );
    });

    use tokio_stream::StreamExt as _;
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(move |text| {
        let chunk = ChatCompletionStreamChunk {
            id: format!(
                "chatcmpl-{}-{}",
                model_path.split('/').next_back().unwrap_or("default"),
                chrono::Utc::now().timestamp()
            ),
            object: "chat.completion.chunk",
            created: chrono::Utc::now().timestamp(),
            model: model_name2.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    content: Some(text),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        Ok::<_, std::convert::Infallible>(Event::default().data(json))
    });

    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ── Anthropic /v1/messages ─────────────────────────────────────────────────────

async fn create_messages(
    State(state): State<Arc<Mutex<ModelManager>>>,
    Json(req): Json<MessagesRequest>,
) -> Response {
    let guard = state.lock().await;
    let entry = match guard.get(&req.model) {
        Some(e) => e,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Model '{}' not found", req.model),
                "invalid_model",
            );
        }
    };
    let e = entry.lock().await;

    if req.stream {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "Anthropic streaming not yet implemented.",
            "not_implemented",
        );
    }

    let prompt = format_chat_messages(&req.messages, e.chat_template);
    let model_path = e.model_path.clone();
    let max_tokens = req.max_tokens;
    let config = e.config.clone();
    let tok = e.tokenizer.clone();
    let template = e.chat_template;
    let model_name = req.model.clone();
    let cpu_weights2 = Arc::clone(&e.cpu_weights);
    #[cfg(feature = "gpu")]
    let gpu_weights2 = e.gpu_weights.clone();
    #[cfg(feature = "gpu")]
    let speculative_engine2 = e.speculative_engine.clone();
    let sem = Arc::clone(&e.inference_sem);
    drop(e);
    drop(guard);

    let _permit = match sem.acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Model inference is unavailable (semaphore closed)",
                "service_unavailable",
            );
        }
    };

    let applied = template.apply(&prompt);
    let prompt_tokens = tok.encode(&applied, false);
    let prompt_tokens_len = prompt_tokens.len();

    let model_path2 = model_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = _permit;
        run_sync_inference(
            &cpu_weights2,
            #[cfg(feature = "gpu")]
            &gpu_weights2,
            #[cfg(feature = "gpu")]
            &speculative_engine2,
            &model_path2,
            &config,
            &tok,
            &prompt_tokens,
            max_tokens,
            1.0, // temperature — Anthropic default
            0.9, // top_p
        )
    })
    .await;

    let (generated, completion_tokens) = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {}", e),
                "inference_error",
            );
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task panic: {}", e),
                "inference_error",
            );
        }
    };

    let response = MessagesResponse {
        id: format!(
            "msg_{}_{}",
            model_path.split('/').next_back().unwrap_or("default"),
            chrono::Utc::now().timestamp()
        ),
        msg_type: "message",
        role: "assistant",
        model: model_name,
        content: vec![MessageContent {
            content_type: "text",
            text: generated,
        }],
        usage: MessagesUsage {
            input_tokens: prompt_tokens_len,
            output_tokens: completion_tokens,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

async fn ready(State(state): State<Arc<Mutex<ModelManager>>>) -> impl IntoResponse {
    let guard = state.lock().await;
    let ready = !guard.keys().is_empty();
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(json!({"status": if ready { "ready" } else { "no_models_loaded" }})),
    )
}

// ── Helpers ──────────────────────────────────────────────────────────────────────

fn error_response(status: StatusCode, msg: impl Into<String>, typ: impl Into<String>) -> Response {
    (status, Json(ApiError::new(msg, typ))).into_response()
}

fn format_chat_messages(messages: &[ChatMessage], template: ChatTemplate) -> String {
    let pairs: Vec<(String, String)> = messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    template.apply_messages(&pairs)
}

// ── Synchronous inference dispatcher ─────────────────────────────────────────────

#[expect(
    clippy::too_many_arguments,
    reason = "dispatcher matches the callers (server handlers with fixed params)"
)]
fn run_sync_inference(
    cpu_weights: &Arc<CpuModelWeights>,
    #[cfg(feature = "gpu")] gpu_weights: &Option<Arc<crate::gpu::GpuModelWeights>>,
    #[cfg(feature = "gpu")] speculative_engine: &Option<Arc<Mutex<crate::gpu::SpeculativeEngine>>>,
    model_path: &str,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<(String, usize), String> {
    #[cfg(feature = "gpu")]
    {
        if let Some(spec_engine_arc) = speculative_engine {
            let gpu_caps = crate::gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
            let device = crate::gpu::GpuDevice::get_or_init(gpu_caps.device_id)
                .map_err(|e| format!("gpu init: {}", e))?;
            let mut engine = spec_engine_arc.blocking_lock();
            let orchestrator = crate::gpu::SpeculativeOrchestrator::new(4)?;
            return orchestrator.generate(&device, &mut engine, tok, prompt_tokens, max_tokens);
        }
        if let Some(gw) = gpu_weights {
            crate::api::gpu_inference::run_gpu_sync_inference(
                gw,
                cpu_weights,
                config,
                tok,
                prompt_tokens,
                max_tokens,
                temperature,
                top_p,
            )
        } else {
            Err("GPU feature enabled but model has no loaded GPU weights".to_string())
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = model_path; // unused in CPU-only path
        run_cpu_sync_inference(
            cpu_weights,
            config,
            tok,
            prompt_tokens,
            max_tokens,
            temperature,
            top_p,
        )
    }
}

// ── Streaming inference dispatcher ───────────────────────────────────────────────

#[expect(
    clippy::too_many_arguments,
    reason = "dispatcher matches streaming SSE callers (fixed params)"
)]
fn run_stream_inference(
    cpu_weights: &Arc<CpuModelWeights>,
    #[cfg(feature = "gpu")] gpu_weights: &Option<Arc<crate::gpu::GpuModelWeights>>,
    #[cfg(feature = "gpu")] speculative_engine: &Option<Arc<Mutex<crate::gpu::SpeculativeEngine>>>,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<(), String> {
    #[cfg(feature = "gpu")]
    {
        if let Some(spec_engine_arc) = speculative_engine {
            let gpu_caps = crate::gpu::detect().ok_or("GPU requested but no AMD GPU detected")?;
            let device = crate::gpu::GpuDevice::get_or_init(gpu_caps.device_id)
                .map_err(|e| format!("gpu init: {}", e))?;
            let mut engine = spec_engine_arc.blocking_lock();
            let orchestrator = crate::gpu::SpeculativeOrchestrator::new(4)?;
            return orchestrator.generate_stream(
                &device,
                &mut engine,
                tok,
                prompt_tokens,
                max_tokens,
                tx,
            );
        }
        if let Some(gw) = gpu_weights {
            crate::api::gpu_inference::run_gpu_stream_inference(
                gw,
                cpu_weights,
                config,
                tok,
                prompt_tokens,
                max_tokens,
                temperature,
                top_p,
                tx,
            )
        } else {
            Err("GPU feature enabled but model has no loaded GPU weights".to_string())
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        run_cpu_stream_inference(
            cpu_weights,
            config,
            tok,
            prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            tx,
        )
    }
}

// ── Synchronous CPU inference wrapper ────────────────────────────────────────────

use crate::cpu::SimdKernels;
use crate::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    prefill::cpu_prefill_forward_parallel,
    sampler::{cpu_sample_greedy, cpu_sample_top_p},
};
use crate::hardware::{derive_batch_config, detect, CpuCapabilities};

#[expect(
    clippy::too_many_arguments,
    reason = "CPU sync inference entrypoint, fixed params from callers"
)]
fn run_cpu_sync_inference(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<(String, usize), String> {
    let caps: CpuCapabilities = detect().map_err(|e| format!("hw detection: {}", e))?;
    let batch_config = derive_batch_config(&caps, config);
    let _simd = SimdKernels::new(caps.simd.kernel_preference());

    let max_seq = (prompt_tokens.len() + max_tokens).min(config.max_seq_len);
    let mut kv = CpuKvCache::new(config, max_seq);
    let mut scratch = CpuForwardScratch::new(config);
    let use_greedy = top_p >= 1.0;

    // Prefill all prompt tokens
    let n_prompt = prompt_tokens.len();
    cpu_prefill_forward_parallel(
        prompt_tokens,
        &weights,
        &mut kv,
        &mut scratch,
        0,
        config,
        &batch_config,
    )
    .map_err(|e| format!("prefill: {}", e))?;

    // Sample first token from prefill output
    let mut seed = 0xdeadbeefu64;
    let mut next_token = if use_greedy {
        cpu_sample_greedy(&scratch.logits)
    } else {
        seed = seed.wrapping_add(1);
        cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
    };

    // Decode loop
    let mut hidden = vec![0.0f32; config.hidden_size];
    let mut pos = n_prompt;
    let mut output_tokens = Vec::with_capacity(max_tokens);

    for _ in 0..max_tokens {
        if tok.is_eog(next_token) {
            break;
        }

        output_tokens.push(next_token);
        cpu_embed_token(next_token, &weights, &mut hidden, config);
        cpu_full_forward(&mut hidden, &weights, &mut kv, &mut scratch, pos, config)
            .map_err(|e| format!("decode: {}", e))?;

        next_token = if use_greedy {
            cpu_sample_greedy(&scratch.logits)
        } else {
            seed = seed.wrapping_add(1);
            cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
        };
        pos += 1;
    }

    let text = tok.decode(&output_tokens, false);
    Ok((text, output_tokens.len()))
}

// ── Streaming CPU inference wrapper ──────────────────────────────────────────────

#[expect(
    clippy::too_many_arguments,
    reason = "CPU stream inference entrypoint, fixed params from callers"
)]
fn run_cpu_stream_inference(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tok: &BpeTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<(), String> {
    let caps: CpuCapabilities = detect().map_err(|e| format!("hw detection: {}", e))?;
    let batch_config = derive_batch_config(&caps, config);
    let _simd = SimdKernels::new(caps.simd.kernel_preference());

    let max_seq = (prompt_tokens.len() + max_tokens).min(config.max_seq_len);
    let mut kv = CpuKvCache::new(config, max_seq);
    let mut scratch = CpuForwardScratch::new(config);
    let use_greedy = top_p >= 1.0;

    let n_prompt = prompt_tokens.len();
    cpu_prefill_forward_parallel(
        prompt_tokens,
        &weights,
        &mut kv,
        &mut scratch,
        0,
        config,
        &batch_config,
    )
    .map_err(|e| format!("prefill: {}", e))?;

    let mut seed = 0xdeadbeefu64;
    let mut next_token = if use_greedy {
        cpu_sample_greedy(&scratch.logits)
    } else {
        seed = seed.wrapping_add(1);
        cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
    };

    let mut hidden = vec![0.0f32; config.hidden_size];
    let mut pos = n_prompt;
    let mut output_tokens = Vec::with_capacity(max_tokens);
    let mut previous_text = String::new();

    for _ in 0..max_tokens {
        if tok.is_eog(next_token) {
            break;
        }

        output_tokens.push(next_token);

        // Decode incremental text for this batch of tokens
        let text = tok.decode(&output_tokens, false);
        let new_chars = &text[previous_text.len().min(text.len())..];
        if !new_chars.is_empty() {
            let _ = tx.send(new_chars.to_string());
        }
        previous_text = text;

        cpu_embed_token(next_token, &weights, &mut hidden, config);
        cpu_full_forward(&mut hidden, &weights, &mut kv, &mut scratch, pos, config)
            .map_err(|e| format!("decode: {}", e))?;

        next_token = if use_greedy {
            cpu_sample_greedy(&scratch.logits)
        } else {
            seed = seed.wrapping_add(1);
            cpu_sample_top_p(&scratch.logits, temperature, top_p, seed)
        };
        pos += 1;
    }

    Ok(())
}
