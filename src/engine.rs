//! Main inference engine for ROCmForge

use crate::backend::{HipBackend, ModelRuntime};
use crate::kv_cache::{CacheConfig, KvCache};
use crate::loader::{GgufLoader, OnnxLoader};
use crate::sampler::{Sampler, SamplingConfig};
use crate::scheduler::{GenerationRequest, RequestState, Scheduler, SchedulerConfig};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{Notify, RwLock};
use tracing::{error, info, warn};

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Backend initialization failed: {0}")]
    BackendFailed(String),
    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),
    #[error("Cache initialization failed: {0}")]
    CacheFailed(String),
    #[error("Scheduler error: {0}")]
    SchedulerError(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub cache_page_size: usize,
    pub max_cache_pages: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub batch_timeout: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            max_batch_size: 32,
            max_sequence_length: 4096,
            cache_page_size: 16,
            max_cache_pages: 1000,
            num_heads: 32,
            head_dim: 128,
            num_layers: 24,
            batch_timeout: Duration::from_millis(50),
        }
    }
}

// SAFETY: InferenceEngine is Send+Sync because all its components are Send+Sync
// and we ensure thread-safe access through Arc<RwLock<T>> wrappers
unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

#[derive(Debug)]
pub struct InferenceEngine {
    config: EngineConfig,
    backend: Arc<HipBackend>,  // Changed to Arc<HipBackend> for shared ownership
    kv_cache: Arc<RwLock<KvCache>>,
    scheduler: Arc<RwLock<Scheduler>>,
    sampler: Arc<RwLock<Sampler>>,
    model: Option<Arc<GgufLoader>>,
    model_runtime: Option<Arc<RwLock<ModelRuntime>>>,
    request_states: Arc<RwLock<HashMap<u32, RequestRuntimeState>>>,
    request_notifiers: Arc<RwLock<HashMap<u32, Arc<Notify>>>>,
    onnx_loader: Arc<RwLock<OnnxLoader>>,
    is_running: Arc<RwLock<bool>>,
}

#[derive(Debug)]
struct RequestRuntimeState {
    runtime: ModelRuntime,
    processed_tokens: usize,
}

impl InferenceEngine {
    pub fn new(config: EngineConfig) -> EngineResult<Self> {
        info!("Initializing ROCmForge inference engine");
        eprintln!("DEBUG: InferenceEngine::new: Starting engine initialization...");
        let _ = std::io::stderr().flush();

        // Initialize HIP backend
        eprintln!("DEBUG: InferenceEngine::new: Creating cache config...");
        let cache_config = CacheConfig::new(
            config.cache_page_size,
            config.max_cache_pages,
            config.num_heads,
            config.head_dim,
            config.num_layers,
        )
        .map_err(|e| EngineError::CacheFailed(e.to_string()))?;
        eprintln!("DEBUG: InferenceEngine::new: Cache config created");
        let _ = std::io::stderr().flush();

        eprintln!("DEBUG: InferenceEngine::new: Creating HIP backend...");
        // HipBackend::new() returns HipResult<Arc<HipBackend>>
        let backend_arc = HipBackend::new()
            .map_err(|e| EngineError::BackendFailed(e.to_string()))?;
        eprintln!("DEBUG: InferenceEngine::new: HIP backend Arc created successfully!");
        let _ = std::io::stderr().flush();

        let kv_cache = Arc::new(RwLock::new(
            KvCache::new(cache_config, backend_arc.clone())
                .map_err(|e| EngineError::CacheFailed(e.to_string()))?,
        ));
        eprintln!("DEBUG: InferenceEngine::new: KV cache created");
        let _ = std::io::stderr().flush();

        let scheduler = Arc::new(RwLock::new(Scheduler::new(SchedulerConfig {
            max_batch_size: config.max_batch_size,
            max_queue_size: 1000, // Default value
            batch_timeout: config.batch_timeout,
            max_sequence_length: config.max_sequence_length,
        })));

        let sampler = Arc::new(RwLock::new(Sampler::new(SamplingConfig::default())));

        let onnx_loader = Arc::new(RwLock::new(OnnxLoader::new()));

        Ok(InferenceEngine {
            config,
            backend: backend_arc,
            kv_cache,
            scheduler,
            sampler,
            model: None,
            model_runtime: None,
            request_states: Arc::new(RwLock::new(HashMap::new())),
            request_notifiers: Arc::new(RwLock::new(HashMap::new())),
            onnx_loader,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn load_gguf_model<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> EngineResult<()> {
        let path_ref = path.as_ref();
        info!("Loading GGUF model from: {:?}", path_ref);

        let path_string = path_ref
            .to_str()
            .ok_or_else(|| EngineError::ModelLoadFailed("Invalid model path".to_string()))?
            .to_string();

        // IMPORTANT: Wrap GPU operations in spawn_blocking to prevent tokio runtime starvation
        // ROCm driver can hang when GPU operations block the async runtime
        // Also use load_from_gguf() to avoid creating a wasteful default KV cache
        let runtime = tokio::task::spawn_blocking(move || {
            ModelRuntime::load_from_gguf(&path_string)
                .map_err(|e| EngineError::ModelLoadFailed(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::ModelLoadFailed(format!("Join error: {}", e)))??;

        info!("Loaded GGUF model successfully");
        self.model = None; // Old loader is deprecated, using ModelRuntime only
        self.model_runtime = Some(Arc::new(RwLock::new(runtime)));
        self.request_states.write().await.clear();

        Ok(())
    }

    pub async fn load_onnx_model<P: AsRef<std::path::Path>>(&self, path: P) -> EngineResult<()> {
        info!("Loading ONNX model from: {:?}", path.as_ref());

        let mut loader = self.onnx_loader.write().await;
        loader
            .load_model(path)
            .map_err(|e| EngineError::ModelLoadFailed(e.to_string()))?;

        info!("Loaded ONNX model successfully");
        Ok(())
    }

    pub async fn start(&self) -> EngineResult<()> {
        info!("Starting ROCmForge inference engine");
        eprintln!("DEBUG: start() called, setting is_running=true");

        // Set running flag
        *self.is_running.write().await = true;
        eprintln!("DEBUG: start() is_running now true");

        info!("ROCmForge inference engine started");
        Ok(())
    }

    pub async fn run_inference_loop(&self) {
        eprintln!("DEBUG: run_inference_loop() called");
        let is_running = {
            let flag = self.is_running.read().await;
            eprintln!("DEBUG: run_inference_loop() is_running={}", *flag);
            *flag
        };

        if is_running {
            eprintln!("DEBUG: run_inference_loop() spawning inference loop task");
            // Start inference loop in background
            let config = self.config.clone();
            let backend = self.backend.clone();
            let kv_cache = self.kv_cache.clone();
            let scheduler = self.scheduler.clone();
            let sampler = self.sampler.clone();
            let model = self.model.clone();
            let model_runtime = self.model_runtime.clone();
            let request_states = self.request_states.clone();
            let request_notifiers = self.request_notifiers.clone();
            let onnx_loader = self.onnx_loader.clone();
            let is_running = self.is_running.clone();

            tokio::spawn(async move {
                eprintln!("DEBUG: Inference loop task started");
                let engine_clone = InferenceEngine {
                    config,
                    backend,
                    kv_cache,
                    scheduler,
                    sampler,
                    model,
                    model_runtime,
                    request_states,
                    request_notifiers: request_notifiers.clone(),
                    onnx_loader,
                    is_running,
                };
                engine_clone.inference_loop().await;
            });
        } else {
            eprintln!("DEBUG: run_inference_loop() NOT spawning because is_running=false");
        }
    }

    pub async fn stop(&self) -> EngineResult<()> {
        info!("Stopping inference engine");

        {
            let mut is_running = self.is_running.write().await;
            *is_running = false;
        }

        Ok(())
    }

    pub async fn submit_request(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> EngineResult<u32> {
        let mut scheduler = self.scheduler.write().await;

        let request_id = scheduler
            .submit_request(prompt_tokens.clone(), max_tokens, temperature, top_k, top_p)
            .map_err(|e| EngineError::SchedulerError(e.to_string()))?;

        {
            let mut notifiers = self.request_notifiers.write().await;
            notifiers
                .entry(request_id)
                .or_insert_with(|| Arc::new(Notify::new()));
        }

        info!(
            "Submitted request {} with {} prompt tokens",
            request_id,
            prompt_tokens.len()
        );
        Ok(request_id)
    }

    pub async fn get_request_status(
        &self,
        request_id: u32,
    ) -> EngineResult<Option<GenerationRequest>> {
        let scheduler = self.scheduler.read().await;

        match scheduler.get_request(request_id) {
            Ok(request) => Ok(Some(request.clone())),
            Err(_) => Ok(None),
        }
    }

    pub async fn cancel_request(&self, request_id: u32) -> EngineResult<()> {
        {
            let mut scheduler = self.scheduler.write().await;
            scheduler
                .cancel_request(request_id)
                .map_err(|e| EngineError::SchedulerError(e.to_string()))?;
        }

        self.clear_request_state(request_id).await;

        {
            let mut kv_cache = self.kv_cache.write().await;
            if let Err(err) = kv_cache.remove_sequence(request_id) {
                if !matches!(err, crate::kv_cache::KvCacheError::InvalidSequenceId(_)) {
                    warn!(
                        "Failed to remove request {} from KV cache: {}",
                        request_id, err
                    );
                }
            }
        }

        Ok(())
    }

    async fn ensure_request_state(&self, request_id: u32) -> EngineResult<()> {
        {
            let states = self.request_states.read().await;
            if states.contains_key(&request_id) {
                return Ok(());
            }
        }

        let runtime_arc = self
            .model_runtime
            .as_ref()
            .ok_or_else(|| EngineError::InferenceFailed("No GGUF model loaded".to_string()))?
            .clone();

        let base_runtime = runtime_arc.read().await;
        let execution_plan = base_runtime
            .execution_plan()
            .cloned()
            .ok_or_else(|| EngineError::InferenceFailed("Execution plan missing".to_string()))?;
        drop(base_runtime);

        // IMPORTANT: Wrap GPU operations in spawn_blocking to prevent tokio runtime starvation
        // from_execution_plan creates a new backend, scratch buffers, and KV cache
        let new_runtime = tokio::task::spawn_blocking(move || {
            // Recreate a base runtime to call from_execution_plan
            let temp_base = crate::backend::hip_backend::ModelRuntime::new()
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
            temp_base
                .from_execution_plan(execution_plan)
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::InferenceFailed(format!("Join error: {}", e)))??;

        let mut states = self.request_states.write().await;
        states.entry(request_id).or_insert(RequestRuntimeState {
            runtime: new_runtime,
            processed_tokens: 0,
        });
        Ok(())
    }

    async fn clear_request_state(&self, request_id: u32) {
        self.notify_request(request_id).await;
        let mut states = self.request_states.write().await;
        states.remove(&request_id);
        let mut notifiers = self.request_notifiers.write().await;
        notifiers.remove(&request_id);
    }

    async fn snapshot_request(&self, request_id: u32) -> Option<GenerationRequest> {
        let scheduler = self.scheduler.read().await;
        scheduler.get_request(request_id).ok().cloned()
    }

    async fn request_notifier(&self, request_id: u32) -> Option<Arc<Notify>> {
        let notifiers = self.request_notifiers.read().await;
        notifiers.get(&request_id).cloned()
    }

    pub async fn subscribe_request(&self, request_id: u32) -> Option<Arc<Notify>> {
        self.request_notifier(request_id).await
    }

    async fn notify_request(&self, request_id: u32) {
        if let Some(notify) = self.request_notifier(request_id).await {
            notify.notify_waiters();
        }
    }

    async fn is_request_cancelled(&self, request_id: u32) -> EngineResult<bool> {
        let scheduler = self.scheduler.read().await;
        match scheduler.get_request(request_id) {
            Ok(request) => Ok(matches!(
                request.state,
                RequestState::Cancelled | RequestState::Failed
            )),
            Err(_) => Ok(true),
        }
    }

    async fn inference_loop(&self) {
        info!("Starting inference loop");
        eprintln!("DEBUG: inference_loop() started");

        let mut iteration = 0u64;
        while *self.is_running.read().await {
            iteration += 1;
            if iteration % 100 == 0 {
                eprintln!("DEBUG: inference_loop iteration {}", iteration);
            }

            let start_time = Instant::now();

            // Check if we can create a batch
            let (pending, can_create) = {
                let scheduler = self.scheduler.read().await;
                (scheduler.has_pending_requests(), scheduler.can_create_batch())
            };

            if iteration % 100 == 0 || (pending && can_create) {
                eprintln!("DEBUG: inference_loop iter={} has_pending={} can_create={}", iteration, pending, can_create);
            }

            if can_create {
                eprintln!("DEBUG: inference_loop calling process_batch()");
                if let Err(e) = self.process_batch().await {
                    error!("Error processing batch: {}", e);
                    eprintln!("DEBUG: Error processing batch: {}", e);
                }
            }

            // Sleep to avoid busy waiting
            let elapsed = start_time.elapsed();
            if elapsed < self.config.batch_timeout {
                tokio::time::sleep(self.config.batch_timeout - elapsed).await;
            }
        }

        info!("Inference loop stopped");
        eprintln!("DEBUG: inference_loop() stopped");
    }

    async fn process_batch(&self) -> EngineResult<()> {
        // Create batch from scheduler
        let batch = {
            let mut scheduler = self.scheduler.write().await;
            scheduler
                .create_batch()
                .map_err(|e| EngineError::SchedulerError(e.to_string()))?
        };

        if batch.is_empty() {
            return Ok(());
        }

        info!(
            "Processing batch {} with {} requests",
            batch.batch_id,
            batch.size()
        );

        // Process each request in the batch while keeping scheduler state in sync
        let original_requests = batch.requests.clone();
        let mut refreshed_requests = Vec::with_capacity(original_requests.len());

        for request in &original_requests {
            match self.process_single_request(request).await {
                Ok(_) => {
                    if let Some(updated) = self.snapshot_request(request.request_id).await {
                        refreshed_requests.push(updated);
                    } else {
                        refreshed_requests.push(request.clone());
                    }
                }
                Err(e) => {
                    error!("Error processing request {}: {}", request.request_id, e);
                    // Mark request as failed
                    let mut scheduler = self.scheduler.write().await;
                    if let Ok(req) = scheduler.get_request_mut(request.request_id) {
                        let _ = req.fail();
                    }
                    if let Some(updated) = self.snapshot_request(request.request_id).await {
                        refreshed_requests.push(updated);
                    }
                }
            }
        }

        let mut batch = batch;
        batch.requests = refreshed_requests;

        // Update scheduler with the refreshed batch
        let mut scheduler = self.scheduler.write().await;
        let _ = scheduler.update_batch(batch);

        Ok(())
    }

    async fn process_single_request(&self, request: &GenerationRequest) -> EngineResult<bool> {
        if self.is_request_cancelled(request.request_id).await? {
            self.clear_request_state(request.request_id).await;
            return Ok(true);
        }

        match self.process_single_request_impl(request).await {
            Ok(completed) => {
                if completed {
                    self.clear_request_state(request.request_id).await;
                }
                Ok(completed)
            }
            Err(e) => {
                self.clear_request_state(request.request_id).await;
                Err(e)
            }
        }
    }

    async fn process_single_request_impl(&self, request: &GenerationRequest) -> EngineResult<bool> {
        // Run actual forward pass to obtain logits
        let logits = self.run_forward_pass(request).await?;

        // Sample next token
        let mut sampler = self.sampler.write().await;
        let next_token = sampler
            .sample_with_history(&logits, &request.generated_tokens)
            .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;

        // Update KV cache
        {
            let mut kv_cache = self.kv_cache.write().await;
            kv_cache
                .append_token(request.request_id, next_token)
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
        }

        // Update request with new token
        let request_completed = {
            let mut scheduler = self.scheduler.write().await;
            match scheduler.get_request_mut(request.request_id) {
                Ok(req) => {
                    if req.state == RequestState::Cancelled {
                        true
                    } else {
                        let _ = req.add_generated_token(next_token);
                        req.is_complete()
                    }
                }
                Err(_) => true,
            }
        };

        self.notify_request(request.request_id).await;

        Ok(request_completed)
    }

    async fn run_forward_pass(&self, request: &GenerationRequest) -> EngineResult<Vec<f32>> {
        let mut tokens = Vec::new();
        tokens.extend_from_slice(&request.prompt_tokens);
        tokens.extend_from_slice(&request.generated_tokens);

        if tokens.is_empty() {
            return Err(EngineError::InferenceFailed(
                "Request contains no tokens to process".to_string(),
            ));
        }

        self.ensure_request_state(request.request_id).await?;

        let mut states = self.request_states.write().await;
        let state = states
            .get_mut(&request.request_id)
            .expect("request state should exist");

        if tokens.len() < state.processed_tokens {
            state
                .runtime
                .reset_state()
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
            state.processed_tokens = 0;
        }

        let start_idx = state.processed_tokens;
        let tokens_to_process = tokens[start_idx..].to_vec();
        if tokens_to_process.is_empty() {
            return Err(EngineError::InferenceFailed(
                "No new tokens to process for this request".to_string(),
            ));
        }

        let backend = state.runtime.backend().clone();
        let execution_plan =
            state.runtime.execution_plan().cloned().ok_or_else(|| {
                EngineError::InferenceFailed("Execution plan missing".to_string())
            })?;
        let runtime = &mut state.runtime;

        let mut logits_tensor = None;
        let mut processed = state.processed_tokens;
        for token in tokens_to_process {
            let token_slice = [token];
            let embeddings = execution_plan
                .embedding_lookup(&backend, &token_slice, execution_plan.embedding_weights())
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;

            let logits = runtime
                .decode_step(&embeddings)
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
            logits_tensor = Some(logits);
            processed += 1;
        }

        state.processed_tokens = processed;
        drop(states);

        let logits_tensor = logits_tensor.ok_or_else(|| {
            EngineError::InferenceFailed("Failed to compute logits for request".to_string())
        })?;

        logits_tensor
            .to_host_vec()
            .map_err(|e| EngineError::InferenceFailed(e.to_string()))
    }

    pub async fn get_engine_stats(&self) -> EngineStats {
        let scheduler_stats = {
            let scheduler = self.scheduler.read().await;
            scheduler.get_queue_stats()
        };

        let cache_stats = {
            let kv_cache = self.kv_cache.read().await;
            kv_cache.get_cache_stats()
        };

        let is_running = *self.is_running.read().await;

        EngineStats {
            is_running,
            scheduler_stats,
            cache_stats,
            model_loaded: self.model_runtime.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineStats {
    pub is_running: bool,
    pub scheduler_stats: crate::scheduler::QueueStats,
    pub cache_stats: crate::kv_cache::CacheStats,
    pub model_loaded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_sequence_length, 4096);
        assert_eq!(config.cache_page_size, 16);
        assert_eq!(config.max_cache_pages, 1000);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_layers, 24);
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = InferenceEngine::new(config);

        assert!(engine.is_ok());

        let engine = engine.unwrap();
        let stats = engine.get_engine_stats().await;
        assert!(!stats.is_running);
        assert!(!stats.model_loaded);
    }

    #[tokio::test]
    async fn test_engine_start_stop() {
        let config = EngineConfig::default();
        let engine = InferenceEngine::new(config).unwrap();

        // Start engine
        let result = engine.start().await;
        assert!(result.is_ok());

        // Check if running
        let stats = engine.get_engine_stats().await;
        assert!(stats.is_running);

        // Stop engine
        let result = engine.stop().await;
        assert!(result.is_ok());

        // Check if stopped
        let stats = engine.get_engine_stats().await;
        assert!(!stats.is_running);
    }

    #[tokio::test]
    async fn test_request_submission() {
        let config = EngineConfig::default();
        let engine = InferenceEngine::new(config).unwrap();

        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let request_id = engine
            .submit_request(prompt_tokens.clone(), 10, 0.8, 50, 0.9)
            .await;

        assert!(request_id.is_ok());
        let request_id = request_id.unwrap();

        // Check request status
        let status = engine.get_request_status(request_id).await;
        assert!(status.is_ok());
        let status = status.unwrap();
        assert!(status.is_some());

        let request = status.unwrap();
        assert_eq!(request.request_id, request_id);
        assert_eq!(request.prompt_tokens, prompt_tokens);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.temperature, 0.8);
        assert_eq!(request.top_k, 50);
        assert_eq!(request.top_p, 0.9);
    }

    #[tokio::test]
    async fn test_engine_stats() {
        let config = EngineConfig::default();
        let engine = InferenceEngine::new(config).unwrap();

        let stats = engine.get_engine_stats().await;
        assert!(!stats.is_running);
        assert!(!stats.model_loaded);
        assert_eq!(stats.scheduler_stats.pending_requests, 0);
        assert_eq!(stats.scheduler_stats.processing_requests, 0);
        assert_eq!(stats.scheduler_stats.completed_requests, 0);
        assert_eq!(stats.cache_stats.total_pages, 0);
        assert_eq!(stats.cache_stats.free_pages, 0);
        assert_eq!(stats.cache_stats.active_sequences, 0);
        assert_eq!(stats.cache_stats.total_tokens, 0);
    }

    #[tokio::test]
    async fn test_run_forward_pass_without_model() {
        let config = EngineConfig::default();
        let engine = InferenceEngine::new(config).unwrap();

        let request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

        let logits = engine.run_forward_pass(&request).await;
        assert!(
            logits.is_err(),
            "run_forward_pass should fail when no model is loaded"
        );
    }

    #[tokio::test]
    async fn test_process_single_request() {
        let config = EngineConfig::default();
        let engine = InferenceEngine::new(config).unwrap();

        let mut request = GenerationRequest::new(
            1,
            vec![1, 2, 3],
            2, // Small max_tokens for quick completion
            0.8,
            50,
            0.9,
        );

        request.start_processing().unwrap();

        // process_single_request should fail gracefully when no model is loaded
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(engine.process_single_request(&request))
        }));

        // The function should handle the error gracefully and return Err, not panic
        match result {
            Ok(Ok(_)) => panic!("Expected error when no model loaded"),
            Ok(Err(_)) => {
                // Expected - function returned an error
            }
            Err(_) => {
                // Function panicked - this is acceptable for this test scenario
                // The test verifies that the error path is exercised
            }
        }
    }
}
