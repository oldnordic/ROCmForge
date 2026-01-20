//! Core types for execution plan
//!
//! Contains the ExecutionPlan struct definition and LoadingStats.
//! Most methods are implemented in other modules.

#![allow(deprecated)] // TODO: Migrate from to_host_vec() to copy_from_device_safe() (Phase 13-03-02)

use super::{Architecture, LayerPlan};
use super::ggml_plan::{EmbeddingGgmlPlan, RopeCache, LayerGgmlPlan};

use crate::attention::rope::RopeConfig;
use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::loader::gguf::GgufLoader;
use crate::loader::lazy_tensor::LazyTensor;
use crate::ggml::Layout;
use crate::model::{config::ModelConfig, glm_position::GlmPositionHandler, kv_cache::KVCache};
use once_cell::sync::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
 // Renamed to avoid conflict with once_cell::sync

/// Loading statistics for debugging/observability
///
/// Provides information about which tensors are loaded vs unloaded.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadingStats {
    /// Total number of tensors in the model
    pub total_tensors: usize,
    /// Number of tensors currently loaded on GPU
    pub loaded_tensors: usize,
    /// Number of tensors not yet loaded
    pub unloaded_tensors: usize,
    /// Number of tensors cached (OnceCell hits)
    pub cached_tensors: usize,
}

/// Static execution plan for a transformer model
///
/// Contains lazy tensor handles and execution information for all layers.
/// Tensors are loaded on-demand during first forward pass.
#[derive(Debug)]
pub struct ExecutionPlan {
    layers: Vec<LayerPlan>,
    config: ModelConfig,

    // LAZY TENSOR FIELDS (Phase 2)
    /// Lazy tensor handle for embedding weights (loaded on-demand)
    embedding_weights_lazy: Arc<LazyTensor>,

    /// Embedding layout as stored in GGUF
    #[allow(dead_code)] // Stored for potential future use in layout-aware operations
    embedding_layout: Layout,

    /// Lazy tensor handle for LM head (loaded on-demand)
    lm_head_lazy: Arc<LazyTensor>,

    /// Reference to GGUF loader for on-demand loading (kept alive)
    loader: Arc<GgufLoader>,

    /// Reference to HIP backend for GPU operations
    backend: Arc<HipBackend>,

    // CACHED GPU TENSORS (after first access) - using OnceCell for thread-safe single initialization
    /// Cached embedding weights (loaded on first access)
    embedding_weights_cached: OnceCell<Arc<DeviceTensor>>,

    /// Cached LM head (loaded on first access)
    lm_head_cached: OnceCell<DeviceTensor>,

    /// Cached ggml embedding plan (persistent buffers + graph)
    embedding_plan: OnceCell<EmbeddingGgmlPlan>,

    /// Cached ggml layer plans for decode path
    layer_ggml_plans: OnceCell<Vec<LayerGgmlPlan>>,

    /// Cached RoPE tables on GPU (if configured)
    rope_cache: OnceCell<RopeCache>,

    /// Position encoding handler for applying RoPE embeddings
    position_handler: Option<GlmPositionHandler>,
}

impl ExecutionPlan {
    /// Create a new execution plan from model configuration
    ///
    /// **DEPRECATED**: This method is deprecated due to Phase 2 lazy loading.
    /// Use `ExecutionPlan::from_gguf()` instead which properly initializes
    /// lazy tensor handles for on-demand loading.
    #[deprecated(note = "Use ExecutionPlan::from_gguf() instead for lazy loading")]
    pub fn new(_backend: &HipBackend, _config: &ModelConfig) -> HipResult<Self> {
        Err(HipError::GenericError(
            "ExecutionPlan::new() is deprecated. Use ExecutionPlan::from_gguf() instead."
                .to_string(),
        ))
    }

    /// Get reference to all layer plans
    pub fn layers(&self) -> &[LayerPlan] {
        &self.layers
    }

    /// Get reference to model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get reference to loader
    pub fn loader(&self) -> &Arc<GgufLoader> {
        &self.loader
    }

    /// Get reference to backend
    pub fn backend(&self) -> &Arc<HipBackend> {
        &self.backend
    }

    /// Get reference to position handler
    pub fn position_handler(&self) -> Option<&GlmPositionHandler> {
        self.position_handler.as_ref()
    }

    /// Detect model architecture from available tensor names
    ///
    /// Scans tensor names to identify the architecture pattern:
    /// - Qwen2: tensors start with `blk.N.`
    /// - LLaMA: tensors start with `transformer.layers.N.`
    /// - Mistral: tensors start with `model.layers.N.`
    fn detect_architecture(tensor_names: &HashSet<String>) -> HipResult<Architecture> {
        Architecture::detect(tensor_names)
    }

    /// Create execution plan from GGUF loader using helper functions
    ///
    /// # Phase 2 Lazy Loading
    ///
    /// This method now creates lazy tensor handles instead of loading all tensors:
    /// - Wraps loader and backend in Arc for on-demand loading
    /// - Creates Arc<LazyTensor> handles for all weights
    /// - No GPU uploads occur during construction (<5s initialization)
    /// - Tensors are loaded on-demand during first forward pass
    pub fn from_gguf(backend: &HipBackend, loader: &GgufLoader) -> HipResult<Self> {
        eprintln!("ExecutionPlan::from_gguf: Starting...");
        let config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create model config: {}", e)))?;
        eprintln!(
            "ExecutionPlan::from_gguf: Config created, layers={}",
            config.num_hidden_layers
        );

        // Get lazy tensor handles (metadata only, <1s)
        let lazy_tensors: &HashMap<String, LazyTensor> = &loader.lazy_tensors;
        eprintln!(
            "ExecutionPlan::from_gguf: Got lazy tensors, count={}",
            lazy_tensors.len()
        );

        // Wrap loader and backend in Arc for lazy loading
        let loader_arc = Arc::new(loader.clone());
        let backend_arc = Arc::new(backend.clone());

        // Detect architecture from lazy tensor names
        let tensor_names: HashSet<_> = lazy_tensors.keys().cloned().collect();
        eprintln!("ExecutionPlan::from_gguf: Detecting architecture...");
        let architecture = Self::detect_architecture(&tensor_names)?;
        println!("Using {} architecture mapping", architecture.name());
        eprintln!("ExecutionPlan::from_gguf: Architecture detected, mapping embedding...");

        // Import mapping functions from embedding module
        use super::embedding::{map_embedding_lazy, map_lm_head_lazy};
        use super::layer_tensors::create_layer_plan_lazy;

        // Map embedding and LM head to LazyTensor handles
        eprintln!("ExecutionPlan::from_gguf: Mapping embedding weights...");
        let (embedding_weights_lazy, embedding_layout) =
            map_embedding_lazy(lazy_tensors, &config, &architecture)?;
        eprintln!("ExecutionPlan::from_gguf: Mapping LM head...");
        let lm_head_lazy = map_lm_head_lazy(lazy_tensors, &config, &architecture)?;
        eprintln!("ExecutionPlan::from_gguf: Creating layers...");

        // Create layers using LazyTensor handles
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            if layer_idx % 8 == 0 || layer_idx == config.num_hidden_layers - 1 {
                eprintln!(
                    "ExecutionPlan::from_gguf: Creating layer {}/{}",
                    layer_idx + 1,
                    config.num_hidden_layers
                );
            }
            let layer_plan =
                create_layer_plan_lazy(&config, lazy_tensors, layer_idx, &architecture)?;
            layers.push(layer_plan);
        }
        eprintln!("ExecutionPlan::from_gguf: All layers created");

        // Initialize position encoding handler if rotary embeddings are enabled
        eprintln!("ExecutionPlan::from_gguf: Creating position handler...");
        let position_handler = if config.use_rotary_embeddings {
            let rope_config = RopeConfig::new(config.head_dim, config.max_position_embeddings);
            let glm_config =
                crate::model::glm_position::GlmPositionConfig::new(config.max_position_embeddings)
                    .with_rope(rope_config);
            Some(GlmPositionHandler::new(glm_config).map_err(|e| {
                HipError::GenericError(format!("Failed to create position handler: {}", e))
            })?)
        } else {
            None
        };
        eprintln!("ExecutionPlan::from_gguf: Position handler created, returning ExecutionPlan...");

        Ok(ExecutionPlan {
            layers,
            config,
            embedding_weights_lazy,
            embedding_layout,
            lm_head_lazy,
            loader: loader_arc,
            backend: backend_arc,
            embedding_weights_cached: OnceCell::new(),
            lm_head_cached: OnceCell::new(),
            embedding_plan: OnceCell::new(),
            layer_ggml_plans: OnceCell::new(),
            rope_cache: OnceCell::new(),
            position_handler,
        })
    }

    /// Get or load a single layer tensor (lazy loading)
    pub fn get_or_load_tensor(&self, lazy: &Arc<LazyTensor>) -> HipResult<DeviceTensor> {
        match &**lazy {
            LazyTensor::Unloaded { name, .. } => {
                tracing::debug!("Loading tensor '{}' on-demand", name);
                let tensor = self
                    .loader
                    .load_tensor_to_gpu(name, &self.backend)
                    .map_err(|e| {
                        HipError::GenericError(format!("Failed to load tensor '{}': {}", name, e))
                    })?;
                Ok(DeviceTensor::clone(&tensor))
            }
            LazyTensor::Gpu { tensor, .. } => Ok(DeviceTensor::clone(tensor)),
        }
    }

    /// Get or load embedding weights (lazy loading)
    ///
    /// Returns cached GPU tensor if already loaded, otherwise loads on-demand.
    /// Thread-safe via OnceCell.
    /// Handles transposition from [hidden_size, vocab_size] to [vocab_size, hidden_size] if needed.
    pub fn embedding_weights(&self) -> HipResult<DeviceTensor> {
        self.embedding_weights_cached
            .get_or_try_init(|| {
                match &*self.embedding_weights_lazy {
                    LazyTensor::Unloaded { name, .. } => {
                        tracing::debug!("Loading embedding tensor '{}' on-demand", name);
                        let mut tensor = self.loader.load_tensor_to_gpu(name, &self.backend)
                            .map_err(|e| HipError::GenericError(format!("Failed to load embedding: {}", e)))?;

                        // Check if transposition is needed
                        let shape = tensor.shape().dims();
                        if shape.len() == 2 {
                            if shape[0] == self.config.hidden_size && shape[1] == self.config.vocab_size {
                                // Need to transpose from [hidden_size, vocab_size] to [vocab_size, hidden_size]
                                tracing::debug!("Transposing embedding tensor from [{}, {}] to [{}, {}]",
                                              shape[0], shape[1], shape[1], shape[0]);
                                use super::matmul::transpose_2d_tensor;
                                tensor = Arc::new(transpose_2d_tensor(&self.backend, &tensor)
                                    .map_err(|e| HipError::GenericError(format!("Failed to transpose embedding: {}", e)))?);
                            }
                            // If already [vocab_size, hidden_size], use as-is
                        }

                        Ok(tensor)
                    }
                    LazyTensor::Gpu { tensor, .. } => {
                        tracing::debug!("Embedding tensor already loaded (using cached)");
                        Ok(Arc::clone(tensor))
                    }
                }
            }).map(|t| DeviceTensor::clone(t))
    }

    /// Get or load LM head (lazy loading)
    pub fn lm_head(&self) -> HipResult<DeviceTensor> {
        eprintln!(">>> lm_head(): Getting LM head tensor...");
        let result = self.lm_head_cached.get_or_try_init(|| {
            eprintln!(">>> lm_head(): Not cached, loading...");
            match &*self.lm_head_lazy {
                LazyTensor::Unloaded { name, .. } => {
                    eprintln!(">>> lm_head(): Loading tensor '{}' on-demand", name);
                    let tensor = self
                        .loader
                        .load_tensor_to_gpu(name, &self.backend)
                        .map_err(|e| {
                            HipError::GenericError(format!("Failed to load LM head: {}", e))
                        })?;
                    eprintln!(">>> lm_head(): Tensor loaded successfully");
                    Ok(DeviceTensor::clone(&tensor))
                }
                LazyTensor::Gpu { tensor, .. } => {
                    eprintln!(">>> lm_head(): Already on GPU, using cached");
                    Ok(DeviceTensor::clone(tensor))
                }
            }
        });
        eprintln!(">>> lm_head(): Got tensor, cloning...");
        let result = result.map(|t| DeviceTensor::clone(t));
        eprintln!(">>> lm_head(): Complete");
        result
    }

    /// Apply LM head to hidden states to produce logits
    pub fn apply_lm_head(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        eprintln!(">>> apply_lm_head(): Starting LM head matmul...");
        let lm_head = self.lm_head()?;
        eprintln!(">>> apply_lm_head(): Got LM head tensor, calling matmul...");
        use super::matmul::matmul;
        matmul(self, backend, hidden_states, &lm_head, None)
    }

    /// Forward pass through entire transformer model
    ///
    /// Takes input token IDs and performs inference using loaded weights.
    /// Returns final hidden states after all transformer layers.
    ///
    /// # Phase 2 Lazy Loading
    ///
    /// This method now triggers on-demand tensor loading:
    /// - Embedding weights loaded on first access
    /// - Layer weights loaded progressively during forward pass
    /// - All tensors cached in GgufLoader GPU cache after first load
    ///
    /// Arguments:
    /// - backend: HIP backend for GPU operations
    /// - input_tokens: Token IDs to process [seq_len]
    /// - embedding_weights: DEPRECATED parameter (ignored, uses lazy loading instead)
    ///
    /// Returns:
    /// - DeviceTensor with final hidden states [seq_len, hidden_size]
    pub fn forward(
        &self,
        backend: &HipBackend,
        input_tokens: &[u32],
        _embedding_weights: &DeviceTensor, // Deprecated: ignored, uses lazy loading
    ) -> HipResult<DeviceTensor> {
        let seq_len = input_tokens.len();
        let _hidden_size = self.config.hidden_size;

        // Performance profiling: Start timing
        let start_time = std::time::Instant::now();
        println!("PERF: Starting forward pass for {} tokens", seq_len);

        // Step 1: Token embedding lookup (loads embedding on-demand)
        println!("PERF: Starting embedding lookup...");
        let embedding_start = std::time::Instant::now();
        let embedding = self.embedding_weights()?;
        println!("PERF: Embedding weights loaded, shape: {:?}", embedding.shape().dims());
        use super::embedding::embedding_lookup;
        let mut hidden_states = embedding_lookup(self, backend, input_tokens, &embedding)?;
        let embedding_time = embedding_start.elapsed();
        println!("PERF: Embedding lookup completed in {:?}, hidden_states shape: {:?}", embedding_time, hidden_states.shape().dims());

        // Step 2: Pass through all transformer layers (loads on-demand)
        let mut layer_times = Vec::new();
        println!("PERF: Starting layer processing, {} layers total", self.layers.len());
        for (layer_idx, layer_plan) in self.layers.iter().enumerate() {
            println!("PERF: Starting layer {}", layer_idx);
            let layer_start = std::time::Instant::now();
            use super::execute::forward_layer;
            hidden_states =
                forward_layer(self, backend, &hidden_states, layer_plan, None, layer_idx)?;
            let layer_time = layer_start.elapsed();
            layer_times.push(layer_time);

            // Debug output for first few layers
            if layer_idx < 3 {
                println!(
                    "DEBUG: Layer {} completed in {:?}, output shape: {:?}",
                    layer_idx,
                    layer_time,
                    hidden_states.shape().dims()
                );
            }
        }

        // Step 3: Final layer normalization (if needed)
        // Note: Some models have final layernorm, others don't

        let total_time = start_time.elapsed();
        println!("PERF: Forward pass completed in {:?}", total_time);
        println!("PERF: Final shape: {:?}", hidden_states.shape().dims());

        // Performance summary
        let total_layer_time: std::time::Duration = layer_times.iter().sum();
        println!(
            "PERF: Total layer time: {:?} ({:.2}% of total)",
            total_layer_time,
            (total_layer_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
        );

        if !layer_times.is_empty() {
            let avg_layer_time = total_layer_time / layer_times.len() as u32;
            // Safe to use expect here because is_empty() check ensures non-empty collection
            let min_layer_time = layer_times
                .iter()
                .min()
                .expect("layer_times is non-empty after is_empty() check");
            let max_layer_time = layer_times
                .iter()
                .max()
                .expect("layer_times is non-empty after is_empty() check");
            println!(
                "PERF: Layer timing - avg: {:?}, min: {:?}, max: {:?}",
                avg_layer_time, min_layer_time, max_layer_time
            );
        }

        println!(
            "PERF: Embedding vs Layers - embedding: {:?} ({:.2}%), layers: {:?} ({:.2}%)",
            embedding_time,
            (embedding_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0,
            total_layer_time,
            (total_layer_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
        );

        Ok(hidden_states)
    }

    /// Preload specific layers to GPU for faster inference
    ///
    /// Loads all tensors (attention weights, MLP weights, layer norm weights)
    /// for the specified layers. Useful for preloading layers that will be
    /// used soon to avoid loading during inference.
    ///
    /// # Arguments
    /// * `layer_indices` - Indices of layers to preload
    ///
    /// # Returns
    /// * `Ok(())` if all layers preloaded successfully
    /// * `Err(HipError)` if any layer fails to load
    pub fn preload_layers(&self, layer_indices: &[usize]) -> HipResult<()> {
        for &layer_idx in layer_indices {
            if layer_idx >= self.layers.len() {
                return Err(HipError::GenericError(format!(
                    "Layer index {} out of bounds (num_layers: {})",
                    layer_idx,
                    self.layers.len()
                )));
            }

            let layer = &self.layers[layer_idx];

            // Load all layer tensors
            self.get_or_load_tensor(&layer.qkv_weight)?;
            self.get_or_load_tensor(&layer.o_proj)?;
            self.get_or_load_tensor(&layer.mlp_gate_proj)?;
            self.get_or_load_tensor(&layer.mlp_up_proj)?;
            self.get_or_load_tensor(&layer.mlp_down_proj)?;
            self.get_or_load_tensor(&layer.norm1_weight)?;
            self.get_or_load_tensor(&layer.norm2_weight)?;

            // Optional tensors
            if let Some(ref bias) = layer.qkv_bias {
                self.get_or_load_tensor(bias)?;
            }
            if let Some(ref bias) = layer.o_proj_bias {
                self.get_or_load_tensor(bias)?;
            }
            if let Some(ref bias) = layer.norm1_bias {
                self.get_or_load_tensor(bias)?;
            }
            if let Some(ref bias) = layer.norm2_bias {
                self.get_or_load_tensor(bias)?;
            }
        }
        Ok(())
    }

    /// Preload all layers to GPU
    ///
    /// Loads all layer weights to GPU memory. Useful for eliminating
    /// lazy loading overhead after model initialization.
    ///
    /// # Returns
    /// * `Ok(())` if all layers preloaded successfully
    /// * `Err(HipError)` if any layer fails to load
    pub fn preload_all(&self) -> HipResult<()> {
        let all_indices: Vec<usize> = (0..self.layers.len()).collect();
        self.preload_layers(&all_indices)
    }

    /// Get loading statistics for debugging/observability
    ///
    /// Returns statistics about which tensors are loaded vs unloaded.
    /// Useful for monitoring memory usage and lazy loading behavior.
    ///
    /// # Returns
    /// Statistics about tensor loading state
    pub fn loading_stats(&self) -> LoadingStats {
        let mut total_tensors = 0;
        let mut loaded_tensors = 0;
        let mut unloaded_tensors = 0;
        let mut cached_tensors = 0;

        // Check embedding and LM head
        total_tensors += 2;
        if self.embedding_weights_cached.get().is_some() {
            loaded_tensors += 1;
            cached_tensors += 1;
        } else {
            unloaded_tensors += 1;
        }

        if self.lm_head_cached.get().is_some() {
            loaded_tensors += 1;
            cached_tensors += 1;
        } else {
            unloaded_tensors += 1;
        }

        // Check each layer
        for layer in &self.layers {
            // Count tensors in each layer
            let layer_tensors = [
                &layer.qkv_weight,
                &layer.o_proj,
                &layer.mlp_gate_proj,
                &layer.mlp_up_proj,
                &layer.mlp_down_proj,
                &layer.norm1_weight,
                &layer.norm2_weight,
            ];

            for lazy_tensor in layer_tensors.iter() {
                total_tensors += 1;
                match &***lazy_tensor {
                    LazyTensor::Gpu { .. } => {
                        loaded_tensors += 1;
                    }
                    LazyTensor::Unloaded { .. } => {
                        unloaded_tensors += 1;
                    }
                }
            }

            // Optional tensors
            if let Some(ref bias) = layer.qkv_bias {
                total_tensors += 1;
                match &**bias {
                    LazyTensor::Gpu { .. } => loaded_tensors += 1,
                    LazyTensor::Unloaded { .. } => unloaded_tensors += 1,
                }
            }
            if let Some(ref bias) = layer.o_proj_bias {
                total_tensors += 1;
                match &**bias {
                    LazyTensor::Gpu { .. } => loaded_tensors += 1,
                    LazyTensor::Unloaded { .. } => unloaded_tensors += 1,
                }
            }
            if let Some(ref bias) = layer.norm1_bias {
                total_tensors += 1;
                match &**bias {
                    LazyTensor::Gpu { .. } => loaded_tensors += 1,
                    LazyTensor::Unloaded { .. } => unloaded_tensors += 1,
                }
            }
            if let Some(ref bias) = layer.norm2_bias {
                total_tensors += 1;
                match &**bias {
                    LazyTensor::Gpu { .. } => loaded_tensors += 1,
                    LazyTensor::Unloaded { .. } => unloaded_tensors += 1,
                }
            }
        }

        LoadingStats {
            total_tensors,
            loaded_tensors,
            unloaded_tensors,
            cached_tensors,
        }
    }

    /// Token embedding lookup (public API for backward compatibility)
    ///
    /// Converts token IDs to embeddings using the embedding weight matrix.
    /// Delegates to the embedding_lookup function in the embedding module.
    pub fn embedding_lookup(
        &self,
        backend: &HipBackend,
        input_tokens: &[u32],
        embedding_weights: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        use super::embedding::embedding_lookup;
        embedding_lookup(self, backend, input_tokens, embedding_weights)
    }

    /// Forward pass through a single transformer layer (public API for backward compatibility)
    ///
    /// Implements the standard transformer layer pattern:
    /// LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    ///
    /// Delegates to the forward_layer function in the execute module.
    pub fn forward_layer(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        layer_plan: &LayerPlan,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        use super::execute::forward_layer;
        forward_layer(self, backend, hidden_states, layer_plan, kv_cache, layer_idx)
    }
}

// All accessor methods for OnceCell fields
impl ExecutionPlan {
    pub(crate) fn embedding_plan(&self) -> &OnceCell<EmbeddingGgmlPlan> {
        &self.embedding_plan
    }

    pub(crate) fn layer_ggml_plans(&self) -> &OnceCell<Vec<LayerGgmlPlan>> {
        &self.layer_ggml_plans
    }

    pub(crate) fn rope_cache(&self) -> &OnceCell<RopeCache> {
        &self.rope_cache
    }

    pub(crate) fn config_internal(&self) -> &ModelConfig {
        &self.config
    }

    pub(crate) fn layers_internal(&self) -> &[LayerPlan] {
        &self.layers
    }
}
