// This file contains the original execution_plan.rs code
// It will be gradually modularized into separate files
// See mod.rs for module-level documentation

#![allow(deprecated)] // TODO: Migrate from to_host_vec() to copy_from_device_safe() (Phase 13-03-02)

use super::{Architecture, LayerPlan};
use super::ggml_plan::{EmbeddingGgmlPlan, RopeCache, LayerGgmlPlan};

use crate::attention::rope::RopeConfig;
use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::loader::gguf::GgufLoader;
use crate::loader::lazy_tensor::LazyTensor;
use crate::loader::TensorShape;
use crate::ggml::backend::GgmlBackend;
use crate::ggml::{executor::execute_graph, Graph, Layout, Op, TensorDesc, DType};
use crate::ggml::hip_backend::HipGgmlBackend;
use crate::model::{config::ModelConfig, glm_position::GlmPositionHandler, kv_cache::KVCache};
use once_cell::sync::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex as StdMutex; // Renamed to avoid conflict with once_cell::sync

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
                                tensor = Arc::new(Self::transpose_2d_tensor(&self.backend, &tensor)
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
        self.matmul(backend, hidden_states, &lm_head, None)
    }

    /// Get or load a single layer tensor (lazy loading)
    fn get_or_load_tensor(&self, lazy: &Arc<LazyTensor>) -> HipResult<DeviceTensor> {
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

    /// Get or load fused QKV tensor, handling separate Q/K/V weights for Qwen2
    ///
    /// NOTE: This method is currently unused. The Qwen2 attention weights are loaded
    /// separately in get_layer_weights() and concatenated there. This method was
    /// intended for lazy QKV loading but the current implementation loads weights eagerly.
    /// Kept for potential future lazy loading optimization.
    #[allow(dead_code)]
    fn get_or_load_fused_qkv(&self, q_lazy: &Arc<LazyTensor>) -> HipResult<DeviceTensor> {
        match &**q_lazy {
            LazyTensor::Unloaded { name, .. } => {
                if name.contains(".attn_q.weight") {
                    // Try to load fused QKV first (LLaMA style)
                    let fused_name = name.replace(".attn_q.weight", ".attention.wq.weight");
                    if let Some(fused_lazy) = self.loader.lazy_tensors.get(&fused_name) {
                        tracing::debug!("Loading fused QKV tensor '{}' for layer", fused_name);
                        return self.get_or_load_tensor(&Arc::new(fused_lazy.clone()));
                    }

                    // If no fused, load separate Q/K/V and concatenate (Qwen2 style)
                    let k_name = name.replace(".attn_q.weight", ".attn_k.weight");
                    let v_name = name.replace(".attn_q.weight", ".attn_v.weight");

                    if let (Some(k_lazy), Some(v_lazy)) = (
                        self.loader.lazy_tensors.get(&k_name),
                        self.loader.lazy_tensors.get(&v_name),
                    ) {
                        tracing::debug!("Loading separate Q/K/V tensors and concatenating for layer");
                        let q_tensor = self.get_or_load_tensor(q_lazy)?;
                        let k_tensor = self.get_or_load_tensor(&Arc::new(k_lazy.clone()))?;
                        let v_tensor = self.get_or_load_tensor(&Arc::new(v_lazy.clone()))?;
                        return ExecutionPlan::concatenate_qkv_tensors(self.backend.as_ref(), &q_tensor, &k_tensor, &v_tensor, &self.config);
                    }
                }

                // Fall back to loading the tensor as-is
                self.get_or_load_tensor(q_lazy)
            }
            LazyTensor::Gpu { tensor, .. } => {
                Ok(DeviceTensor::clone(tensor))
            }
        }
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

        // ❌ REMOVE: Load all tensors to GPU (~55s)
        // Phase 2: Use lazy loading instead - no eager loading
        // let gpu_tensors = loader.load_to_gpu(backend)?;

        // ✅ NEW: Get lazy tensor handles (metadata only, <1s)
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

        // Map embedding and LM head to LazyTensor handles
        eprintln!("ExecutionPlan::from_gguf: Mapping embedding weights...");
        let (embedding_weights_lazy, embedding_layout) =
            Self::map_embedding_lazy(lazy_tensors, &config, &architecture)?;
        eprintln!("ExecutionPlan::from_gguf: Mapping LM head...");
        let lm_head_lazy = Self::map_lm_head_lazy(lazy_tensors, &config, &architecture)?;
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
                Self::create_layer_plan_lazy(&config, lazy_tensors, layer_idx, &architecture)?;
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

    /// Map embedding weights to LazyTensor handle
    ///
    /// This function implements llama.cpp-compatible embedding detection:
    /// - vocab_size == 0 means "unknown", not "invalid"
    /// - Accepts both [vocab_size, hidden] and [hidden, vocab_size] layouts
    /// - Infers vocab_size from tensor shape when metadata is missing
    fn map_embedding_lazy(
        lazy_tensors: &HashMap<String, LazyTensor>,
        config: &ModelConfig,
        _architecture: &Architecture,
    ) -> HipResult<(Arc<LazyTensor>, Layout)> {
        let embedding_names = [
            "token_embd.weight",
            "embed_tokens.weight",
            "word_embeddings.weight",
        ];

        eprintln!(">>> map_embedding_lazy: config.hidden_size={}, config.vocab_size={}",
            config.hidden_size, config.vocab_size);

        // Try to find and validate embedding tensor
        for name in &embedding_names {
            if let Some(lazy) = lazy_tensors.get(*name) {
                if let Some(shape) = lazy.shape() {
                    eprintln!(">>> map_embedding_lazy: Found tensor '{}' with shape {:?}", name, shape);
                    if shape.len() != 2 {
                        eprintln!(">>> map_embedding_lazy: Skipping (not 2D)");
                        continue;
                    }

                    let (d0, d1) = (shape[0], shape[1]);
                    let hidden_size = config.hidden_size;
                    eprintln!(">>> map_embedding_lazy: d0={}, d1={}, hidden_size={}", d0, d1, hidden_size);

                    // Determine vocab_size if unknown (== 0)
                    let actual_vocab_size = if config.vocab_size == 0 {
                        // Infer from shape: whichever dimension ISN'T hidden_size is vocab_size
                        if d0 == hidden_size && d1 != hidden_size {
                            d1 // [hidden, vocab] layout
                        } else if d1 == hidden_size && d0 != hidden_size {
                            d0 // [vocab, hidden] layout
                        } else {
                            // Can't determine - use larger dimension as vocab (llama.cpp heuristic)
                            d0.max(d1)
                        }
                    } else {
                        config.vocab_size
                    };
                    eprintln!(">>> map_embedding_lazy: actual_vocab_size={}", actual_vocab_size);

                    // Check if this tensor matches expected patterns
                    // Accept: [vocab, hidden] OR [hidden, vocab]
                    if d0 == actual_vocab_size && d1 == hidden_size {
                        eprintln!(">>> map_embedding_lazy: Match [vocab, hidden] pattern -> RowMajor");
                        tracing::info!(
                            "Found embedding tensor '{}' with shape {:?}, inferred vocab_size={}",
                            name,
                            shape,
                            actual_vocab_size
                        );
                        return Ok((Arc::new(lazy.clone()), Layout::RowMajor));
                    }

                    if d0 == hidden_size && d1 == actual_vocab_size {
                        eprintln!(">>> map_embedding_lazy: Match [hidden, vocab] pattern -> ColMajor");
                        tracing::info!(
                            "Found embedding tensor '{}' with shape {:?}, inferred vocab_size={}",
                            name,
                            shape,
                            actual_vocab_size
                        );
                        return Ok((Arc::new(lazy.clone()), Layout::ColMajor));
                    }
                }
            }
        }

        // Fail with evidence (llama.cpp style - list what we found)
        let mut found_tensors = Vec::new();
        for name in &embedding_names {
            if let Some(lazy) = lazy_tensors.get(*name) {
                if let Some(shape) = lazy.shape() {
                    found_tensors.push(format!("{}: {:?}", name, shape));
                }
            }
        }

        let error_msg = if found_tensors.is_empty() {
            format!("No embedding tensor found (tried: {}). lazy_tensors has {} total keys. vocab_size={}, hidden_size={}",
                   embedding_names.join(", "), lazy_tensors.len(), config.vocab_size, config.hidden_size)
        } else {
            format!("Found embedding tensors but shape validation failed. Found: {}. Expected 2D tensor with vocab_size={} and hidden_size={}",
                   found_tensors.join(", "), config.vocab_size, config.hidden_size)
        };

        Err(HipError::GenericError(error_msg))
    }

    /// Map LM head to LazyTensor handle
    ///
    /// This function implements llama.cpp-compatible LM head detection:
    /// - Accepts both [vocab, hidden] and [hidden, vocab] layouts
    /// - Falls back to tied embeddings (token_embd.weight) when no separate LM head exists
    fn map_lm_head_lazy(
        lazy_tensors: &HashMap<String, LazyTensor>,
        config: &ModelConfig,
        _architecture: &Architecture,
    ) -> HipResult<Arc<LazyTensor>> {
        let lm_head_names = ["output.weight", "lm_head.weight", "logits.weight"];

        // Try explicit LM head tensors first
        for name in &lm_head_names {
            if let Some(lazy) = lazy_tensors.get(*name) {
                if let Some(shape) = lazy.shape() {
                    if shape.len() == 2 {
                        // Accept either layout - validation happens at usage time
                        tracing::info!("Found LM head tensor '{}' with shape {:?}", name, shape);
                        return Ok(Arc::new(lazy.clone()));
                    }
                }
            }
        }

        // For tied embeddings (Qwen2 style), try embedding tensors
        let tied_names = ["token_embd.weight", "embed_tokens.weight"];
        for name in &tied_names {
            if let Some(lazy) = lazy_tensors.get(*name) {
                tracing::info!("Using tied embedding '{}' as LM head", name);
                return Ok(Arc::new(lazy.clone()));
            }
        }

        Err(HipError::GenericError(format!(
            "No LM head tensor found (tried: {}). vocab_size={}, hidden_size={}",
            lm_head_names.join(", "),
            config.vocab_size,
            config.hidden_size
        )))
    }

    /// Create layer plan with LazyTensor handles
    fn create_layer_plan_lazy(
        _config: &ModelConfig,
        lazy_tensors: &HashMap<String, LazyTensor>,
        layer_idx: usize,
        architecture: &Architecture,
    ) -> HipResult<LayerPlan> {
        eprintln!(
            ">>> CREATE_LAYER_PLAN_LAZY: layer_idx={}, starting...",
            layer_idx
        );
        let prefix = architecture.layer_prefix(layer_idx);
        eprintln!(">>> CREATE_LAYER_PLAN_LAZY: prefix='{}'", prefix);

        // Helper to get lazy tensor or error
        let get_lazy = |name: &str| -> HipResult<Arc<LazyTensor>> {
            lazy_tensors
                .get(name)
                .cloned()
                .map(Arc::new)
                .ok_or_else(|| HipError::GenericError(format!("Tensor '{}' not found", name)))
        };

        let get_lazy_optional = |name: &str| -> Option<Arc<LazyTensor>> {
            lazy_tensors.get(name).cloned().map(Arc::new)
        };

        // Detect attention tensor format:
        // - Some models (LLaMA) use fused QKV: attn_qkv.weight
        // - Some models (Qwen2) use separate: attn_q.weight, attn_k.weight, attn_v.weight
        let qkv_key = &format!("{}.attn_qkv.weight", prefix);
        let q_key = &format!("{}.attn_q.weight", prefix);
        let k_key = &format!("{}.attn_k.weight", prefix);
        let v_key = &format!("{}.attn_v.weight", prefix);

        let has_fused_qkv = lazy_tensors.contains_key(qkv_key);
        let has_q = lazy_tensors.contains_key(q_key);
        let has_k = lazy_tensors.contains_key(k_key);
        let has_v = lazy_tensors.contains_key(v_key);
        let has_separate_qkv = has_q && has_k && has_v;

        eprintln!(">>> Layer {}: Detection - prefix='{}', has_fused_qkv={}, has_separate_qkv={} (q={}, k={}, v={})",
                 layer_idx, prefix, has_fused_qkv, has_separate_qkv, has_q, has_k, has_v);

        let (qkv_weight, q_weight, k_weight, v_weight, qkv_bias, q_bias, k_bias, v_bias) =
            if has_fused_qkv {
                // Model uses fused QKV weight
                eprintln!(">>> Layer {}: Using fused QKV weight", layer_idx);
                (
                    get_lazy(&format!("{}.attn_qkv.weight", prefix))?,
                    None,
                    None,
                    None,
                    get_lazy_optional(&format!("{}.attn_qkv.bias", prefix)),
                    None,
                    None,
                    None,
                )
            } else if has_separate_qkv {
                // Model uses separate Q, K, V weights (e.g., Qwen2 with GQA)
                eprintln!(">>> Layer {}: Using separate Q, K, V weights", layer_idx);
                let q_weight = get_lazy(&format!("{}.attn_q.weight", prefix))?;
                let k_weight = get_lazy(&format!("{}.attn_k.weight", prefix))?;
                let v_weight = get_lazy(&format!("{}.attn_v.weight", prefix))?;
                let q_bias = get_lazy_optional(&format!("{}.attn_q.bias", prefix));
                let k_bias = get_lazy_optional(&format!("{}.attn_k.bias", prefix));
                let v_bias = get_lazy_optional(&format!("{}.attn_v.bias", prefix));

                // Create a placeholder qkv_weight (required by LayerPlan struct)
                // It won't be used since q_weight, k_weight, v_weight are present
                let qkv_weight = q_weight.clone();

                (
                    qkv_weight,
                    Some(q_weight),
                    Some(k_weight),
                    Some(v_weight),
                    None,
                    q_bias,
                    k_bias,
                    v_bias,
                )
            } else {
                return Err(HipError::GenericError(format!(
                "Layer {}: Neither fused QKV ({0}.attn_qkv.weight) nor separate Q,K,V ({0}.attn_q/k/v.weight) found",
                prefix
            )));
            };

        let o_proj = get_lazy(&format!("{}.attn_output.weight", prefix))?;
        let o_proj_bias = get_lazy_optional(&format!("{}.attn_output.bias", prefix));

        // Map MLP weights
        let mlp_gate = get_lazy(&format!("{}.ffn_gate.weight", prefix))?;
        let mlp_up = get_lazy(&format!("{}.ffn_up.weight", prefix))?;
        let mlp_down = get_lazy(&format!("{}.ffn_down.weight", prefix))?;

        // Map layer norm weights
        let norm1_weight = get_lazy(&format!("{}.attn_norm.weight", prefix))?;
        let norm1_bias = get_lazy_optional(&format!("{}.attn_norm.bias", prefix));
        let norm2_weight = get_lazy(&format!("{}.ffn_norm.weight", prefix))?;
        let norm2_bias = get_lazy_optional(&format!("{}.ffn_norm.bias", prefix));

        Ok(LayerPlan {
            qkv_weight,
            q_weight,
            k_weight,
            v_weight,
            qkv_bias,
            q_bias,
            k_bias,
            v_bias,
            o_proj,
            o_proj_bias,
            mlp_gate_proj: mlp_gate.clone(),
            mlp_up_proj: mlp_up,
            mlp_down_proj: mlp_down.clone(),
            mlp_fc1: mlp_gate.clone(),
            mlp_fc1_bias: None,
            mlp_fc2: mlp_down,
            mlp_fc2_bias: None,
            norm1_weight,
            norm1_bias,
            norm2_weight,
            norm2_bias,
        })
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
        let mut hidden_states = self.embedding_lookup(backend, input_tokens, &embedding)?;
        let embedding_time = embedding_start.elapsed();
        println!("PERF: Embedding lookup completed in {:?}, hidden_states shape: {:?}", embedding_time, hidden_states.shape().dims());

        // Step 2: Pass through all transformer layers (loads on-demand)
        let mut layer_times = Vec::new();
        println!("PERF: Starting layer processing, {} layers total", self.layers.len());
        for (layer_idx, layer_plan) in self.layers.iter().enumerate() {
            println!("PERF: Starting layer {}", layer_idx);
            let layer_start = std::time::Instant::now();
            hidden_states =
                self.forward_layer(backend, &hidden_states, layer_plan, None, layer_idx)?;
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

    /// Token embedding lookup
    ///
    /// Converts token IDs to embeddings using the embedding weight matrix.
    fn build_embedding_plan(
        &self,
        backend: &HipBackend,
        embedding_weights: &DeviceTensor,
    ) -> HipResult<EmbeddingGgmlPlan> {
        let embed_shape = embedding_weights.shape().dims();
        eprintln!(">>> build_embedding_plan: embed_shape={:?}", embed_shape);
        eprintln!(">>> build_embedding_plan: self.config.hidden_size={}, self.config.vocab_size={}",
            self.config.hidden_size, self.config.vocab_size);

        if embed_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Embedding weight shape must be 2D, got {:?}",
                embed_shape
            )));
        }

        // Detect layout from actual tensor shape (not self.embedding_layout which may be stale)
        // After transpose in embedding_weights(), the tensor is always [vocab, hidden]
        let (n_embd, vocab_size) = (embed_shape[1], embed_shape[0]);
        eprintln!(">>> build_embedding_plan: n_embd={}, vocab_size={}", n_embd, vocab_size);

        if n_embd != self.config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Embedding hidden size mismatch: expected {}, got {}",
                self.config.hidden_size, n_embd
            )));
        }

        let max_seq_len = std::cmp::max(1, self.config.max_position_embeddings);
        let tokens_bytes = max_seq_len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                HipError::GenericError("Token buffer size overflow".to_string())
            })?;
        let output_elems = max_seq_len.checked_mul(n_embd).ok_or_else(|| {
            HipError::GenericError("Output buffer element overflow".to_string())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipError::GenericError("Output buffer size overflow".to_string()))?;

        let mut graph = Graph::new();
        // After transpose, embedding is [vocab, hidden] which is RowMajor layout
        // Do NOT use self.embedding_layout which is for the original GGUF [hidden, vocab] format
        let weights_id = graph.add_tensor(TensorDesc::new(
            embed_shape.to_vec(),
            DType::F32,
            Layout::RowMajor,  // Transposed format is always RowMajor [vocab, hidden]
        ));
        let tokens_id = graph.add_tensor(TensorDesc::new(
            vec![max_seq_len],
            DType::U32,
            Layout::RowMajor,
        ));
        let output_id = graph.add_tensor(TensorDesc::new(
            vec![max_seq_len, n_embd],
            DType::F32,
            Layout::RowMajor,
        ));
        graph.add_node(Op::GetRows, vec![weights_id, tokens_id], vec![output_id]);

        let mut ggml_backend = HipGgmlBackend::new(Arc::clone(&self.backend));
        let weights_desc = graph.tensors[weights_id.0].clone();
        ggml_backend
            .bind(&weights_desc, embedding_weights.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("GetRows bind weights failed: {:?}", e)))?;

        let tokens_desc = graph.tensors[tokens_id.0].clone();
        let tokens_buffer = backend.allocate_buffer(tokens_bytes)?;
        ggml_backend
            .bind(&tokens_desc, tokens_buffer.clone())
            .map_err(|e| HipError::GenericError(format!("GetRows bind tokens failed: {:?}", e)))?;

        let output_desc = graph.tensors[output_id.0].clone();
        let output_buffer = backend.allocate_buffer(output_bytes)?;
        ggml_backend
            .bind(&output_desc, output_buffer.clone())
            .map_err(|e| HipError::GenericError(format!("GetRows bind output failed: {:?}", e)))?;

        Ok(EmbeddingGgmlPlan {
            graph,
            backend: StdMutex::new(ggml_backend),
            tokens_buffer,
            output_buffer,
            max_seq_len,
            hidden_size: n_embd,
        })
    }

    fn rope_cache(&self) -> HipResult<Option<&RopeCache>> {
        let Some(ref position_handler) = self.position_handler else {
            return Ok(None);
        };
        let Some(rope) = position_handler.rope() else {
            return Ok(None);
        };

        let cache = self.rope_cache.get_or_try_init(|| {
            let half_dim = rope.config().head_dim / 2;
            let max_seq_len = rope.config().max_seq_len;
            let cos_shape = TensorShape::from_dims(&[max_seq_len, half_dim]);
            let sin_shape = TensorShape::from_dims(&[max_seq_len, half_dim]);
            let cos_tensor =
                DeviceTensor::from_host_vec(&self.backend, rope.cos().to_vec(), cos_shape)?;
            let sin_tensor =
                DeviceTensor::from_host_vec(&self.backend, rope.sin().to_vec(), sin_shape)?;
            Ok::<RopeCache, HipError>(RopeCache {
                cos: cos_tensor,
                sin: sin_tensor,
                half_dim,
                max_seq_len,
            })
        })?;

        Ok(Some(cache))
    }

    fn build_layer_ggml_plans(&self, _backend: &HipBackend) -> HipResult<Vec<LayerGgmlPlan>> {
        let mut plans = Vec::with_capacity(self.layers.len());
        let rope_cache = self.rope_cache()?;

        for layer_plan in &self.layers {
            let qkv_weight = self.get_or_load_tensor(&layer_plan.qkv_weight)?;
            let q_weight = layer_plan
                .q_weight
                .as_ref()
                .map(|w| self.get_or_load_tensor(w))
                .transpose()?;
            let k_weight = layer_plan
                .k_weight
                .as_ref()
                .map(|w| self.get_or_load_tensor(w))
                .transpose()?;
            let v_weight = layer_plan
                .v_weight
                .as_ref()
                .map(|w| self.get_or_load_tensor(w))
                .transpose()?;
            let qkv_bias = layer_plan
                .qkv_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let q_bias = layer_plan
                .q_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let k_bias = layer_plan
                .k_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let v_bias = layer_plan
                .v_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;

            let o_proj = self.get_or_load_tensor(&layer_plan.o_proj)?;
            let o_proj_bias = layer_plan
                .o_proj_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let mlp_gate_proj = self.get_or_load_tensor(&layer_plan.mlp_gate_proj)?;
            let mlp_up_proj = self.get_or_load_tensor(&layer_plan.mlp_up_proj)?;
            let mlp_down_proj = self.get_or_load_tensor(&layer_plan.mlp_down_proj)?;
            let norm1_weight = self.get_or_load_tensor(&layer_plan.norm1_weight)?;
            let norm1_bias = layer_plan
                .norm1_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let norm2_weight = self.get_or_load_tensor(&layer_plan.norm2_weight)?;
            let norm2_bias = layer_plan
                .norm2_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;

            let use_separate_qkv = q_weight.is_some() && k_weight.is_some() && v_weight.is_some();

            let num_heads = self.config.num_attention_heads;
            let head_dim = self.config.head_dim;
            let hidden_size = self.config.hidden_size;
            let max_seq_len = self.config.max_position_embeddings.max(1);

            let mut graph = Graph::new();

            let input_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let norm1_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let norm2_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));

            let norm1_w_id = graph.add_tensor(TensorDesc::new(
                norm1_weight.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let norm1_bias_id = norm1_bias.as_ref().map(|bias| {
                graph.add_tensor(TensorDesc::new(
                    bias.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ))
            });
            let mut norm1_inputs = vec![input_id, norm1_w_id];
            if let Some(bias_id) = norm1_bias_id {
                norm1_inputs.push(bias_id);
            }
            graph.add_node(Op::LayerNorm { eps: 1e-6 }, norm1_inputs, vec![norm1_id]);

            let q_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let k_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let v_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let q_id = graph.add_tensor(
                TensorDesc::new(vec![1, num_heads, head_dim], DType::F32, Layout::RowMajor)
                    .view_of(q_flat_id, 0),
            );
            let k_id = graph.add_tensor(
                TensorDesc::new(vec![1, num_heads, head_dim], DType::F32, Layout::RowMajor)
                    .view_of(k_flat_id, 0),
            );
            let v_id = graph.add_tensor(
                TensorDesc::new(vec![1, num_heads, head_dim], DType::F32, Layout::RowMajor)
                    .view_of(v_flat_id, 0),
            );

            let mut q_w_id = None;
            let mut k_w_id = None;
            let mut v_w_id = None;
            let mut qkv_w_id = None;
            let mut qkv_bias_id = None;
            let mut q_bias_id = None;
            let mut k_bias_id = None;
            let mut v_bias_id = None;
            let mut o_proj_bias_id = None;

            let (q_src_id, k_src_id, v_src_id) = if use_separate_qkv {
                // Safe to use expect here because use_separate_qkv check ensures all weights are Some
                let q_ref = q_weight.as_ref().expect("q_weight is Some when use_separate_qkv is true");
                let k_ref = k_weight.as_ref().expect("k_weight is Some when use_separate_qkv is true");
                let v_ref = v_weight.as_ref().expect("v_weight is Some when use_separate_qkv is true");

                let q_id_local = graph.add_tensor(TensorDesc::new(
                    q_ref.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                let k_id_local = graph.add_tensor(TensorDesc::new(
                    k_ref.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                let v_id_local = graph.add_tensor(TensorDesc::new(
                    v_ref.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                q_w_id = Some(q_id_local);
                k_w_id = Some(k_id_local);
                v_w_id = Some(v_id_local);
                if q_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    q_bias_id = Some(bias_id);
                    let q_mm_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    graph.add_node(Op::MatMul, vec![norm1_id, q_id_local], vec![q_mm_id]);
                    graph.add_node(Op::Add, vec![q_mm_id, bias_id], vec![q_flat_id]);
                } else {
                    graph.add_node(Op::MatMul, vec![norm1_id, q_id_local], vec![q_flat_id]);
                }

                if k_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    k_bias_id = Some(bias_id);
                    let k_mm_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    graph.add_node(Op::MatMul, vec![norm1_id, k_id_local], vec![k_mm_id]);
                    graph.add_node(Op::Add, vec![k_mm_id, bias_id], vec![k_flat_id]);
                } else {
                    graph.add_node(Op::MatMul, vec![norm1_id, k_id_local], vec![k_flat_id]);
                }

                if v_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    v_bias_id = Some(bias_id);
                    let v_mm_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    graph.add_node(Op::MatMul, vec![norm1_id, v_id_local], vec![v_mm_id]);
                    graph.add_node(Op::Add, vec![v_mm_id, bias_id], vec![v_flat_id]);
                } else {
                    graph.add_node(Op::MatMul, vec![norm1_id, v_id_local], vec![v_flat_id]);
                }

                (q_flat_id, k_flat_id, v_flat_id)
            } else {
                let qkv_id = graph.add_tensor(TensorDesc::new(
                    vec![1, hidden_size * 3],
                    DType::F32,
                    Layout::RowMajor,
                ));
                let qkv_id_local = graph.add_tensor(TensorDesc::new(
                    qkv_weight.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                qkv_w_id = Some(qkv_id_local);
                let qkv_mm_id = if qkv_bias.is_some() {
                    graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size * 3],
                        DType::F32,
                        Layout::RowMajor,
                    ))
                } else {
                    qkv_id
                };
                graph.add_node(Op::MatMul, vec![norm1_id, qkv_id_local], vec![qkv_mm_id]);
                if qkv_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size * 3],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    qkv_bias_id = Some(bias_id);
                    graph.add_node(Op::Add, vec![qkv_mm_id, bias_id], vec![qkv_id]);
                }
                graph.add_node(
                    Op::SplitQkv,
                    vec![qkv_id],
                    vec![q_flat_id, k_flat_id, v_flat_id],
                );
                (q_flat_id, k_flat_id, v_flat_id)
            };

            let cos_id = graph.add_tensor(TensorDesc::new(
                vec![1, head_dim / 2],
                DType::F32,
                Layout::RowMajor,
            ));
            let sin_id = graph.add_tensor(TensorDesc::new(
                vec![1, head_dim / 2],
                DType::F32,
                Layout::RowMajor,
            ));

            let q_rope_id = graph.add_tensor(TensorDesc::new(
                vec![1, num_heads, head_dim],
                DType::F32,
                Layout::RowMajor,
            ));
            let k_rope_id = graph.add_tensor(TensorDesc::new(
                vec![1, num_heads, head_dim],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Reshape, vec![q_src_id], vec![q_id]);
            graph.add_node(Op::Reshape, vec![k_src_id], vec![k_id]);
            graph.add_node(Op::Reshape, vec![v_src_id], vec![v_id]);

            if rope_cache.is_some() {
                graph.add_node(Op::Rope, vec![q_id, cos_id, sin_id], vec![q_rope_id]);
                graph.add_node(Op::Rope, vec![k_id, cos_id, sin_id], vec![k_rope_id]);
            } else {
                graph.add_node(Op::Copy, vec![q_id], vec![q_rope_id]);
                graph.add_node(Op::Copy, vec![k_id], vec![k_rope_id]);
            }

            let kv_read_k_id = graph.add_tensor(
                TensorDesc::new(
                    vec![max_seq_len, num_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );
            let kv_read_v_id = graph.add_tensor(
                TensorDesc::new(
                    vec![max_seq_len, num_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );
            let kv_write_k_id = graph.add_tensor(
                TensorDesc::new(
                    vec![1, num_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );
            let kv_write_v_id = graph.add_tensor(
                TensorDesc::new(
                    vec![1, num_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );

            graph.add_node(Op::Copy, vec![k_rope_id], vec![kv_write_k_id]);
            graph.add_node(Op::Copy, vec![v_id], vec![kv_write_v_id]);

            let scores_id = graph.add_tensor(TensorDesc::new(
                vec![1, max_seq_len],
                DType::F32,
                Layout::RowMajor,
            ));
            let softmax_id = graph.add_tensor(TensorDesc::new(
                vec![1, max_seq_len],
                DType::F32,
                Layout::RowMajor,
            ));
            let attn_out_id = graph.add_tensor(TensorDesc::new(
                vec![1, num_heads, head_dim],
                DType::F32,
                Layout::RowMajor,
            ));

            graph.add_node(
                Op::Attention,
                vec![q_rope_id, kv_read_k_id, kv_read_v_id, scores_id, softmax_id],
                vec![attn_out_id],
            );

            let attn_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Reshape, vec![attn_out_id], vec![attn_flat_id]);

            let o_proj_id = graph.add_tensor(TensorDesc::new(
                o_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let attn_proj_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let attn_proj_mm_id = if o_proj_bias.is_some() {
                graph.add_tensor(TensorDesc::new(
                    vec![1, hidden_size],
                    DType::F32,
                    Layout::RowMajor,
                ))
            } else {
                attn_proj_id
            };
            graph.add_node(
                Op::MatMul,
                vec![attn_flat_id, o_proj_id],
                vec![attn_proj_mm_id],
            );
            if o_proj_bias.is_some() {
                let bias_id = graph.add_tensor(TensorDesc::new(
                    vec![1, hidden_size],
                    DType::F32,
                    Layout::RowMajor,
                ));
                o_proj_bias_id = Some(bias_id);
                graph.add_node(Op::Add, vec![attn_proj_mm_id, bias_id], vec![attn_proj_id]);
            }

            let attn_residual_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Add, vec![attn_proj_id, input_id], vec![attn_residual_id]);

            let norm2_w_id = graph.add_tensor(TensorDesc::new(
                norm2_weight.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let norm2_bias_id = norm2_bias.as_ref().map(|bias| {
                graph.add_tensor(TensorDesc::new(
                    bias.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ))
            });
            let mut norm2_inputs = vec![attn_residual_id, norm2_w_id];
            if let Some(bias_id) = norm2_bias_id {
                norm2_inputs.push(bias_id);
            }
            graph.add_node(Op::LayerNorm { eps: 1e-6 }, norm2_inputs, vec![norm2_id]);

            let mlp_gate_id = graph.add_tensor(TensorDesc::new(
                mlp_gate_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let mlp_up_id = graph.add_tensor(TensorDesc::new(
                mlp_up_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let mlp_down_id = graph.add_tensor(TensorDesc::new(
                mlp_down_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let mlp_out_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(
                Op::MlpSwiglu,
                vec![norm2_id, mlp_gate_id, mlp_up_id, mlp_down_id],
                vec![mlp_out_id],
            );

            let output_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Add, vec![mlp_out_id, attn_residual_id], vec![output_id]);

            let mut ggml_backend = HipGgmlBackend::new(Arc::clone(&self.backend));
            ggml_backend.bind(&graph.tensors[norm1_w_id.0], norm1_weight.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[norm2_w_id.0], norm2_weight.buffer().clone())?;
            if let (Some(bias), Some(bias_id)) = (norm1_bias.as_ref(), norm1_bias_id) {
                ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
            }
            if let (Some(bias), Some(bias_id)) = (norm2_bias.as_ref(), norm2_bias_id) {
                ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
            }
            ggml_backend.bind(&graph.tensors[o_proj_id.0], o_proj.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[mlp_gate_id.0], mlp_gate_proj.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[mlp_up_id.0], mlp_up_proj.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[mlp_down_id.0], mlp_down_proj.buffer().clone())?;

            if use_separate_qkv {
                // Safe to use expect here because use_separate_qkv ensures all weights are Some
                let q_buf = q_weight.as_ref().expect("q_weight is Some when use_separate_qkv is true");
                let k_buf = k_weight.as_ref().expect("k_weight is Some when use_separate_qkv is true");
                let v_buf = v_weight.as_ref().expect("v_weight is Some when use_separate_qkv is true");

                ggml_backend.bind(
                    &graph.tensors[q_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing Q weight id".to_string())
                    })?.0],
                    q_buf.buffer().clone(),
                )?;
                ggml_backend.bind(
                    &graph.tensors[k_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing K weight id".to_string())
                    })?.0],
                    k_buf.buffer().clone(),
                )?;
                ggml_backend.bind(
                    &graph.tensors[v_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing V weight id".to_string())
                    })?.0],
                    v_buf.buffer().clone(),
                )?;
                if let (Some(bias), Some(bias_id)) = (q_bias.as_ref(), q_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
                if let (Some(bias), Some(bias_id)) = (k_bias.as_ref(), k_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
                if let (Some(bias), Some(bias_id)) = (v_bias.as_ref(), v_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
            } else {
                ggml_backend.bind(
                    &graph.tensors[qkv_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing QKV weight id".to_string())
                    })?.0],
                    qkv_weight.buffer().clone(),
                )?;
                if let (Some(bias), Some(bias_id)) = (qkv_bias.as_ref(), qkv_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
            }

            if let Some(rope_cache) = rope_cache {
                ggml_backend.bind(&graph.tensors[cos_id.0], rope_cache.cos.buffer().clone())?;
                ggml_backend.bind(&graph.tensors[sin_id.0], rope_cache.sin.buffer().clone())?;
            }

            if let (Some(bias), Some(bias_id)) = (o_proj_bias.as_ref(), o_proj_bias_id) {
                ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
            }

            let scores_bytes = max_seq_len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| HipError::GenericError("Scores buffer size overflow".to_string()))?;
            let scores_buffer = self.backend.allocate_buffer(scores_bytes)?;
            let softmax_buffer = self.backend.allocate_buffer(scores_bytes)?;
            ggml_backend.bind(&graph.tensors[scores_id.0], scores_buffer)?;
            ggml_backend.bind(&graph.tensors[softmax_id.0], softmax_buffer)?;

            plans.push(LayerGgmlPlan {
                graph: StdMutex::new(graph),
                backend: StdMutex::new(ggml_backend),
                input_id,
                output_id,
                kv_read_k_id,
                kv_read_v_id,
                kv_write_k_id,
                kv_write_v_id,
                scores_id,
                softmax_id,
                cos_id,
                sin_id,
                num_heads,
                head_dim,
                hidden_size,
                max_seq_len,
            });
        }

        Ok(plans)
    }

    pub(crate) fn forward_layer_ggml_decode(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        let plans = self
            .layer_ggml_plans
            .get_or_try_init(|| self.build_layer_ggml_plans(backend))?;
        let plan = plans.get(layer_idx).ok_or_else(|| {
            HipError::GenericError(format!("Missing ggml plan for layer {}", layer_idx))
        })?;

        let current_len = kv_cache.current_seq_len(layer_idx)?;
        if current_len >= plan.max_seq_len {
            return Err(HipError::GenericError(format!(
                "KV cache capacity exceeded at layer {} (len={}, max={})",
                layer_idx, current_len, plan.max_seq_len
            )));
        }
        let new_len = current_len + 1;
        let stride = plan.num_heads * plan.head_dim;
        let elem_bytes = std::mem::size_of::<f32>();
        let write_bytes = stride
            .checked_mul(elem_bytes)
            .ok_or_else(|| HipError::GenericError("KV write size overflow".to_string()))?;
        let _read_bytes = new_len
            .checked_mul(stride)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| HipError::GenericError("KV read size overflow".to_string()))?;
        let write_offset = current_len
            .checked_mul(stride)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| HipError::GenericError("KV write offset overflow".to_string()))?;

        let (kv_keys, kv_values) = kv_cache.get(layer_idx)
            .map_err(|e| HipError::GenericError(format!("KV cache get failed: {}", e)))?;

        let graph = plan
            .graph
            .lock()
            .map_err(|_| HipError::GenericError("Layer graph lock poisoned".to_string()))?;

        // PHASE 2: Fixed-shape tensors - NO set_shape() calls!
        // Tensors are pre-allocated with max_seq_len at graph construction.
        // Removing set_shape() eliminates O(tokens) graph rebuilds.
        // Binding uses offset-based views (sub_buffer_view) for correct positioning.
        // Previously:
        //   graph.tensors[plan.kv_read_k_id.0].set_shape(vec![new_len, plan.num_heads, plan.head_dim]);
        //   graph.tensors[plan.kv_read_v_id.0].set_shape(vec![new_len, plan.num_heads, plan.head_dim]);
        //   graph.tensors[plan.scores_id.0].set_shape(vec![1, new_len]);
        //   graph.tensors[plan.softmax_id.0].set_shape(vec![1, new_len]);

        let mut ggml_backend = plan
            .backend
            .lock()
            .map_err(|_| HipError::GenericError("Layer backend lock poisoned".to_string()))?;

        ggml_backend
            .bind(&graph.tensors[plan.input_id.0], hidden_states.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("Bind input failed: {:?}", e)))?;

        ggml_backend
            .bind(&graph.tensors[plan.kv_read_k_id.0], kv_keys.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("Bind KV read K failed: {:?}", e)))?;
        ggml_backend
            .bind(&graph.tensors[plan.kv_read_v_id.0], kv_values.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("Bind KV read V failed: {:?}", e)))?;

        let kv_write_k_view = kv_keys.buffer().sub_buffer_view(write_offset, write_bytes)?;
        let kv_write_v_view = kv_values.buffer().sub_buffer_view(write_offset, write_bytes)?;
        ggml_backend
            .bind(&graph.tensors[plan.kv_write_k_id.0], kv_write_k_view)
            .map_err(|e| HipError::GenericError(format!("Bind KV write K failed: {:?}", e)))?;
        ggml_backend
            .bind(&graph.tensors[plan.kv_write_v_id.0], kv_write_v_view)
            .map_err(|e| HipError::GenericError(format!("Bind KV write V failed: {:?}", e)))?;

        if let Some(rope_cache) = self.rope_cache()? {
            let rope_offset = current_len
                .checked_mul(rope_cache.half_dim)
                .and_then(|v| v.checked_mul(elem_bytes))
                .ok_or_else(|| HipError::GenericError("RoPE offset overflow".to_string()))?;
            let rope_bytes = rope_cache
                .half_dim
                .checked_mul(elem_bytes)
                .ok_or_else(|| HipError::GenericError("RoPE view size overflow".to_string()))?;

            let cos_view = rope_cache
                .cos
                .buffer()
                .sub_buffer_view(rope_offset, rope_bytes)?;
            let sin_view = rope_cache
                .sin
                .buffer()
                .sub_buffer_view(rope_offset, rope_bytes)?;

            ggml_backend
                .bind(&graph.tensors[plan.cos_id.0], cos_view)
                .map_err(|e| HipError::GenericError(format!("Bind RoPE cos failed: {:?}", e)))?;
            ggml_backend
                .bind(&graph.tensors[plan.sin_id.0], sin_view)
                .map_err(|e| HipError::GenericError(format!("Bind RoPE sin failed: {:?}", e)))?;
        }

        execute_graph(&mut *ggml_backend, &graph)
            .map_err(|e| HipError::GenericError(format!("GGML layer exec failed: {:?}", e)))?;

        kv_cache.advance(layer_idx, 1)?;

        let output_buf = ggml_backend
            .buffer(plan.output_id)
            .ok_or_else(|| HipError::GenericError("Missing output buffer".to_string()))?
            .clone();
        let output_shape = TensorShape::from_dims(&[1, plan.hidden_size]);
        DeviceTensor::from_buffer(backend, output_buf, output_shape)
    }

    pub fn embedding_lookup(
        &self,
        backend: &HipBackend,
        input_tokens: &[u32],
        embedding_weights: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        eprintln!(
            ">>> embedding_lookup: Starting with {} tokens",
            input_tokens.len()
        );
        eprintln!(">>> embedding_lookup: Getting seq_len and hidden_size...");
        let seq_len = input_tokens.len();
        let hidden_size = self.config.hidden_size;
        eprintln!(
            ">>> embedding_lookup: seq_len={}, hidden_size={}",
            seq_len, hidden_size
        );

        eprintln!(">>> embedding_lookup: About to call embedding_weights.shape().dims()...");
        let embed_shape = embedding_weights.shape().dims();
        eprintln!(">>> embedding_lookup: Got shape {:?}", embed_shape);

        eprintln!(">>> embedding_lookup: About to call get_or_try_init on embedding_plan...");
        let plan = match self.embedding_plan.get_or_try_init(|| {
            eprintln!(">>> embedding_lookup: Inside get_or_try_init closure, calling build_embedding_plan...");
            let result = self.build_embedding_plan(backend, embedding_weights);
            match &result {
                Ok(_) => eprintln!(">>> embedding_lookup: build_embedding_plan returned Ok"),
                Err(e) => eprintln!(">>> embedding_lookup: build_embedding_plan returned Err: {}", e),
            }
            result
        }) {
            Ok(p) => {
                eprintln!(">>> embedding_lookup: get_or_try_init succeeded");
                p
            }
            Err(e) => {
                eprintln!(">>> embedding_lookup: get_or_try_init failed: {}", e);
                return Err(e);
            }
        };

        if seq_len > plan.max_seq_len {
            return Err(HipError::GenericError(format!(
                "Sequence length {} exceeds embedding capacity {}",
                seq_len, plan.max_seq_len
            )));
        }

        if plan.hidden_size != hidden_size {
            return Err(HipError::GenericError(format!(
                "Embedding hidden size mismatch: expected {}, got {}",
                hidden_size, plan.hidden_size
            )));
        }

        // Validate token IDs
        // After transpose, embedding is always [vocab, hidden], so vocab is embed_shape[0]
        let vocab_size = embed_shape[0];
        for &token_id in input_tokens {
            if token_id as usize >= vocab_size {
                return Err(HipError::GenericError(format!(
                    "Token ID {} exceeds vocabulary size {}",
                    token_id, vocab_size
                )));
            }
        }

        // Create output tensor: [seq_len, hidden_size]
        let output_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        eprintln!(">>> embedding_lookup: Creating output tensor and executing GetRows...");
        eprintln!(
            ">>> embedding_lookup: About to write {} tokens into shared buffer",
            input_tokens.len()
        );
        let mut padded_tokens = vec![0u32; plan.max_seq_len];
        padded_tokens[..seq_len].copy_from_slice(input_tokens);
        eprintln!(">>> embedding_lookup: About to call copy_to_device...");
        backend.copy_to_device(&plan.tokens_buffer, &padded_tokens)?;
        eprintln!(">>> embedding_lookup: copy_to_device complete");

        eprintln!(">>> embedding_lookup: About to lock ggml_backend...");
        let mut ggml_backend = plan.backend.lock().map_err(|_| {
            HipError::GenericError("GetRows backend lock poisoned".to_string())
        })?;
        eprintln!(">>> embedding_lookup: ggml_backend locked, about to execute_graph...");
        execute_graph(&mut *ggml_backend, &plan.graph)
            .map_err(|e| HipError::GenericError(format!("GetRows execute failed: {:?}", e)))?;
        eprintln!(">>> embedding_lookup: execute_graph complete");

        let output_bytes = seq_len
            .checked_mul(hidden_size)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HipError::GenericError("Output view size overflow".to_string()))?;
        let output_view = plan.output_buffer.sub_buffer_view(0, output_bytes)?;
        let hidden_states = DeviceTensor::from_buffer(backend, output_view, output_shape)?;

        eprintln!(">>> embedding_lookup: Complete");
        Ok(hidden_states)
    }

    /// Forward pass through a single transformer layer
    ///
    /// Implements the standard transformer layer pattern:
    /// LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    ///
    /// # Phase 2 Lazy Loading
    ///
    /// This method now loads layer weights on-demand:
    /// - All layer tensors loaded on first access for this layer
    /// - Subsequent accesses use cached tensors from GgufLoader
    /// - Loading is transparent to the caller
    pub fn forward_layer(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        layer_plan: &LayerPlan,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        let layer_start = std::time::Instant::now();
        eprintln!(">>> forward_layer({}): START", layer_idx);

        let input_shape = hidden_states.shape().dims();
        let _seq_len = input_shape[0];
        let _hidden_size = input_shape[1];

        // Load all layer weights on-demand (cached after first access)
        let load_start = std::time::Instant::now();
        // Load attention weights - handle both fused and separate QKV formats
        let qkv_weight = self.get_or_load_tensor(&layer_plan.qkv_weight)?;
        let qkv_bias = layer_plan
            .qkv_bias
            .as_ref()
            .map(|b| self.get_or_load_tensor(b))
            .transpose()?;

        // Load separate Q, K, V weights if model uses them (e.g., Qwen2)
        let q_weight = layer_plan
            .q_weight
            .as_ref()
            .map(|w| self.get_or_load_tensor(w))
            .transpose()?;
        let k_weight = layer_plan
            .k_weight
            .as_ref()
            .map(|w| self.get_or_load_tensor(w))
            .transpose()?;
        let v_weight = layer_plan
            .v_weight
            .as_ref()
            .map(|w| self.get_or_load_tensor(w))
            .transpose()?;
        let q_bias = layer_plan
            .q_bias
            .as_ref()
            .map(|b| self.get_or_load_tensor(b))
            .transpose()?;
        let k_bias = layer_plan
            .k_bias
            .as_ref()
            .map(|b| self.get_or_load_tensor(b))
            .transpose()?;
        let v_bias = layer_plan
            .v_bias
            .as_ref()
            .map(|b| self.get_or_load_tensor(b))
            .transpose()?;

        let o_proj = self.get_or_load_tensor(&layer_plan.o_proj)?;
        let o_proj_bias = layer_plan
            .o_proj_bias
            .as_ref()
            .map(|b| self.get_or_load_tensor(b))
            .transpose()?;
        let mlp_gate_proj = self.get_or_load_tensor(&layer_plan.mlp_gate_proj)?;
        let mlp_up_proj = self.get_or_load_tensor(&layer_plan.mlp_up_proj)?;
        let mlp_down_proj = self.get_or_load_tensor(&layer_plan.mlp_down_proj)?;
        let norm1_weight = self.get_or_load_tensor(&layer_plan.norm1_weight)?;
        let norm1_bias = layer_plan
            .norm1_bias
            .as_ref()
            .map(|b| self.get_or_load_tensor(b))
            .transpose()?;
        let norm2_weight = self.get_or_load_tensor(&layer_plan.norm2_weight)?;
        let norm2_bias = layer_plan
            .norm2_bias
            .as_ref()
            .map(|b| self.get_or_load_tensor(b))
            .transpose()?;
        eprintln!(
            ">>> forward_layer({}): weights loaded in {:?}",
            layer_idx,
            load_start.elapsed()
        );

        // Store input for residual connection
        let residual = hidden_states.clone();

        // Step 1: Pre-attention LayerNorm
        eprintln!(
            ">>> forward_layer({}): Step 1/6 - Pre-attention LayerNorm starting...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let normed_hidden =
            self.layer_norm(backend, hidden_states, &norm1_weight, norm1_bias.as_ref())?;
        eprintln!(
            ">>> forward_layer({}): Step 1/6 - LayerNorm complete ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 2: Self-attention
        eprintln!(
            ">>> forward_layer({}): Step 2/6 - Self-attention starting...",
            layer_idx
        );
        let step_start = std::time::Instant::now();

        // Determine if using separate QKV or fused QKV based on what's available
        let use_separate_qkv = q_weight.is_some() && k_weight.is_some() && v_weight.is_some();

        let attention_output = if use_separate_qkv {
            // Model uses separate Q, K, V projections (e.g., Qwen2)
            // Safe to use expect here because use_separate_qkv check ensures all weights are Some
            let q_w = q_weight.as_ref().expect("q_weight is Some when use_separate_qkv is true");
            let k_w = k_weight.as_ref().expect("k_weight is Some when use_separate_qkv is true");
            let v_w = v_weight.as_ref().expect("v_weight is Some when use_separate_qkv is true");

            self.self_attention_separate(
                backend,
                &normed_hidden,
                q_w,
                k_w,
                v_w,
                q_bias.as_ref(),
                k_bias.as_ref(),
                v_bias.as_ref(),
                &o_proj,
                o_proj_bias.as_ref(),
                kv_cache,
                layer_idx,
            )?
        } else {
            // Model uses fused QKV projection (e.g., LLaMA)
            self.self_attention(
                backend,
                &normed_hidden,
                &qkv_weight,
                qkv_bias.as_ref(),
                &o_proj,
                o_proj_bias.as_ref(),
                kv_cache,
                layer_idx,
            )?
        };
        eprintln!(
            ">>> forward_layer({}): Step 2/6 - Self-attention complete ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 3: Add residual connection
        eprintln!(
            ">>> forward_layer({}): Step 3/6 - Add residual (attention)...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let attention_with_residual = self.add_residual(backend, &attention_output, &residual)?;
        eprintln!(
            ">>> forward_layer({}): Step 3/6 - Residual complete ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Store attention output for second residual
        let attention_residual = attention_with_residual.clone();

        // Step 4: Pre-MLP LayerNorm
        eprintln!(
            ">>> forward_layer({}): Step 4/6 - Pre-MLP LayerNorm starting...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let normed_attention = self.layer_norm(
            backend,
            &attention_with_residual,
            &norm2_weight,
            norm2_bias.as_ref(),
        )?;
        eprintln!(
            ">>> forward_layer({}): Step 4/6 - Pre-MLP LayerNorm complete ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 5: MLP (SwiGLU)
        eprintln!(
            ">>> forward_layer({}): Step 5/6 - MLP SwiGLU starting...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let mlp_output = self.mlp_swiglu(
            backend,
            &normed_attention,
            &mlp_gate_proj,
            &mlp_up_proj,
            &mlp_down_proj,
        )?;
        eprintln!(
            ">>> forward_layer({}): Step 5/6 - MLP SwiGLU complete ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 6: Add residual connection
        eprintln!(
            ">>> forward_layer({}): Step 6/6 - Add residual (MLP)...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let final_output = self.add_residual(backend, &mlp_output, &attention_residual)?;
        eprintln!(
            ">>> forward_layer({}): Step 6/6 - Residual complete ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        eprintln!(
            ">>> forward_layer({}): COMPLETE (total time: {:?})",
            layer_idx,
            layer_start.elapsed()
        );
        Ok(final_output)
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        backend: &HipBackend,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        bias: Option<&DeviceTensor>,
    ) -> HipResult<DeviceTensor> {
        // Use GPU-accelerated layer normalization from HIP backend
        let output_shape = input.shape().clone();
        let mut output = DeviceTensor::empty(backend, output_shape)?;

        backend.layernorm(
            input,
            weight,
            bias,
            &mut output,
            1e-6, // epsilon
        )?;

        Ok(output)
    }

    /// Self-attention computation
    fn self_attention(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        qkv_weight: &DeviceTensor,
        qkv_bias: Option<&DeviceTensor>,
        o_proj: &DeviceTensor,
        o_proj_bias: Option<&DeviceTensor>,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        eprintln!(">>>   self_attention({}): START", layer_idx);
        let attn_start = std::time::Instant::now();

        let input_shape = hidden_states.shape().dims();
        let seq_len = input_shape[0];
        let hidden_size = input_shape[1];
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Step 1: Project to Q, K, V using GPU matrix multiplication
        // QKV projection: [seq_len, hidden_size] x [hidden_size, 3*hidden_size] -> [seq_len, 3*hidden_size]
        eprintln!(
            ">>>   self_attention({}): Step 1/6 - QKV projection matmul...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let qkv_proj = self.matmul(backend, hidden_states, qkv_weight, qkv_bias)?;
        eprintln!(
            ">>>   self_attention({}): Step 1/6 - QKV projection done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 2: Split Q, K, V directly on GPU
        eprintln!(
            ">>>   self_attention({}): Step 2/6 - Extract Q,K,V tensors...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let (mut q_reshaped, mut k_reshaped, v_reshaped) =
            self.extract_qkv_tensors(backend, &qkv_proj, seq_len, num_heads, head_dim)?;
        eprintln!(
            ">>>   self_attention({}): Step 2/6 - Q,K,V extracted ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 3: Apply position encoding to Q and K tensors (FIX-1)
        // This is critical for correct model behavior - RoPE adds positional information
        eprintln!(
            ">>>   self_attention({}): Step 3/6 - Apply RoPE position embeddings...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        if let Some(ref position_handler) = self.position_handler {
            // Generate sequential position IDs: [0, 1, 2, ..., seq_len-1]
            let position_ids: Vec<usize> = (0..seq_len).collect();

            // Apply RoPE position embeddings to Q and K
            // Use GPU method when available, otherwise use CPU fallback
            #[cfg(feature = "rocm")]
            {
                let (q_with_pos, k_with_pos) = position_handler
                    .apply_position_embeddings_device(
                        q_reshaped.clone(),
                        k_reshaped.clone(),
                        &position_ids,
                        num_heads,
                    )
                    .map_err(|e| {
                        HipError::GenericError(format!(
                            "Failed to apply position embeddings: {}",
                            e
                        ))
                    })?;
                q_reshaped = q_with_pos;
                k_reshaped = k_with_pos;
            }

            #[cfg(not(feature = "rocm"))]
            {
                // CPU fallback: download tensors, apply RoPE, upload back
                let q_host = q_reshaped
                    .to_host_vec()
                    .map_err(|e| HipError::GenericError(format!("Failed to download Q: {}", e)))?;
                let k_host = k_reshaped
                    .to_host_vec()
                    .map_err(|e| HipError::GenericError(format!("Failed to download K: {}", e)))?;

                let (q_with_pos, k_with_pos) = position_handler
                    .apply_position_embeddings(q_host, k_host, &position_ids, num_heads)
                    .map_err(|e| {
                        HipError::GenericError(format!(
                            "Failed to apply position embeddings: {}",
                            e
                        ))
                    })?;

                // Upload position-encoded tensors back to GPU
                let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
                q_reshaped = DeviceTensor::from_host_vec(backend, q_with_pos, q_shape)
                    .map_err(|e| HipError::GenericError(format!("Failed to upload Q: {}", e)))?;

                let k_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
                k_reshaped = DeviceTensor::from_host_vec(backend, k_with_pos, k_shape)
                    .map_err(|e| HipError::GenericError(format!("Failed to upload K: {}", e)))?;
            }
        }
        eprintln!(
            ">>>   self_attention({}): Step 3/6 - RoPE done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 4: Scaled dot-product attention (still CPU fallback for now)
        // TODO: Replace with GPU attention kernel
        eprintln!(
            ">>>   self_attention({}): Step 4/6 - Scaled dot-product attention...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let attention_output = self.scaled_dot_product_attention(
            backend,
            &q_reshaped,
            &k_reshaped,
            &v_reshaped,
            kv_cache,
            layer_idx,
        )?;
        eprintln!(
            ">>>   self_attention({}): Step 4/6 - Attention done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 5: Reshape back: [seq_len, hidden_size]
        eprintln!(
            ">>>   self_attention({}): Step 5/6 - Flatten attention output...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let output_reshaped = self.flatten_attention_output(
            backend,
            &attention_output,
            seq_len,
            num_heads,
            head_dim,
        )?;
        eprintln!(
            ">>>   self_attention({}): Step 5/6 - Flatten done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 6: Output projection using GPU matrix multiplication
        eprintln!(
            ">>>   self_attention({}): Step 6/6 - Output projection matmul...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let final_output = self.matmul(backend, &output_reshaped, o_proj, o_proj_bias)?;
        eprintln!(
            ">>>   self_attention({}): Step 6/6 - Output projection done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        eprintln!(
            ">>>   self_attention({}): COMPLETE (total: {:?})",
            layer_idx,
            attn_start.elapsed()
        );
        Ok(final_output)
    }

    /// Self-attention computation with separate Q, K, V projections
    ///
    /// Used by models like Qwen2 that store attention weights separately
    /// rather than as a fused QKV matrix. This is common with GQA (Grouped Query Attention)
    /// where K and V have fewer dimensions than Q.
    fn self_attention_separate(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        q_weight: &DeviceTensor,
        k_weight: &DeviceTensor,
        v_weight: &DeviceTensor,
        q_bias: Option<&DeviceTensor>,
        k_bias: Option<&DeviceTensor>,
        v_bias: Option<&DeviceTensor>,
        o_proj: &DeviceTensor,
        o_proj_bias: Option<&DeviceTensor>,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        eprintln!(">>>   self_attention_separate({}): START", layer_idx);
        let attn_start = std::time::Instant::now();

        let input_shape = hidden_states.shape().dims();
        let seq_len = input_shape[0];
        let hidden_size = input_shape[1];
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Step 1: Separate Q, K, V projections using GPU matrix multiplication
        // Q: [seq_len, hidden_size] x [hidden_size, hidden_size] -> [seq_len, hidden_size]
        // K: [seq_len, hidden_size] x [hidden_size, kv_dim] -> [seq_len, kv_dim]
        // V: [seq_len, hidden_size] x [hidden_size, kv_dim] -> [seq_len, kv_dim]
        eprintln!(
            ">>>   self_attention_separate({}): Step 1/7 - Q,K,V projection matmuls...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let q_proj = self.matmul(backend, hidden_states, q_weight, q_bias)?;
        let k_proj = self.matmul(backend, hidden_states, k_weight, k_bias)?;
        let v_proj = self.matmul(backend, hidden_states, v_weight, v_bias)?;
        eprintln!(
            ">>>   self_attention_separate({}): Step 1/7 - Projections done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 2: Reshape Q, K, V for multi-head attention
        // Q: [seq_len, hidden_size] -> [seq_len, num_heads, head_dim]
        // K, V: [seq_len, kv_dim] -> [seq_len, num_kv_heads, head_dim]
        eprintln!(
            ">>>   self_attention_separate({}): Step 2/7 - Reshape Q,K,V for attention...",
            layer_idx
        );
        let _step_start = std::time::Instant::now();

        let q_reshaped =
            self.reshape_for_attention(backend, &q_proj, seq_len, num_heads, head_dim)?;

        // For K and V, we need to determine kv_dim from their shape
        let k_shape = k_proj.shape().dims();
        let kv_dim = k_shape[1];
        // GQA: num_kv_heads might be less than num_heads
        let num_kv_heads = kv_dim / head_dim;

        let k_reshaped =
            self.reshape_for_attention(backend, &k_proj, seq_len, num_kv_heads, head_dim)?;
        let v_reshaped =
            self.reshape_for_attention(backend, &v_proj, seq_len, num_kv_heads, head_dim)?;

        eprintln!(">>>   self_attention_separate({}): Step 2/7 - Reshape done (num_heads={}, num_kv_heads={})",
                 layer_idx, num_heads, num_kv_heads);

        // Step 3: Apply position encoding to Q and K tensors (RoPE)
        // NOTE: Using CPU path for now due to GPU RoPE backend synchronization issue with GQA
        eprintln!(
            ">>>   self_attention_separate({}): Step 3/7 - Apply RoPE position embeddings...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let (q_reshaped, k_reshaped) = if let Some(ref position_handler) = self.position_handler {
            let position_ids: Vec<usize> = (0..seq_len).collect();

            // For GQA: apply RoPE separately to Q and K with different head counts
            let rope = position_handler
                .rope()
                .ok_or_else(|| HipError::GenericError("RoPE not configured".to_string()))?;

            // Download Q and K to host, apply RoPE separately, upload back
            let mut q_host = q_reshaped
                .to_host_vec()
                .map_err(|e| HipError::GenericError(format!("Failed to download Q: {}", e)))?;
            let mut k_host = k_reshaped
                .to_host_vec()
                .map_err(|e| HipError::GenericError(format!("Failed to download K: {}", e)))?;

            // Apply RoPE to Q with num_heads (in-place)
            rope.apply_q(&mut q_host, &position_ids, num_heads)
                .map_err(|e| HipError::GenericError(format!("Failed to apply RoPE to Q: {}", e)))?;

            // Apply RoPE to K with num_kv_heads (in-place, different for GQA!)
            rope.apply_k(&mut k_host, &position_ids, num_kv_heads)
                .map_err(|e| HipError::GenericError(format!("Failed to apply RoPE to K: {}", e)))?;

            let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
            let q_with_pos = DeviceTensor::from_host_vec(backend, q_host, q_shape)
                .map_err(|e| HipError::GenericError(format!("Failed to upload Q: {}", e)))?;

            let k_shape = TensorShape::from_dims(&[seq_len, num_kv_heads, head_dim]);
            let k_with_pos = DeviceTensor::from_host_vec(backend, k_host, k_shape)
                .map_err(|e| HipError::GenericError(format!("Failed to upload K: {}", e)))?;
            (q_with_pos, k_with_pos)
        } else {
            (q_reshaped, k_reshaped)
        };
        eprintln!(
            ">>>   self_attention_separate({}): Step 3/7 - RoPE done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 4: Scaled dot-product attention
        eprintln!(
            ">>>   self_attention_separate({}): Step 4/7 - Scaled dot-product attention...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let attention_output = self.scaled_dot_product_attention(
            backend,
            &q_reshaped,
            &k_reshaped,
            &v_reshaped,
            kv_cache,
            layer_idx,
        )?;
        eprintln!(
            ">>>   self_attention_separate({}): Step 4/7 - Attention done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 5: Reshape back: [seq_len, hidden_size]
        eprintln!(
            ">>>   self_attention_separate({}): Step 5/7 - Flatten attention output...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let output_reshaped = self.flatten_attention_output(
            backend,
            &attention_output,
            seq_len,
            num_heads,
            head_dim,
        )?;
        eprintln!(
            ">>>   self_attention_separate({}): Step 5/7 - Flatten done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        // Step 6: Output projection using GPU matrix multiplication
        eprintln!(
            ">>>   self_attention_separate({}): Step 6/7 - Output projection matmul...",
            layer_idx
        );
        let step_start = std::time::Instant::now();
        let final_output = self.matmul(backend, &output_reshaped, o_proj, o_proj_bias)?;
        eprintln!(
            ">>>   self_attention_separate({}): Step 6/7 - Output projection done ({:?})",
            layer_idx,
            step_start.elapsed()
        );

        eprintln!(
            ">>>   self_attention_separate({}): COMPLETE (total: {:?})",
            layer_idx,
            attn_start.elapsed()
        );
        Ok(final_output)
    }

    /// MLP with SwiGLU activation
    fn mlp_swiglu(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        gate_weight: &DeviceTensor,
        up_weight: &DeviceTensor,
        down_weight: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        // Use existing HIP backend MLP implementation
        let input_shape = hidden_states.shape().dims();
        let output_shape = TensorShape::from_dims(input_shape);
        let mut output = DeviceTensor::empty(backend, output_shape)?;

        backend.mlp_swiglu(
            hidden_states,
            gate_weight,
            up_weight,
            down_weight,
            &mut output,
        )?;

        Ok(output)
    }

    /// Matrix multiplication with optional bias
    fn matmul(
        &self,
        backend: &HipBackend,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        bias: Option<&DeviceTensor>,
    ) -> HipResult<DeviceTensor> {
        eprintln!(">>>       matmul: ENTRY - getting shapes...");
        use crate::backend::hip_blas::HipBlasHandle;
        use crate::tensor::matmul::matmul_f32;

        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();
        let _batch_size = input_shape[0];
        let _input_dim = input_shape[1];
        let output_dim = weight_shape[1];

        eprintln!(
            ">>>       matmul: input_shape={:?}, weight_shape={:?}, expecting output_dim={}",
            input_shape, weight_shape, output_dim
        );

        let batch_size = input_shape[0];
        let input_dim = input_shape[1];
        let output_dim = weight_shape[1];

        // Validate shapes
        if weight_shape[0] != input_dim {
            return Err(HipError::GenericError(format!(
                "Weight shape {:?} incompatible with input shape {:?}",
                weight_shape, input_shape
            )));
        }

        eprintln!(">>>       matmul: input_shape={:?}, weight_shape={:?}, expecting output_dim=3*hidden={}",
                 input.shape().dims(), weight.shape().dims(), 3 * input_dim);
        let matmul_start = std::time::Instant::now();

        // Create hipBLAS handle for matrix operations
        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;
        eprintln!(">>>       matmul: hipBLAS handle created");

        // CRITICAL: Associate hipBLAS handle with our HIP stream
        // Without this, hipBLAS uses the default stream while our kernels use a custom stream,
        // causing synchronization issues and hangs.
        blas_handle
            .set_stream(backend.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;
        eprintln!(">>>       matmul: hipBLAS stream set");

        // Perform matrix multiplication: input @ weight -> output
        // input: [batch_size, input_dim], weight: [input_dim, output_dim] -> output: [batch_size, output_dim]
        eprintln!(">>>       matmul: calling matmul_f32...",);
        let matmul_call_start = std::time::Instant::now();
        let output_buffer = matmul_f32(
            backend,
            &blas_handle,
            input.buffer(),
            weight.buffer(),
            batch_size as i32,
            output_dim as i32,
            input_dim as i32,
        )
        .map_err(|e| HipError::GenericError(format!("Matrix multiplication failed: {}", e)))?;
        eprintln!(
            ">>>       matmul: matmul_f32 done in {:?}",
            matmul_call_start.elapsed()
        );

        // CRITICAL: Synchronize after matmul_f32 to ensure GPU completes the operation
        // before we try to copy the data. Without this, the memcpy blocks indefinitely.
        eprintln!(">>>       matmul: synchronizing GPU...",);
        let sync_start = std::time::Instant::now();
        backend
            .synchronize()
            .map_err(|e| HipError::GenericError(format!("GPU sync failed: {}", e)))?;
        eprintln!(
            ">>>       matmul: GPU synchronized in {:?}",
            sync_start.elapsed()
        );

        let output_shape = TensorShape::from_dims(&[batch_size, output_dim]);
        let mut output_tensor = DeviceTensor::empty(backend, output_shape)?;
        eprintln!(">>>       matmul: copying to output tensor...",);
        output_tensor.copy_from_device_buffer(&output_buffer)?;
        // Synchronize again after copy to ensure data is actually transferred
        backend
            .synchronize()
            .map_err(|e| HipError::GenericError(format!("Post-copy sync failed: {}", e)))?;
        eprintln!(">>>       matmul: copy done",);

        if let Some(bias_tensor) = bias {
            eprintln!(">>>       matmul: adding bias...",);
            backend.add_row_bias(&mut output_tensor, bias_tensor)?;
            eprintln!(">>>       matmul: bias added",);
        }

        eprintln!(">>>       matmul: COMPLETE in {:?}", matmul_start.elapsed());
        Ok(output_tensor)
    }

    /// Reshape a projected tensor for multi-head attention
    ///
    /// Takes a 2D tensor [seq_len, dim] and reshapes it to 3D [seq_len, num_heads, head_dim]
    /// where dim = num_heads * head_dim. This is a simple reshape that changes the stride
    /// interpretation - no data movement is required.
    fn reshape_for_attention(
        &self,
        backend: &HipBackend,
        proj: &DeviceTensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<DeviceTensor> {
        let expected_dim = num_heads * head_dim;
        let proj_shape = proj.shape().dims();

        eprintln!(">>>         reshape_for_attention: proj shape={:?}, expected=[{}, {}], target=[{}, {}, {}]",
                 proj_shape, seq_len, expected_dim, seq_len, num_heads, head_dim);

        if proj_shape[0] != seq_len || proj_shape[1] != expected_dim {
            return Err(HipError::GenericError(format!(
                "reshape_for_attention: proj shape {:?} doesn't match expected [{}, {}] where {} = {} * {}",
                proj_shape, seq_len, expected_dim, expected_dim, num_heads, head_dim
            )));
        }

        // For tensors that are already contiguous with the right total size,
        // we can create a new DeviceTensor with a different shape interpretation.
        // This is a view/reshape operation that doesn't copy data.

        // Read the data and create a new tensor with the 3D shape
        let proj_data = proj.to_host_vec().map_err(|e| {
            HipError::GenericError(format!(
                "reshape_for_attention: failed to download tensor: {}",
                e
            ))
        })?;

        let new_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        DeviceTensor::from_host_vec(backend, proj_data, new_shape).map_err(|e| {
            HipError::GenericError(format!(
                "reshape_for_attention: failed to upload tensor: {}",
                e
            ))
        })
    }

    fn extract_qkv_tensors(
        &self,
        backend: &HipBackend,
        qkv_proj: &DeviceTensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor)> {
        let hidden_size = num_heads * head_dim;
        let chunk_elements = seq_len * hidden_size;

        fn copy_chunk(
            backend: &HipBackend,
            src: &DeviceTensor,
            offset_elements: usize,
            seq_len: usize,
            num_heads: usize,
            head_dim: usize,
            name: &str,
        ) -> HipResult<DeviceTensor> {
            eprintln!(
                ">>>         copy_chunk({}): creating tensor [{},{},{}], offset={}",
                name, seq_len, num_heads, head_dim, offset_elements
            );
            let step_start = std::time::Instant::now();
            let shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
            eprintln!(
                ">>>         copy_chunk({}): calling DeviceTensor::empty...",
                name
            );
            let mut tensor = DeviceTensor::empty(backend, shape)?;
            eprintln!(
                ">>>         copy_chunk({}): empty done, calling copy_from_device_slice...",
                name
            );
            tensor.copy_from_device_slice(src, offset_elements)?;
            eprintln!(
                ">>>         copy_chunk({}): COMPLETE ({:?})",
                name,
                step_start.elapsed()
            );
            Ok(tensor)
        }

        eprintln!(">>>       extract_qkv_tensors: Starting Q extraction...");
        let q = copy_chunk(backend, qkv_proj, 0, seq_len, num_heads, head_dim, "Q")?;
        eprintln!(">>>       extract_qkv_tensors: Q extracted, starting K extraction...");
        let k = copy_chunk(
            backend,
            qkv_proj,
            chunk_elements,
            seq_len,
            num_heads,
            head_dim,
            "K",
        )?;
        eprintln!(">>>       extract_qkv_tensors: K extracted, starting V extraction...");
        let v = copy_chunk(
            backend,
            qkv_proj,
            chunk_elements * 2,
            seq_len,
            num_heads,
            head_dim,
            "V",
        )?;
        eprintln!(">>>       extract_qkv_tensors: All Q,K,V extracted successfully");

        Ok((q, k, v))
    }

    fn flatten_attention_output(
        &self,
        backend: &HipBackend,
        attention_output: &DeviceTensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<DeviceTensor> {
        let hidden_size = num_heads * head_dim;
        let shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let mut tensor = DeviceTensor::empty(backend, shape)?;
        tensor.copy_from_device_slice(attention_output, 0)?;
        Ok(tensor)
    }

    /// Scaled dot-product attention with detailed tracing
    fn scaled_dot_product_attention(
        &self,
        backend: &HipBackend,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        _kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        eprintln!(">>>     scaled_dot_product_attention({}): START", layer_idx);

        // Validate input shapes
        let q_shape = q.shape().dims();
        let k_shape = k.shape().dims();
        let v_shape = v.shape().dims();

        tracing::debug!("scaled_dot_product_attention: Q shape={:?}, K shape={:?}, V shape={:?}",
                       q_shape, k_shape, v_shape);

        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(HipError::GenericError(
                "Q, K, V must be 3D tensors [seq_len, num_heads, head_dim]".to_string(),
            ));
        }

        let seq_len = q_shape[0];
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];

        // For GQA: K and V may have fewer heads than Q (num_kv_heads <= num_heads)
        // Validate seq_len and head_dim match, but allow different num_heads
        let num_kv_heads = k_shape[1];

        if k_shape[0] != seq_len || k_shape[2] != head_dim {
            return Err(HipError::GenericError(format!(
                "K tensor shape {:?} incompatible with Q shape {:?}",
                k_shape, q_shape
            )));
        }

        if v_shape[0] != seq_len || v_shape[2] != head_dim || v_shape[1] != num_kv_heads {
            return Err(HipError::GenericError(format!(
                "V tensor shape {:?} incompatible with K shape {:?}",
                v_shape, k_shape
            )));
        }

        tracing::debug!("scaled_dot_product_attention: shape validation passed (GQA: {} KV heads for {} Q heads)",
                       num_kv_heads, num_heads);

        // Use CPU fallback for attention computation
        tracing::debug!("scaled_dot_product_attention: using CPU fallback path");
        self.compute_attention_cpu_fallback(
            backend,
            q,
            k,
            v,
            seq_len,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }

    /// Compute attention with CPU fallback (for when GPU path fails or GQA CPU path is needed)
    fn compute_attention_cpu_fallback(
        &self,
        backend: &HipBackend,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        kv_seq_len: usize,
        q_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<DeviceTensor> {
        // For GQA with different head counts, use CPU-side attention computation
        // This is simpler than expanding K/V on GPU for the CPU fallback path
        tracing::debug!("compute_attention_cpu_fallback: CPU path with num_heads={}, num_kv_heads={}",
                       num_heads, num_kv_heads);

        let mut output_host = vec![0.0f32; q_seq_len * num_heads * head_dim];

        // Download Q, K, V to host
        let q_host = q.to_host_vec()?;
        let k_host = k.to_host_vec()?;
        let v_host = v.to_host_vec()?;

        // Compute attention for each head
        for head in 0..num_heads {
            let kv_head = head / (num_heads / num_kv_heads); // Map Q head to KV head for GQA
            let _head_offset = head * q_seq_len * head_dim;
            let _kv_head_offset = kv_head * kv_seq_len * head_dim;

            for q_pos in 0..q_seq_len {
                let mut attention_weights = vec![0.0f32; kv_seq_len];

                // Compute attention scores
                let mut max_score = f32::NEG_INFINITY;
                for kv_pos in 0..kv_seq_len {
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let q_idx = q_pos * num_heads * head_dim + head * head_dim + dim;
                        let k_idx = kv_pos * num_kv_heads * head_dim + kv_head * head_dim + dim;
                        score += q_host[q_idx] * k_host[k_idx];
                    }
                    score /= (head_dim as f32).sqrt();
                    attention_weights[kv_pos] = score;
                    max_score = max_score.max(score);
                }

                // Softmax with causal mask
                let mut sum = 0.0f32;
                for kv_pos in 0..kv_seq_len {
                    if kv_pos > q_pos {
                        attention_weights[kv_pos] = f32::NEG_INFINITY; // Causal mask
                    } else {
                        attention_weights[kv_pos] = (attention_weights[kv_pos] - max_score).exp();
                        sum += attention_weights[kv_pos];
                    }
                }

                // Normalize
                for kv_pos in 0..kv_seq_len {
                    if attention_weights[kv_pos].is_finite() {
                        attention_weights[kv_pos] /= sum;
                    }
                }

                // Compute weighted sum of V
                for dim in 0..head_dim {
                    let mut value = 0.0f32;
                    for kv_pos in 0..kv_seq_len {
                        if attention_weights[kv_pos].is_finite() {
                            let v_idx = kv_pos * num_kv_heads * head_dim + kv_head * head_dim + dim;
                            value += attention_weights[kv_pos] * v_host[v_idx];
                        }
                    }
                    let out_idx = q_pos * num_heads * head_dim + head * head_dim + dim;
                    output_host[out_idx] = value;
                }
            }
        }

        // Upload result to GPU
        let output_shape = TensorShape::from_dims(&[q_seq_len, num_heads, head_dim]);
        DeviceTensor::from_host_vec(backend, output_host, output_shape)
    }

    /// Add residual connection using optimized GPU operations
    fn add_residual(
        &self,
        backend: &HipBackend,
        input: &DeviceTensor,
        residual: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        // Validate shapes match
        if input.shape().dims() != residual.shape().dims() {
            return Err(HipError::GenericError(
                "Input and residual tensors must have the same shape".to_string(),
            ));
        }

        let output_shape = input.shape().clone();
        let mut output = DeviceTensor::empty(backend, output_shape)?;

        output.buffer().copy_from_buffer(residual.buffer())?;
        backend.add_inplace(input, &mut output)?;

        Ok(output)
    }

    /// Map embedding weights from GGUF tensors
    ///
    /// Extracts token embedding weights from GGUF and validates shape.
    /// Supports multiple naming conventions: token_embd, embed_tokens, word_embeddings.
    /// llama.cpp-compatible: accepts both layouts, infers vocab_size when 0.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_embedding(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
    ) -> HipResult<DeviceTensor> {
        let embedding_names = [
            "token_embd.weight",
            "embed_tokens.weight",
            "word_embeddings.weight",
        ];

        for name in &embedding_names {
            if let Some(tensor) = gpu_tensors.get(*name) {
                let shape = tensor.shape().dims();

                if shape.len() != 2 {
                    return Err(HipError::GenericError(format!(
                        "Embedding tensor '{}' should be 2D, got {}D",
                        name,
                        shape.len()
                    )));
                }

                let (d0, d1) = (shape[0], shape[1]);
                let hidden_size = config.hidden_size;

                // Infer vocab_size if unknown (vocab_size == 0)
                let actual_vocab_size = if config.vocab_size == 0 {
                    // Use hidden_size as anchor to disambiguate
                    if d0 == hidden_size && d1 != hidden_size {
                        d1 // [hidden, vocab] layout
                    } else if d1 == hidden_size && d0 != hidden_size {
                        d0 // [vocab, hidden] layout
                    } else {
                        // Fallback: larger dimension is vocab
                        d0.max(d1)
                    }
                } else {
                    config.vocab_size
                };

                // Accept: [vocab, hidden] OR [hidden, vocab] (transpose if needed)
                if d0 == actual_vocab_size && d1 == hidden_size {
                    tracing::info!(
                        "Found embedding '{}' with shape [{}, {}], vocab_size={}",
                        name,
                        d0,
                        d1,
                        actual_vocab_size
                    );
                    return Ok(tensor.clone());
                }

                if d0 == hidden_size && d1 == actual_vocab_size {
                    tracing::info!(
                        "Found transposed embedding '{}' with shape [{}, {}], vocab_size={}",
                        name,
                        d0,
                        d1,
                        actual_vocab_size
                    );
                    let transposed = Self::transpose_2d_tensor(backend, tensor)?;
                    return Ok(transposed);
                }
            }
        }

        Err(HipError::GenericError(format!(
            "No embedding tensor found (tried: {}). vocab_size={}, hidden_size={}",
            embedding_names.join(", "),
            config.vocab_size,
            config.hidden_size
        )))
    }

    /// Map language model head weights from GGUF tensors
    ///
    /// Extracts LM head weights from GGUF and validates shape.
    /// Supports multiple naming conventions: output.weight, lm_head.weight, logits.weight.
    /// llama.cpp-compatible: accepts both layouts, infers vocab_size when 0, supports tied embeddings.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_lm_head(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
    ) -> HipResult<DeviceTensor> {
        let lm_head_names = ["output.weight", "lm_head.weight", "logits.weight"];

        for name in &lm_head_names {
            if let Some(tensor) = gpu_tensors.get(*name) {
                let shape = tensor.shape().dims();

                if shape.len() != 2 {
                    continue;
                }

                let (d0, d1) = (shape[0], shape[1]);
                let hidden_size = config.hidden_size;

                // Infer vocab_size if unknown (vocab_size == 0)
                let actual_vocab_size = if config.vocab_size == 0 {
                    if d0 == hidden_size && d1 != hidden_size {
                        d1
                    } else if d1 == hidden_size && d0 != hidden_size {
                        d0
                    } else {
                        d0.max(d1)
                    }
                } else {
                    config.vocab_size
                };

                // Accept: [vocab, hidden] OR [hidden, vocab]
                if d0 == actual_vocab_size && d1 == hidden_size {
                    tracing::info!(
                        "Found LM head '{}' with shape [{}, {}], vocab_size={}",
                        name,
                        d0,
                        d1,
                        actual_vocab_size
                    );
                    // Transpose to [hidden, vocab] format
                    let transposed = Self::transpose_2d_tensor(backend, tensor)?;
                    return Ok(transposed);
                }

                if d0 == hidden_size && d1 == actual_vocab_size {
                    tracing::info!(
                        "Found LM head '{}' with shape [{}, {}], vocab_size={}",
                        name,
                        d0,
                        d1,
                        actual_vocab_size
                    );
                    return Ok(tensor.clone());
                }
            }
        }

        // For tied embeddings (Qwen2 style), try embedding tensors
        let tied_names = ["token_embd.weight", "embed_tokens.weight"];
        for name in &tied_names {
            if let Some(tensor) = gpu_tensors.get(*name) {
                let shape = tensor.shape().dims();

                if shape.len() != 2 {
                    continue;
                }

                let (d0, d1) = (shape[0], shape[1]);
                let hidden_size = config.hidden_size;

                // Infer vocab_size if unknown
                let actual_vocab_size = if config.vocab_size == 0 {
                    if d0 == hidden_size && d1 != hidden_size {
                        d1
                    } else if d1 == hidden_size && d0 != hidden_size {
                        d0
                    } else {
                        d0.max(d1)
                    }
                } else {
                    config.vocab_size
                };

                // Accept: [vocab, hidden] OR [hidden, vocab]
                if d0 == actual_vocab_size && d1 == hidden_size {
                    tracing::info!(
                        "Using tied embedding '{}' as LM head, shape [{}, {}], vocab_size={}",
                        name,
                        d0,
                        d1,
                        actual_vocab_size
                    );
                    let transposed = Self::transpose_2d_tensor(backend, tensor)?;
                    return Ok(transposed);
                }

                if d0 == hidden_size && d1 == actual_vocab_size {
                    tracing::info!(
                        "Using tied embedding '{}' as LM head, shape [{}, {}], vocab_size={}",
                        name,
                        d0,
                        d1,
                        actual_vocab_size
                    );
                    return Ok(tensor.clone());
                }
            }
        }

        Err(HipError::GenericError(format!(
            "No LM head tensor found (tried: {}). vocab_size={}, hidden_size={}",
            lm_head_names.join(", "),
            config.vocab_size,
            config.hidden_size
        )))
    }

    /// Transpose a 2D tensor on the host and upload it back to the device
    fn transpose_2d_tensor(backend: &HipBackend, tensor: &DeviceTensor) -> HipResult<DeviceTensor> {
        let shape = tensor.shape().dims();
        if shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Expected 2D tensor for transpose, got {}D",
                shape.len()
            )));
        }

        let rows = shape[0];
        let cols = shape[1];
        tracing::debug!(
            "transpose_2d_tensor: shape=[{}, {}], size={} bytes",
            rows,
            cols,
            tensor.len() * std::mem::size_of::<f32>()
        );
        let host = tensor.to_host_vec()?;
        let mut transposed = vec![0.0f32; host.len()];

        for r in 0..rows {
            for c in 0..cols {
                let src_idx = r * cols + c;
                let dst_idx = c * rows + r;
                transposed[dst_idx] = host[src_idx];
            }
        }

        let new_shape = TensorShape::from_dims(&[cols, rows]);
        DeviceTensor::from_host_vec(backend, transposed, new_shape)
    }

    /// Map attention weights from GGUF tensors for a specific layer
    ///
    /// Extracts QKV projection and output projection weights for transformer attention.
    /// Adapts to the detected model architecture.
    ///
    /// **Architecture-Agnostic QKV Handling:**
    /// - If the model has separate Q, K, V tensors, concatenates them into a fused QKV matrix
    /// - If the model has a fused QKV tensor, uses it directly
    ///
    /// **Supported Architectures:**
    /// - **Qwen2**: Uses `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight`
    /// - **LLaMA**: Uses `transformer.layers.N.attention.wq.weight` (fused or separate)
    /// - **Mistral**: Uses `model.layers.N.self_attn.q_proj.weight` (fused or separate)
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_attention_weights(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
        architecture: &Architecture,
    ) -> HipResult<(DeviceTensor, DeviceTensor)> {
        let prefix = architecture.layer_prefix(layer_idx);

        // Try to find separate Q, K, V tensors first
        let q_name = format!("{}.attn_q.weight", prefix);
        let k_name = format!("{}.attn_k.weight", prefix);
        let v_name = format!("{}.attn_v.weight", prefix);

        let q_weight = gpu_tensors.get(&q_name);
        let k_weight = gpu_tensors.get(&k_name);
        let v_weight = gpu_tensors.get(&v_name);

        // If separate Q, K, V tensors exist, concatenate them
        if let (Some(q), Some(k), Some(v)) = (q_weight, k_weight, v_weight) {
            println!(
                "Layer {}: Found separate Q, K, V tensors - concatenating",
                layer_idx
            );
            let qkv_weight = Self::concatenate_qkv_tensors(backend, q, k, v, config)?;

            // Get output projection
            let o_name = format!("{}.attn_output.weight", prefix);
            let o_name_alt = format!("{}.attn.o_proj.weight", prefix);
            let o_weight = gpu_tensors
                .get(&o_name)
                .or_else(|| gpu_tensors.get(&o_name_alt))
                .ok_or_else(|| {
                    HipError::GenericError(format!(
                        "Output projection not found (tried: {}, {})",
                        o_name, o_name_alt
                    ))
                })?;

            return Ok((qkv_weight, o_weight.clone()));
        }

        // Try fused QKV tensor
        let qkv_variants = vec![
            format!("{}.attention.wq.weight", prefix), // LLaMA-style
            format!("{}.attention.query_key_value.weight", prefix), // Falcon-style
            format!("{}.self_attn.q_proj.weight", prefix), // Mistral-style
            format!("{}.attn.qkv.weight", prefix),     // Generic
        ];

        let qkv_weight = qkv_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No QKV projection found for layer {} (tried separate Q/K/V and: {})",
                    layer_idx,
                    qkv_variants.join(", ")
                ))
            })?;

        // Validate QKV weight shape: [hidden_size, 3 * hidden_size] or [3 * hidden_size, hidden_size]
        let qkv_shape = qkv_weight.shape().dims();
        if qkv_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "QKV projection weights should be 2D, got {}D",
                qkv_shape.len()
            )));
        }

        let expected_qkv_size = config.hidden_size * 3;
        if !((qkv_shape[0] == config.hidden_size && qkv_shape[1] == expected_qkv_size)
            || (qkv_shape[0] == expected_qkv_size && qkv_shape[1] == config.hidden_size))
        {
            return Err(HipError::GenericError(format!(
                "QKV projection weights shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
                qkv_shape[0],
                qkv_shape[1],
                config.hidden_size,
                expected_qkv_size,
                expected_qkv_size,
                config.hidden_size
            )));
        }

        // Get output projection
        let o_variants = vec![
            format!("{}.attention.wo.weight", prefix), // LLaMA-style
            format!("{}.self_attn.o_proj.weight", prefix), // Mistral-style
            format!("{}.attn.o_proj.weight", prefix),  // Generic
            format!("{}.attn_output.weight", prefix),  // Qwen2-style
        ];

        let o_weight = o_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No output projection found for layer {} (tried: {})",
                    layer_idx,
                    o_variants.join(", ")
                ))
            })?;

        // Validate output projection weight shape: [hidden_size, hidden_size]
        let o_shape = o_weight.shape().dims();
        if o_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Output projection weights should be 2D, got {}D",
                o_shape.len()
            )));
        }

        if o_shape[0] != config.hidden_size || o_shape[1] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Output projection weights shape [{}, {}] doesn't match expected [{}, {}]",
                o_shape[0], o_shape[1], config.hidden_size, config.hidden_size
            )));
        }

        println!(
            "Layer {}: Found fused QKV tensor - using directly",
            layer_idx
        );
        Ok((qkv_weight.clone(), o_weight.clone()))
    }

    /// Try to map Qwen2-style attention weights (separate Q, K, V with blk.N. prefix)
    ///
    /// Qwen2 tensor names:
    /// - `blk.N.attn_q.weight` [hidden_size, hidden_size]
    /// - `blk.N.attn_k.weight` [hidden_size, head_dim] or [hidden_size, hidden_size]
    /// - `blk.N.attn_v.weight` [hidden_size, head_dim] or [hidden_size, hidden_size]
    /// - `blk.N.attn_output.weight` [hidden_size, hidden_size]
    ///
    /// Returns `Err` if Qwen2-style tensors are not found.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn try_map_qwen2_attention_weights(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor)> {
        let blk_prefix = format!("blk.{}", layer_idx);

        // Qwen2 tensor names
        let q_name = format!("{}.attn_q.weight", blk_prefix);
        let k_name = format!("{}.attn_k.weight", blk_prefix);
        let v_name = format!("{}.attn_v.weight", blk_prefix);
        let o_name = format!("{}.attn_output.weight", blk_prefix);

        // Try to find all Q, K, V tensors
        let q_weight = gpu_tensors.get(&q_name);
        let k_weight = gpu_tensors.get(&k_name);
        let v_weight = gpu_tensors.get(&v_name);
        let o_weight = gpu_tensors.get(&o_name);

        // If any tensor is missing, this is not a Qwen2 model
        let q_weight = match q_weight {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 Q tensor not found".to_string(),
                ))
            }
        };
        let k_weight = match k_weight {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 K tensor not found".to_string(),
                ))
            }
        };
        let v_weight = match v_weight {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 V tensor not found".to_string(),
                ))
            }
        };
        let o_weight = match o_weight {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 O tensor not found".to_string(),
                ))
            }
        };

        // Validate output projection shape: [hidden_size, hidden_size]
        let o_shape = o_weight.shape().dims();
        if o_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Qwen2 output projection should be 2D, got {}D",
                o_shape.len()
            )));
        }
        if o_shape[0] != config.hidden_size || o_shape[1] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Qwen2 output projection shape [{}, {}] doesn't match expected [{}, {}]",
                o_shape[0], o_shape[1], config.hidden_size, config.hidden_size
            )));
        }

        // Concatenate Q, K, V into fused QKV matrix
        let qkv_weight =
            Self::concatenate_qkv_tensors(backend, q_weight, k_weight, v_weight, config)?;

        Ok((qkv_weight, o_weight.clone()))
    }

    /// Map LLaMA-style attention weights (fused QKV with transformer.layers.N. prefix)
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_llama_attention_weights(
        _backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor)> {
        let layer_prefix = format!("transformer.layers.{}", layer_idx);

        // Try different attention weight naming conventions
        let qkv_names = [
            format!("{}.attention.wq.weight", layer_prefix),
            format!("{}.attention.query_key_value.weight", layer_prefix),
            format!("{}.self_attn.q_proj.weight", layer_prefix),
            format!("{}.attn.q_proj.weight", layer_prefix),
        ];

        let o_names = [
            format!("{}.attention.wo.weight", layer_prefix),
            format!("{}.attention.o_proj.weight", layer_prefix),
            format!("{}.self_attn.o_proj.weight", layer_prefix),
            format!("{}.attn.o_proj.weight", layer_prefix),
        ];

        // Find QKV projection weights
        let qkv_weight = qkv_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No QKV projection weights found for layer {} (tried: {})",
                    layer_idx,
                    qkv_names.join(", ")
                ))
            })?;

        // Find output projection weights
        let o_weight = o_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No output projection weights found for layer {} (tried: {})",
                    layer_idx,
                    o_names.join(", ")
                ))
            })?;

        // Validate QKV weight shape: [hidden_size, 3 * hidden_size] or [3 * hidden_size, hidden_size]
        let qkv_shape = qkv_weight.shape().dims();
        if qkv_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "QKV projection weights should be 2D, got {}D",
                qkv_shape.len()
            )));
        }

        let expected_qkv_size = config.hidden_size * 3;
        if !((qkv_shape[0] == config.hidden_size && qkv_shape[1] == expected_qkv_size)
            || (qkv_shape[0] == expected_qkv_size && qkv_shape[1] == config.hidden_size))
        {
            return Err(HipError::GenericError(format!(
                "QKV projection weights shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
                qkv_shape[0],
                qkv_shape[1],
                config.hidden_size,
                expected_qkv_size,
                expected_qkv_size,
                config.hidden_size
            )));
        }

        // Validate output projection weight shape: [hidden_size, hidden_size]
        let o_shape = o_weight.shape().dims();
        if o_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Output projection weights should be 2D, got {}D",
                o_shape.len()
            )));
        }

        if o_shape[0] != config.hidden_size || o_shape[1] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Output projection weights shape [{}, {}] doesn't match expected [{}, {}]",
                o_shape[0], o_shape[1], config.hidden_size, config.hidden_size
            )));
        }

        Ok((qkv_weight.clone(), o_weight.clone()))
    }

    /// Concatenate separate Q, K, V tensors into a fused QKV matrix
    ///
    /// **Input shapes:**
    /// - Q: [hidden_size, hidden_size]
    /// - K: [hidden_size, head_dim] or [hidden_size, hidden_size]
    /// - V: [hidden_size, head_dim] or [hidden_size, hidden_size]
    ///
    /// **Output shape:**
    /// - QKV: [hidden_size, 3 * hidden_size]
    ///
    /// **Concatenation strategy:**
    /// If K and V have shape [hidden_size, head_dim], we pad them to [hidden_size, hidden_size]
    /// before concatenation. This ensures the output always has the expected shape.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn concatenate_qkv_tensors(
        backend: &HipBackend,
        q_weight: &DeviceTensor,
        k_weight: &DeviceTensor,
        v_weight: &DeviceTensor,
        config: &ModelConfig,
    ) -> HipResult<DeviceTensor> {
        let q_shape = q_weight.shape().dims();
        let k_shape = k_weight.shape().dims();
        let v_shape = v_weight.shape().dims();

        // Validate Q shape
        if q_shape.len() != 2 || q_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Q weight shape {:?} invalid, expected [hidden_size, hidden_size]",
                q_shape
            )));
        }

        // For Qwen2, K and V may be [hidden_size, head_dim] or [hidden_size, hidden_size]
        let _head_dim = config.head_dim;

        // Check if K and V need padding
        let _k_needs_padding = k_shape[1] != config.hidden_size;
        let _v_needs_padding = v_shape[1] != config.hidden_size;

        // Helper function to pad tensor if needed
        let pad_tensor = |tensor: &DeviceTensor, target_cols: usize| -> HipResult<DeviceTensor> {
            let current_shape = tensor.shape().dims();
            if current_shape[1] == target_cols {
                return Ok(tensor.clone());
            }

            // Create padded tensor: [hidden_size, target_cols]
            let padded_shape = TensorShape::from_dims(&[current_shape[0], target_cols]);
            let _padded = DeviceTensor::empty(backend, padded_shape.clone())?;

            // Copy original data (column-major layout: copy column by column)
            // For now, copy row by row from source
            let host_data = tensor.to_host_vec()?;
            let mut padded_data = vec![0.0f32; current_shape[0] * target_cols];

            for row in 0..current_shape[0] {
                for col in 0..current_shape[1] {
                    let src_idx = row * current_shape[1] + col;
                    let dst_idx = row * target_cols + col;
                    padded_data[dst_idx] = host_data[src_idx];
                }
            }

            DeviceTensor::from_host_vec(backend, padded_data, padded_shape)
        };

        // Pad K and V if necessary
        let k_padded = pad_tensor(k_weight, config.hidden_size)?;
        let v_padded = pad_tensor(v_weight, config.hidden_size)?;

        // Concatenate Q, K, V along dimension 1 (columns)
        // Output shape: [hidden_size, 3 * hidden_size]
        let qkv_shape = TensorShape::from_dims(&[config.hidden_size, 3 * config.hidden_size]);
        let _qkv_weight = DeviceTensor::empty(backend, qkv_shape.clone())?;

        // Copy Q, K, V data into QKV tensor
        let q_host = q_weight.to_host_vec()?;
        let k_host = k_padded.to_host_vec()?;
        let v_host = v_padded.to_host_vec()?;

        let mut qkv_host = vec![0.0f32; config.hidden_size * 3 * config.hidden_size];

        // Copy Q to first third
        for i in 0..q_host.len() {
            qkv_host[i] = q_host[i];
        }

        // Copy K to middle third
        let k_offset = config.hidden_size * config.hidden_size;
        for i in 0..k_host.len() {
            qkv_host[k_offset + i] = k_host[i];
        }

        // Copy V to last third
        let v_offset = 2 * config.hidden_size * config.hidden_size;
        for i in 0..v_host.len() {
            qkv_host[v_offset + i] = v_host[i];
        }

        DeviceTensor::from_host_vec(backend, qkv_host, qkv_shape.clone())
    }

    /// Validate and optionally transpose MLP weight tensor
    ///
    /// Validates that the tensor has shape [dim1, dim2] or [dim2, dim1].
    /// If transposed, returns a transposed copy. Otherwise returns the original.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn validate_and_transpose_mlp_weight(
        backend: &HipBackend,
        weight: &DeviceTensor,
        name: &str,
        expected_dim1: usize,
        expected_dim2: usize,
    ) -> HipResult<DeviceTensor> {
        let shape = weight.shape().dims();

        if shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "{} weight should be 2D, got {}D",
                name,
                shape.len()
            )));
        }

        // Check if shape matches [expected_dim1, expected_dim2]
        if shape[0] == expected_dim1 && shape[1] == expected_dim2 {
            return Ok(weight.clone());
        }

        // Check if shape matches [expected_dim2, expected_dim1] (transposed)
        if shape[0] == expected_dim2 && shape[1] == expected_dim1 {
            // Transpose the tensor
            return Self::transpose_2d_tensor(backend, weight);
        }

        // Shape doesn't match either orientation
        Err(HipError::GenericError(format!(
            "{} weight shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
            name, shape[0], shape[1], expected_dim1, expected_dim2, expected_dim2, expected_dim1
        )))
    }

    /// Map MLP weights from GGUF tensors for a specific layer
    ///
    /// Extracts feed-forward network weights for transformer MLP layers.
    /// Adapts to the detected model architecture.
    ///
    /// **Supported Architectures:**
    /// - **Qwen2**: Uses `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight`, `blk.N.ffn_down.weight`
    /// - **LLaMA**: Uses `transformer.layers.N.mlp.gate_proj.weight`, etc.
    /// - **Mistral**: Uses `model.layers.N.mlp.gate_proj.weight`, etc.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_mlp_weights(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
        architecture: &Architecture,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor)> {
        let prefix = architecture.layer_prefix(layer_idx);

        // Try multiple naming variants for each component
        let gate_variants = vec![
            format!("{}.ffn_gate.weight", prefix),        // Qwen2-style
            format!("{}.mlp.gate_proj.weight", prefix),   // LLaMA/Mistral-style
            format!("{}.feed_forward.w1.weight", prefix), // Alternative
            format!("{}.mlp.c_fc.weight", prefix),        // GPT-style
        ];

        let up_variants = vec![
            format!("{}.ffn_up.weight", prefix),          // Qwen2-style
            format!("{}.mlp.up_proj.weight", prefix),     // LLaMA/Mistral-style
            format!("{}.feed_forward.w3.weight", prefix), // Alternative
            format!("{}.mlp.c_proj.weight", prefix),      // GPT-style
        ];

        let down_variants = vec![
            format!("{}.ffn_down.weight", prefix),        // Qwen2-style
            format!("{}.mlp.down_proj.weight", prefix),   // LLaMA/Mistral-style
            format!("{}.feed_forward.w2.weight", prefix), // Alternative
        ];

        let gate_weight = gate_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No gate projection weights found for layer {} (tried: {})",
                    layer_idx,
                    gate_variants.join(", ")
                ))
            })?;

        let up_weight = up_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No up projection weights found for layer {} (tried: {})",
                    layer_idx,
                    up_variants.join(", ")
                ))
            })?;

        let down_weight = down_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No down projection weights found for layer {} (tried: {})",
                    layer_idx,
                    down_variants.join(", ")
                ))
            })?;

        // Validate and potentially transpose tensors to match expected shapes
        let validated_gate = Self::validate_and_transpose_mlp_weight(
            backend,
            gate_weight,
            "gate",
            config.hidden_size,
            config.intermediate_size,
        )?;

        let validated_up = Self::validate_and_transpose_mlp_weight(
            backend,
            up_weight,
            "up",
            config.hidden_size,
            config.intermediate_size,
        )?;

        let validated_down = Self::validate_and_transpose_mlp_weight(
            backend,
            down_weight,
            "down",
            config.intermediate_size,
            config.hidden_size,
        )?;

        Ok((validated_gate, validated_up, validated_down))
    }

    /// Try to map Qwen2-style MLP weights (blk.N.ffn_* prefix)
    ///
    /// Qwen2 tensor names:
    /// - `blk.N.ffn_gate.weight` [intermediate_size, hidden_size] or [hidden_size, intermediate_size]
    /// - `blk.N.ffn_up.weight` [intermediate_size, hidden_size] or [hidden_size, intermediate_size]
    /// - `blk.N.ffn_down.weight` [hidden_size, intermediate_size] or [intermediate_size, hidden_size]
    ///
    /// Returns `Err` if Qwen2-style tensors are not found.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn try_map_qwen2_mlp_weights(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor)> {
        let blk_prefix = format!("blk.{}", layer_idx);

        // Qwen2 tensor names
        let gate_name = format!("{}.ffn_gate.weight", blk_prefix);
        let up_name = format!("{}.ffn_up.weight", blk_prefix);
        let down_name = format!("{}.ffn_down.weight", blk_prefix);

        // Try to find all MLP tensors
        let gate_weight = gpu_tensors.get(&gate_name);
        let up_weight = gpu_tensors.get(&up_name);
        let down_weight = gpu_tensors.get(&down_name);

        // If any tensor is missing, this is not a Qwen2 model
        let gate_weight = match gate_weight {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 FFN gate tensor not found".to_string(),
                ))
            }
        };
        let up_weight = match up_weight {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 FFN up tensor not found".to_string(),
                ))
            }
        };
        let down_weight = match down_weight {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 FFN down tensor not found".to_string(),
                ))
            }
        };

        // Validate and potentially transpose tensors to match expected shapes
        let validated_gate = Self::validate_and_transpose_mlp_weight(
            backend,
            gate_weight,
            "ffn_gate",
            config.hidden_size,
            config.intermediate_size,
        )?;
        let validated_up = Self::validate_and_transpose_mlp_weight(
            backend,
            up_weight,
            "ffn_up",
            config.hidden_size,
            config.intermediate_size,
        )?;
        let validated_down = Self::validate_and_transpose_mlp_weight(
            backend,
            down_weight,
            "ffn_down",
            config.intermediate_size,
            config.hidden_size,
        )?;

        Ok((validated_gate, validated_up, validated_down))
    }

    /// Map LLaMA-style MLP weights (transformer.layers.N.* prefix)
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_llama_mlp_weights(
        _backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor)> {
        let layer_prefix = format!("transformer.layers.{}", layer_idx);

        // Try different MLP weight naming conventions
        let gate_names = [
            format!("{}.feed_forward.w1.weight", layer_prefix),
            format!("{}.mlp.gate_proj.weight", layer_prefix),
            format!("{}.mlp.c_fc.weight", layer_prefix),
            format!("{}.ffn.gate.weight", layer_prefix),
        ];

        let up_names = [
            format!("{}.feed_forward.w3.weight", layer_prefix),
            format!("{}.mlp.up_proj.weight", layer_prefix),
            format!("{}.mlp.c_proj.weight", layer_prefix),
            format!("{}.ffn.up.weight", layer_prefix),
        ];

        let down_names = [
            format!("{}.feed_forward.w2.weight", layer_prefix),
            format!("{}.mlp.down_proj.weight", layer_prefix),
            format!("{}.mlp.c_proj_2.weight", layer_prefix),
            format!("{}.ffn.down.weight", layer_prefix),
        ];

        // Find gate projection weights
        let gate_weight = gate_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No gate projection weights found for layer {} (tried: {})",
                    layer_idx,
                    gate_names.join(", ")
                ))
            })?;

        // Find up projection weights
        let up_weight = up_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No up projection weights found for layer {} (tried: {})",
                    layer_idx,
                    up_names.join(", ")
                ))
            })?;

        // Find down projection weights
        let down_weight = down_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No down projection weights found for layer {} (tried: {})",
                    layer_idx,
                    down_names.join(", ")
                ))
            })?;

        // Validate gate projection weight shape: [hidden_size, intermediate_size] or [intermediate_size, hidden_size]
        let gate_shape = gate_weight.shape().dims();
        if gate_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Gate projection weights should be 2D, got {}D",
                gate_shape.len()
            )));
        }

        if !((gate_shape[0] == config.hidden_size && gate_shape[1] == config.intermediate_size)
            || (gate_shape[0] == config.intermediate_size && gate_shape[1] == config.hidden_size))
        {
            return Err(HipError::GenericError(format!(
                "Gate projection weights shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
                gate_shape[0], gate_shape[1], config.hidden_size, config.intermediate_size,
                config.intermediate_size, config.hidden_size
            )));
        }

        // Validate up projection weight shape: [hidden_size, intermediate_size] or [intermediate_size, hidden_size]
        let up_shape = up_weight.shape().dims();
        if up_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Up projection weights should be 2D, got {}D",
                up_shape.len()
            )));
        }

        if !((up_shape[0] == config.hidden_size && up_shape[1] == config.intermediate_size)
            || (up_shape[0] == config.intermediate_size && up_shape[1] == config.hidden_size))
        {
            return Err(HipError::GenericError(format!(
                "Up projection weights shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
                up_shape[0],
                up_shape[1],
                config.hidden_size,
                config.intermediate_size,
                config.intermediate_size,
                config.hidden_size
            )));
        }

        // Validate down projection weight shape: [intermediate_size, hidden_size] or [hidden_size, intermediate_size]
        let down_shape = down_weight.shape().dims();
        if down_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Down projection weights should be 2D, got {}D",
                down_shape.len()
            )));
        }

        if !((down_shape[0] == config.intermediate_size && down_shape[1] == config.hidden_size)
            || (down_shape[0] == config.hidden_size && down_shape[1] == config.intermediate_size))
        {
            return Err(HipError::GenericError(format!(
                "Down projection weights shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
                down_shape[0], down_shape[1], config.intermediate_size, config.hidden_size,
                config.hidden_size, config.intermediate_size
            )));
        }

        Ok((gate_weight.clone(), up_weight.clone(), down_weight.clone()))
    }

    /// Map layer normalization weights from GGUF tensors for a specific layer
    ///
    /// Extracts attention and post-attention layer normalization weights.
    /// Adapts to the detected model architecture.
    ///
    /// **Supported Architectures:**
    /// - **Qwen2**: Uses `blk.N.attn_norm.weight` and `blk.N.ffn_norm.weight`
    /// - **LLaMA**: Uses `transformer.layers.N.attention_norm.weight` and `transformer.layers.N.ffn_norm.weight`
    /// - **Mistral**: Uses `model.layers.N.input_layernorm.weight` and `model.layers.N.post_attention_layernorm.weight`
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_layer_norm_weights(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
        architecture: &Architecture,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor, DeviceTensor)> {
        let prefix = architecture.layer_prefix(layer_idx);

        // Try multiple naming variants for attention norm (input/first layer norm)
        let attn_norm_variants = vec![
            format!("{}.attn_norm.weight", prefix),       // Qwen2-style
            format!("{}.attention_norm.weight", prefix),  // LLaMA-style
            format!("{}.input_layernorm.weight", prefix), // Mistral-style
            format!("{}.ln_1.weight", prefix),            // GPT-style
            format!("{}.pre_attention_layernorm.weight", prefix), // Alternative
        ];

        // Try multiple naming variants for FFN norm (output/second layer norm)
        let ffn_norm_variants = vec![
            format!("{}.ffn_norm.weight", prefix), // Qwen2-style
            format!("{}.post_attention_layernorm.weight", prefix), // Mistral-style
            format!("{}.ln_2.weight", prefix),     // GPT-style
        ];

        let attn_norm_weight = attn_norm_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No attention layer norm weights found for layer {} (tried: {})",
                    layer_idx,
                    attn_norm_variants.join(", ")
                ))
            })?;

        let ffn_norm_weight = ffn_norm_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No FFN layer norm weights found for layer {} (tried: {})",
                    layer_idx,
                    ffn_norm_variants.join(", ")
                ))
            })?;

        // Validate attention norm weight shape: [hidden_size]
        let attn_norm_shape = attn_norm_weight.shape().dims();
        if attn_norm_shape.len() != 1 || attn_norm_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Attention layer norm weight shape {:?} doesn't match expected [{}]",
                attn_norm_shape, config.hidden_size
            )));
        }

        // Validate FFN norm weight shape: [hidden_size]
        let ffn_norm_shape = ffn_norm_weight.shape().dims();
        if ffn_norm_shape.len() != 1 || ffn_norm_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "FFN layer norm weight shape {:?} doesn't match expected [{}]",
                ffn_norm_shape, config.hidden_size
            )));
        }

        // Try to find bias tensors (optional - create zero bias if not found)
        let create_zero_bias = || -> HipResult<DeviceTensor> {
            let bias_shape = TensorShape::from_dims(&[config.hidden_size]);
            let zeros = vec![0.0f32; config.hidden_size];
            DeviceTensor::from_host_vec(backend, zeros, bias_shape)
        };

        let attn_norm_bias_variants = [
            format!("{}.attn_norm.bias", prefix),
            format!("{}.attention_norm.bias", prefix),
            format!("{}.input_layernorm.bias", prefix),
        ];

        let ffn_norm_bias_variants = [
            format!("{}.ffn_norm.bias", prefix),
            format!("{}.post_attention_layernorm.bias", prefix),
        ];

        // Handle optional bias tensors with fallback to zero bias
        let attn_norm_bias = match attn_norm_bias_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .cloned()
        {
            Some(bias) => bias,
            None => create_zero_bias()
                .map_err(|e| HipError::GenericError(format!("Failed to create zero bias for attention norm: {}", e)))?,
        };

        let ffn_norm_bias = match ffn_norm_bias_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .cloned()
        {
            Some(bias) => bias,
            None => create_zero_bias()
                .map_err(|e| HipError::GenericError(format!("Failed to create zero bias for FFN norm: {}", e)))?,
        };

        Ok((
            attn_norm_weight.clone(),
            attn_norm_bias,
            ffn_norm_weight.clone(),
            ffn_norm_bias,
        ))
    }

    /// Try to map Qwen2-style layer norm weights (blk.N.attn_norm and blk.N.ffn_norm)
    ///
    /// Qwen2 tensor names:
    /// - `blk.N.attn_norm.weight` [hidden_size]
    /// - `blk.N.attn_norm.bias` [hidden_size] (optional)
    /// - `blk.N.ffn_norm.weight` [hidden_size]
    /// - `blk.N.ffn_norm.bias` [hidden_size] (optional)
    ///
    /// Returns `Err` if Qwen2-style tensors are not found.
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn try_map_qwen2_layer_norm_weights(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor, DeviceTensor)> {
        let blk_prefix = format!("blk.{}", layer_idx);

        // Qwen2 tensor names
        let attn_norm_weight_name = format!("{}.attn_norm.weight", blk_prefix);
        let attn_norm_bias_name = format!("{}.attn_norm.bias", blk_prefix);
        let ffn_norm_weight_name = format!("{}.ffn_norm.weight", blk_prefix);
        let ffn_norm_bias_name = format!("{}.ffn_norm.bias", blk_prefix);

        // Try to find layer norm tensors (bias is optional)
        let attn_norm_weight = match gpu_tensors.get(&attn_norm_weight_name) {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 attn_norm.weight not found".to_string(),
                ))
            }
        };
        let attn_norm_bias = gpu_tensors.get(&attn_norm_bias_name);
        let ffn_norm_weight = match gpu_tensors.get(&ffn_norm_weight_name) {
            Some(t) => t,
            None => {
                return Err(HipError::GenericError(
                    "Qwen2 ffn_norm.weight not found".to_string(),
                ))
            }
        };
        let ffn_norm_bias = gpu_tensors.get(&ffn_norm_bias_name);

        // Validate shapes
        let attn_norm_shape = attn_norm_weight.shape().dims();
        if attn_norm_shape.len() != 1 || attn_norm_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Qwen2 attn_norm.weight shape {:?} doesn't match expected [{}]",
                attn_norm_shape, config.hidden_size
            )));
        }

        let ffn_norm_shape = ffn_norm_weight.shape().dims();
        if ffn_norm_shape.len() != 1 || ffn_norm_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Qwen2 ffn_norm.weight shape {:?} doesn't match expected [{}]",
                ffn_norm_shape, config.hidden_size
            )));
        }

        // Create zero bias if not present
        let create_zero_bias = || -> HipResult<DeviceTensor> {
            let bias_shape = TensorShape::from_dims(&[config.hidden_size]);
            let _bias_tensor = DeviceTensor::empty(backend, bias_shape.clone())?;
            // Fill with zeros by uploading a zero-filled host buffer
            let zeros = vec![0.0f32; config.hidden_size];
            DeviceTensor::from_host_vec(backend, zeros, bias_shape)
        };

        let attn_norm_bias = match attn_norm_bias {
            Some(bias) => {
                let bias_shape = bias.shape().dims();
                if bias_shape.len() != 1 || bias_shape[0] != config.hidden_size {
                    return Err(HipError::GenericError(format!(
                        "Qwen2 attn_norm.bias shape {:?} doesn't match expected [{}]",
                        bias_shape, config.hidden_size
                    )));
                }
                bias.clone()
            }
            None => create_zero_bias()?,
        };

        let ffn_norm_bias = match ffn_norm_bias {
            Some(bias) => {
                let bias_shape = bias.shape().dims();
                if bias_shape.len() != 1 || bias_shape[0] != config.hidden_size {
                    return Err(HipError::GenericError(format!(
                        "Qwen2 ffn_norm.bias shape {:?} doesn't match expected [{}]",
                        bias_shape, config.hidden_size
                    )));
                }
                bias.clone()
            }
            None => create_zero_bias()?,
        };

        Ok((
            attn_norm_weight.clone(),
            attn_norm_bias,
            ffn_norm_weight.clone(),
            ffn_norm_bias,
        ))
    }

    /// Map LLaMA-style layer norm weights (transformer.layers.N.*)
    #[allow(dead_code)] // Reserved for alternative weight mapping strategies
    fn map_llama_layer_norm_weights(
        _backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor, DeviceTensor)> {
        let layer_prefix = format!("transformer.layers.{}", layer_idx);

        // Try different layer norm weight naming conventions
        let attention_norm_names = [
            format!("{}.attention_norm.weight", layer_prefix),
            format!("{}.input_layernorm.weight", layer_prefix),
            format!("{}.ln_1.weight", layer_prefix),
            format!("{}.pre_attention_layernorm.weight", layer_prefix),
        ];

        let attention_bias_names = [
            format!("{}.attention_norm.bias", layer_prefix),
            format!("{}.input_layernorm.bias", layer_prefix),
            format!("{}.ln_1.bias", layer_prefix),
            format!("{}.pre_attention_layernorm.bias", layer_prefix),
        ];

        let ffn_norm_names = [
            format!("{}.ffn_norm.weight", layer_prefix),
            format!("{}.post_attention_layernorm.weight", layer_prefix),
            format!("{}.ln_2.weight", layer_prefix),
            format!("{}.post_attention_layernorm.weight", layer_prefix),
        ];

        let ffn_bias_names = [
            format!("{}.ffn_norm.bias", layer_prefix),
            format!("{}.post_attention_layernorm.bias", layer_prefix),
            format!("{}.ln_2.bias", layer_prefix),
            format!("{}.post_attention_layernorm.bias", layer_prefix),
        ];

        // Find attention layer norm weights
        let attention_norm_weight = attention_norm_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No attention layer norm weights found for layer {} (tried: {})",
                    layer_idx,
                    attention_norm_names.join(", ")
                ))
            })?;

        // Find attention layer norm bias
        let attention_norm_bias = attention_bias_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No attention layer norm bias found for layer {} (tried: {})",
                    layer_idx,
                    attention_bias_names.join(", ")
                ))
            })?;

        // Find FFN layer norm weights
        let ffn_norm_weight = ffn_norm_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No FFN layer norm weights found for layer {} (tried: {})",
                    layer_idx,
                    ffn_norm_names.join(", ")
                ))
            })?;

        // Find FFN layer norm bias
        let ffn_norm_bias = ffn_bias_names
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .ok_or_else(|| {
                HipError::GenericError(format!(
                    "No FFN layer norm bias found for layer {} (tried: {})",
                    layer_idx,
                    ffn_bias_names.join(", ")
                ))
            })?;

        // Validate attention layer norm weight shape: [hidden_size]
        let attention_norm_shape = attention_norm_weight.shape().dims();
        if attention_norm_shape.len() != 1 || attention_norm_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Attention layer norm weight shape {:?} doesn't match expected [{}]",
                attention_norm_shape, config.hidden_size
            )));
        }

        // Validate attention layer norm bias shape: [hidden_size]
        let attention_bias_shape = attention_norm_bias.shape().dims();
        if attention_bias_shape.len() != 1 || attention_bias_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "Attention layer norm bias shape {:?} doesn't match expected [{}]",
                attention_bias_shape, config.hidden_size
            )));
        }

        // Validate FFN layer norm weight shape: [hidden_size]
        let ffn_norm_shape = ffn_norm_weight.shape().dims();
        if ffn_norm_shape.len() != 1 || ffn_norm_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "FFN layer norm weight shape {:?} doesn't match expected [{}]",
                ffn_norm_shape, config.hidden_size
            )));
        }

        // Validate FFN layer norm bias shape: [hidden_size]
        let ffn_bias_shape = ffn_norm_bias.shape().dims();
        if ffn_bias_shape.len() != 1 || ffn_bias_shape[0] != config.hidden_size {
            return Err(HipError::GenericError(format!(
                "FFN layer norm bias shape {:?} doesn't match expected [{}]",
                ffn_bias_shape, config.hidden_size
            )));
        }

        Ok((
            attention_norm_weight.clone(),
            attention_norm_bias.clone(),
            ffn_norm_weight.clone(),
            ffn_norm_bias.clone(),
        ))
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
}

// Note: LayerPlan impl is now in layer_plan.rs
// Note: Test includes are now in mod.rs
