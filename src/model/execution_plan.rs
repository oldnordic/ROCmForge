//! Execution Plan for Transformer Layers
//!
//! Static execution plan describing how each transformer layer executes.
//! Minimal design - no dynamic graph, no heavyweight abstractions.

use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::loader::gguf::GgufLoader;
use crate::loader::TensorShape;
use crate::model::{config::ModelConfig, kv_cache::KVCache};
use crate::ops::attention_gpu::HipAttentionKernels;
use std::collections::HashSet;

/// Detected model architecture based on tensor naming patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Qwen2-style: tensors start with `blk.N.`
    Qwen2,
    /// LLaMA-style: tensors start with `transformer.layers.N.`
    LLaMA,
    /// Mistral-style: tensors start with `model.layers.N.`
    Mistral,
}

impl Architecture {
    /// Get the layer prefix pattern for this architecture
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        match self {
            Architecture::Qwen2 => format!("blk.{}", layer_idx),
            Architecture::LLaMA => format!("transformer.layers.{}", layer_idx),
            Architecture::Mistral => format!("model.layers.{}", layer_idx),
        }
    }

    /// Get the architecture name for logging
    pub fn name(&self) -> &'static str {
        match self {
            Architecture::Qwen2 => "Qwen2",
            Architecture::LLaMA => "LLaMA",
            Architecture::Mistral => "Mistral",
        }
    }
}

/// Static execution plan for a transformer model
///
/// Contains pre-loaded weights and execution information for all layers.
/// No dynamic allocation during inference.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    layers: Vec<LayerPlan>,
    config: ModelConfig,
    embedding_weights: DeviceTensor,
    lm_head: DeviceTensor,
}

/// Execution plan for a single transformer layer
///
/// Contains all weight tensors needed for layer execution:
/// - QKV projection (fused Q, K, V)
/// - Output projection
/// - MLP layers (gate_proj, up_proj, down_proj for GLM)
/// - Layer normalization weights
#[derive(Debug, Clone)]
pub struct LayerPlan {
    /// Fused QKV projection weight matrix [3 * hidden_size, hidden_size]
    pub qkv_weight: DeviceTensor,
    /// Optional QKV bias [3 * hidden_size]
    pub qkv_bias: Option<DeviceTensor>,
    /// Output projection weight [hidden_size, hidden_size]
    pub o_proj: DeviceTensor,
    /// Optional output projection bias [hidden_size]
    pub o_proj_bias: Option<DeviceTensor>,
    /// MLP gate projection weight [intermediate_size, hidden_size] (GLM)
    pub mlp_gate_proj: DeviceTensor,
    /// MLP up projection weight [intermediate_size, hidden_size] (GLM)
    pub mlp_up_proj: DeviceTensor,
    /// MLP down projection weight [hidden_size, intermediate_size] (GLM)
    pub mlp_down_proj: DeviceTensor,
    /// Legacy MLP first layer weight [intermediate_size, hidden_size]
    pub mlp_fc1: DeviceTensor,
    /// Optional MLP first layer bias [intermediate_size]
    pub mlp_fc1_bias: Option<DeviceTensor>,
    /// Legacy MLP second layer weight [hidden_size, intermediate_size]
    pub mlp_fc2: DeviceTensor,
    /// Optional MLP second layer bias [hidden_size]
    pub mlp_fc2_bias: Option<DeviceTensor>,
    /// First layer norm weight [hidden_size]
    pub norm1_weight: DeviceTensor,
    /// Optional first layer norm bias [hidden_size]
    pub norm1_bias: Option<DeviceTensor>,
    /// Second layer norm weight [hidden_size]
    pub norm2_weight: DeviceTensor,
    /// Optional second layer norm bias [hidden_size]
    pub norm2_bias: Option<DeviceTensor>,
}

impl ExecutionPlan {
    /// Create a new execution plan from model configuration
    ///
    /// Loads all required weights and creates layer plans.
    /// For now, creates synthetic weights for testing.
    pub fn new(backend: &HipBackend, config: &ModelConfig) -> HipResult<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for layer_idx in 0..config.num_hidden_layers {
            let layer_plan = LayerPlan::new(backend, config, layer_idx)?;
            layers.push(layer_plan);
        }

        let embedding_shape = TensorShape::from_dims(&[config.vocab_size, config.hidden_size]);
        let embedding_weights = DeviceTensor::empty(backend, embedding_shape)?;

        let lm_head_shape = TensorShape::from_dims(&[config.hidden_size, config.vocab_size]);
        let lm_head = DeviceTensor::empty(backend, lm_head_shape)?;

        Ok(ExecutionPlan {
            layers,
            config: config.clone(),
            embedding_weights,
            lm_head,
        })
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

    /// Get embedding weights tensor
    pub fn embedding_weights(&self) -> &DeviceTensor {
        &self.embedding_weights
    }

    /// Get LM head tensor (hidden_size x vocab_size)
    pub fn lm_head(&self) -> &DeviceTensor {
        &self.lm_head
    }

    /// Apply LM head to hidden states to produce logits
    pub fn apply_lm_head(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        self.matmul(backend, hidden_states, &self.lm_head, None)
    }

    /// Detect model architecture from available tensor names
    ///
    /// Scans tensor names to identify the architecture pattern:
    /// - Qwen2: tensors start with `blk.N.`
    /// - LLaMA: tensors start with `transformer.layers.N.`
    /// - Mistral: tensors start with `model.layers.N.`
    fn detect_architecture(
        tensor_names: &HashSet<String>,
    ) -> HipResult<Architecture> {
        // Check for Qwen2 pattern: blk.0.*
        let qwen2_pattern = "blk.0.";
        let has_qwen2 = tensor_names
            .iter()
            .any(|name| name.starts_with(qwen2_pattern));

        if has_qwen2 {
            println!("Detected architecture: Qwen2 (pattern: {})", qwen2_pattern);
            return Ok(Architecture::Qwen2);
        }

        // Check for LLaMA pattern: transformer.layers.0.*
        let llama_pattern = "transformer.layers.0.";
        let has_llama = tensor_names
            .iter()
            .any(|name| name.starts_with(llama_pattern));

        if has_llama {
            println!("Detected architecture: LLaMA (pattern: {})", llama_pattern);
            return Ok(Architecture::LLaMA);
        }

        // Check for Mistral pattern: model.layers.0.*
        let mistral_pattern = "model.layers.0.";
        let has_mistral = tensor_names
            .iter()
            .any(|name| name.starts_with(mistral_pattern));

        if has_mistral {
            println!("Detected architecture: Mistral (pattern: {})", mistral_pattern);
            return Ok(Architecture::Mistral);
        }

        // Unknown architecture - try to provide helpful error
        let sample_tensors: Vec<_> = tensor_names
            .iter()
            .filter(|name| name.contains('.'))
            .take(10)
            .collect();

        Err(HipError::GenericError(format!(
            "Unable to detect model architecture from tensor names. \
             Expected patterns like 'blk.0.*', 'transformer.layers.0.*', or 'model.layers.0.*'. \
             Sample tensors found: {:?}",
            sample_tensors
        )))
    }

    /// Create execution plan from GGUF loader using helper functions
    pub fn from_gguf(backend: &HipBackend, loader: &GgufLoader) -> HipResult<Self> {
        let config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create model config: {}", e)))?;

        // Load all tensors to GPU
        let gpu_tensors = loader
            .load_to_gpu(backend)
            .map_err(|e| HipError::GenericError(format!("Failed to load tensors to GPU: {}", e)))?;

        // Detect architecture from actual tensor names
        let tensor_names: HashSet<_> = gpu_tensors.keys().cloned().collect();
        let architecture = Self::detect_architecture(&tensor_names)?;
        println!("Using {} architecture mapping", architecture.name());

        // Map embedding and LM head using helper functions and store them
        let embedding_weights = Self::map_embedding(backend, &config, &gpu_tensors)?;
        let lm_head = Self::map_lm_head(backend, &config, &gpu_tensors)?;

        // Create layers using detected architecture
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            // Map weights using detected architecture
            let (qkv_weight, o_proj) =
                Self::map_attention_weights(backend, &config, &gpu_tensors, layer_idx, &architecture)?;

            let (mlp_gate, mlp_up, mlp_down) =
                Self::map_mlp_weights(backend, &config, &gpu_tensors, layer_idx, &architecture)?;

            let (ln1_weight, ln1_bias, ln2_weight, ln2_bias) =
                Self::map_layer_norm_weights(backend, &config, &gpu_tensors, layer_idx, &architecture)?;

            // Create layer plan with mapped weights
            // Note: Helper functions don't provide QKV/O biases, so set them to None
            let layer_plan = LayerPlan {
                qkv_weight,
                qkv_bias: None, // Helper functions don't map biases
                o_proj,
                o_proj_bias: None, // Helper functions don't map biases
                mlp_gate_proj: mlp_gate.clone(),
                mlp_up_proj: mlp_up.clone(),
                mlp_down_proj: mlp_down.clone(),
                mlp_fc1: mlp_gate.clone(), // Legacy compatibility
                mlp_fc1_bias: None,
                mlp_fc2: mlp_down.clone(), // Legacy compatibility
                mlp_fc2_bias: None,
                norm1_weight: ln1_weight,
                norm1_bias: Some(ln1_bias), // Layer norm biases are provided
                norm2_weight: ln2_weight,
                norm2_bias: Some(ln2_bias), // Layer norm biases are provided
            };

            layers.push(layer_plan);
        }

        Ok(ExecutionPlan {
            layers,
            config,
            embedding_weights,
            lm_head,
        })
    }

    /// Forward pass through entire transformer model
    ///
    /// Takes input token IDs and performs inference using loaded weights.
    /// Returns final hidden states after all transformer layers.
    ///
    /// Arguments:
    /// - backend: HIP backend for GPU operations
    /// - input_tokens: Token IDs to process [seq_len]
    /// - embedding_weights: Token embedding weights [vocab_size, hidden_size]
    ///
    /// Returns:
    /// - DeviceTensor with final hidden states [seq_len, hidden_size]
    pub fn forward(
        &self,
        backend: &HipBackend,
        input_tokens: &[u32],
        embedding_weights: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        let seq_len = input_tokens.len();
        let hidden_size = self.config.hidden_size;

        // Performance profiling: Start timing
        let start_time = std::time::Instant::now();
        println!("PERF: Starting forward pass for {} tokens", seq_len);

        // Step 1: Token embedding lookup
        let embedding_start = std::time::Instant::now();
        let mut hidden_states = self.embedding_lookup(backend, input_tokens, embedding_weights)?;
        let embedding_time = embedding_start.elapsed();
        println!("PERF: Embedding lookup: {:?}", embedding_time);

        // Step 2: Pass through all transformer layers
        let mut layer_times = Vec::new();
        for (layer_idx, layer_plan) in self.layers.iter().enumerate() {
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
            let min_layer_time = layer_times.iter().min().unwrap();
            let max_layer_time = layer_times.iter().max().unwrap();
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
    pub fn embedding_lookup(
        &self,
        backend: &HipBackend,
        input_tokens: &[u32],
        embedding_weights: &DeviceTensor,
    ) -> HipResult<DeviceTensor> {
        let seq_len = input_tokens.len();
        let hidden_size = self.config.hidden_size;

        // Validate embedding weight shape: [vocab_size, hidden_size]
        let embed_shape = embedding_weights.shape().dims();
        if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
            return Err(HipError::GenericError(format!(
                "Embedding weight shape must be [vocab_size, hidden_size], got {:?}",
                embed_shape
            )));
        }

        // Validate token IDs
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

        // Efficient GPU embedding lookup via device-to-device copies
        let mut hidden_states = DeviceTensor::empty(backend, output_shape)?;
        for (i, &token_id) in input_tokens.iter().enumerate() {
            let token_index = token_id as usize;
            if token_index >= vocab_size {
                return Err(HipError::GenericError(format!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_index, vocab_size
                )));
            }

            let src_offset = token_index * hidden_size;
            let dst_offset = i * hidden_size;
            hidden_states.copy_from_device_region(
                dst_offset,
                embedding_weights,
                src_offset,
                hidden_size,
            )?;
        }

        Ok(hidden_states)
    }

    /// Forward pass through a single transformer layer
    ///
    /// Implements the standard transformer layer pattern:
    /// LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    pub fn forward_layer(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        layer_plan: &LayerPlan,
        mut kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        let input_shape = hidden_states.shape().dims();
        let seq_len = input_shape[0];
        let hidden_size = input_shape[1];

        // Store input for residual connection
        let residual = hidden_states.clone();

        // Step 1: Pre-attention LayerNorm
        let normed_hidden = self.layer_norm(
            backend,
            hidden_states,
            &layer_plan.norm1_weight,
            layer_plan.norm1_bias.as_ref(),
        )?;

        // Step 2: Self-attention
        let attention_output = self.self_attention(
            backend,
            &normed_hidden,
            &layer_plan.qkv_weight,
            layer_plan.qkv_bias.as_ref(),
            &layer_plan.o_proj,
            layer_plan.o_proj_bias.as_ref(),
            kv_cache,
            layer_idx,
        )?;

        // Step 3: Add residual connection
        let attention_with_residual = self.add_residual(backend, &attention_output, &residual)?;

        // Store attention output for second residual
        let attention_residual = attention_with_residual.clone();

        // Step 4: Pre-MLP LayerNorm
        let normed_attention = self.layer_norm(
            backend,
            &attention_with_residual,
            &layer_plan.norm2_weight,
            layer_plan.norm2_bias.as_ref(),
        )?;

        // Step 5: MLP (SwiGLU)
        let mlp_output = self.mlp_swiglu(
            backend,
            &normed_attention,
            &layer_plan.mlp_gate_proj,
            &layer_plan.mlp_up_proj,
            &layer_plan.mlp_down_proj,
        )?;

        // Step 6: Add residual connection
        let final_output = self.add_residual(backend, &mlp_output, &attention_residual)?;

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
        mut kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        let input_shape = hidden_states.shape().dims();
        let seq_len = input_shape[0];
        let hidden_size = input_shape[1];
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Step 1: Project to Q, K, V using GPU matrix multiplication
        // QKV projection: [seq_len, hidden_size] x [hidden_size, 3*hidden_size] -> [seq_len, 3*hidden_size]
        let qkv_proj = self.matmul(backend, hidden_states, qkv_weight, qkv_bias)?;

        // Step 2: Split Q, K, V directly on GPU
        let (q_reshaped, k_reshaped, v_reshaped) =
            self.extract_qkv_tensors(backend, &qkv_proj, seq_len, num_heads, head_dim)?;

        // Step 4: Scaled dot-product attention (still CPU fallback for now)
        // TODO: Replace with GPU attention kernel
        let attention_output = self.scaled_dot_product_attention(
            backend,
            &q_reshaped,
            &k_reshaped,
            &v_reshaped,
            kv_cache.as_deref_mut(),
            layer_idx,
        )?;

        // Step 5: Reshape back: [seq_len, hidden_size]
        let output_reshaped = self.flatten_attention_output(
            backend,
            &attention_output,
            seq_len,
            num_heads,
            head_dim,
        )?;

        // Step 6: Output projection using GPU matrix multiplication
        let final_output = self.matmul(backend, &output_reshaped, o_proj, o_proj_bias)?;

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
        let output_shape = TensorShape::from_dims(&input_shape);
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
        use crate::backend::hip_blas::{sgemm, HipBlasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T};
        use crate::tensor::matmul::matmul_f32;

        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();

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

        // Create hipBLAS handle for matrix operations
        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        // Perform matrix multiplication: input @ weight -> output
        // input: [batch_size, input_dim], weight: [input_dim, output_dim] -> output: [batch_size, output_dim]
        let output_buffer = matmul_f32(
            &blas_handle,
            input.buffer(),
            weight.buffer(),
            batch_size as i32,
            output_dim as i32,
            input_dim as i32,
        )
        .map_err(|e| HipError::GenericError(format!("Matrix multiplication failed: {}", e)))?;

        let output_shape = TensorShape::from_dims(&[batch_size, output_dim]);
        let mut output_tensor = DeviceTensor::empty(backend, output_shape)?;
        output_tensor.copy_from_device_buffer(&output_buffer)?;

        if let Some(bias_tensor) = bias {
            backend.add_row_bias(&mut output_tensor, bias_tensor)?;
        }

        Ok(output_tensor)
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
        ) -> HipResult<DeviceTensor> {
            let shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
            let mut tensor = DeviceTensor::empty(backend, shape)?;
            tensor.copy_from_device_slice(src, offset_elements)?;
            Ok(tensor)
        }

        let q = copy_chunk(backend, qkv_proj, 0, seq_len, num_heads, head_dim)?;
        let k = copy_chunk(
            backend,
            qkv_proj,
            chunk_elements,
            seq_len,
            num_heads,
            head_dim,
        )?;
        let v = copy_chunk(
            backend,
            qkv_proj,
            chunk_elements * 2,
            seq_len,
            num_heads,
            head_dim,
        )?;

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

    /// Scaled dot-product attention using GPU kernels
    fn scaled_dot_product_attention(
        &self,
        backend: &HipBackend,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        // Validate input shapes
        let q_shape = q.shape().dims();
        let k_shape = k.shape().dims();
        let v_shape = v.shape().dims();

        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(HipError::GenericError(
                "Q, K, V must be 3D tensors [seq_len, num_heads, head_dim]".to_string(),
            ));
        }

        let seq_len = q_shape[0];
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];

        // Validate all tensors have compatible shapes
        if k_shape[0] != seq_len || k_shape[1] != num_heads || k_shape[2] != head_dim {
            return Err(HipError::GenericError(
                "K tensor shape must match Q tensor shape".to_string(),
            ));
        }

        if v_shape[0] != seq_len || v_shape[1] != num_heads || v_shape[2] != head_dim {
            return Err(HipError::GenericError(
                "V tensor shape must match Q tensor shape".to_string(),
            ));
        }

        // Create GPU attention kernels
        let attention_kernels = HipAttentionKernels::new(backend)?;

        if let Some(cache) = kv_cache {
            cache.append(layer_idx, k, v)?;
            let current_len = cache.get_current_length(layer_idx)?;
            let attention_shape = TensorShape::from_dims(&[seq_len, current_len]);
            let mut attention_scores = DeviceTensor::empty(backend, attention_shape.clone())?;
            let softmax_temp = DeviceTensor::empty(backend, attention_shape)?;
            let cache_ref: &KVCache = &*cache;
            return attention_kernels.compute_attention(
                q,
                &attention_scores,
                &softmax_temp,
                cache_ref,
                layer_idx,
                current_len,
            );
        }

        // Create temporary buffers for attention computation
        let attention_shape = TensorShape::from_dims(&[seq_len, seq_len]);
        let mut attention_scores = DeviceTensor::empty(backend, attention_shape)?;

        let softmax_temp_shape = TensorShape::from_dims(&[seq_len, seq_len]);
        let softmax_temp = DeviceTensor::empty(backend, softmax_temp_shape)?;

        // Step 1: Compute QK^T attention scores
        attention_kernels.compute_qk_t(q, k, &mut attention_scores)?;

        // Step 2: Scale by 1/sqrt(head_dim) - manual scaling
        let scale = 1.0 / (head_dim as f32).sqrt();
        backend.scale_inplace(&mut attention_scores, scale)?;

        // Step 3: Apply causal mask (for decoder-only models)
        attention_kernels.apply_causal_mask(&mut attention_scores, seq_len, seq_len)?;

        // Step 4: Compute softmax
        attention_kernels.compute_softmax(&mut attention_scores, &softmax_temp)?;

        // Step 5: Compute attention-weighted V
        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut output = DeviceTensor::empty(backend, output_shape)?;
        attention_kernels.compute_attention_weighted_v(&attention_scores, v, &mut output)?;

        Ok(output)
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
    fn map_embedding(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
    ) -> HipResult<DeviceTensor> {
        // Try different embedding tensor naming conventions
        let embedding_names = [
            "token_embd.weight",
            "embed_tokens.weight",
            "word_embeddings.weight",
        ];

        for name in &embedding_names {
            if let Some(tensor) = gpu_tensors.get(*name) {
                let shape = tensor.shape().dims();

                // Validate embedding tensor shape: [vocab_size, hidden_size] or transposed
                if shape.len() != 2 {
                    return Err(HipError::GenericError(format!(
                        "Embedding tensor '{}' should be 2D, got {}D",
                        name,
                        shape.len()
                    )));
                }

                if shape[0] == config.vocab_size && shape[1] == config.hidden_size {
                    return Ok(tensor.clone());
                }

                if shape[0] == config.hidden_size && shape[1] == config.vocab_size {
                    let transposed = Self::transpose_2d_tensor(backend, tensor)?;
                    return Ok(transposed);
                }

                return Err(HipError::GenericError(format!(
                    "Embedding tensor '{}' shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
                    name,
                    shape[0],
                    shape[1],
                    config.vocab_size,
                    config.hidden_size,
                    config.hidden_size,
                    config.vocab_size
                )));
            }
        }

        Err(HipError::GenericError(
            "No embedding tensor found (tried: token_embd.weight, embed_tokens.weight, word_embeddings.weight)".to_string()
        ))
    }

    /// Map language model head weights from GGUF tensors
    ///
    /// Extracts LM head weights from GGUF and validates shape.
    /// Supports multiple naming conventions: output.weight, lm_head.weight, logits.weight.
    /// Accepts weights stored as [hidden_size, vocab_size] or [vocab_size, hidden_size]
    /// and ensures the execution plan stores them in [hidden_size, vocab_size] form.
    fn map_lm_head(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
    ) -> HipResult<DeviceTensor> {
        // Try different LM head tensor naming conventions
        let lm_head_names = ["output.weight", "lm_head.weight", "logits.weight"];

        for name in &lm_head_names {
            if let Some(tensor) = gpu_tensors.get(*name) {
                let shape = tensor.shape().dims();

                // Validate LM head tensor shape: [vocab_size, hidden_size]
                if shape.len() != 2 {
                    return Err(HipError::GenericError(format!(
                        "LM head tensor '{}' should be 2D, got {}D",
                        name,
                        shape.len()
                    )));
                }

                let expected = (config.hidden_size, config.vocab_size);
                if shape[0] == expected.0 && shape[1] == expected.1 {
                    return Ok(tensor.clone());
                }

                if shape[0] == expected.1 && shape[1] == expected.0 {
                    // Convert from [vocab_size, hidden_size] to [hidden_size, vocab_size]
                    let transposed = Self::transpose_2d_tensor(backend, tensor)?;
                    return Ok(transposed);
                }

                return Err(HipError::GenericError(format!(
                    "LM head tensor '{}' shape [{}, {}] doesn't match expected [{}, {}] or [{}, {}]",
                    name,
                    shape[0],
                    shape[1],
                    expected.0,
                    expected.1,
                    expected.1,
                    expected.0
                )));
            }
        }

        Err(HipError::GenericError(
            "No LM head tensor found (tried: output.weight, lm_head.weight, logits.weight)"
                .to_string(),
        ))
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
            println!("Layer {}: Found separate Q, K, V tensors - concatenating", layer_idx);
            let qkv_weight = Self::concatenate_qkv_tensors(backend, q, k, v, config)?;

            // Get output projection
            let o_name = format!("{}.attn_output.weight", prefix);
            let o_name_alt = format!("{}.attn.o_proj.weight", prefix);
            let o_weight = gpu_tensors.get(&o_name)
                .or_else(|| gpu_tensors.get(&o_name_alt))
                .ok_or_else(|| HipError::GenericError(format!(
                    "Output projection not found (tried: {}, {})",
                    o_name, o_name_alt
                )))?;

            return Ok((qkv_weight, o_weight.clone()));
        }

        // Try fused QKV tensor
        let qkv_variants = vec![
            format!("{}.attention.wq.weight", prefix),  // LLaMA-style
            format!("{}.attention.query_key_value.weight", prefix),  // Falcon-style
            format!("{}.self_attn.q_proj.weight", prefix),  // Mistral-style
            format!("{}.attn.qkv.weight", prefix),  // Generic
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
            format!("{}.attention.wo.weight", prefix),  // LLaMA-style
            format!("{}.self_attn.o_proj.weight", prefix),  // Mistral-style
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

        println!("Layer {}: Found fused QKV tensor - using directly", layer_idx);
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
            None => return Err(HipError::GenericError("Qwen2 Q tensor not found".to_string())),
        };
        let k_weight = match k_weight {
            Some(t) => t,
            None => return Err(HipError::GenericError("Qwen2 K tensor not found".to_string())),
        };
        let v_weight = match v_weight {
            Some(t) => t,
            None => return Err(HipError::GenericError("Qwen2 V tensor not found".to_string())),
        };
        let o_weight = match o_weight {
            Some(t) => t,
            None => return Err(HipError::GenericError("Qwen2 O tensor not found".to_string())),
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
        let qkv_weight = Self::concatenate_qkv_tensors(backend, q_weight, k_weight, v_weight, config)?;

        Ok((qkv_weight, o_weight.clone()))
    }

    /// Map LLaMA-style attention weights (fused QKV with transformer.layers.N. prefix)
    fn map_llama_attention_weights(
        backend: &HipBackend,
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
        let head_dim = config.head_dim;

        // Check if K and V need padding
        let k_needs_padding = k_shape[1] != config.hidden_size;
        let v_needs_padding = v_shape[1] != config.hidden_size;

        // Helper function to pad tensor if needed
        let pad_tensor = |tensor: &DeviceTensor, target_cols: usize| -> HipResult<DeviceTensor> {
            let current_shape = tensor.shape().dims();
            if current_shape[1] == target_cols {
                return Ok(tensor.clone());
            }

            // Create padded tensor: [hidden_size, target_cols]
            let padded_shape = TensorShape::from_dims(&[current_shape[0], target_cols]);
            let mut padded = DeviceTensor::empty(backend, padded_shape.clone())?;

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
        let mut qkv_weight = DeviceTensor::empty(backend, qkv_shape.clone())?;

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
            format!("{}.ffn_gate.weight", prefix),  // Qwen2-style
            format!("{}.mlp.gate_proj.weight", prefix),  // LLaMA/Mistral-style
            format!("{}.feed_forward.w1.weight", prefix),  // Alternative
            format!("{}.mlp.c_fc.weight", prefix),  // GPT-style
        ];

        let up_variants = vec![
            format!("{}.ffn_up.weight", prefix),  // Qwen2-style
            format!("{}.mlp.up_proj.weight", prefix),  // LLaMA/Mistral-style
            format!("{}.feed_forward.w3.weight", prefix),  // Alternative
            format!("{}.mlp.c_proj.weight", prefix),  // GPT-style
        ];

        let down_variants = vec![
            format!("{}.ffn_down.weight", prefix),  // Qwen2-style
            format!("{}.mlp.down_proj.weight", prefix),  // LLaMA/Mistral-style
            format!("{}.feed_forward.w2.weight", prefix),  // Alternative
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
            None => return Err(HipError::GenericError("Qwen2 FFN gate tensor not found".to_string())),
        };
        let up_weight = match up_weight {
            Some(t) => t,
            None => return Err(HipError::GenericError("Qwen2 FFN up tensor not found".to_string())),
        };
        let down_weight = match down_weight {
            Some(t) => t,
            None => return Err(HipError::GenericError("Qwen2 FFN down tensor not found".to_string())),
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
    fn map_llama_mlp_weights(
        backend: &HipBackend,
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
            format!("{}.attn_norm.weight", prefix),  // Qwen2-style
            format!("{}.attention_norm.weight", prefix),  // LLaMA-style
            format!("{}.input_layernorm.weight", prefix),  // Mistral-style
            format!("{}.ln_1.weight", prefix),  // GPT-style
            format!("{}.pre_attention_layernorm.weight", prefix),  // Alternative
        ];

        // Try multiple naming variants for FFN norm (output/second layer norm)
        let ffn_norm_variants = vec![
            format!("{}.ffn_norm.weight", prefix),  // Qwen2-style
            format!("{}.post_attention_layernorm.weight", prefix),  // Mistral-style
            format!("{}.ln_2.weight", prefix),  // GPT-style
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

        let attn_norm_bias_variants = vec![
            format!("{}.attn_norm.bias", prefix),
            format!("{}.attention_norm.bias", prefix),
            format!("{}.input_layernorm.bias", prefix),
        ];

        let ffn_norm_bias_variants = vec![
            format!("{}.ffn_norm.bias", prefix),
            format!("{}.post_attention_layernorm.bias", prefix),
        ];

        let attn_norm_bias = attn_norm_bias_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .cloned()
            .unwrap_or_else(|| create_zero_bias().unwrap());

        let ffn_norm_bias = ffn_norm_bias_variants
            .iter()
            .find_map(|name| gpu_tensors.get(name))
            .cloned()
            .unwrap_or_else(|| create_zero_bias().unwrap());

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
            None => return Err(HipError::GenericError("Qwen2 attn_norm.weight not found".to_string())),
        };
        let attn_norm_bias = gpu_tensors.get(&attn_norm_bias_name);
        let ffn_norm_weight = match gpu_tensors.get(&ffn_norm_weight_name) {
            Some(t) => t,
            None => return Err(HipError::GenericError("Qwen2 ffn_norm.weight not found".to_string())),
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
    fn map_llama_layer_norm_weights(
        backend: &HipBackend,
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
}

impl LayerPlan {
    /// Create a new layer plan
    ///
    /// Initializes all weight tensors for a single transformer layer.
    /// Currently creates synthetic weights for testing.
    fn new(backend: &HipBackend, config: &ModelConfig, layer_idx: usize) -> HipResult<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        // Create synthetic weights for testing
        // In a real implementation, these would be loaded from model files

        // QKV projection weight: [3 * hidden_size, hidden_size]
        let qkv_weight_shape = TensorShape::from_dims(&[3 * hidden_size, hidden_size]);
        let qkv_weight = DeviceTensor::empty(backend, qkv_weight_shape)?;

        // QKV bias: [3 * hidden_size]
        let qkv_bias_shape = TensorShape::from_dims(&[3 * hidden_size]);
        let qkv_bias = Some(DeviceTensor::empty(backend, qkv_bias_shape)?);

        // Output projection: [hidden_size, hidden_size]
        let o_proj_shape = TensorShape::from_dims(&[hidden_size, hidden_size]);
        let o_proj = DeviceTensor::empty(backend, o_proj_shape)?;

        // Output projection bias: [hidden_size]
        let o_proj_bias_shape = TensorShape::from_dims(&[hidden_size]);
        let o_proj_bias = Some(DeviceTensor::empty(backend, o_proj_bias_shape)?);

        // MLP FC1: [intermediate_size, hidden_size]
        let mlp_fc1_shape = TensorShape::from_dims(&[intermediate_size, hidden_size]);
        let mlp_fc1 = DeviceTensor::empty(backend, mlp_fc1_shape)?;

        // MLP FC1 bias: [intermediate_size]
        let mlp_fc1_bias_shape = TensorShape::from_dims(&[intermediate_size]);
        let mlp_fc1_bias = Some(DeviceTensor::empty(backend, mlp_fc1_bias_shape)?);

        // MLP FC2: [hidden_size, intermediate_size]
        let mlp_fc2_shape = TensorShape::from_dims(&[hidden_size, intermediate_size]);
        let mlp_fc2 = DeviceTensor::empty(backend, mlp_fc2_shape)?;

        // MLP FC2 bias: [hidden_size]
        let mlp_fc2_bias_shape = TensorShape::from_dims(&[hidden_size]);
        let mlp_fc2_bias = Some(DeviceTensor::empty(backend, mlp_fc2_bias_shape)?);

        // Layer norm weights: [hidden_size]
        let norm_shape = TensorShape::from_dims(&[hidden_size]);
        let norm1_weight = DeviceTensor::empty(backend, norm_shape.clone())?;
        let norm2_weight = DeviceTensor::empty(backend, norm_shape.clone())?;

        // Layer norm biases: [hidden_size]
        let norm1_bias = Some(DeviceTensor::empty(backend, norm_shape.clone())?);
        let norm2_bias = Some(DeviceTensor::empty(backend, norm_shape)?);

        Ok(LayerPlan {
            qkv_weight,
            qkv_bias,
            o_proj,
            o_proj_bias,
            mlp_gate_proj: mlp_fc1.clone(), // Use fc1 as gate_proj for now
            mlp_up_proj: mlp_fc1.clone(),   // Use fc1 as up_proj for now
            mlp_down_proj: mlp_fc2.clone(), // Use fc2 as down_proj for now
            mlp_fc1,
            mlp_fc1_bias,
            mlp_fc2,
            mlp_fc2_bias,
            norm1_weight,
            norm1_bias,
            norm2_weight,
            norm2_bias,
        })
    }

    /// Get QKV projection weight
    pub fn qkv_weight(&self) -> &DeviceTensor {
        &self.qkv_weight
    }

    /// Get QKV projection bias
    pub fn qkv_bias(&self) -> Option<&DeviceTensor> {
        self.qkv_bias.as_ref()
    }

    /// Get output projection weight
    pub fn o_proj(&self) -> &DeviceTensor {
        &self.o_proj
    }

    /// Get output projection bias
    pub fn o_proj_bias(&self) -> Option<&DeviceTensor> {
        self.o_proj_bias.as_ref()
    }

    /// Get MLP first layer weight
    pub fn mlp_fc1(&self) -> &DeviceTensor {
        &self.mlp_fc1
    }

    /// Get MLP first layer bias
    pub fn mlp_fc1_bias(&self) -> Option<&DeviceTensor> {
        self.mlp_fc1_bias.as_ref()
    }

    /// Get MLP second layer weight
    pub fn mlp_fc2(&self) -> &DeviceTensor {
        &self.mlp_fc2
    }

    /// Get MLP second layer bias
    pub fn mlp_fc2_bias(&self) -> Option<&DeviceTensor> {
        self.mlp_fc2_bias.as_ref()
    }

    /// Get first layer norm weight
    pub fn norm1_weight(&self) -> &DeviceTensor {
        &self.norm1_weight
    }

    /// Get first layer norm bias
    pub fn norm1_bias(&self) -> Option<&DeviceTensor> {
        self.norm1_bias.as_ref()
    }

    /// Get second layer norm weight
    pub fn norm2_weight(&self) -> &DeviceTensor {
        &self.norm2_weight
    }

    /// Get second layer norm bias
    pub fn norm2_bias(&self) -> Option<&DeviceTensor> {
        self.norm2_bias.as_ref()
    }

    /// Create layer plan from GGUF tensors
    fn from_gguf_tensors(
        backend: &HipBackend,
        config: &ModelConfig,
        gpu_tensors: &std::collections::HashMap<String, DeviceTensor>,
        layer_idx: usize,
    ) -> HipResult<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        // Helper to get tensor with error handling
        let get_tensor = |name: &str| -> HipResult<DeviceTensor> {
            gpu_tensors
                .get(name)
                .map(|t| t.clone())
                .ok_or_else(|| HipError::GenericError(format!("Tensor '{}' not found", name)))
        };

        let get_optional_tensor =
            |name: &str| -> Option<DeviceTensor> { gpu_tensors.get(name).map(|t| t.clone()) };

        // Map GGUF tensor names to layer plan tensors
        let layer_prefix = format!("layers.{}", layer_idx);

        // QKV projection
        let qkv_weight_name = format!("{}.attention.qkv.weight", layer_prefix);
        let qkv_weight = get_tensor(&qkv_weight_name)?;

        let qkv_bias_name = format!("{}.attention.qkv.bias", layer_prefix);
        let qkv_bias = get_optional_tensor(&qkv_bias_name);

        // Output projection
        let o_proj_name = format!("{}.attention.o_proj.weight", layer_prefix);
        let o_proj = get_tensor(&o_proj_name)?;

        let o_proj_bias_name = format!("{}.attention.o_proj.bias", layer_prefix);
        let o_proj_bias = get_optional_tensor(&o_proj_bias_name);

        // MLP projections (GLM style)
        let gate_proj_name = format!("{}.mlp.gate_proj.weight", layer_prefix);
        let up_proj_name = format!("{}.mlp.up_proj.weight", layer_prefix);
        let down_proj_name = format!("{}.mlp.down_proj.weight", layer_prefix);

        // Try GLM-style names first, fall back to generic names
        let mlp_gate_proj = if let Some(tensor) = get_optional_tensor(&gate_proj_name) {
            tensor
        } else {
            let fc1_name = format!("{}.mlp.fc1.weight", layer_prefix);
            get_tensor(&fc1_name)?
        };

        let mlp_up_proj = if let Some(tensor) = get_optional_tensor(&up_proj_name) {
            tensor
        } else {
            // Use gate_proj as up_proj if not available
            mlp_gate_proj.clone()
        };

        let mlp_down_proj = if let Some(tensor) = get_optional_tensor(&down_proj_name) {
            tensor
        } else {
            let fc2_name = format!("{}.mlp.fc2.weight", layer_prefix);
            get_tensor(&fc2_name)?
        };

        // Legacy MLP tensors for compatibility
        let mlp_fc1_name = format!("{}.mlp.fc1.weight", layer_prefix);
        let mlp_fc1 = get_tensor(&mlp_fc1_name)?;

        let mlp_fc1_bias_name = format!("{}.mlp.fc1.bias", layer_prefix);
        let mlp_fc1_bias = get_optional_tensor(&mlp_fc1_bias_name);

        let mlp_fc2_name = format!("{}.mlp.fc2.weight", layer_prefix);
        let mlp_fc2 = get_tensor(&mlp_fc2_name)?;

        let mlp_fc2_bias_name = format!("{}.mlp.fc2.bias", layer_prefix);
        let mlp_fc2_bias = get_optional_tensor(&mlp_fc2_bias_name);

        // Layer norms (GLM uses attention_norm and ffn_norm)
        let norm1_name = format!("{}.attention_norm.weight", layer_prefix);
        let norm1_weight = if let Some(tensor) = get_optional_tensor(&norm1_name) {
            tensor
        } else {
            let input_norm_name = format!("{}.input_layernorm.weight", layer_prefix);
            get_tensor(&input_norm_name)?
        };

        let norm1_bias_name = format!("{}.attention_norm.bias", layer_prefix);
        let norm1_bias = get_optional_tensor(&norm1_bias_name);

        let norm2_name = format!("{}.ffn_norm.weight", layer_prefix);
        let norm2_weight = if let Some(tensor) = get_optional_tensor(&norm2_name) {
            tensor
        } else {
            let post_norm_name = format!("{}.post_attention_layernorm.weight", layer_prefix);
            get_tensor(&post_norm_name)?
        };

        let norm2_bias_name = format!("{}.ffn_norm.bias", layer_prefix);
        let norm2_bias = get_optional_tensor(&norm2_bias_name);

        Ok(LayerPlan {
            qkv_weight,
            qkv_bias,
            o_proj,
            o_proj_bias,
            mlp_gate_proj,
            mlp_up_proj,
            mlp_down_proj,
            mlp_fc1,
            mlp_fc1_bias,
            mlp_fc2,
            mlp_fc2_bias,
            norm1_weight,
            norm1_bias,
            norm2_weight,
            norm2_bias,
        })
    }
}
