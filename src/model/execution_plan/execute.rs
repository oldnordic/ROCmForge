//! Layer execution logic
//!
//! Handles forward pass through transformer layers, attention, MLP, etc.

use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::loader::TensorShape;
use crate::model::kv_cache::KVCache;

use super::layer_tensors::create_layer_plan_lazy;
use super::matmul::{matmul, reshape_for_attention, extract_qkv_tensors, flatten_attention_output, add_residual};
use super::rope::rope_cache;
use super::types::ExecutionPlan;

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
    plan: &ExecutionPlan,
    backend: &HipBackend,
    hidden_states: &DeviceTensor,
    layer_plan: &crate::model::execution_plan::LayerPlan,
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
    let qkv_weight = plan.get_or_load_tensor(&layer_plan.qkv_weight)?;
    let qkv_bias = layer_plan
        .qkv_bias
        .as_ref()
        .map(|b| plan.get_or_load_tensor(b))
        .transpose()?;

    // Load separate Q, K, V weights if model uses them (e.g., Qwen2)
    let q_weight = layer_plan
        .q_weight
        .as_ref()
        .map(|w| plan.get_or_load_tensor(w))
        .transpose()?;
    let k_weight = layer_plan
        .k_weight
        .as_ref()
        .map(|w| plan.get_or_load_tensor(w))
        .transpose()?;
    let v_weight = layer_plan
        .v_weight
        .as_ref()
        .map(|w| plan.get_or_load_tensor(w))
        .transpose()?;
    let q_bias = layer_plan
        .q_bias
        .as_ref()
        .map(|b| plan.get_or_load_tensor(b))
        .transpose()?;
    let k_bias = layer_plan
        .k_bias
        .as_ref()
        .map(|b| plan.get_or_load_tensor(b))
        .transpose()?;
    let v_bias = layer_plan
        .v_bias
        .as_ref()
        .map(|b| plan.get_or_load_tensor(b))
        .transpose()?;

    let o_proj = plan.get_or_load_tensor(&layer_plan.o_proj)?;
    let o_proj_bias = layer_plan
        .o_proj_bias
        .as_ref()
        .map(|b| plan.get_or_load_tensor(b))
        .transpose()?;
    let mlp_gate_proj = plan.get_or_load_tensor(&layer_plan.mlp_gate_proj)?;
    let mlp_up_proj = plan.get_or_load_tensor(&layer_plan.mlp_up_proj)?;
    let mlp_down_proj = plan.get_or_load_tensor(&layer_plan.mlp_down_proj)?;
    let norm1_weight = plan.get_or_load_tensor(&layer_plan.norm1_weight)?;
    let norm1_bias = layer_plan
        .norm1_bias
        .as_ref()
        .map(|b| plan.get_or_load_tensor(b))
        .transpose()?;
    let norm2_weight = plan.get_or_load_tensor(&layer_plan.norm2_weight)?;
    let norm2_bias = layer_plan
        .norm2_bias
        .as_ref()
        .map(|b| plan.get_or_load_tensor(b))
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
        layer_norm(backend, hidden_states, &norm1_weight, norm1_bias.as_ref())?;
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

        self_attention_separate(
            plan,
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
        self_attention(
            plan,
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
    let attention_with_residual = add_residual(backend, &attention_output, &residual)?;
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
    let normed_attention = layer_norm(
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
    let mlp_output = mlp_swiglu(
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
    let final_output = add_residual(backend, &mlp_output, &attention_residual)?;
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
    plan: &ExecutionPlan,
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
    let num_heads = plan.config_internal().num_attention_heads;
    let head_dim = hidden_size / num_heads;

    // Step 1: Project to Q, K, V using GPU matrix multiplication
    // QKV projection: [seq_len, hidden_size] x [hidden_size, 3*hidden_size] -> [seq_len, 3*hidden_size]
    eprintln!(
        ">>>   self_attention({}): Step 1/6 - QKV projection matmul...",
        layer_idx
    );
    let step_start = std::time::Instant::now();
    let qkv_proj = matmul(plan, backend, hidden_states, qkv_weight, qkv_bias)?;
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
        extract_qkv_tensors(backend, &qkv_proj, seq_len, num_heads, head_dim)?;
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
    if let Some(ref position_handler) = plan.position_handler() {
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
    let attention_output = scaled_dot_product_attention(
        plan,
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
    let output_reshaped = flatten_attention_output(
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
    let final_output = matmul(plan, backend, &output_reshaped, o_proj, o_proj_bias)?;
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
    plan: &ExecutionPlan,
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
    let num_heads = plan.config_internal().num_attention_heads;
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
    let q_proj = matmul(plan, backend, hidden_states, q_weight, q_bias)?;
    let k_proj = matmul(plan, backend, hidden_states, k_weight, k_bias)?;
    let v_proj = matmul(plan, backend, hidden_states, v_weight, v_bias)?;
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
        reshape_for_attention(backend, &q_proj, seq_len, num_heads, head_dim)?;

    // For K and V, we need to determine kv_dim from their shape
    let k_shape = k_proj.shape().dims();
    let kv_dim = k_shape[1];
    // GQA: num_kv_heads might be less than num_heads
    let num_kv_heads = kv_dim / head_dim;

    let k_reshaped =
        reshape_for_attention(backend, &k_proj, seq_len, num_kv_heads, head_dim)?;
    let v_reshaped =
        reshape_for_attention(backend, &v_proj, seq_len, num_kv_heads, head_dim)?;

    eprintln!(">>>   self_attention_separate({}): Step 2/7 - Reshape done (num_heads={}, num_kv_heads={})",
             layer_idx, num_heads, num_kv_heads);

    // Step 3: Apply position encoding to Q and K tensors (RoPE)
    // NOTE: Using CPU path for now due to GPU RoPE backend synchronization issue with GQA
    eprintln!(
        ">>>   self_attention_separate({}): Step 3/7 - Apply RoPE position embeddings...",
        layer_idx
    );
    let step_start = std::time::Instant::now();
    let (q_reshaped, k_reshaped) = if let Some(ref position_handler) = plan.position_handler() {
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
    let attention_output = scaled_dot_product_attention(
        plan,
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
    let output_reshaped = flatten_attention_output(
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
    let final_output = matmul(plan, backend, &output_reshaped, o_proj, o_proj_bias)?;
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

/// Scaled dot-product attention with detailed tracing
fn scaled_dot_product_attention(
    plan: &ExecutionPlan,
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
    compute_attention_cpu_fallback(
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
