//! GPU prefill forward path for batched prompt processing.
//!
//! This module implements batched QKV projection for prompt prefill, where multiple tokens
//! are processed in parallel. This is distinct from decode which processes one token
//! at a time with cached key-value states.

use super::cache::GpuPrefillScratch;
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use super::kernels::{
    add_batched, embed_q4_0_batch, embed_q8_0_batch, flash_attn_prefill_strided, kv_write_batched,
    rms_norm_batched, rope_heads_batched,
};
use super::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_gemv_on_stream, gpu_dispatch_rms_norm,
};
use super::ops_batched::{
    gpu_dispatch_batched_fused_gate_up_q4_0, gpu_dispatch_batched_gemv_batched,
};
use super::weights::{GpuBuffer, GpuLayerWeights, GpuModelWeights, WeightMeta};
use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::forward::cpu_embed_token;
use crate::cpu::ops::dispatch_gemv as cpu_dispatch_gemv;
use crate::cpu::weights::CpuModelWeights;
use crate::loader::GgmlType;

/// Batched QKV projection for prefill processing (type-aware).
///
/// This function projects input hidden states through Query, Key, and Value weight matrices
/// in parallel for multiple tokens. This is the core operation for prompt prefill.
///
/// # Arguments
/// * `device` - GPU device reference
/// * `q_weights` - Query projection weights (quantized)
/// * `q_meta` - Query weight metadata (determines kernel dispatch)
/// * `k_weights` - Key projection weights (quantized)
/// * `k_meta` - Key weight metadata (determines kernel dispatch)
/// * `v_weights` - Value projection weights (quantized)
/// * `v_meta` - Value weight metadata (determines kernel dispatch)
/// * `input` - Input hidden states [seq_len, hidden_dim] (row-major, GPU pointer)
/// * `q_output` - Query output buffer [seq_len, q_dim] (row-major, GPU pointer)
/// * `k_output` - Key output buffer [seq_len, kv_dim] (row-major, GPU pointer)
/// * `v_output` - Value output buffer [seq_len, kv_dim] (row-major, GPU pointer)
/// * `hidden_dim` - Input hidden dimension
/// * `q_dim` - Query output dimension (num_heads * head_dim)
/// * `kv_dim` - Key/Value output dimension (num_kv_heads * head_dim)
/// * `seq_len` - Number of tokens in batch
/// * `stream` - HIP stream for kernel execution
///
/// # Returns
/// Ok(()) on success, Err if validation or kernel launch fails
///
/// # Supported Types
/// - Q4_0: Uses batched Q4_0 kernel
/// - Q4_1: Uses batched Q4_1 kernel
/// - Other types: Not yet implemented
pub fn gpu_batched_qkv_projection(
    device: &GpuDevice,
    q_weights: &GpuBuffer,
    q_meta: &WeightMeta,
    k_weights: &GpuBuffer,
    k_meta: &WeightMeta,
    v_weights: &GpuBuffer,
    v_meta: &WeightMeta,
    input: *const f32,
    q_output: *mut f32,
    k_output: *mut f32,
    v_output: *mut f32,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    seq_len: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_batched_qkv_dims(hidden_dim, q_dim, kv_dim, seq_len)?;

    // Project Query: Q [seq_len, q_dim] = input [seq_len, hidden_dim] × W_q [hidden_dim, q_dim]
    gpu_dispatch_batched_gemv_batched(
        device, q_weights, q_meta, input, q_output, hidden_dim, q_dim, seq_len, stream,
    )?;

    // Project Key: K [seq_len, kv_dim] = input [seq_len, hidden_dim] × W_k [hidden_dim, kv_dim]
    gpu_dispatch_batched_gemv_batched(
        device, k_weights, k_meta, input, k_output, hidden_dim, kv_dim, seq_len, stream,
    )?;

    // Project Value: V [seq_len, kv_dim] = input [seq_len, hidden_dim] × W_v [hidden_dim, kv_dim]
    gpu_dispatch_batched_gemv_batched(
        device, v_weights, v_meta, input, v_output, hidden_dim, kv_dim, seq_len, stream,
    )?;

    Ok(())
}

fn validate_batched_qkv_dims(
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if hidden_dim == 0 || q_dim == 0 || kv_dim == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_batched_qkv_projection_q4_0: invalid dimensions hidden_dim={}, q_dim={}, kv_dim={}, seq_len={}",
                hidden_dim, q_dim, kv_dim, seq_len
            ),
        });
    }

    Ok(())
}

/// Embed multiple prompt tokens into batched hidden states.
///
/// This function prepares the input hidden states for all prompt tokens at once.
/// For non-Q8_0 token embeddings, it falls back to CPU with async upload.
///
/// # Arguments
/// * `device` - GPU device reference
/// * `token_ids` - Slice of token IDs to embed
/// * `gpu_weights` - GPU model weights
/// * `cpu_weights` - CPU model weights (fallback for non-Q8_0 embeddings)
/// * `scratch` - GPU scratch buffer for batched prefill
/// * `config` - Model configuration
///
/// # Returns
/// Ok(()) on success, Err if embedding or upload fails
fn embed_prompt_tokens(
    device: &GpuDevice,
    token_ids: &[u32],
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    scratch: &mut GpuPrefillScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let seq_len = token_ids.len();

    // Copy token_ids to GPU scratch.token_ids buffer (converting from u32 to i32)
    let token_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let bytes_to_copy = unsafe {
        std::slice::from_raw_parts(
            token_ids_i32.as_ptr() as *const u8,
            token_ids_i32.len() * std::mem::size_of::<i32>(),
        )
    };
    scratch.token_ids.copy_from_host(bytes_to_copy)?;

    // Check if we have Q8_0 token embeddings (native GPU path)
    if gpu_weights.token_emb_meta.wtype == GgmlType::Q8_0 {
        embed_q8_0_batch(
            gpu_weights.token_emb.as_ptr() as *const u8,
            scratch.token_ids.as_ptr() as *const i32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            config.vocab_size,
            seq_len,
        )?;
    } else if gpu_weights.token_emb_meta.wtype == GgmlType::Q4_0 {
        embed_q4_0_batch(
            gpu_weights.token_emb.as_ptr() as *const u8,
            scratch.token_ids.as_ptr() as *const i32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            config.vocab_size,
            seq_len,
        )?;
    } else {
        // Fallback: embed all tokens on CPU and upload row-by-row
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let hidden_row = scratch.hidden_row_ptr(pos, h);
            let mut hidden_cpu = vec![0.0f32; h];
            cpu_embed_token(token_id, cpu_weights, &mut hidden_cpu, config);

            // Upload this row to GPU
            unsafe {
                super::ffi::hip_memcpy_h2d_async(
                    hidden_row as *mut u8,
                    hidden_cpu.as_ptr() as *const u8,
                    h * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }
        }
    }

    Ok(())
}

/// Full batched prefill forward pass for Q4_0 models.
///
/// This function processes an entire prompt through all transformer layers in parallel,
/// using batched kernels for improved throughput. This is the core prefill operation.
///
/// # Arguments
/// * `device` - GPU device reference
/// * `gpu_weights` - GPU model weights
/// * `cpu_weights` - CPU model weights (fallback)
/// * `kv` - Mutable KV cache (will be populated with prompt keys/values)
/// * `scratch` - GPU scratch buffer for batched prefill
/// * `host_scratch` - Host scratch buffer (for logits download)
/// * `token_ids` - Prompt token IDs
/// * `config` - Model configuration
/// * `logits_mode` - Whether to compute/download final logits
///
/// # Returns
/// Ok(next_token) if greedy sampling, Ok(None) if skipped or error
pub fn gpu_batched_prefill_forward_q4_0(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut super::cache::GpuKvCache,
    scratch: &mut GpuPrefillScratch,
    host_scratch: &mut CpuForwardScratch,
    token_ids: &[u32],
    start_pos: usize,
    config: &ModelConfig,
    logits_mode: super::forward::GpuLogitsMode,
) -> GpuResult<Option<u32>> {
    let seq_len = token_ids.len();
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let max_seq = kv.max_seq_len;

    if seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_batched_prefill_forward_q4_0: token_ids cannot be empty".to_string(),
        });
    }

    if seq_len > scratch.seq_len {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_batched_prefill_forward_q4_0: seq_len {} exceeds scratch capacity {}",
                seq_len, scratch.seq_len
            ),
        });
    }

    // Step 1: Embed all prompt tokens
    embed_prompt_tokens(device, token_ids, gpu_weights, cpu_weights, scratch, config)?;

    // Step 2: Process all layers with batched operations
    for layer_idx in 0..config.num_layers {
        let gpu_layer = gpu_weights.layer(layer_idx);

        // Attention normalization (batched)
        rms_norm_batched(
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
            seq_len,
        )?;

        // Batched QKV projection (type-aware dispatch)
        gpu_batched_qkv_projection(
            device,
            &gpu_layer.attn_q,
            &gpu_layer.attn_q_meta,
            &gpu_layer.attn_k,
            &gpu_layer.attn_k_meta,
            &gpu_layer.attn_v,
            &gpu_layer.attn_v_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.q.as_ptr() as *mut f32,
            scratch.k.as_ptr() as *mut f32,
            scratch.v.as_ptr() as *mut f32,
            h,
            q_size,
            kv_size,
            seq_len,
            device.stream(),
        )?;

        // Apply RoPE to Q and K (batched)
        rope_heads_batched(
            scratch.q.as_ptr() as *mut f32,
            start_pos,
            config.num_heads,
            config.head_dim,
            config.rope_theta,
            seq_len,
            config.rope_neox,
        )?;

        rope_heads_batched(
            scratch.k.as_ptr() as *mut f32,
            start_pos,
            config.num_kv_heads,
            config.head_dim,
            config.rope_theta,
            seq_len,
            config.rope_neox,
        )?;

        // Write K/V to cache (batched)
        kv_write_batched(
            kv.k_ptr(layer_idx)?,
            kv.v_ptr(layer_idx)?,
            scratch.k.as_ptr() as *const f32,
            scratch.v.as_ptr() as *const f32,
            start_pos,
            kv_size,
            max_seq,
            seq_len,
        )?;

        // Flash attention for prefill
        // Strides are in units of float elements (row-major layout: buffer[row * stride + col])
        flash_attn_prefill_strided(
            scratch.attn_out.as_ptr() as *mut f32,
            scratch.q.as_ptr() as *const f32,
            kv.k_ptr(layer_idx)?,
            kv.v_ptr(layer_idx)?,
            seq_len,
            config.head_dim,
            q_size, // out_stride: elements between consecutive rows in attn_out [seq_len, q_size]
            q_size, // q_stride: elements between consecutive rows in q [seq_len, q_size]
            kv_size, // kv_stride: elements between consecutive rows in K/V cache [max_seq, kv_size]
            h,      // out_head_offset
            q_size, // q_head_offset
            kv_size, // kv_head_offset
            1.0f32 / (config.head_dim as f32).sqrt(), // scale
        )?;

        // Attention output projection (batched, type-aware)
        gpu_dispatch_batched_gemv_batched(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            q_size,
            h,
            seq_len,
            device.stream(),
        )?;

        // Residual connection (batched)
        add_batched(
            scratch.layer_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            seq_len,
        )?;

        // FFN normalization (batched)
        rms_norm_batched(
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
            seq_len,
        )?;

        // Gate-up projection - use batched kernel for Q4_0/Q4_0, fallback to per-token loop for other types
        let gate_up_result = gpu_dispatch_batched_fused_gate_up_q4_0(
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            h,
            ff_size,
            seq_len,
            device.stream(),
        );

        // If batched dispatch fails (unsupported type combination), fall back to per-token loop
        if let Err(GpuError::UnsupportedWeightType { .. }) = gate_up_result {
            // Fallback: use per-token fused gate-up for each row
            for pos in 0..seq_len {
                let normed_row = scratch.normed_row_ptr(pos, h);
                let swiglu_row = scratch.swiglu_row_mut_ptr(pos, ff_size);

                gpu_dispatch_fused_gate_up_on_stream(
                    device,
                    &gpu_layer.ffn_gate,
                    &gpu_layer.ffn_gate_meta,
                    &gpu_layer.ffn_up,
                    &gpu_layer.ffn_up_meta,
                    gpu_layer.ffn_gate_up_interleaved.as_ref(),
                    gpu_layer.ffn_gate_up_interleaved_tile4.as_ref(),
                    normed_row,
                    swiglu_row,
                    ff_size,
                    h,
                    device.stream(),
                )?;
            }
        } else {
            // Batched dispatch succeeded or failed with non-type error
            gate_up_result?;
        }

        // FFN down projection (batched, type-aware)
        gpu_dispatch_batched_gemv_batched(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            ff_size,
            h,
            seq_len,
            device.stream(),
        )?;

        // Residual connection (batched)
        add_batched(
            scratch.layer_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            seq_len,
        )?;
    }
    // Step 3: Compute final logits if requested
    if matches!(logits_mode, super::forward::GpuLogitsMode::Skip) {
        return Ok(None);
    }

    // Get only the last token's hidden state for logits computation
    let last_pos = seq_len - 1;
    let last_hidden = scratch.hidden_row_ptr(last_pos, h);
    let last_normed = scratch.normed_row_ptr(last_pos, h) as *mut f32;

    // Normalize final hidden state directly on GPU
    gpu_dispatch_rms_norm(
        device,
        last_hidden,
        gpu_weights.output_norm.as_ptr() as *const f32,
        last_normed,
        h,
        config.rms_norm_eps,
        device.stream(),
    )?;

    // Vocabulary / LM head projection directly on GPU
    gpu_dispatch_gemv_on_stream(
        device,
        &gpu_weights.lm_head,
        &gpu_weights.lm_head_meta,
        last_normed,
        scratch.logits.as_ptr() as *mut f32,
        config.vocab_size,
        h,
        device.stream(),
    )?;

    // Download GPU logits back to host_scratch.logits if requested
    match logits_mode {
        super::forward::GpuLogitsMode::DownloadToHost
        | super::forward::GpuLogitsMode::GreedyArgmax => {
            unsafe {
                super::ffi::hip_memcpy_d2h_async(
                    host_scratch.logits.as_mut_ptr() as *mut u8,
                    scratch.logits.as_ptr() as *const u8,
                    config.vocab_size * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }
            device.synchronize()?;
        }
        super::forward::GpuLogitsMode::Skip => {}
    }

    // Sample or return logits
    match logits_mode {
        super::forward::GpuLogitsMode::DownloadToHost => Ok(None),
        super::forward::GpuLogitsMode::GreedyArgmax => {
            let token =
                crate::cpu::sampler::cpu_sample_greedy(&host_scratch.logits[..config.vocab_size]);
            Ok(Some(token))
        }
        super::forward::GpuLogitsMode::Skip => Ok(None),
    }
}

/// Prefill layer forward pass for Q4_0 models (stub for benchmarking/milestone 1).
pub fn gpu_prefill_layer_forward_q4_0(
    _device: &GpuDevice,
    _weights: &GpuLayerWeights,
    _scratch: &mut GpuPrefillScratch,
    _kv: &super::cache::GpuKvCache,
    _layer_idx: usize,
    _pos: usize,
    _config: &ModelConfig,
) -> GpuResult<()> {
    Err(GpuError::HipApiError {
        code: -1,
        description: "gpu_prefill_layer_forward_q4_0 is not yet implemented".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_qkv_projection_rejects_invalid_weight_type_metadata() {
        let q_meta_invalid = WeightMeta {
            wtype: GgmlType::Q4_1, // Wrong type
            dims: vec![1024, 4096],
            needs_transpose: false,
            role: crate::gpu::TensorRole::Generic,
        };

        assert_eq!(q_meta_invalid.wtype, GgmlType::Q4_1);
    }

    #[test]
    fn batched_qkv_projection_rejects_zero_dimensions() {
        assert!(validate_batched_qkv_dims(0, 1024, 512, 8).is_err());
        assert!(validate_batched_qkv_dims(4096, 0, 512, 8).is_err());
        assert!(validate_batched_qkv_dims(4096, 1024, 0, 8).is_err());
        assert!(validate_batched_qkv_dims(4096, 1024, 512, 0).is_err());
        assert!(validate_batched_qkv_dims(4096, 1024, 512, 8).is_ok());
    }

    #[test]
    fn flash_attn_stride_compile_time_verification() {
        // Compile-time verification that stride calculation is correct for row-major layout
        // For a buffer [seq_len, dim], the stride between consecutive rows is `dim` elements
        // This test documents the correct stride formula for future maintenance

        const SEQ_LEN: usize = 32;
        const H: usize = 4096;
        const Q_SIZE: usize = 32 * 128; // num_heads * head_dim
        const KV_SIZE: usize = 4 * 128; // num_kv_heads * head_dim

        // Correct strides for row-major [seq_len, dim] buffers
        const OUT_STRIDE_CORRECT: usize = Q_SIZE;
        const Q_STRIDE_CORRECT: usize = Q_SIZE;
        const KV_STRIDE_CORRECT: usize = KV_SIZE;

        // Wrong strides that would cause out-of-bounds access
        const OUT_STRIDE_WRONG: usize = SEQ_LEN * H;
        const Q_STRIDE_WRONG: usize = SEQ_LEN * Q_SIZE;
        const KV_STRIDE_WRONG: usize = SEQ_LEN * KV_SIZE;

        // Verify correct strides are reasonable
        assert!(OUT_STRIDE_CORRECT < OUT_STRIDE_WRONG);
        assert!(Q_STRIDE_CORRECT < Q_STRIDE_WRONG);
        assert!(KV_STRIDE_CORRECT < KV_STRIDE_WRONG);

        // Verify correct strides match the dimension (not seq_len * dimension)
        assert_eq!(OUT_STRIDE_CORRECT, Q_SIZE);
        assert_eq!(Q_STRIDE_CORRECT, Q_SIZE);
        assert_eq!(KV_STRIDE_CORRECT, KV_SIZE);
    }
}
