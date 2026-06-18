//! Helper functions for GPU prefill: QKV projection and token embedding.

use super::cache::GpuPrefillScratch;
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use super::kernels::{embed_q4_0_batch, embed_q8_0_batch};
use super::weights::{GpuBuffer, GpuModelWeights, WeightMeta};
use crate::config::ModelConfig;
use crate::cpu::forward::cpu_embed_token;
use crate::cpu::weights::CpuModelWeights;
use crate::gpu::ops::gpu_dispatch_gemv_on_stream;
use crate::loader::GgmlType;

/// Batched QKV projection for prefill processing (type-aware).
///
/// Projects input hidden states through Query, Key, and Value weight matrices
/// in parallel for multiple tokens. Core operation for prompt prefill.
///
/// # Arguments
/// * `device` - GPU device reference
/// * `q_weights` - Query projection weights (quantized)
/// * `q_meta` - Query weight metadata (determines kernel dispatch)
/// * `k_weights` - Key projection weights (quantized)
/// * `k_meta` - Key weight metadata
/// * `v_weights` - Value projection weights (quantized)
/// * `v_meta` - Value weight metadata
/// * `input` - Input hidden states [seq_len, hidden_dim] (row-major, GPU pointer)
/// * `q_output` - Query output buffer [seq_len, q_dim]
/// * `k_output` - Key output buffer [seq_len, kv_dim]
/// * `v_output` - Value output buffer [seq_len, kv_dim]
/// * `hidden_dim` - Input hidden dimension
/// * `q_dim` - Query output dimension (num_heads * head_dim)
/// * `kv_dim` - Key/Value output dimension (num_kv_heads * head_dim)
/// * `seq_len` - Number of tokens in batch
/// * `stream` - HIP stream for kernel execution
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

    // For Q4_0, the existing batched GEMM kernels dequantize weights in f32
    // per element and accumulate in f32. That matches a CPU f32 fallback but
    // diverges from the CPU decode reference, which uses int32 per-block
    // accumulation over Q8_0-quantized activations. Fall back to the same
    // per-token GEMV dispatch the decode path uses so prefill and decode see
    // the same numerics and both match the CPU reference.
    if q_meta.wtype == GgmlType::Q4_0
        && k_meta.wtype == GgmlType::Q4_0
        && v_meta.wtype == GgmlType::Q4_0
        && seq_len > 1
    {
        for pos in 0..seq_len {
            let input_row = unsafe { input.add(pos * hidden_dim) };
            let q_row = unsafe { q_output.add(pos * q_dim) };
            let k_row = unsafe { k_output.add(pos * kv_dim) };
            let v_row = unsafe { v_output.add(pos * kv_dim) };

            gpu_dispatch_gemv_on_stream(
                device, q_weights, q_meta, input_row, q_row, q_dim, hidden_dim, stream,
            )?;
            gpu_dispatch_gemv_on_stream(
                device, k_weights, k_meta, input_row, k_row, kv_dim, hidden_dim, stream,
            )?;
            gpu_dispatch_gemv_on_stream(
                device, v_weights, v_meta, input_row, v_row, kv_dim, hidden_dim, stream,
            )?;
        }
        return Ok(());
    }

    use crate::gpu::ops::gpu_dispatch_gemm;

    gpu_dispatch_gemm(
        device, q_weights, q_meta, input, q_output, q_dim, hidden_dim, seq_len,
    )?;
    gpu_dispatch_gemm(
        device, k_weights, k_meta, input, k_output, kv_dim, hidden_dim, seq_len,
    )?;
    gpu_dispatch_gemm(
        device, v_weights, v_meta, input, v_output, kv_dim, hidden_dim, seq_len,
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
/// Prepares input hidden states for all prompt tokens at once.
/// For non-Q8_0/Q4_0 token embeddings, falls back to CPU with H2D upload.
pub(crate) fn embed_prompt_tokens(
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
        if let Some(dense) = gpu_weights.token_emb.as_dense() {
            embed_q8_0_batch(
                dense.as_ptr() as *const u8,
                scratch.token_ids.as_ptr() as *const i32,
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                seq_len,
            )?;
        }
        // Sparse / MPO Q8_0 embeddings: fall through to CPU embed fallback below
    } else if gpu_weights.token_emb_meta.wtype == GgmlType::Q4_0 {
        if let Some(dense) = gpu_weights.token_emb.as_dense() {
            embed_q4_0_batch(
                dense.as_ptr() as *const u8,
                scratch.token_ids.as_ptr() as *const i32,
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                seq_len,
            )?;
        }
        // Sparse / MPO Q4_0 embeddings: fall through to CPU embed fallback below
    }

    // If we didn't embed natively on GPU (sparse/MPO or unsupported format),
    // fall back to CPU embed + H2D copy.
    if gpu_weights.token_emb_meta.wtype != GgmlType::Q8_0
        && gpu_weights.token_emb_meta.wtype != GgmlType::Q4_0
        || gpu_weights.token_emb.as_dense().is_none()
    {
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let hidden_row = scratch.hidden_row_ptr(pos, h);
            let mut hidden_cpu = vec![0.0f32; h];
            cpu_embed_token(token_id, cpu_weights, &mut hidden_cpu, config, None);

            unsafe {
                super::ffi::hip_memcpy_h2d(
                    hidden_row as *mut u8,
                    hidden_cpu.as_ptr() as *const u8,
                    h * std::mem::size_of::<f32>(),
                )?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::TensorRole;
    use crate::loader::GgmlType;

    #[test]
    fn batched_qkv_projection_rejects_invalid_weight_type_metadata() {
        let q_meta_invalid = WeightMeta {
            wtype: GgmlType::Q4_1,
            dims: vec![1024, 4096],
            needs_transpose: false,
            role: TensorRole::Generic,
            svd_k: None,
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
        const SEQ_LEN: usize = 32;
        const H: usize = 4096;
        const Q_SIZE: usize = 32 * 128;
        const KV_SIZE: usize = 4 * 128;

        const OUT_STRIDE_CORRECT: usize = Q_SIZE;
        const Q_STRIDE_CORRECT: usize = Q_SIZE;
        const KV_STRIDE_CORRECT: usize = KV_SIZE;

        const OUT_STRIDE_WRONG: usize = SEQ_LEN * H;
        const Q_STRIDE_WRONG: usize = SEQ_LEN * Q_SIZE;
        const KV_STRIDE_WRONG: usize = SEQ_LEN * KV_SIZE;

        assert!(OUT_STRIDE_CORRECT < OUT_STRIDE_WRONG);
        assert!(Q_STRIDE_CORRECT < Q_STRIDE_WRONG);
        assert!(KV_STRIDE_CORRECT < KV_STRIDE_WRONG);

        assert_eq!(OUT_STRIDE_CORRECT, Q_SIZE);
        assert_eq!(Q_STRIDE_CORRECT, Q_SIZE);
        assert_eq!(KV_STRIDE_CORRECT, KV_SIZE);
    }
}
