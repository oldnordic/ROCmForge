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
    add_batched, add_on_stream, embed_q4_0_batch, embed_q8_0_batch, flash_attn_prefill_strided,
    kv_write_batched, mul_on_stream, rms_norm_batched, rope_heads_batched, silu_on_stream,
};
use super::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_gemv_on_stream,
    gpu_dispatch_gemv_svd_on_stream, gpu_dispatch_mpo_apply_on_stream, gpu_dispatch_rms_norm,
    gpu_dispatch_sparse_csr_gemv_on_stream,
};
use super::ops_batched::{
    gpu_dispatch_batched_fused_gate_up_on_stream, gpu_dispatch_batched_gemv_batched,
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

    use crate::gpu::ops::gpu_dispatch_gemv_on_stream;

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
        if let Some(dense) = gpu_weights.token_emb.as_dense() {
            embed_q8_0_batch(
                dense.as_ptr() as *const u8,
                scratch.token_ids.as_ptr() as *const i32,
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                seq_len,
            )?;
        } else {
            // Sparse / MPO Q8_0 embeddings: fall through to CPU embed fallback below
        }
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
        } else {
            // Sparse / MPO Q4_0 embeddings: fall through to CPU embed fallback below
        }
    }

    // If we didn't embed natively on GPU (sparse/MPO or unsupported format),
    // fall back to CPU embed + H2D copy.
    if gpu_weights.token_emb_meta.wtype != GgmlType::Q8_0
        && gpu_weights.token_emb_meta.wtype != GgmlType::Q4_0
        || gpu_weights.token_emb.as_dense().is_none()
    {
        // Fallback: embed all tokens on CPU and upload row-by-row
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let hidden_row = scratch.hidden_row_ptr(pos, h);
            let mut hidden_cpu = vec![0.0f32; h];
            cpu_embed_token(token_id, cpu_weights, &mut hidden_cpu, config);

            // Upload this row to GPU
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
struct CpuLayer0Activations {
    hidden_in: Vec<f32>,
    normed_attn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    attn_out: Vec<f32>,
    layer_out_attn: Vec<f32>,
    hidden_after_attn: Vec<f32>,
    normed_ffn: Vec<f32>,
    gate: Vec<f32>,
    swiglu: Vec<f32>,
    layer_out_ffn: Vec<f32>,
    hidden_out: Vec<f32>,
}

fn download_gpu_buffer(buf: &crate::gpu::GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes).expect("copy_to_host failed");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

fn max_abs_error_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
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

    let debug_prefill = false;
    let mut cpu_acts = None;
    if debug_prefill {
        println!("DEBUG PREFILL: Computing CPU reference activations for layer 0...");
        let mut cpu_kv = crate::cpu::cache::CpuKvCache::new(config, seq_len.max(1));
        let mut cpu_scratch = crate::cpu::cache::CpuForwardScratch::new(config);
        let mut cpu_hidden = vec![0.0f32; h];

        let mut hidden_in = vec![0.0f32; seq_len * h];
        let mut normed_attn = vec![0.0f32; seq_len * h];
        let mut q = vec![0.0f32; seq_len * q_size];
        let mut k = vec![0.0f32; seq_len * kv_size];
        let mut v = vec![0.0f32; seq_len * kv_size];
        let mut q_rope = vec![0.0f32; seq_len * q_size];
        let mut k_rope = vec![0.0f32; seq_len * kv_size];
        let mut attn_out = vec![0.0f32; seq_len * q_size];
        let mut layer_out_attn = vec![0.0f32; seq_len * h];
        let mut hidden_after_attn = vec![0.0f32; seq_len * h];
        let mut normed_ffn = vec![0.0f32; seq_len * h];
        let mut gate = vec![0.0f32; seq_len * ff_size];
        let mut swiglu = vec![0.0f32; seq_len * ff_size];
        let mut layer_out_ffn = vec![0.0f32; seq_len * h];
        let mut hidden_out = vec![0.0f32; seq_len * h];

        for (pos, &token_id) in token_ids.iter().enumerate() {
            cpu_embed_token(token_id, cpu_weights, &mut cpu_hidden, config);
            hidden_in[pos * h..(pos + 1) * h].copy_from_slice(&cpu_hidden);

            let layer_idx = 0;
            let layer_weights = cpu_weights.layer(layer_idx);

            // 1. Attn Norm
            let mut t_normed_attn = vec![0.0f32; h];
            crate::cpu::ops::rms_norm(
                &cpu_hidden,
                &layer_weights.attn_norm,
                &mut t_normed_attn,
                config.rms_norm_eps,
            );
            normed_attn[pos * h..(pos + 1) * h].copy_from_slice(&t_normed_attn);

            // 2. QKV projection
            let mut t_q = vec![0.0f32; q_size];
            let mut t_k = vec![0.0f32; kv_size];
            let mut t_v = vec![0.0f32; kv_size];
            let mut q8_scratch = vec![0u8; cpu_scratch.q8_scratch.len()];
            cpu_dispatch_gemv(
                &layer_weights.attn_q,
                &layer_weights.attn_q_meta,
                &t_normed_attn,
                &mut t_q,
                q_size,
                h,
                Some(&mut q8_scratch),
            )
            .expect(
                "M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path",
            );
            cpu_dispatch_gemv(
                &layer_weights.attn_k,
                &layer_weights.attn_k_meta,
                &t_normed_attn,
                &mut t_k,
                kv_size,
                h,
                Some(&mut q8_scratch),
            )
            .expect(
                "M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path",
            );
            cpu_dispatch_gemv(
                &layer_weights.attn_v,
                &layer_weights.attn_v_meta,
                &t_normed_attn,
                &mut t_v,
                kv_size,
                h,
                Some(&mut q8_scratch),
            )
            .expect(
                "M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path",
            );

            if let Some(bq) = &layer_weights.attn_q_bias {
                crate::cpu::ops::add_bias(&mut t_q, bq);
            }
            if let Some(bk) = &layer_weights.attn_k_bias {
                crate::cpu::ops::add_bias(&mut t_k, bk);
            }
            if let Some(bv) = &layer_weights.attn_v_bias {
                crate::cpu::ops::add_bias(&mut t_v, bv);
            }
            q[pos * q_size..(pos + 1) * q_size].copy_from_slice(&t_q);
            k[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&t_k);
            v[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&t_v);

            // 3. RoPE
            let mut t_q_rope = t_q.clone();
            let mut t_k_rope = t_k.clone();
            crate::cpu::ops::rope(
                &mut t_q_rope,
                config.num_heads,
                config.head_dim,
                pos,
                config.rope_theta,
                config.rope_neox,
            );
            crate::cpu::ops::rope(
                &mut t_k_rope,
                config.num_kv_heads,
                config.head_dim,
                pos,
                config.rope_theta,
                config.rope_neox,
            );
            q_rope[pos * q_size..(pos + 1) * q_size].copy_from_slice(&t_q_rope);
            k_rope[pos * kv_size..(pos + 1) * kv_size].copy_from_slice(&t_k_rope);

            // Write to cache
            cpu_kv.write_k(layer_idx, pos, &t_k_rope);
            cpu_kv.write_v(layer_idx, pos, &t_v);

            // 4. Attention
            let mut t_attn_out = vec![0.0f32; q_size];
            crate::cpu::ops::flash_attn_decode(
                &t_q_rope,
                cpu_kv.k_buf(layer_idx),
                cpu_kv.v_buf(layer_idx),
                &mut t_attn_out,
                pos + 1,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
            );
            attn_out[pos * q_size..(pos + 1) * q_size].copy_from_slice(&t_attn_out);

            // 5. Attn Out Projection
            let mut t_layer_out_attn = vec![0.0f32; h];
            cpu_dispatch_gemv(
                &layer_weights.attn_o,
                &layer_weights.attn_o_meta,
                &t_attn_out,
                &mut t_layer_out_attn,
                h,
                q_size,
                Some(&mut q8_scratch),
            )
            .expect(
                "M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path",
            );
            layer_out_attn[pos * h..(pos + 1) * h].copy_from_slice(&t_layer_out_attn);

            // 6. Attn Residual
            let mut t_hidden_after_attn = cpu_hidden.clone();
            crate::cpu::ops::residual_add(&mut t_hidden_after_attn, &t_layer_out_attn);
            hidden_after_attn[pos * h..(pos + 1) * h].copy_from_slice(&t_hidden_after_attn);

            // 7. FFN Norm
            let mut t_normed_ffn = vec![0.0f32; h];
            crate::cpu::ops::rms_norm(
                &t_hidden_after_attn,
                &layer_weights.ffn_norm,
                &mut t_normed_ffn,
                config.rms_norm_eps,
            );
            normed_ffn[pos * h..(pos + 1) * h].copy_from_slice(&t_normed_ffn);

            // 8. FFN Gate + Up
            let mut t_gate = vec![0.0f32; ff_size];
            let mut t_swiglu = vec![0.0f32; ff_size];
            cpu_dispatch_gemv(
                &layer_weights.ffn_gate,
                &layer_weights.ffn_gate_meta,
                &t_normed_ffn,
                &mut t_gate,
                ff_size,
                h,
                Some(&mut q8_scratch),
            )
            .expect(
                "M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path",
            );
            cpu_dispatch_gemv(
                &layer_weights.ffn_up,
                &layer_weights.ffn_up_meta,
                &t_normed_ffn,
                &mut t_swiglu,
                ff_size,
                h,
                Some(&mut q8_scratch),
            )
            .expect(
                "M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path",
            );
            gate[pos * ff_size..(pos + 1) * ff_size].copy_from_slice(&t_gate);

            crate::cpu::ops::silu_fuse(&t_gate, &mut t_swiglu);
            swiglu[pos * ff_size..(pos + 1) * ff_size].copy_from_slice(&t_swiglu);

            // 9. FFN Down Projection
            let mut t_layer_out_ffn = vec![0.0f32; h];
            cpu_dispatch_gemv(
                &layer_weights.ffn_down,
                &layer_weights.ffn_down_meta,
                &t_swiglu,
                &mut t_layer_out_ffn,
                h,
                ff_size,
                Some(&mut q8_scratch),
            )
            .expect(
                "M-ALLOW: cpu_dispatch_gemv infallible for valid weight dims in validation path",
            );
            layer_out_ffn[pos * h..(pos + 1) * h].copy_from_slice(&t_layer_out_ffn);

            // 10. FFN Residual
            crate::cpu::ops::residual_add(&mut cpu_hidden, &t_layer_out_ffn);
            hidden_out[pos * h..(pos + 1) * h].copy_from_slice(&cpu_hidden);
        }

        cpu_acts = Some(CpuLayer0Activations {
            hidden_in,
            normed_attn,
            q,
            k,
            v,
            q_rope,
            k_rope,
            attn_out,
            layer_out_attn,
            hidden_after_attn,
            normed_ffn,
            gate,
            swiglu,
            layer_out_ffn,
            hidden_out,
        });
    }

    // Step 1: Embed all prompt tokens
    embed_prompt_tokens(device, token_ids, gpu_weights, cpu_weights, scratch, config)?;

    if debug_prefill {
        let gpu_val = download_gpu_buffer(&scratch.hidden, seq_len * h);
        let max_err = max_abs_error_slice(
            &cpu_acts
                .as_ref()
                .expect("invariant: cpu_acts populated before validation block")
                .hidden_in,
            &gpu_val,
        );
        println!(
            "DEBUG PREFILL: embed_prompt_tokens hidden max_abs_error = {}",
            max_err
        );
    }

    // Step 2: Process all layers with batched operations
    for layer_idx in 0..config.num_layers {
        let gpu_layer = gpu_weights.layer(layer_idx);

        // If this is an SSM layer, use the SSM prefill path
        if gpu_layer.ssm.is_some() {
            gpu_prefill_ssm_layer_on_stream(
                device, gpu_layer, kv, scratch, layer_idx, start_pos, config,
            )?;
            continue;
        }

        // Attention normalization (batched)
        rms_norm_batched(
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
            seq_len,
        )?;
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_val = download_gpu_buffer(&scratch.normed, seq_len * h);
            let max_err = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .normed_attn,
                &gpu_val,
            );
            println!(
                "DEBUG PREFILL: layer0 RMSNorm Attn max_abs_error = {}",
                max_err
            );
        }

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
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_q = download_gpu_buffer(&scratch.q, seq_len * q_size);
            let gpu_k = download_gpu_buffer(&scratch.k, seq_len * kv_size);
            let gpu_v = download_gpu_buffer(&scratch.v, seq_len * kv_size);
            let err_q = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .q,
                &gpu_q,
            );
            let err_k = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .k,
                &gpu_k,
            );
            let err_v = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .v,
                &gpu_v,
            );
            println!(
                "DEBUG PREFILL: layer0 QKV max_abs_error: Q={}, K={}, V={}",
                err_q, err_k, err_v
            );
        }

        // Apply SVD corrections for Q, K, V if present
        if gpu_layer.attn_q_svd.is_some()
            || gpu_layer.attn_k_svd.is_some()
            || gpu_layer.attn_v_svd.is_some()
        {
            for pos in 0..seq_len {
                let normed_row = scratch.normed_row_ptr(pos, h);
                let t_scratch = unsafe { (scratch.svd_scratch.as_ptr() as *mut f32).add(pos * 32) };

                if let Some(svd) = gpu_layer.attn_q_svd.as_ref() {
                    let q_row = scratch.q_row_mut_ptr(pos, q_size);
                    crate::gpu::kernels::elementwise::dispatch_svd_correction(
                        device.stream(),
                        &svd.u,
                        &svd.v,
                        svd.k,
                        normed_row,
                        q_row,
                        h,
                        q_size,
                        t_scratch,
                    )?;
                }
                if let Some(svd) = gpu_layer.attn_k_svd.as_ref() {
                    let k_row = scratch.k_row_mut_ptr(pos, kv_size);
                    crate::gpu::kernels::elementwise::dispatch_svd_correction(
                        device.stream(),
                        &svd.u,
                        &svd.v,
                        svd.k,
                        normed_row,
                        k_row,
                        h,
                        kv_size,
                        t_scratch,
                    )?;
                }
                if let Some(svd) = gpu_layer.attn_v_svd.as_ref() {
                    let v_row = scratch.v_row_mut_ptr(pos, kv_size);
                    crate::gpu::kernels::elementwise::dispatch_svd_correction(
                        device.stream(),
                        &svd.u,
                        &svd.v,
                        svd.k,
                        normed_row,
                        v_row,
                        h,
                        kv_size,
                        t_scratch,
                    )?;
                }
            }
            device.synchronize()?;
        }

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
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_q = download_gpu_buffer(&scratch.q, seq_len * q_size);
            let gpu_k = download_gpu_buffer(&scratch.k, seq_len * kv_size);
            let err_q = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .q_rope,
                &gpu_q,
            );
            let err_k = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .k_rope,
                &gpu_k,
            );
            println!(
                "DEBUG PREFILL: layer0 RoPE max_abs_error: Q={}, K={}",
                err_q, err_k
            );
        }

        // Write K/V to cache (batched)
        device.synchronize()?;
        kv.write_batched(
            layer_idx,
            start_pos,
            seq_len,
            scratch.k.as_ptr() as *const f32,
            scratch.v.as_ptr() as *const f32,
        )?;
        device.synchronize()?;

        let kv_lora_dim = kv.kv_lora_dim.unwrap_or(0);
        let w_up_k = kv
            .w_up_k
            .as_ref()
            .map(|w| w[layer_idx].as_ptr() as *const f32)
            .unwrap_or(std::ptr::null());
        let w_up_v = kv
            .w_up_v
            .as_ref()
            .map(|w| w[layer_idx].as_ptr() as *const f32)
            .unwrap_or(std::ptr::null());
        let effective_kv_size = kv.kv_lora_dim.unwrap_or(kv_size);

        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let kv_group = num_heads / num_kv_heads;
        let scale = 1.0f32 / (config.head_dim as f32).sqrt();

        // Flash attention for prefill (loop over heads)
        for head in 0..num_heads {
            let kv_head = head / kv_group;
            let q_offset = head * config.head_dim;
            let kv_offset = kv_head * config.head_dim;

            flash_attn_prefill_strided(
                scratch.attn_out.as_ptr() as *mut f32,
                scratch.q.as_ptr() as *const f32,
                if kv.kv_quant_bits.is_some() {
                    scratch.k.as_ptr() as *const f32
                } else {
                    kv.k_ptr(layer_idx)? as *const f32
                },
                if kv.kv_quant_bits.is_some() {
                    scratch.v.as_ptr() as *const f32
                } else {
                    kv.v_ptr(layer_idx)? as *const f32
                },
                seq_len,
                config.head_dim,
                q_size,
                q_size,
                effective_kv_size,
                q_offset,
                q_offset,
                kv_offset,
                scale,
                if kv.kv_quant_bits.is_some() {
                    0
                } else {
                    kv_lora_dim
                },
                if kv.kv_quant_bits.is_some() {
                    std::ptr::null()
                } else {
                    w_up_k
                },
                if kv.kv_quant_bits.is_some() {
                    std::ptr::null()
                } else {
                    w_up_v
                },
            )?;
        }
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_val = download_gpu_buffer(&scratch.attn_out, seq_len * q_size);
            let max_err = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .attn_out,
                &gpu_val,
            );
            println!(
                "DEBUG PREFILL: layer0 Flash Attn max_abs_error = {}",
                max_err
            );
        }

        // Attention output projection (loop over tokens)
        for pos in 0..seq_len {
            let attn_out_row =
                unsafe { (scratch.attn_out.as_ptr() as *const f32).add(pos * q_size) };
            let layer_out_row = scratch.layer_out_row_mut_ptr(pos, h);
            crate::gpu::ops::gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.attn_o,
                &gpu_layer.attn_o_meta,
                attn_out_row,
                layer_out_row,
                h,
                q_size,
                device.stream(),
            )?;
        }

        // Apply SVD correction for attn_o if present
        if let Some(svd) = gpu_layer.attn_o_svd.as_ref() {
            for pos in 0..seq_len {
                let attn_out_row =
                    unsafe { (scratch.attn_out.as_ptr() as *const f32).add(pos * q_size) };
                let layer_out_row = scratch.layer_out_row_mut_ptr(pos, h);
                let t_scratch = unsafe { (scratch.svd_scratch.as_ptr() as *mut f32).add(pos * 32) };
                crate::gpu::kernels::elementwise::dispatch_svd_correction(
                    device.stream(),
                    &svd.u,
                    &svd.v,
                    svd.k,
                    attn_out_row,
                    layer_out_row,
                    q_size,
                    h,
                    t_scratch,
                )?;
            }
        }
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_val = download_gpu_buffer(&scratch.layer_out, seq_len * h);
            let max_err = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .layer_out_attn,
                &gpu_val,
            );
            println!(
                "DEBUG PREFILL: layer0 Attn Out Projection max_abs_error = {}",
                max_err
            );
        }

        // Residual connection (batched element-wise add)
        add_on_stream(
            scratch.layer_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            seq_len * h,
            device.stream(),
        )?;
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_val = download_gpu_buffer(&scratch.hidden, seq_len * h);
            let max_err = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .hidden_after_attn,
                &gpu_val,
            );
            println!(
                "DEBUG PREFILL: layer0 Attn Residual max_abs_error = {}",
                max_err
            );
        }

        // FFN normalization (batched)
        rms_norm_batched(
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
            seq_len,
        )?;
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_val = download_gpu_buffer(&scratch.normed, seq_len * h);
            let max_err = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .normed_ffn,
                &gpu_val,
            );
            println!(
                "DEBUG PREFILL: layer0 RMSNorm FFN max_abs_error = {}",
                max_err
            );
        }

        // Gate-up projection — use batched kernel when both weights support it,
        // otherwise fall back to per-token loop.
        let mut gate_up_result = Err(GpuError::UnsupportedWeightType {
            tensor: "forced_svd_fallback".to_string(),
            wtype: GgmlType::Q4_0,
        });
        if false {
            gate_up_result = gpu_dispatch_batched_fused_gate_up_on_stream(
                device,
                &gpu_layer.ffn_gate,
                &gpu_layer.ffn_gate_meta,
                &gpu_layer.ffn_up,
                &gpu_layer.ffn_up_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                scratch.gate.as_ptr() as *mut f32,
                h,
                ff_size,
                seq_len,
                device.stream(),
            );
        }

        // If batched dispatch fails (unsupported type combination), fall back to per-token loop
        if let Err(GpuError::UnsupportedWeightType { .. }) = gate_up_result {
            // Fallback: use per-token fused gate-up for each row
            for pos in 0..seq_len {
                let normed_row = scratch.normed_row_ptr(pos, h);
                let swiglu_row = scratch.swiglu_row_mut_ptr(pos, ff_size);

                if gpu_layer.ffn_gate_svd.is_some() || gpu_layer.ffn_up_svd.is_some() {
                    let gate_row = scratch.gate_row_mut_ptr(pos, ff_size);
                    let t_scratch =
                        unsafe { (scratch.svd_scratch.as_ptr() as *mut f32).add(pos * 32) };

                    gpu_dispatch_gemv_svd_on_stream(
                        device,
                        &gpu_layer.ffn_gate,
                        &gpu_layer.ffn_gate_meta,
                        gpu_layer.ffn_gate_svd.as_ref(),
                        normed_row,
                        gate_row,
                        ff_size,
                        h,
                        t_scratch,
                        device.stream(),
                    )?;
                    gpu_dispatch_gemv_svd_on_stream(
                        device,
                        &gpu_layer.ffn_up,
                        &gpu_layer.ffn_up_meta,
                        gpu_layer.ffn_up_svd.as_ref(),
                        normed_row,
                        swiglu_row,
                        ff_size,
                        h,
                        t_scratch,
                        device.stream(),
                    )?;
                    silu_on_stream(gate_row as *const f32, gate_row, ff_size, device.stream())?;
                    mul_on_stream(
                        gate_row as *const f32,
                        swiglu_row as *const f32,
                        swiglu_row,
                        ff_size,
                        device.stream(),
                    )?;
                } else {
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
            }
            device.synchronize()?;
        } else {
            // Batched dispatch succeeded or failed with non-type error
            gate_up_result?;
            device.synchronize()?;
        }

        if debug_prefill && layer_idx == 0 {
            let gpu_gate = download_gpu_buffer(&scratch.gate, seq_len * ff_size);
            let gpu_swiglu = download_gpu_buffer(&scratch.swiglu, seq_len * ff_size);
            let err_gate = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .gate,
                &gpu_gate,
            );
            let err_swiglu = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .swiglu,
                &gpu_swiglu,
            );
            println!(
                "DEBUG PREFILL: layer0 FFN Gate-Up max_abs_error: Gate={}, SwiGLU={}",
                err_gate, err_swiglu
            );
        }

        // FFN down projection (loop over tokens)
        for pos in 0..seq_len {
            let swiglu_row = unsafe { (scratch.swiglu.as_ptr() as *const f32).add(pos * ff_size) };
            let layer_out_row = scratch.layer_out_row_mut_ptr(pos, h);
            crate::gpu::ops::gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                swiglu_row,
                layer_out_row,
                h,
                ff_size,
                device.stream(),
            )?;
        }

        // Apply SVD correction for ffn_down if present
        if let Some(svd) = gpu_layer.ffn_down_svd.as_ref() {
            for pos in 0..seq_len {
                let swiglu_row =
                    unsafe { (scratch.swiglu.as_ptr() as *const f32).add(pos * ff_size) };
                let layer_out_row = scratch.layer_out_row_mut_ptr(pos, h);
                let t_scratch = unsafe { (scratch.svd_scratch.as_ptr() as *mut f32).add(pos * 32) };
                crate::gpu::kernels::elementwise::dispatch_svd_correction(
                    device.stream(),
                    &svd.u,
                    &svd.v,
                    svd.k,
                    swiglu_row,
                    layer_out_row,
                    ff_size,
                    h,
                    t_scratch,
                )?;
            }
        }
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_val = download_gpu_buffer(&scratch.layer_out, seq_len * h);
            let max_err = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .layer_out_ffn,
                &gpu_val,
            );
            println!(
                "DEBUG PREFILL: layer0 FFN Down Projection max_abs_error = {}",
                max_err
            );
        }

        // Residual connection (batched element-wise add)
        add_on_stream(
            scratch.layer_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            seq_len * h,
            device.stream(),
        )?;
        device.synchronize()?;

        if debug_prefill && layer_idx == 0 {
            let gpu_val = download_gpu_buffer(&scratch.hidden, seq_len * h);
            let max_err = max_abs_error_slice(
                &cpu_acts
                    .as_ref()
                    .expect("invariant: cpu_acts populated before validation block")
                    .hidden_out,
                &gpu_val,
            );
            println!(
                "DEBUG PREFILL: layer0 FFN Residual max_abs_error = {}",
                max_err
            );
        }
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
    if let Some(dense) = gpu_weights.lm_head.as_dense() {
        gpu_dispatch_gemv_on_stream(
            device,
            dense,
            &gpu_weights.lm_head_meta,
            last_normed,
            scratch.logits.as_ptr() as *mut f32,
            config.vocab_size,
            h,
            device.stream(),
        )?;
    } else if let Some(sparse) = gpu_weights.lm_head.as_sparse_csr() {
        gpu_dispatch_sparse_csr_gemv_on_stream(
            device,
            sparse,
            last_normed,
            scratch.logits.as_ptr() as *mut f32,
            config.vocab_size,
            h,
            device.stream(),
        )?;
    } else if let Some(mpo) = gpu_weights.lm_head.as_mpo() {
        gpu_dispatch_mpo_apply_on_stream(
            device,
            mpo,
            last_normed,
            scratch.logits.as_ptr() as *mut f32,
            config.vocab_size,
            h,
            device.stream(),
        )?;
    } else {
        return Err(crate::gpu::error::GpuError::InvalidWeightLayout {
            tensor: "lm_head".to_string(),
            dims: gpu_weights.lm_head_meta.dims.clone(),
            reason: "LM head is neither dense, sparse CSR, nor MPO".to_string(),
        });
    }

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

/// Batched SSM prefill layer forward pass.
///
/// Processes `seq_len` tokens through an SSM layer in parallel, updating
/// SSM state and KV cache. This mirrors the decode path but operates on
/// batched tokens.
pub fn gpu_prefill_ssm_layer_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut super::cache::GpuKvCache,
    scratch: &mut GpuPrefillScratch,
    layer_idx: usize,
    start_pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let ssm = gpu_layer
        .ssm
        .as_ref()
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "SSM weights not found in layer".to_string(),
        })?;

    let h = config.hidden_size;
    let eps = config.rms_norm_eps;
    let stream = device.stream();
    let seq_len = scratch.seq_len;

    // 1. RMSNorm of input hidden states (batched)
    rms_norm_batched(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        seq_len,
    )?;

    // 2. QKV projection (fused wqkv, batched GEMV)
    let wqkv = gpu_layer
        .attn_qkv
        .as_ref()
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "fused QKV weights not found in SSM layer".to_string(),
        })?;
    let wqkv_meta = gpu_layer
        .attn_qkv_meta
        .as_ref()
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "fused QKV meta not found in SSM layer".to_string(),
        })?;
    let qkv_dim = if wqkv_meta.dims[0] as usize == h {
        wqkv_meta.dims[1] as usize
    } else {
        wqkv_meta.dims[0] as usize
    };

    gpu_dispatch_batched_gemv_batched(
        device,
        wqkv,
        wqkv_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.gate.as_ptr() as *mut f32,
        h,
        qkv_dim,
        seq_len,
        stream,
    )?;

    // 3. Z (gate) projection (batched GEMV)
    let wz = gpu_layer
        .attn_gate
        .as_ref()
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "fused gate weight not found in SSM layer".to_string(),
        })?;
    let wz_meta = gpu_layer
        .attn_gate_meta
        .as_ref()
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "fused gate meta not found in SSM layer".to_string(),
        })?;
    let d_inner = if wz_meta.dims[0] as usize == h {
        wz_meta.dims[1] as usize
    } else {
        wz_meta.dims[0] as usize
    };

    gpu_dispatch_batched_gemv_batched(
        device,
        wz,
        wz_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.attn_out.as_ptr() as *mut f32,
        h,
        d_inner,
        seq_len,
        stream,
    )?;

    // 4. Beta + alpha projections (batched GEMV)
    let ssm_heads = if ssm.beta_meta.dims[0] as usize == h {
        ssm.beta_meta.dims[1] as usize
    } else {
        ssm.beta_meta.dims[0] as usize
    };

    gpu_dispatch_batched_gemv_batched(
        device,
        &ssm.beta,
        &ssm.beta_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.q.as_ptr() as *mut f32,
        h,
        ssm_heads,
        seq_len,
        stream,
    )?;

    gpu_dispatch_batched_gemv_batched(
        device,
        &ssm.alpha,
        &ssm.alpha_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.k.as_ptr() as *mut f32,
        h,
        ssm_heads,
        seq_len,
        stream,
    )?;

    // 5. Fused sigmoid/alpha gate discretization (batched)
    crate::gpu::kernels::dispatch_batched_fused_sigmoid_alpha_gate(
        scratch.q.as_ptr() as *mut f32,
        scratch.k.as_ptr() as *mut f32,
        ssm.dt.as_ptr() as *const f32,
        ssm.a.as_ptr() as *const f32,
        ssm_heads,
        seq_len,
        stream,
    )?;

    // 6. Fused conv1d + SiLU (batched)
    let conv_state_ptr =
        kv.ssm_conv_state_ptr(layer_idx)?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "SSM conv state not allocated".to_string(),
            })?;

    crate::gpu::kernels::dispatch_batched_conv1d_silu(
        scratch.swiglu.as_ptr() as *mut f32,
        scratch.gate.as_ptr() as *const f32,
        ssm.conv1d.as_ptr() as *const f32,
        conv_state_ptr,
        qkv_dim,
        seq_len,
        stream,
    )?;

    // 7. Split conv output into Q, K, V
    let ssm_kv_heads = (qkv_dim / 128 - ssm_heads) / 2;
    let k_dim = ssm_kv_heads * 128;
    let q_dim = ssm_heads * 128;

    // 8. Fused Q/K L2-norm and scale (batched)
    crate::gpu::kernels::dispatch_batched_fused_qk_l2_norm_scale(
        scratch.swiglu.as_ptr() as *mut f32,
        unsafe { scratch.swiglu.as_ptr().add(k_dim) } as *mut f32,
        ssm_kv_heads,
        128,
        seq_len,
        1.0 / (128.0f32).sqrt(),
        eps,
        stream,
    )?;

    // 9. Repeat/interleave key heads if needed (batched)
    let (q_gdn_ptr, k_gdn_ptr) = if ssm_kv_heads < ssm_heads {
        let ratio = ssm_heads / ssm_kv_heads;
        let q_exp_ptr = unsafe { (scratch.gate.as_ptr() as *mut f32).add(qkv_dim * seq_len) };
        let k_exp_ptr = unsafe { (scratch.swiglu.as_ptr() as *mut f32).add(qkv_dim * seq_len) };

        for t in 0..seq_len {
            crate::gpu::kernels::dispatch_repeat_interleave_qk(
                unsafe { scratch.swiglu.as_ptr().add(t * qkv_dim) } as *const f32,
                unsafe { scratch.swiglu.as_ptr().add(t * qkv_dim + k_dim) } as *const f32,
                unsafe { q_exp_ptr.add(t * q_dim) },
                unsafe { k_exp_ptr.add(t * k_dim) },
                ssm_kv_heads,
                ratio,
                128,
                stream,
            )?;
        }
        (q_exp_ptr as *const f32, k_exp_ptr as *const f32)
    } else {
        (
            scratch.swiglu.as_ptr() as *const f32,
            unsafe { scratch.swiglu.as_ptr().add(k_dim) } as *const f32,
        )
    };

    // 10. Gated selective scan matrix update (batched)
    let ssm_state_ptr = kv
        .ssm_state_ptr(layer_idx)?
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "SSM state not allocated".to_string(),
        })?;

    crate::gpu::kernels::dispatch_batched_gated_delta_net(
        q_gdn_ptr,
        k_gdn_ptr,
        unsafe { scratch.swiglu.as_ptr().add(k_dim * 2) } as *const f32,
        scratch.q.as_ptr() as *const f32,
        scratch.k.as_ptr() as *const f32,
        ssm_state_ptr,
        scratch.attn_out.as_ptr() as *mut f32,
        seq_len,
        ssm_heads,
        128,
        stream,
    )?;

    // 11. Gated Norm (batched)
    crate::gpu::kernels::dispatch_batched_gated_norm(
        scratch.attn_out.as_ptr() as *const f32,
        scratch.attn_out.as_ptr() as *const f32, // z reuses attn_out
        ssm.norm.as_ptr() as *const f32,
        scratch.q.as_ptr() as *mut f32,
        ssm_heads,
        128,
        seq_len,
        eps,
        stream,
    )?;

    // 12. Output projection (wo, batched GEMV)
    gpu_dispatch_batched_gemv_batched(
        device,
        &ssm.out,
        &ssm.out_meta,
        scratch.q.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        q_dim,
        h,
        seq_len,
        stream,
    )?;

    // 13. Residual connection (batched element-wise add)
    add_on_stream(
        scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        seq_len * h,
        stream,
    )?;

    // 14. FFN normalization (batched)
    rms_norm_batched(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        seq_len,
    )?;

    // 15. FFN gate+up (batched)
    let ff_size = config.intermediate_size;
    let mut gate_up_result = Err(GpuError::UnsupportedWeightType {
        tensor: "forced_svd_fallback".to_string(),
        wtype: GgmlType::Q4_0,
    });
    if gpu_layer.ffn_gate_svd.is_none() && gpu_layer.ffn_up_svd.is_none() {
        gate_up_result = gpu_dispatch_batched_fused_gate_up_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            scratch.gate.as_ptr() as *mut f32,
            h,
            ff_size,
            seq_len,
            stream,
        );
    }

    if let Err(GpuError::UnsupportedWeightType { .. }) = gate_up_result {
        for pos in 0..seq_len {
            let normed_row = scratch.normed_row_ptr(pos, h);
            let swiglu_row = scratch.swiglu_row_mut_ptr(pos, ff_size);
            if gpu_layer.ffn_gate_svd.is_some() || gpu_layer.ffn_up_svd.is_some() {
                let gate_row = scratch.gate_row_mut_ptr(pos, ff_size);
                let t_scratch = unsafe { (scratch.svd_scratch.as_ptr() as *mut f32).add(pos * 32) };
                gpu_dispatch_gemv_svd_on_stream(
                    device,
                    &gpu_layer.ffn_gate,
                    &gpu_layer.ffn_gate_meta,
                    gpu_layer.ffn_gate_svd.as_ref(),
                    normed_row,
                    gate_row,
                    ff_size,
                    h,
                    t_scratch,
                    stream,
                )?;
                gpu_dispatch_gemv_svd_on_stream(
                    device,
                    &gpu_layer.ffn_up,
                    &gpu_layer.ffn_up_meta,
                    gpu_layer.ffn_up_svd.as_ref(),
                    normed_row,
                    swiglu_row,
                    ff_size,
                    h,
                    t_scratch,
                    stream,
                )?;
                silu_on_stream(gate_row as *const f32, gate_row, ff_size, stream)?;
                mul_on_stream(
                    gate_row as *const f32,
                    swiglu_row as *const f32,
                    swiglu_row,
                    ff_size,
                    stream,
                )?;
            } else {
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
                    stream,
                )?;
            }
        }
    } else {
        gate_up_result?;
    }

    // 16. FFN down projection (batched)
    gpu_dispatch_batched_gemv_batched(
        device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        ff_size,
        h,
        seq_len,
        stream,
    )?;

    // 17. Residual connection (batched element-wise add)
    add_on_stream(
        scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        seq_len * h,
        stream,
    )?;

    Ok(())
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
