//! GPU prefill forward path for batched prompt processing.
//!
//! This module implements batched QKV projection for prompt prefill, where multiple tokens
//! are processed in parallel. This is distinct from decode which processes one token
//! at a time with cached key-value states.

use super::cache::GpuPrefillScratch;
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::kernels::{
    add_on_stream, flash_attn_prefill_strided, gelu_on_stream, mul_on_stream, rms_norm_batched,
    rope_heads_batched, silu_on_stream,
};
use super::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_gemv_on_stream,
    gpu_dispatch_gemv_with_fallback_on_stream, gpu_dispatch_mpo_apply_on_stream,
    gpu_dispatch_rms_norm, gpu_dispatch_sparse_csr_gemv_on_stream,
};
use super::ops_batched::gpu_dispatch_batched_fused_gate_up_on_stream;
use super::prefill_debug::{
    compute_layer0_cpu_reference, download_gpu_buffer, max_abs_error_slice, CpuLayer0Activations,
};
use super::prefill_helpers::{embed_prompt_tokens, gpu_batched_qkv_projection};
use super::prefill_layer::gpu_prefill_ssm_layer_on_stream;
use super::weights::{GpuLayerType, GpuLayerWeights, GpuModelWeights};
use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::weights::CpuModelWeights;
use crate::loader::GgmlType;

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
pub fn gpu_batched_prefill_forward(
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
            description: "gpu_batched_prefill_forward: token_ids cannot be empty".to_string(),
        });
    }

    if seq_len > scratch.seq_len {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_batched_prefill_forward: seq_len {} exceeds scratch capacity {}",
                seq_len, scratch.seq_len
            ),
        });
    }

    let debug_prefill = true;
    let cpu_acts: Option<CpuLayer0Activations> = if debug_prefill {
        Some(compute_layer0_cpu_reference(token_ids, cpu_weights, config))
    } else {
        None
    };

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

        match gpu_layer.layer_type {
            GpuLayerType::Ssm => {
                gpu_prefill_ssm_layer_on_stream(
                    device, gpu_layer, kv, scratch, layer_idx, start_pos, config,
                )?;
                continue;
            }
            GpuLayerType::Shortconv => {
                super::prefill_layer::gpu_prefill_shortconv_layer_on_stream(
                    device, gpu_layer, kv, scratch, layer_idx, start_pos, config,
                )?;
                continue;
            }
            _ => {}
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

        // Attention output projection
        crate::gpu::ops::gpu_dispatch_gemm(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            q_size,
            seq_len,
        )?;

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
        if let (Some(gate_buf), Some(gate_meta)) = (
            gpu_layer.ffn_gate.as_ref(),
            gpu_layer.ffn_gate_meta.as_ref(),
        ) {
            if false {
                gate_up_result = gpu_dispatch_batched_fused_gate_up_on_stream(
                    device,
                    gate_buf,
                    gate_meta,
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

                        gpu_dispatch_gemv_with_fallback_on_stream(
                            device,
                            gate_buf,
                            gate_meta,
                            gpu_layer.ffn_gate_svd.as_ref(),
                            gpu_layer.ffn_gate_sparse.as_ref(),
                            gpu_layer.ffn_gate_mpo.as_ref(),
                            normed_row,
                            gate_row,
                            ff_size,
                            h,
                            t_scratch,
                            device.stream(),
                        )?;
                        gpu_dispatch_gemv_with_fallback_on_stream(
                            device,
                            &gpu_layer.ffn_up,
                            &gpu_layer.ffn_up_meta,
                            gpu_layer.ffn_up_svd.as_ref(),
                            gpu_layer.ffn_up_sparse.as_ref(),
                            gpu_layer.ffn_up_mpo.as_ref(),
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
                            gate_buf,
                            gate_meta,
                            &gpu_layer.ffn_up,
                            &gpu_layer.ffn_up_meta,
                            gpu_layer.ffn_gate_up_interleaved.as_ref(),
                            gpu_layer.ffn_gate_up_interleaved_tile4.as_ref(),
                            normed_row,
                            std::ptr::null_mut(),
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
        } else {
            // Standard FFN (non-SwiGLU)
            if gpu_layer.ffn_up_svd.is_none()
                && gpu_layer.ffn_up_sparse.is_none()
                && gpu_layer.ffn_up_mpo.is_none()
            {
                crate::gpu::ops::gpu_dispatch_gemm(
                    device,
                    &gpu_layer.ffn_up,
                    &gpu_layer.ffn_up_meta,
                    scratch.normed.as_ptr() as *const f32,
                    scratch.swiglu.as_ptr() as *mut f32,
                    ff_size,
                    h,
                    seq_len,
                )?;
                gelu_on_stream(
                    scratch.swiglu.as_ptr() as *const f32,
                    scratch.swiglu.as_ptr() as *mut f32,
                    ff_size * seq_len,
                    device.stream(),
                )?;
            } else {
                // per-token up -> gelu
                for pos in 0..seq_len {
                    let normed_row = scratch.normed_row_ptr(pos, h);
                    let swiglu_row = scratch.swiglu_row_mut_ptr(pos, ff_size);
                    let t_scratch =
                        unsafe { (scratch.svd_scratch.as_ptr() as *mut f32).add(pos * 32) };
                    gpu_dispatch_gemv_with_fallback_on_stream(
                        device,
                        &gpu_layer.ffn_up,
                        &gpu_layer.ffn_up_meta,
                        gpu_layer.ffn_up_svd.as_ref(),
                        gpu_layer.ffn_up_sparse.as_ref(),
                        gpu_layer.ffn_up_mpo.as_ref(),
                        normed_row,
                        swiglu_row,
                        ff_size,
                        h,
                        t_scratch,
                        device.stream(),
                    )?;
                    gelu_on_stream(
                        swiglu_row as *const f32,
                        swiglu_row,
                        ff_size,
                        device.stream(),
                    )?;
                }
            }
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
            let cpu_swiglu = &cpu_acts
                .as_ref()
                .expect("invariant: cpu_acts populated before validation block")
                .swiglu;
            println!(
                "DEBUG PREFILL: layer0 SwiGLU[0..4] CPU: {:?}",
                &cpu_swiglu[0..4]
            );
            println!(
                "DEBUG PREFILL: layer0 SwiGLU[0..4] GPU: {:?}",
                &gpu_swiglu[0..4]
            );
            let mut max_err = 0.0f32;
            let mut max_err_idx = 0;
            for (idx, (c, g)) in cpu_swiglu.iter().zip(gpu_swiglu.iter()).enumerate() {
                let err = (c - g).abs();
                if err > max_err {
                    max_err = err;
                    max_err_idx = idx;
                }
            }
            println!(
                "DEBUG PREFILL: Max SwiGLU error at index {}: CPU={}, GPU={}, diff={}",
                max_err_idx, cpu_swiglu[max_err_idx], gpu_swiglu[max_err_idx], max_err
            );
            println!(
                "DEBUG PREFILL: layer0 FFN Gate-Up max_abs_error: Gate={}, SwiGLU={}",
                err_gate, err_swiglu
            );
        }

        // FFN down projection
        if gpu_layer.ffn_down_svd.is_none()
            && gpu_layer.ffn_down_sparse.is_none()
            && gpu_layer.ffn_down_mpo.is_none()
        {
            crate::gpu::ops::gpu_dispatch_gemm(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                ff_size,
                seq_len,
            )?;
        } else {
            // Fallback to loop for complex layouts
            for pos in 0..seq_len {
                let swiglu_row =
                    unsafe { (scratch.swiglu.as_ptr() as *const f32).add(pos * ff_size) };
                let layer_out_row = scratch.layer_out_row_mut_ptr(pos, h);
                let t_scratch = unsafe { (scratch.svd_scratch.as_ptr() as *mut f32).add(pos * 32) };
                gpu_dispatch_gemv_with_fallback_on_stream(
                    device,
                    &gpu_layer.ffn_down,
                    &gpu_layer.ffn_down_meta,
                    gpu_layer.ffn_down_svd.as_ref(),
                    gpu_layer.ffn_down_sparse.as_ref(),
                    gpu_layer.ffn_down_mpo.as_ref(),
                    swiglu_row,
                    layer_out_row,
                    h,
                    ff_size,
                    t_scratch,
                    device.stream(),
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
