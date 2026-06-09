//! Per-layer prefill forward passes: stub layer and batched SSM layer.

use super::cache::GpuPrefillScratch;
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::kernels::{add_on_stream, mul_on_stream, rms_norm_batched, silu_on_stream};
use super::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_gemv_with_fallback_on_stream,
};
use super::ops_batched::{
    gpu_dispatch_batched_fused_gate_up_on_stream, gpu_dispatch_batched_gemv_batched,
};
use super::weights::{GpuLayerWeights, GpuModelWeights};
use crate::config::ModelConfig;
use crate::loader::GgmlType;

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
        scratch.attn_out.as_ptr() as *const f32,
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
                gpu_dispatch_gemv_with_fallback_on_stream(
                    device,
                    &gpu_layer.ffn_gate,
                    &gpu_layer.ffn_gate_meta,
                    gpu_layer.ffn_gate_svd.as_ref(),
                    gpu_layer.ffn_gate_sparse.as_ref(),
                    gpu_layer.ffn_gate_mpo.as_ref(),
                    normed_row,
                    gate_row,
                    ff_size,
                    h,
                    t_scratch,
                    stream,
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
                    std::ptr::null_mut(),
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

    let _ = start_pos; // used by caller for KV cache positioning
    Ok(())
}
