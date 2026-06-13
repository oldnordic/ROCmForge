use crate::config::ModelConfig;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::{gelu_on_stream, silu_on_stream, mul_on_stream};
use crate::gpu::ops::{gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_gemv_on_stream, gpu_dispatch_gemv_with_fallback_on_stream, gpu_dispatch_rms_norm};
use crate::gpu::weights::GpuLayerWeights;
use super::gpu_dispatch_moe_ffn_on_stream;
use crate::gpu::forward::utils::residual_add_inplace;

/// Native GPU forward pass for shortconv layers.
pub(crate) fn gpu_shortconv_native_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    _pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let sc = gpu_layer
        .shortconv
        .as_ref()
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "Shortconv weights not found in layer".to_string(),
        })?;

    let h = config.hidden_size;
    let eps = config.rms_norm_eps;
    let stream = device.stream();
    let l_cache = config.shortconv_l_cache.unwrap_or(3);

    // 1. RMSNorm
    gpu_dispatch_rms_norm(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        stream,
    )?;

    // 2. in_proj: [h] → [3h]
    // We use scratch.gate for [3h]
    let ff_size = config.intermediate_size;
    if ff_size < 3 * h {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("intermediate_size {} too small for shortconv in_proj 3*h {}", ff_size, 3 * h),
        });
    }

    gpu_dispatch_gemv_on_stream(
        device,
        &sc.in_proj,
        &sc.in_proj_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.gate.as_ptr() as *mut f32,
        3 * h,
        h,
        stream,
    )?;

    // 3. Shortconv logic: B*x, causal conv1d, C*out
    let conv_state_ptr = kv.conv_state_ptr(layer_idx)?
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "shortconv state not allocated".to_string(),
        })?;

    crate::gpu::kernels::shortconv::dispatch_shortconv(
        scratch.swiglu.as_ptr() as *mut f32,
        scratch.gate.as_ptr() as *const f32,
        sc.conv.as_ptr() as *const f32,
        conv_state_ptr,
        h,
        l_cache,
        stream,
    )?;

    // 4. out_proj: [h] → [h]
    // We use swiglu as input, layer_out as output.
    gpu_dispatch_gemv_on_stream(
        device,
        &sc.out_proj,
        &sc.out_proj_meta,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        h,
        h,
        stream,
    )?;

    // 5. Residual
    residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;

    // --- FFN Part ---

    // 6. FFN RMSNorm
    gpu_dispatch_rms_norm(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        stream,
    )?;

    // 7. FFN (dense or MoE)
    if gpu_dispatch_moe_ffn_on_stream(device, gpu_layer, scratch, h, ff_size, config)? {
        return Ok(());
    }

    if let (Some(gate_buf), Some(gate_meta)) = (
        gpu_layer.ffn_gate.as_ref(),
        gpu_layer.ffn_gate_meta.as_ref(),
    ) {
        if gpu_layer.ffn_gate_svd.is_some() || gpu_layer.ffn_up_svd.is_some() {
            gpu_dispatch_gemv_with_fallback_on_stream(
                device,
                gate_buf,
                gate_meta,
                gpu_layer.ffn_gate_svd.as_ref(),
                gpu_layer.ffn_gate_sparse.as_ref(),
                gpu_layer.ffn_gate_mpo.as_ref(),
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                ff_size,
                h,
                scratch.svd_scratch.as_ptr() as *mut f32,
                stream,
            )?;
            gpu_dispatch_gemv_with_fallback_on_stream(
                device,
                &gpu_layer.ffn_up,
                &gpu_layer.ffn_up_meta,
                gpu_layer.ffn_up_svd.as_ref(),
                gpu_layer.ffn_up_sparse.as_ref(),
                gpu_layer.ffn_up_mpo.as_ref(),
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                ff_size,
                h,
                scratch.svd_scratch.as_ptr() as *mut f32,
                stream,
            )?;
            silu_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                ff_size,
                stream,
            )?;
            mul_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                ff_size,
                stream,
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
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                scratch.swiglu.as_ptr() as *mut f32,
                ff_size,
                h,
                stream,
            )?;
        }
    } else {
        // Standard FFN (non-SwiGLU): up -> gelu
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            h,
            stream,
        )?;
        gelu_on_stream(
            scratch.swiglu.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            stream,
        )?;
    }

    // 8. FFN output projection + residual
    gpu_dispatch_gemv_with_fallback_on_stream(
        device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        gpu_layer.ffn_down_svd.as_ref(),
        gpu_layer.ffn_down_sparse.as_ref(),
        gpu_layer.ffn_down_mpo.as_ref(),
        scratch.swiglu.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        h,
        ff_size,
        scratch.svd_scratch.as_ptr() as *mut f32,
        stream,
    )?;
    residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;

    Ok(())
}
