use super::super::utils::{cpu_fallback_gemv_and_upload, ensure_size, residual_add_inplace};
use super::gpu_dispatch_moe_ffn_on_stream;
use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::weights::CpuLayerWeights;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::{mul_on_stream, silu_on_stream};
use crate::gpu::ops::{
    gpu_dispatch_gemv_on_stream, gpu_dispatch_gemv_residual_on_stream,
    gpu_dispatch_gemv_svd_on_stream, gpu_dispatch_rms_norm, supports_gemv_type,
};
use crate::gpu::weights::GpuLayerWeights;

pub(super) fn gpu_layer_forward_ssm_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    cpu_layer: Option<&CpuLayerWeights>,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    mut cpu_scratch: Option<&mut CpuForwardScratch>,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let ssm = gpu_layer
        .ssm
        .as_ref()
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "SSM weights not found in layer".to_string(),
        })?;

    let h = config.hidden_size; // 2048
    let eps = config.rms_norm_eps;
    let stream = device.stream();

    // 1. RMSNorm of input hidden states
    gpu_dispatch_rms_norm(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        stream,
    )?;

    // 2. QKV projection (linear projection wqkv)
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

    // Dynamically retrieve actual QKV output dimension directly from weight shape
    let qkv_dim = if wqkv_meta.dims[0] as usize == h {
        wqkv_meta.dims[1] as usize
    } else {
        wqkv_meta.dims[0] as usize
    }; // e.g. 8192
    let qkv_ptr = scratch.gate.as_ptr() as *mut f32;

    if supports_gemv_type(wqkv_meta.wtype) {
        gpu_dispatch_gemv_svd_on_stream(
            device,
            wqkv,
            wqkv_meta,
            gpu_layer.attn_qkv_svd.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            qkv_ptr,
            qkv_dim,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
    } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
        let cpu_w = cpu_l
            .attn_qkv
            .as_ref()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "attn_qkv CPU weight missing".to_string(),
            })?;
        let cpu_m = cpu_l
            .attn_qkv_meta
            .as_ref()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "attn_qkv CPU meta missing".to_string(),
            })?;
        ensure_size(&mut cpu_s.normed, h);
        ensure_size(&mut cpu_s.gate, qkv_dim);
        cpu_fallback_gemv_and_upload(
            "attn_qkv",
            cpu_w,
            cpu_m,
            &scratch.normed,
            &mut cpu_s.normed,
            &mut cpu_s.gate,
            &mut scratch.gate,
            qkv_dim,
            h,
            &mut cpu_s.q8_scratch,
        )?;
    } else {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "attn_qkv".to_string(),
            wtype: wqkv_meta.wtype,
        });
    }

    // 3. Z (gate) projection
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

    // Dynamically retrieve actual inner gate output dimension directly from weight shape
    let d_inner = if wz_meta.dims[0] as usize == h {
        wz_meta.dims[1] as usize
    } else {
        wz_meta.dims[0] as usize
    }; // e.g. 4096
    let z_ptr = scratch.attn_out.as_ptr() as *mut f32; // size 4096

    if supports_gemv_type(wz_meta.wtype) {
        gpu_dispatch_gemv_svd_on_stream(
            device,
            wz,
            wz_meta,
            gpu_layer.attn_gate_svd.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            z_ptr,
            d_inner,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
    } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
        let cpu_w = cpu_l
            .attn_gate
            .as_ref()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "attn_gate CPU weight missing".to_string(),
            })?;
        let cpu_m = cpu_l
            .attn_gate_meta
            .as_ref()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "attn_gate CPU meta missing".to_string(),
            })?;
        ensure_size(&mut cpu_s.normed, h);
        ensure_size(&mut cpu_s.attn_out, d_inner);
        cpu_fallback_gemv_and_upload(
            "attn_gate",
            cpu_w,
            cpu_m,
            &scratch.normed,
            &mut cpu_s.normed,
            &mut cpu_s.attn_out,
            &mut scratch.attn_out,
            d_inner,
            h,
            &mut cpu_s.q8_scratch,
        )?;
    } else {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "attn_gate".to_string(),
            wtype: wz_meta.wtype,
        });
    }

    // 4. Beta + alpha projections
    let beta_out_ptr = scratch.q.as_ptr() as *mut f32;
    let alpha_out_ptr = scratch.k.as_ptr() as *mut f32;

    // Dynamically retrieve actual SSM heads count directly from weight shape
    let ssm_heads = if ssm.beta_meta.dims[0] as usize == h {
        ssm.beta_meta.dims[1] as usize
    } else {
        ssm.beta_meta.dims[0] as usize
    }; // e.g. 32

    if supports_gemv_type(ssm.beta_meta.wtype) {
        gpu_dispatch_gemv_svd_on_stream(
            device,
            &ssm.beta,
            &ssm.beta_meta,
            ssm.beta_svd.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            beta_out_ptr,
            ssm_heads,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
    } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
        let cpu_ssm = cpu_l.ssm.as_ref().ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "SSM CPU weights missing".to_string(),
        })?;
        ensure_size(&mut cpu_s.normed, h);
        ensure_size(&mut cpu_s.q, ssm_heads);
        cpu_fallback_gemv_and_upload(
            "ssm.beta",
            &cpu_ssm.beta,
            &cpu_ssm.beta_meta,
            &scratch.normed,
            &mut cpu_s.normed,
            &mut cpu_s.q,
            &mut scratch.q,
            ssm_heads,
            h,
            &mut cpu_s.q8_scratch,
        )?;
    } else {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "ssm.beta".to_string(),
            wtype: ssm.beta_meta.wtype,
        });
    }

    if supports_gemv_type(ssm.alpha_meta.wtype) {
        gpu_dispatch_gemv_svd_on_stream(
            device,
            &ssm.alpha,
            &ssm.alpha_meta,
            ssm.alpha_svd.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            alpha_out_ptr,
            ssm_heads,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
    } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
        let cpu_ssm = cpu_l.ssm.as_ref().ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "SSM CPU weights missing".to_string(),
        })?;
        ensure_size(&mut cpu_s.normed, h);
        ensure_size(&mut cpu_s.k, ssm_heads);
        cpu_fallback_gemv_and_upload(
            "ssm.alpha",
            &cpu_ssm.alpha,
            &cpu_ssm.alpha_meta,
            &scratch.normed,
            &mut cpu_s.normed,
            &mut cpu_s.k,
            &mut scratch.k,
            ssm_heads,
            h,
            &mut cpu_s.q8_scratch,
        )?;
    } else {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "ssm.alpha".to_string(),
            wtype: ssm.alpha_meta.wtype,
        });
    }

    // 5. Fused sigmoid/alpha gate discretization
    crate::gpu::kernels::dispatch_fused_sigmoid_alpha_gate(
        beta_out_ptr,
        alpha_out_ptr,
        ssm.dt.as_ptr() as *const f32,
        ssm.a.as_ptr() as *const f32,
        ssm_heads,
        1, // batch size = 1
        stream,
    )?;

    // 6. Fused conv1d + SiLU
    let conv_out_ptr = scratch.swiglu.as_ptr() as *mut f32;
    let conv_state_ptr =
        kv.ssm_conv_state_ptr(layer_idx)?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "SSM conv state not allocated".to_string(),
            })?;

    crate::gpu::kernels::dispatch_conv1d_silu(
        conv_out_ptr,
        qkv_ptr,
        ssm.conv1d.as_ptr() as *const f32,
        conv_state_ptr,
        qkv_dim,
        stream,
    )?;

    // 7. Split conv output into Q, K, V
    let ssm_kv_heads = (qkv_dim / 128 - ssm_heads) / 2; // e.g. 16
    let k_dim = ssm_kv_heads * 128; // e.g. 2048
    let q_dim = ssm_heads * 128; // e.g. 4096
    let q_part_ptr = conv_out_ptr;
    let k_part_ptr = unsafe { conv_out_ptr.add(k_dim) };
    let v_part_ptr = unsafe { conv_out_ptr.add(k_dim * 2) };

    // 8. Fused Q/K L2-norm and scale
    crate::gpu::kernels::dispatch_fused_qk_l2_norm_scale(
        q_part_ptr,
        k_part_ptr,
        ssm_kv_heads,
        128, // head dim
        1,   // batch size = 1
        1.0 / (128.0f32).sqrt(),
        eps,
        stream,
    )?;

    // 9. Repeat/interleave key heads if ssm_kv_heads < ssm_heads
    let (q_gdn_ptr, k_gdn_ptr) = if ssm_kv_heads < ssm_heads {
        let ratio = ssm_heads / ssm_kv_heads;
        let q_exp_ptr = unsafe { (scratch.gate.as_ptr() as *mut f32).add(qkv_dim) };
        let k_exp_ptr = unsafe { (scratch.swiglu.as_ptr() as *mut f32).add(qkv_dim) };

        crate::gpu::kernels::dispatch_repeat_interleave_qk(
            q_part_ptr,
            k_part_ptr,
            q_exp_ptr,
            k_exp_ptr,
            ssm_kv_heads,
            ratio,
            128, // head dim
            stream,
        )?;
        (q_exp_ptr as *const f32, k_exp_ptr as *const f32)
    } else {
        (q_part_ptr as *const f32, k_part_ptr as *const f32)
    };

    // 10. Gated selective scan matrix update
    let ssm_state_ptr = kv
        .ssm_state_ptr(layer_idx)?
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "SSM state not allocated".to_string(),
        })?;

    let attn_out_ptr = scratch.q.as_ptr() as *mut f32;

    crate::gpu::kernels::dispatch_gated_delta_net(
        q_gdn_ptr,
        k_gdn_ptr,
        v_part_ptr,
        alpha_out_ptr,
        beta_out_ptr,
        ssm_state_ptr,
        attn_out_ptr,
        1, // n_tokens = 1
        ssm_heads,
        128, // head dim
        stream,
    )?;

    // 11. Gated Norm
    let normed_out_ptr = scratch.attn_out.as_ptr() as *mut f32;

    crate::gpu::kernels::dispatch_gated_norm(
        attn_out_ptr,
        z_ptr,
        ssm.norm.as_ptr() as *const f32,
        normed_out_ptr,
        ssm_heads,
        128, // head_dim
        1,   // batch_size = 1
        eps,
        stream,
    )?;

    // 12. Output projection (wo)
    let attn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
        device,
        &ssm.out,
        &ssm.out_meta,
        normed_out_ptr,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        q_dim, // in_dim: SSM output (ssm_heads * head_dim)
        h,     // out_dim: hidden size
        stream,
    )?;
    if !attn_residual_fused {
        if supports_gemv_type(ssm.out_meta.wtype) {
            gpu_dispatch_gemv_svd_on_stream(
                device,
                &ssm.out,
                &ssm.out_meta,
                ssm.out_svd.as_ref(),
                normed_out_ptr,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                q_dim,
                scratch.svd_scratch.as_ptr() as *mut f32,
                stream,
            )?;
        } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
            let cpu_ssm = cpu_l.ssm.as_ref().ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "SSM CPU weights missing".to_string(),
            })?;
            ensure_size(&mut cpu_s.attn_out, q_dim);
            ensure_size(&mut cpu_s.layer_out, h);
            cpu_fallback_gemv_and_upload(
                "ssm.out",
                &cpu_ssm.out,
                &cpu_ssm.out_meta,
                &scratch.attn_out,
                &mut cpu_s.attn_out,
                &mut cpu_s.layer_out,
                &mut scratch.layer_out,
                h,
                q_dim,
                &mut cpu_s.q8_scratch,
            )?;
        } else {
            return Err(GpuError::UnsupportedWeightType {
                tensor: "ssm.out".to_string(),
                wtype: ssm.out_meta.wtype,
            });
        }
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    // 13. FFN execution
    let ff_size = config.intermediate_size;
    gpu_dispatch_rms_norm(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        stream,
    )?;
    if gpu_dispatch_moe_ffn_on_stream(device, gpu_layer, scratch, h, ff_size)? {
        return Ok(());
    }

    if gpu_layer.ffn_gate_svd.is_some() || gpu_layer.ffn_up_svd.is_some() {
        gpu_dispatch_gemv_svd_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            gpu_layer.ffn_gate_svd.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            scratch.gate.as_ptr() as *mut f32,
            ff_size,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
        gpu_dispatch_gemv_svd_on_stream(
            device,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            gpu_layer.ffn_up_svd.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
    } else {
        if supports_gemv_type(gpu_layer.ffn_gate_meta.wtype)
            && supports_gemv_type(gpu_layer.ffn_up_meta.wtype)
        {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.ffn_gate,
                &gpu_layer.ffn_gate_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                ff_size,
                h,
                stream,
            )?;
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
        } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
            ensure_size(&mut cpu_s.normed, h);
            ensure_size(&mut cpu_s.gate, ff_size);
            cpu_fallback_gemv_and_upload(
                "ffn_gate",
                &cpu_l.ffn_gate,
                &cpu_l.ffn_gate_meta,
                &scratch.normed,
                &mut cpu_s.normed,
                &mut cpu_s.gate,
                &mut scratch.gate,
                ff_size,
                h,
                &mut cpu_s.q8_scratch,
            )?;

            ensure_size(&mut cpu_s.swiglu, ff_size);
            cpu_fallback_gemv_and_upload(
                "ffn_up",
                &cpu_l.ffn_up,
                &cpu_l.ffn_up_meta,
                &scratch.normed,
                &mut cpu_s.normed,
                &mut cpu_s.swiglu,
                &mut scratch.swiglu,
                ff_size,
                h,
                &mut cpu_s.q8_scratch,
            )?;
        } else {
            return Err(GpuError::UnsupportedWeightType {
                tensor: "ffn_gate".to_string(),
                wtype: gpu_layer.ffn_gate_meta.wtype,
            });
        }
    }

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

    let mut ffn_residual_fused = false;
    if gpu_layer.ffn_down_svd.is_none() {
        ffn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            ff_size, // in_dim: post-swiglu intermediate
            h,       // out_dim: hidden size
            stream,
        )?;
    }
    if !ffn_residual_fused {
        if supports_gemv_type(gpu_layer.ffn_down_meta.wtype) {
            gpu_dispatch_gemv_svd_on_stream(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                gpu_layer.ffn_down_svd.as_ref(),
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                ff_size,
                scratch.svd_scratch.as_ptr() as *mut f32,
                stream,
            )?;
        } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
            ensure_size(&mut cpu_s.swiglu, ff_size);
            ensure_size(&mut cpu_s.layer_out, h);
            cpu_fallback_gemv_and_upload(
                "ffn_down",
                &cpu_l.ffn_down,
                &cpu_l.ffn_down_meta,
                &scratch.swiglu,
                &mut cpu_s.swiglu,
                &mut cpu_s.layer_out,
                &mut scratch.layer_out,
                h,
                ff_size,
                &mut cpu_s.q8_scratch,
            )?;
        } else {
            return Err(GpuError::UnsupportedWeightType {
                tensor: "ffn_down".to_string(),
                wtype: gpu_layer.ffn_down_meta.wtype,
            });
        }
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    Ok(())
}
