use crate::config::ModelConfig;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::GpuResult;
use crate::gpu::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, kv_write_from_state_on_stream,
    kv_write_rope_from_state_on_stream, kv_write_rope_on_stream,
};
use crate::gpu::kernels::gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream;
use crate::gpu::kernels::rope::{rope_heads_from_state_on_stream, rope_heads_on_stream};
use crate::gpu::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_fused_qkv_gqa_on_stream,
    gpu_dispatch_fused_qkv_on_stream, gpu_dispatch_gemv_on_stream,
    gpu_dispatch_gemv_residual_on_stream, gpu_dispatch_rms_norm,
};
use crate::gpu::weights::GpuLayerWeights;
use crate::loader::GgmlType;

use super::utils::residual_add_inplace;
use crate::gpu::decode_profile::{
    decode_stage_profiling_enabled, profile_decode_stage, record_layer_invocation, DecodeStage,
};

pub(super) fn gpu_attention_decode(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let seq_len = pos + 1;
    let head_dim = config.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    flash_attn_decode_strided_multi_head_on_stream(
        out_base,
        q_base,
        k_cache,
        v_cache,
        seq_len,
        config.num_heads,
        config.num_kv_heads,
        head_dim,
        scale,
        device.stream(),
    )
}

pub(super) fn gpu_attention_decode_from_state(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let head_dim = config.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    flash_attn_decode_strided_multi_head_from_state_on_stream(
        out_base,
        q_base,
        k_cache,
        v_cache,
        scratch.decode_seq_len_ptr(),
        config.num_heads,
        config.num_kv_heads,
        head_dim,
        scale,
        device.stream(),
    )
}

pub(super) fn gpu_layer_forward_from_state_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    // Check GPU features for DP4A support and check Q4_0 layout compatibility
    let _features = crate::gpu::features::GpuFeatures::detect(device)?;
    let use_dp4a = false;

    if use_dp4a {
        let k_cache = kv.k_ptr(layer_idx)?;
        let v_cache = kv.v_ptr(layer_idx)?;
        gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            eps,
            gpu_layer.attn_q.as_ptr() as *const u8,
            gpu_layer.attn_k.as_ptr() as *const u8,
            gpu_layer.attn_v.as_ptr() as *const u8,
            gpu_layer
                .attn_q_bias
                .as_ref()
                .map(|b| b.as_ptr() as *const f32),
            gpu_layer
                .attn_k_bias
                .as_ref()
                .map(|b| b.as_ptr() as *const f32),
            gpu_layer
                .attn_v_bias
                .as_ref()
                .map(|b| b.as_ptr() as *const f32),
            scratch.q.as_ptr() as *mut f32,
            k_cache,
            v_cache,
            h,
            config.num_heads,
            config.num_kv_heads,
            scratch.decode_pos_ptr(),
            config.head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )?;
    } else {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )?;

        // Always use standard fused QKV followed by separate RoPE/KV-write, which has 100% correct GQA support
        if true {
            // MHA/GQA: use existing fused QKV
            gpu_dispatch_fused_qkv_on_stream(
                device,
                &gpu_layer.attn_q,
                &gpu_layer.attn_q_meta,
                gpu_layer.attn_q_bias.as_ref(),
                &gpu_layer.attn_k,
                &gpu_layer.attn_k_meta,
                gpu_layer.attn_k_bias.as_ref(),
                &gpu_layer.attn_v,
                &gpu_layer.attn_v_meta,
                gpu_layer.attn_v_bias.as_ref(),
                scratch.normed.as_ptr() as *const f32,
                scratch.q.as_ptr() as *mut f32,
                scratch.k.as_ptr() as *mut f32,
                scratch.v.as_ptr() as *mut f32,
                q_size,
                kv_size,
                h,
                device.stream(),
            )?;

            // Apply RoPE separately for MHA
            rope_heads_from_state_on_stream(
                scratch.q.as_ptr() as *mut f32,
                scratch.decode_pos_ptr(),
                config.num_heads,
                config.head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )?;

            kv_write_rope_from_state_on_stream(
                kv.k_ptr(layer_idx)?,
                kv.v_ptr(layer_idx)?,
                scratch.k.as_ptr() as *const f32,
                scratch.v.as_ptr() as *const f32,
                scratch.decode_pos_ptr(),
                config.num_kv_heads,
                config.head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )?;
        } else {
            // GQA: use new GQA-aware fusion (includes RoPE)
            gpu_dispatch_fused_qkv_gqa_on_stream(
                device,
                &gpu_layer.attn_q,
                &gpu_layer.attn_q_meta,
                gpu_layer.attn_q_bias.as_ref(),
                &gpu_layer.attn_k,
                &gpu_layer.attn_k_meta,
                gpu_layer.attn_k_bias.as_ref(),
                &gpu_layer.attn_v,
                &gpu_layer.attn_v_meta,
                gpu_layer.attn_v_bias.as_ref(),
                scratch.normed.as_ptr() as *const f32,
                scratch.q.as_ptr() as *mut f32,
                scratch.k.as_ptr() as *mut f32,
                scratch.v.as_ptr() as *mut f32,
                q_size,
                kv_size,
                config.head_dim,
                scratch.decode_pos_ptr(),
                device.stream(),
            )?;

            // KV-write without RoPE (already applied in fusion)
            kv_write_from_state_on_stream(
                kv.k_ptr(layer_idx)?,
                kv.v_ptr(layer_idx)?,
                scratch.k.as_ptr() as *const f32,
                scratch.v.as_ptr() as *const f32,
                scratch.decode_pos_ptr(),
                config.num_kv_heads,
                config.head_dim,
                device.stream(),
            )?;
        }
    }

    // Note: RoPE and KV-write are now handled inside the conditional blocks above

    gpu_attention_decode_from_state(device, scratch, kv, layer_idx, config)?;

    let attn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
        device,
        &gpu_layer.attn_o,
        &gpu_layer.attn_o_meta,
        scratch.attn_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        h,
        q_size,
        device.stream(),
    )?;
    if !attn_residual_fused {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            q_size,
            device.stream(),
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    gpu_dispatch_rms_norm(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        device.stream(),
    )?;
    gpu_dispatch_fused_gate_up_on_stream(
        device,
        &gpu_layer.ffn_gate,
        &gpu_layer.ffn_gate_meta,
        &gpu_layer.ffn_up,
        &gpu_layer.ffn_up_meta,
        gpu_layer.ffn_gate_up_interleaved.as_ref(), // w_gate_up_interleaved
        gpu_layer.ffn_gate_up_interleaved_tile4.as_ref(), // w_gate_up_interleaved_tile4
        scratch.normed.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )?;

    let ffn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
        device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )?;
    if !ffn_residual_fused {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            ff_size,
            device.stream(),
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    Ok(())
}

/// Hybrid single-layer decode step used by the CLI path and GPU integration tests.
pub fn gpu_layer_forward_hybrid(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    if decode_stage_profiling_enabled() {
        record_layer_invocation();
    }

    // Check GPU features for DP4A support and check Q4_0 layout compatibility
    let _features = crate::gpu::features::GpuFeatures::detect(device)?;
    let use_dp4a = false;

    if use_dp4a {
        // Upload decode state first so pos_ptr has correct pos
        scratch.upload_decode_state(pos, pos + 1, device.stream())?;

        let k_cache = kv.k_ptr(layer_idx)?;
        let v_cache = kv.v_ptr(layer_idx)?;

        profile_decode_stage(device, DecodeStage::Qkv, || {
            gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream(
                device,
                scratch.hidden.as_ptr() as *const f32,
                gpu_layer.attn_norm.as_ptr() as *const f32,
                eps,
                gpu_layer.attn_q.as_ptr() as *const u8,
                gpu_layer.attn_k.as_ptr() as *const u8,
                gpu_layer.attn_v.as_ptr() as *const u8,
                gpu_layer
                    .attn_q_bias
                    .as_ref()
                    .map(|b| b.as_ptr() as *const f32),
                gpu_layer
                    .attn_k_bias
                    .as_ref()
                    .map(|b| b.as_ptr() as *const f32),
                gpu_layer
                    .attn_v_bias
                    .as_ref()
                    .map(|b| b.as_ptr() as *const f32),
                scratch.q.as_ptr() as *mut f32,
                k_cache,
                v_cache,
                h,                   // n_rows (hidden size)
                config.num_heads,    // n_q
                config.num_kv_heads, // n_kv
                scratch.decode_pos_ptr(),
                config.head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )
        })?;
    } else {
        profile_decode_stage(device, DecodeStage::AttnNorm, || {
            gpu_dispatch_rms_norm(
                device,
                scratch.hidden.as_ptr() as *const f32,
                gpu_layer.attn_norm.as_ptr() as *const f32,
                scratch.normed.as_ptr() as *mut f32,
                h,
                eps,
                device.stream(),
            )
        })?;

        profile_decode_stage(device, DecodeStage::Qkv, || {
            gpu_dispatch_fused_qkv_on_stream(
                device,
                &gpu_layer.attn_q,
                &gpu_layer.attn_q_meta,
                gpu_layer.attn_q_bias.as_ref(),
                &gpu_layer.attn_k,
                &gpu_layer.attn_k_meta,
                gpu_layer.attn_k_bias.as_ref(),
                &gpu_layer.attn_v,
                &gpu_layer.attn_v_meta,
                gpu_layer.attn_v_bias.as_ref(),
                scratch.normed.as_ptr() as *const f32,
                scratch.q.as_ptr() as *mut f32,
                scratch.k.as_ptr() as *mut f32,
                scratch.v.as_ptr() as *mut f32,
                q_size,
                kv_size,
                h,
                device.stream(),
            )
        })?;

        profile_decode_stage(device, DecodeStage::QRope, || {
            rope_heads_on_stream(
                scratch.q.as_ptr() as *mut f32,
                pos,
                config.num_heads,
                config.head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )
        })?;

        profile_decode_stage(device, DecodeStage::KvWrite, || {
            kv_write_rope_on_stream(
                kv,
                layer_idx,
                scratch.k.as_ptr() as *mut f32,
                scratch.v.as_ptr() as *mut f32,
                pos,
                config.num_kv_heads,
                config.head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )
        })?;
    }

    profile_decode_stage(device, DecodeStage::Attention, || {
        gpu_attention_decode(device, scratch, kv, layer_idx, pos, config)
    })?;

    let attn_residual_fused = profile_decode_stage(device, DecodeStage::AttnProj, || {
        gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            q_size,
            device.stream(),
        )
    })?;
    if !attn_residual_fused {
        profile_decode_stage(device, DecodeStage::AttnProj, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.attn_o,
                &gpu_layer.attn_o_meta,
                scratch.attn_out.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                q_size,
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::AttnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    profile_decode_stage(device, DecodeStage::FfnNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::GateUp, || {
        gpu_dispatch_fused_gate_up_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            gpu_layer.ffn_gate_up_interleaved.as_ref(), // w_gate_up_interleaved
            gpu_layer.ffn_gate_up_interleaved_tile4.as_ref(), // w_gate_up_interleaved_tile4
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            h,
            device.stream(),
        )
    })?;

    let ffn_residual_fused = profile_decode_stage(device, DecodeStage::FfnDown, || {
        gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            ff_size, // CORRECT: in_dim (input dimension = swiglu size)
            h,       // CORRECT: out_dim (output dimension = hidden size)
            device.stream(),
        )
    })?;
    if !ffn_residual_fused {
        profile_decode_stage(device, DecodeStage::FfnDown, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                ff_size,
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::FfnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    Ok(())
}
