use super::super::utils::residual_add_inplace;
use super::attention::gpu_attention_decode_from_state;
use super::gpu_dispatch_moe_ffn_on_stream;
use super::gpu_layer_forward_ssm_on_stream;
use super::gpu_shortconv_native_on_stream;
use crate::config::ModelConfig;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::decode_profile::{profile_decode_stage, record_layer_invocation, DecodeStage};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream, kv_write_from_state_on_stream,
    kv_write_rope_from_state_on_stream,
};
use crate::gpu::kernels::gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream;
use crate::gpu::kernels::gemv_norm_qkv_rope_kvwrite_q4_0_f32_wmma_on_stream;
use crate::gpu::kernels::rope::rope_heads_from_state_on_stream;
use crate::gpu::kernels::{gelu_on_stream, mul_on_stream, rms_norm_batched, silu_on_stream};
use crate::gpu::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_fused_qkv_gqa_on_stream,
    gpu_dispatch_fused_qkv_on_stream, gpu_dispatch_gemv_on_stream,
    gpu_dispatch_gemv_residual_on_stream, gpu_dispatch_gemv_svd_on_stream,
    gpu_dispatch_gemv_with_fallback_on_stream, gpu_dispatch_rms_norm,
};
use crate::gpu::weights::{GpuLayerType, GpuLayerWeights};
use crate::loader::GgmlType;

pub(in crate::gpu::forward) fn gpu_layer_forward_from_state_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    record_layer_invocation();

    match gpu_layer.layer_type {
        GpuLayerType::Ssm => {
            return gpu_layer_forward_ssm_on_stream(
                device, gpu_layer, None, kv, scratch, None, layer_idx, config,
            );
        }
        GpuLayerType::Shortconv => {
            // Shortconv can be captured in a graph since it's just a sequence of kernels.
            // Note: we use pos=0 as a dummy since it's not used by native on_stream.
            return gpu_shortconv_native_on_stream(
                device, gpu_layer, kv, scratch, layer_idx, 0, config,
            );
        }
        GpuLayerType::AttentionFusedQkv => {
            let h = config.hidden_size;
            let attn_head_dim = config.head_dim;
            let num_q_heads = config.num_heads;
            let num_kv_heads = config.num_kv_heads;
            let q_size = num_q_heads * attn_head_dim;
            let kv_size = num_kv_heads * attn_head_dim;
            let eps = config.rms_norm_eps;

            // 1. RMSNorm
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

            // 2. Fused QKV GEMV → [Q|K|V] concatenated output
            let wqkv =
                gpu_layer
                    .attn_qkv
                    .as_ref()
                    .ok_or_else(|| GpuError::InvalidWeightLayout {
                        tensor: "attn_qkv".to_string(),
                        dims: vec![],
                        reason: "fused QKV buffer missing".to_string(),
                    })?;
            let wqkv_meta =
                gpu_layer
                    .attn_qkv_meta
                    .as_ref()
                    .ok_or_else(|| GpuError::InvalidWeightLayout {
                        tensor: "attn_qkv_meta".to_string(),
                        dims: vec![],
                        reason: "fused QKV metadata missing".to_string(),
                    })?;
            let qkv_dim = if wqkv_meta.dims[0] as usize == h {
                wqkv_meta.dims[1] as usize
            } else {
                wqkv_meta.dims[0] as usize
            };
            let qkv_ptr = scratch.gate.as_ptr() as *mut f32;

            profile_decode_stage(device, DecodeStage::Qkv, || {
                gpu_dispatch_gemv_svd_on_stream(
                    device,
                    wqkv,
                    wqkv_meta,
                    None,
                    scratch.normed.as_ptr() as *const f32,
                    qkv_ptr,
                    qkv_dim,
                    h,
                    scratch.svd_scratch.as_ptr() as *mut f32,
                    device.stream(),
                )
            })?;

            // 3. Split QKV into separate Q, K, V buffers
            // Layout: [Q (q_size)| K (kv_size) | V (kv_size)]
            let q_offset = 0;
            let k_offset = q_size;
            let v_offset = q_size + kv_size;

            // Copy Q
            unsafe {
                crate::gpu::ffi::hip_memcpy_d2d_async(
                    scratch.q.as_ptr(),
                    qkv_ptr.add(q_offset) as *const u8,
                    q_size * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }
            // Copy K
            unsafe {
                crate::gpu::ffi::hip_memcpy_d2d_async(
                    scratch.k.as_ptr(),
                    qkv_ptr.add(k_offset) as *const u8,
                    kv_size * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }
            // Copy V
            unsafe {
                crate::gpu::ffi::hip_memcpy_d2d_async(
                    scratch.v.as_ptr(),
                    qkv_ptr.add(v_offset) as *const u8,
                    kv_size * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }

            // 4. Apply per-head QK norms if present
            if let Some(q_norm_w) = gpu_layer.attn_q_norm.as_ref() {
                rms_norm_batched(
                    scratch.q.as_ptr() as *const f32,
                    q_norm_w.as_ptr() as *const f32,
                    scratch.q.as_ptr() as *mut f32,
                    attn_head_dim,
                    eps,
                    num_q_heads,
                )?;
            }
            if let Some(k_norm_w) = gpu_layer.attn_k_norm.as_ref() {
                rms_norm_batched(
                    scratch.k.as_ptr() as *const f32,
                    k_norm_w.as_ptr() as *const f32,
                    scratch.k.as_ptr() as *mut f32,
                    attn_head_dim,
                    eps,
                    num_kv_heads,
                )?;
            }

            // 5. Apply RoPE
            profile_decode_stage(device, DecodeStage::QRope, || {
                rope_heads_from_state_on_stream(
                    scratch.q.as_ptr() as *mut f32,
                    scratch.positions_ptr(),
                    num_q_heads,
                    attn_head_dim,
                    config.rope_theta,
                    config.rope_neox,
                    device.stream(),
                )
            })?;

            profile_decode_stage(device, DecodeStage::KvWrite, || {
                kv_write_rope_from_state_on_stream(
                    kv,
                    layer_idx,
                    scratch.k.as_ptr() as *const f32,
                    scratch.v.as_ptr() as *const f32,
                    scratch.positions_ptr(),
                    num_kv_heads,
                    attn_head_dim,
                    config.rope_theta,
                    config.rope_neox,
                    device.stream(),
                )
            })?;

            // 6. Attention decode
            profile_decode_stage(device, DecodeStage::Attention, || {
                gpu_attention_decode_from_state(device, scratch, kv, layer_idx, 0, config)
            })?;

            // 7. Attention output projection
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

            // 8. Residual add
            profile_decode_stage(device, DecodeStage::AttnResidual, || {
                residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
            })?;

            // 9. FFN RMSNorm
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

            // 10. FFN
            let ff_size = config.intermediate_size;
            if let (Some(gate_buf), Some(gate_meta)) = (
                gpu_layer.ffn_gate.as_ref(),
                gpu_layer.ffn_gate_meta.as_ref(),
            ) {
                profile_decode_stage(device, DecodeStage::GateUp, || {
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
                        device.stream(),
                    )
                })?;
            } else {
                // Standard FFN (non-SwiGLU): up -> gelu
                profile_decode_stage(device, DecodeStage::GateUp, || {
                    gpu_dispatch_gemv_on_stream(
                        device,
                        &gpu_layer.ffn_up,
                        &gpu_layer.ffn_up_meta,
                        scratch.normed.as_ptr() as *const f32,
                        scratch.swiglu.as_ptr() as *mut f32,
                        ff_size,
                        h,
                        device.stream(),
                    )?;
                    gelu_on_stream(
                        scratch.swiglu.as_ptr() as *const f32,
                        scratch.swiglu.as_ptr() as *mut f32,
                        ff_size,
                        device.stream(),
                    )
                })?;
            }

            // 11. FFN output projection + residual
            profile_decode_stage(device, DecodeStage::FfnDown, || {
                gpu_dispatch_gemv_on_stream(
                    device,
                    &gpu_layer.ffn_down,
                    &gpu_layer.ffn_down_meta,
                    scratch.swiglu.as_ptr() as *const f32,
                    scratch.gate.as_ptr() as *mut f32,
                    h,
                    ff_size,
                    device.stream(),
                )
            })?;
            profile_decode_stage(device, DecodeStage::FfnResidual, || {
                residual_add_inplace(device, &scratch.hidden, &scratch.gate, h)
            })?;

            return Ok(());
        }
        GpuLayerType::Attention => {}
    }

    let h = config.hidden_size;
    // Derive actual attention KV dims from K weight shape.
    // For hybrid models (e.g. Qwen 3.5), config.num_kv_heads may reflect SSM head count
    // while attention layers have fewer KV heads. The K weight is authoritative for kv_size.
    // Use config.head_dim as attn_head_dim (256 for qwen35, correct for attention layers).
    let attn_head_dim = config.head_dim; // 256 for qwen35 (correct for attention)
    let (q_size, kv_size) = {
        let q_dims = &gpu_layer.attn_q_meta.dims;
        let k_dims = &gpu_layer.attn_k_meta.dims;
        let qs = if q_dims[0] as usize == h {
            q_dims[1] as usize
        } else {
            q_dims[0] as usize
        };
        let ks = if k_dims[0] as usize == h {
            k_dims[1] as usize
        } else {
            k_dims[0] as usize
        };
        (qs, ks)
    };
    // num_q_heads for attention uses config value (metadata authoritative for attention output size)
    let num_q_heads = config.num_heads; // e.g. 16 for qwen35 attention
                                        // num_kv_heads derived from K weight (not config, which may be stale for hybrid models)
    let num_kv_heads = kv_size / attn_head_dim;
    // Attention output size: what flash_attn produces and O-proj takes as input
    let attn_out_size = num_q_heads * attn_head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    // Check GPU features for DP4A/WMMA support and check Q4_0 layout compatibility
    let features = crate::gpu::features::GpuFeatures::detect(device)?;
    // The fused norm+QKV+RoPE+KV-write kernel is optimized for MHA. On GQA it
    // launches only one block per KV head, leaving most of the GPU idle, so
    // fall through to the separate RMSNorm + fused QKV + RoPE + KV-write path
    // which has full GQA support and historically hits the documented baseline.
    let use_fused = (gpu_layer.attn_q_meta.wtype == GgmlType::Q4_0)
        && (gpu_layer.attn_k_meta.wtype == GgmlType::Q4_0)
        && (gpu_layer.attn_v_meta.wtype == GgmlType::Q4_0)
        && (features.has_dp4a || features.has_wmma)
        && crate::gpu::safety::use_dp4a_enabled()
        && (num_q_heads == num_kv_heads);

    if use_fused {
        let k_cache = kv.k_ptr(layer_idx)?;
        let v_cache = kv.v_ptr(layer_idx)?;

        if features.has_wmma {
            profile_decode_stage(device, DecodeStage::Qkv, || {
                crate::gpu::kernels::gemv_norm_qkv_rope_kvwrite_q4_0_f32_wmma_on_stream(
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
                )
            })?;
        } else {
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
                    h,
                    config.num_heads,
                    config.num_kv_heads,
                    scratch.decode_pos_ptr(),
                    config.head_dim,
                    config.rope_theta,
                    config.rope_neox,
                    device.stream(),
                )
            })?;
        }
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

        // Always use standard fused QKV followed by separate RoPE/KV-write, which has 100% correct GQA support
        if true {
            // MHA/GQA: use existing fused QKV
            profile_decode_stage(device, DecodeStage::Qkv, || {
                gpu_dispatch_fused_qkv_on_stream(
                    device,
                    &gpu_layer.attn_q,
                    &gpu_layer.attn_q_meta,
                    gpu_layer.attn_q_svd.as_ref(),
                    gpu_layer.attn_q_bias.as_ref(),
                    &gpu_layer.attn_k,
                    &gpu_layer.attn_k_meta,
                    gpu_layer.attn_k_svd.as_ref(),
                    gpu_layer.attn_k_bias.as_ref(),
                    &gpu_layer.attn_v,
                    &gpu_layer.attn_v_meta,
                    gpu_layer.attn_v_svd.as_ref(),
                    gpu_layer.attn_v_bias.as_ref(),
                    scratch.normed.as_ptr() as *const f32,
                    scratch.q.as_ptr() as *mut f32,
                    scratch.k.as_ptr() as *mut f32,
                    scratch.v.as_ptr() as *mut f32,
                    q_size,
                    kv_size,
                    h,
                    scratch.svd_scratch.as_ptr() as *mut f32,
                    device.stream(),
                )
            })?;

            // Apply per-head QK norms if present (e.g. Qwen 3.5 attention layers)
            if let Some(q_norm_w) = gpu_layer.attn_q_norm.as_ref() {
                rms_norm_batched(
                    scratch.q.as_ptr() as *const f32,
                    q_norm_w.as_ptr() as *const f32,
                    scratch.q.as_ptr() as *mut f32,
                    attn_head_dim,
                    eps,
                    num_q_heads,
                )?;
            }
            if let Some(k_norm_w) = gpu_layer.attn_k_norm.as_ref() {
                rms_norm_batched(
                    scratch.k.as_ptr() as *const f32,
                    k_norm_w.as_ptr() as *const f32,
                    scratch.k.as_ptr() as *mut f32,
                    attn_head_dim,
                    eps,
                    num_kv_heads,
                )?;
            }

            // Apply RoPE separately for MHA
            profile_decode_stage(device, DecodeStage::QRope, || {
                rope_heads_from_state_on_stream(
                    scratch.q.as_ptr() as *mut f32,
                    scratch.positions_ptr(),
                    num_q_heads,
                    attn_head_dim,
                    config.rope_theta,
                    config.rope_neox,
                    device.stream(),
                )
            })?;

            profile_decode_stage(device, DecodeStage::KvWrite, || {
                kv_write_rope_from_state_on_stream(
                    kv,
                    layer_idx,
                    scratch.k.as_ptr() as *const f32,
                    scratch.v.as_ptr() as *const f32,
                    scratch.positions_ptr(),
                    num_kv_heads,
                    attn_head_dim,
                    config.rope_theta,
                    config.rope_neox,
                    device.stream(),
                )
            })?;
        } else {
            // GQA: use new GQA-aware fusion (includes RoPE)
            profile_decode_stage(device, DecodeStage::Qkv, || {
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
                    attn_head_dim,
                    scratch.decode_pos_ptr(),
                    device.stream(),
                )
            })?;

            // KV-write without RoPE (already applied in fusion)
            profile_decode_stage(device, DecodeStage::KvWrite, || {
                kv_write_from_state_on_stream(
                    kv.k_ptr(layer_idx)?,
                    kv.v_ptr(layer_idx)?,
                    scratch.k.as_ptr() as *const f32,
                    scratch.v.as_ptr() as *const f32,
                    scratch.decode_pos_ptr(),
                    num_kv_heads,
                    attn_head_dim,
                    device.stream(),
                )
            })?;
        }
    }

    // Note: RoPE and KV-write are now handled inside the conditional blocks above

    profile_decode_stage(device, DecodeStage::Attention, || {
        gpu_attention_decode_from_state(device, scratch, kv, layer_idx, 0, config)
    })?;

    let mut attn_residual_fused = false;
    if gpu_layer.attn_o_svd.is_none() {
        attn_residual_fused = profile_decode_stage(device, DecodeStage::AttnProj, || {
            gpu_dispatch_gemv_residual_on_stream(
                device,
                &gpu_layer.attn_o,
                &gpu_layer.attn_o_meta,
                scratch.attn_out.as_ptr() as *const f32,
                scratch.hidden.as_ptr() as *const f32,
                scratch.hidden.as_ptr() as *mut f32,
                attn_out_size, // in_dim: attention output (num_q_heads * head_dim)
                h,             // out_dim: hidden size
                device.stream(),
            )
        })?;
    }
    if !attn_residual_fused {
        profile_decode_stage(device, DecodeStage::AttnProj, || {
            gpu_dispatch_gemv_svd_on_stream(
                device,
                &gpu_layer.attn_o,
                &gpu_layer.attn_o_meta,
                gpu_layer.attn_o_svd.as_ref(),
                scratch.attn_out.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,             // out_dim: hidden size
                attn_out_size, // in_dim: attention output (num_q_heads * head_dim)
                scratch.svd_scratch.as_ptr() as *mut f32,
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
    if gpu_dispatch_moe_ffn_on_stream(device, gpu_layer, scratch, h, ff_size, config)? {
        return Ok(());
    }
    if let (Some(gate_buf), Some(gate_meta)) = (
        gpu_layer.ffn_gate.as_ref(),
        gpu_layer.ffn_gate_meta.as_ref(),
    ) {
        if gpu_layer.ffn_gate_svd.is_some() || gpu_layer.ffn_up_svd.is_some() {
            profile_decode_stage(device, DecodeStage::GateUp, || {
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
                    device.stream(),
                )
            })?;
            profile_decode_stage(device, DecodeStage::GateUp, || {
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
                    device.stream(),
                )
            })?;
            profile_decode_stage(device, DecodeStage::GateUp, || {
                silu_on_stream(
                    scratch.gate.as_ptr() as *const f32,
                    scratch.gate.as_ptr() as *mut f32,
                    ff_size,
                    device.stream(),
                )
            })?;
            profile_decode_stage(device, DecodeStage::GateUp, || {
                mul_on_stream(
                    scratch.gate.as_ptr() as *const f32,
                    scratch.swiglu.as_ptr() as *const f32,
                    scratch.swiglu.as_ptr() as *mut f32,
                    ff_size,
                    device.stream(),
                )
            })?;
        } else {
            profile_decode_stage(device, DecodeStage::GateUp, || {
                gpu_dispatch_fused_gate_up_on_stream(
                    device,
                    gate_buf,
                    gate_meta,
                    &gpu_layer.ffn_up,
                    &gpu_layer.ffn_up_meta,
                    gpu_layer.ffn_gate_up_interleaved.as_ref(), // w_gate_up_interleaved
                    gpu_layer.ffn_gate_up_interleaved_tile4.as_ref(), // w_gate_up_interleaved_tile4
                    scratch.normed.as_ptr() as *const f32,
                    scratch.gate.as_ptr() as *mut f32,
                    scratch.swiglu.as_ptr() as *mut f32,
                    ff_size,
                    h,
                    device.stream(),
                )
            })?;
        }
    } else {
        // Standard FFN (non-SwiGLU): up -> gelu
        profile_decode_stage(device, DecodeStage::GateUp, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.ffn_up,
                &gpu_layer.ffn_up_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                ff_size,
                h,
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::GateUp, || {
            gelu_on_stream(
                scratch.swiglu.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                ff_size,
                device.stream(),
            )
        })?;
    }

    let mut ffn_residual_fused = false;
    if gpu_layer.ffn_down_svd.is_none() {
        ffn_residual_fused = profile_decode_stage(device, DecodeStage::FfnResidual, || {
            gpu_dispatch_gemv_residual_on_stream(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.hidden.as_ptr() as *const f32,
                scratch.hidden.as_ptr() as *mut f32,
                ff_size,
                h,
                device.stream(),
            )
        })?;
    }
    if !ffn_residual_fused {
        profile_decode_stage(device, DecodeStage::FfnDown, || {
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
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::FfnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    Ok(())
}
