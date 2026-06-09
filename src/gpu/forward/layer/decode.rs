use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::weights::CpuLayerWeights;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::decode_profile::{
    decode_stage_profiling_enabled, profile_decode_stage, record_layer_invocation, DecodeStage,
};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, flash_attn_decode_turboquant,
    kv_write_from_state_on_stream, kv_write_rope_from_state_on_stream, kv_write_rope_on_stream,
};
use crate::gpu::kernels::gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream;
use crate::gpu::kernels::rope::{rope_heads_from_state_on_stream, rope_heads_on_stream};
use crate::gpu::kernels::{mul_on_stream, rms_norm_batched, silu_on_stream};
use crate::gpu::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_fused_qkv_gqa_on_stream,
    gpu_dispatch_fused_qkv_on_stream, gpu_dispatch_gemv_on_stream,
    gpu_dispatch_gemv_residual_on_stream, gpu_dispatch_gemv_svd_on_stream,
    gpu_dispatch_gemv_with_fallback_on_stream, gpu_dispatch_rms_norm, supports_gemv_type,
};
use crate::gpu::weights::GpuLayerWeights;
use super::gpu_dispatch_moe_ffn_on_stream;
use super::gpu_layer_forward_ssm_on_stream;
use super::super::utils::{cpu_fallback_gemv_and_upload, ensure_size, residual_add_inplace};

pub(in crate::gpu::forward) fn gpu_attention_decode(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> GpuResult<()> {
    let seq_len = pos + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

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

    if kv.kv_quant_bits.is_some() {
        let centroids = kv.centroids_ptr()?;
        flash_attn_decode_turboquant(
            out_base,
            q_base,
            k_cache as *const u8,
            v_cache as *const u8,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            centroids,
            kv.qjl_scale,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    } else {
        flash_attn_decode_strided_multi_head_on_stream(
            out_base,
            q_base,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            kv.adastate_anchors_enabled,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    }
}

pub(in crate::gpu::forward) fn gpu_attention_decode_from_state(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> GpuResult<()> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

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

    if kv.kv_quant_bits.is_some() {
        let seq_len = scratch.decode_state_next_pos().unwrap_or(0) + 1;
        let centroids = kv.centroids_ptr()?;
        flash_attn_decode_turboquant(
            out_base,
            q_base,
            k_cache as *const u8,
            v_cache as *const u8,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            centroids,
            kv.qjl_scale,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    } else {
        flash_attn_decode_strided_multi_head_from_state_on_stream(
            out_base,
            q_base,
            k_cache,
            v_cache,
            scratch.decode_seq_len_ptr(),
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            kv_lora_dim,
            kv.adastate_anchors_enabled,
            w_up_k,
            w_up_v,
            device.stream(),
        )
    }
}

pub(in crate::gpu::forward) fn gpu_layer_forward_from_state_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    if gpu_layer.ssm.is_some() {
        return gpu_layer_forward_ssm_on_stream(
            device, gpu_layer, None, kv, scratch, None, layer_idx, config,
        );
    }

    // Fused QKV input projection for non-SSM layers: split into Q/K/V via GEMV
    // and then proceed with standard attention path.
    if gpu_layer.attn_qkv.is_some() {
        let h = config.hidden_size;
        let attn_head_dim = config.head_dim;
        let num_q_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let q_size = num_q_heads * attn_head_dim;
        let kv_size = num_kv_heads * attn_head_dim;
        let eps = config.rms_norm_eps;

        // 1. RMSNorm
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )?;

        // 2. Fused QKV GEMV → [Q|K|V] concatenated output
        let wqkv = gpu_layer
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
        )?;

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
        rope_heads_from_state_on_stream(
            scratch.q.as_ptr() as *mut f32,
            scratch.decode_pos_ptr(),
            num_q_heads,
            attn_head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )?;

        kv_write_rope_from_state_on_stream(
            kv,
            layer_idx,
            scratch.k.as_ptr() as *const f32,
            scratch.v.as_ptr() as *const f32,
            scratch.decode_pos_ptr(),
            num_kv_heads,
            attn_head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )?;

        // 6. Attention decode
        gpu_attention_decode_from_state(
            device,
            scratch,
            kv,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            attn_head_dim,
        )?;

        // 7. Attention output projection
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

        // 8. Residual add
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;

        // 9. FFN RMSNorm
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )?;

        // 10. FFN
        let ff_size = config.intermediate_size;
        gpu_dispatch_fused_gate_up_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
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
        )?;

        // 11. FFN output projection + residual
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.gate.as_ptr() as *mut f32,
            h,
            ff_size,
            device.stream(),
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.gate, h)?;

        return Ok(());
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
            )?;

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
            rope_heads_from_state_on_stream(
                scratch.q.as_ptr() as *mut f32,
                scratch.decode_pos_ptr(),
                num_q_heads,
                attn_head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )?;

            kv_write_rope_from_state_on_stream(
                kv,
                layer_idx,
                scratch.k.as_ptr() as *const f32,
                scratch.v.as_ptr() as *const f32,
                scratch.decode_pos_ptr(),
                num_kv_heads,
                attn_head_dim,
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
                attn_head_dim,
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
                num_kv_heads,
                attn_head_dim,
                device.stream(),
            )?;
        }
    }

    // Note: RoPE and KV-write are now handled inside the conditional blocks above

    gpu_attention_decode_from_state(
        device,
        scratch,
        kv,
        layer_idx,
        num_q_heads,
        num_kv_heads,
        attn_head_dim,
    )?;

    let mut attn_residual_fused = false;
    if gpu_layer.attn_o_svd.is_none() {
        attn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            attn_out_size, // in_dim: attention output (num_q_heads * head_dim)
            h,             // out_dim: hidden size
            device.stream(),
        )?;
    }
    if !attn_residual_fused {
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
    if gpu_dispatch_moe_ffn_on_stream(device, gpu_layer, scratch, h, ff_size)? {
        return Ok(());
    }
    if gpu_layer.ffn_gate_svd.is_some() || gpu_layer.ffn_up_svd.is_some() {
        gpu_dispatch_gemv_with_fallback_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            gpu_layer.ffn_gate_svd.as_ref(),
            gpu_layer.ffn_gate_sparse.as_ref(),
            gpu_layer.ffn_gate_mpo.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            scratch.gate.as_ptr() as *mut f32,
            ff_size,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            device.stream(),
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
            device.stream(),
        )?;
        silu_on_stream(
            scratch.gate.as_ptr() as *const f32,
            scratch.gate.as_ptr() as *mut f32,
            ff_size,
            device.stream(),
        )?;
        mul_on_stream(
            scratch.gate.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
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
            gpu_layer.ffn_gate_up_interleaved.as_ref(), // w_gate_up_interleaved
            gpu_layer.ffn_gate_up_interleaved_tile4.as_ref(), // w_gate_up_interleaved_tile4
            scratch.normed.as_ptr() as *const f32,
            scratch.gate.as_ptr() as *mut f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            h,
            device.stream(),
        )?;
    }

    let mut ffn_residual_fused = false;
    if gpu_layer.ffn_down_svd.is_none() {
        ffn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
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
    }
    if !ffn_residual_fused {
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
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    Ok(())
}

/// Hybrid single-layer decode step used by the CLI path and GPU integration tests.
pub fn gpu_layer_forward_hybrid(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    cpu_layer: Option<&CpuLayerWeights>,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    mut cpu_scratch: Option<&mut CpuForwardScratch>,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    if gpu_layer.ssm.is_some() {
        // Upload decode state first so pos_ptr has correct pos
        scratch.upload_decode_state(pos, pos + 1, device.stream())?;
        return gpu_layer_forward_ssm_on_stream(
            device,
            gpu_layer,
            cpu_layer,
            kv,
            scratch,
            cpu_scratch,
            layer_idx,
            config,
        );
    }

    // Fused QKV input projection for non-SSM layers: split into Q/K/V via GEMV
    // and then proceed with standard attention path.
    if gpu_layer.attn_qkv.is_some() {
        let h = config.hidden_size;
        let attn_head_dim = config.head_dim;
        let num_q_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let q_size = num_q_heads * attn_head_dim;
        let kv_size = num_kv_heads * attn_head_dim;
        let eps = config.rms_norm_eps;

        // Upload decode state first so pos_ptr has correct pos
        scratch.upload_decode_state(pos, pos + 1, device.stream())?;

        // 1. RMSNorm
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )?;

        // 2. Fused QKV GEMV → [Q|K|V] concatenated output
        let wqkv = gpu_layer
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

        if supports_gemv_type(wqkv_meta.wtype) {
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

        // 3. Split QKV into separate Q, K, V buffers
        let q_offset = 0;
        let k_offset = q_size;
        let v_offset = q_size + kv_size;

        unsafe {
            crate::gpu::ffi::hip_memcpy_d2d_async(
                scratch.q.as_ptr(),
                qkv_ptr.add(q_offset) as *const u8,
                q_size * std::mem::size_of::<f32>(),
                device.stream(),
            )?;
            crate::gpu::ffi::hip_memcpy_d2d_async(
                scratch.k.as_ptr(),
                qkv_ptr.add(k_offset) as *const u8,
                kv_size * std::mem::size_of::<f32>(),
                device.stream(),
            )?;
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
        rope_heads_from_state_on_stream(
            scratch.q.as_ptr() as *mut f32,
            scratch.decode_pos_ptr(),
            num_q_heads,
            attn_head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )?;

        kv_write_rope_from_state_on_stream(
            kv,
            layer_idx,
            scratch.k.as_ptr() as *const f32,
            scratch.v.as_ptr() as *const f32,
            scratch.decode_pos_ptr(),
            num_kv_heads,
            attn_head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )?;

        // 6. Attention decode
        gpu_attention_decode_from_state(
            device,
            scratch,
            kv,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            attn_head_dim,
        )?;

        // 7. Attention output projection
        if supports_gemv_type(gpu_layer.attn_o_meta.wtype) {
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
        } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
            ensure_size(&mut cpu_s.attn_out, q_size);
            ensure_size(&mut cpu_s.layer_out, h);
            cpu_fallback_gemv_and_upload(
                "attn_o",
                &cpu_l.attn_o,
                &cpu_l.attn_o_meta,
                &scratch.attn_out,
                &mut cpu_s.attn_out,
                &mut cpu_s.layer_out,
                &mut scratch.layer_out,
                h,
                q_size,
                &mut cpu_s.q8_scratch,
            )?;
        } else {
            return Err(GpuError::UnsupportedWeightType {
                tensor: "attn_o".to_string(),
                wtype: gpu_layer.attn_o_meta.wtype,
            });
        }

        // 8. Residual add
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;

        // 9. FFN RMSNorm
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )?;

        // 10. FFN
        let ff_size = config.intermediate_size;
        if supports_gemv_type(gpu_layer.ffn_gate_meta.wtype)
            && supports_gemv_type(gpu_layer.ffn_up_meta.wtype)
        {
            gpu_dispatch_fused_gate_up_on_stream(
                device,
                &gpu_layer.ffn_gate,
                &gpu_layer.ffn_gate_meta,
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

            ensure_size(&mut cpu_s.layer_out, ff_size);
            cpu_fallback_gemv_and_upload(
                "ffn_up",
                &cpu_l.ffn_up,
                &cpu_l.ffn_up_meta,
                &scratch.normed,
                &mut cpu_s.normed,
                &mut cpu_s.layer_out,
                &mut scratch.layer_out,
                ff_size,
                h,
                &mut cpu_s.q8_scratch,
            )?;

            silu_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                ff_size,
                device.stream(),
            )?;
            mul_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                ff_size,
                device.stream(),
            )?;
        } else {
            return Err(GpuError::UnsupportedWeightType {
                tensor: "ffn_gate".to_string(),
                wtype: gpu_layer.ffn_gate_meta.wtype,
            });
        }

        // 11. FFN output projection + residual
        if supports_gemv_type(gpu_layer.ffn_down_meta.wtype) {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                h,
                ff_size,
                device.stream(),
            )?;
        } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
            ensure_size(&mut cpu_s.layer_out, ff_size);
            ensure_size(&mut cpu_s.gate, h);
            cpu_fallback_gemv_and_upload(
                "ffn_down",
                &cpu_l.ffn_down,
                &cpu_l.ffn_down_meta,
                &scratch.layer_out,
                &mut cpu_s.layer_out,
                &mut cpu_s.gate,
                &mut scratch.gate,
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
        residual_add_inplace(device, &scratch.hidden, &scratch.gate, h)?;

        return Ok(());
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
            if supports_gemv_type(gpu_layer.attn_q_meta.wtype)
                && supports_gemv_type(gpu_layer.attn_k_meta.wtype)
                && supports_gemv_type(gpu_layer.attn_v_meta.wtype)
            {
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
            } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
                ensure_size(&mut cpu_s.normed, h);

                ensure_size(&mut cpu_s.q, q_size);
                cpu_fallback_gemv_and_upload(
                    "attn_q",
                    &cpu_l.attn_q,
                    &cpu_l.attn_q_meta,
                    &scratch.normed,
                    &mut cpu_s.normed,
                    &mut cpu_s.q,
                    &mut scratch.q,
                    q_size,
                    h,
                    &mut cpu_s.q8_scratch,
                )?;
                if let Some(bias) = gpu_layer.attn_q_bias.as_ref() {
                    unsafe {
                        crate::gpu::kernels::add_on_stream(
                            scratch.q.as_ptr() as *mut f32,
                            bias.as_ptr() as *const f32,
                            scratch.q.as_ptr() as *mut f32,
                            q_size,
                            device.stream(),
                        )?;
                    }
                }

                ensure_size(&mut cpu_s.k, kv_size);
                cpu_fallback_gemv_and_upload(
                    "attn_k",
                    &cpu_l.attn_k,
                    &cpu_l.attn_k_meta,
                    &scratch.normed,
                    &mut cpu_s.normed,
                    &mut cpu_s.k,
                    &mut scratch.k,
                    kv_size,
                    h,
                    &mut cpu_s.q8_scratch,
                )?;
                if let Some(bias) = gpu_layer.attn_k_bias.as_ref() {
                    unsafe {
                        crate::gpu::kernels::add_on_stream(
                            scratch.k.as_ptr() as *mut f32,
                            bias.as_ptr() as *const f32,
                            scratch.k.as_ptr() as *mut f32,
                            kv_size,
                            device.stream(),
                        )?;
                    }
                }

                ensure_size(&mut cpu_s.v, kv_size);
                cpu_fallback_gemv_and_upload(
                    "attn_v",
                    &cpu_l.attn_v,
                    &cpu_l.attn_v_meta,
                    &scratch.normed,
                    &mut cpu_s.normed,
                    &mut cpu_s.v,
                    &mut scratch.v,
                    kv_size,
                    h,
                    &mut cpu_s.q8_scratch,
                )?;
                if let Some(bias) = gpu_layer.attn_v_bias.as_ref() {
                    unsafe {
                        crate::gpu::kernels::add_on_stream(
                            scratch.v.as_ptr() as *mut f32,
                            bias.as_ptr() as *const f32,
                            scratch.v.as_ptr() as *mut f32,
                            kv_size,
                            device.stream(),
                        )?;
                    }
                }
                Ok(())
            } else {
                Err(GpuError::UnsupportedWeightType {
                    tensor: "attn_q".to_string(),
                    wtype: gpu_layer.attn_q_meta.wtype,
                })
            }
        })?;

        // Apply per-head QK norms if present (e.g. Qwen 3.5 attention layers)
        {
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
        }

        profile_decode_stage(device, DecodeStage::QRope, || {
            rope_heads_on_stream(
                scratch.q.as_ptr() as *mut f32,
                pos,
                num_q_heads,
                attn_head_dim,
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
                num_kv_heads,
                attn_head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )
        })?;
    }

    profile_decode_stage(device, DecodeStage::Attention, || {
        gpu_attention_decode(
            device,
            scratch,
            kv,
            layer_idx,
            pos,
            num_q_heads,
            num_kv_heads,
            attn_head_dim,
        )
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
            if supports_gemv_type(gpu_layer.attn_o_meta.wtype) {
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
            } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
                ensure_size(&mut cpu_s.attn_out, attn_out_size);
                ensure_size(&mut cpu_s.layer_out, h);
                cpu_fallback_gemv_and_upload(
                    "attn_o",
                    &cpu_l.attn_o,
                    &cpu_l.attn_o_meta,
                    &scratch.attn_out,
                    &mut cpu_s.attn_out,
                    &mut cpu_s.layer_out,
                    &mut scratch.layer_out,
                    h,
                    attn_out_size,
                    &mut cpu_s.q8_scratch,
                )
            } else {
                Err(GpuError::UnsupportedWeightType {
                    tensor: "attn_o".to_string(),
                    wtype: gpu_layer.attn_o_meta.wtype,
                })
            }
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
    if profile_decode_stage(device, DecodeStage::GateUp, || {
        gpu_dispatch_moe_ffn_on_stream(device, gpu_layer, scratch, h, ff_size)
    })? {
        return Ok(());
    }
    profile_decode_stage(device, DecodeStage::GateUp, || {
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
                device.stream(),
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
                device.stream(),
            )?;
            silu_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                ff_size,
                device.stream(),
            )?;
            mul_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                ff_size,
                device.stream(),
            )
        } else {
            if supports_gemv_type(gpu_layer.ffn_gate_meta.wtype)
                && supports_gemv_type(gpu_layer.ffn_up_meta.wtype)
            {
                gpu_dispatch_fused_gate_up_on_stream(
                    device,
                    &gpu_layer.ffn_gate,
                    &gpu_layer.ffn_gate_meta,
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

                silu_on_stream(
                    scratch.gate.as_ptr() as *const f32,
                    scratch.gate.as_ptr() as *mut f32,
                    ff_size,
                    device.stream(),
                )?;
                mul_on_stream(
                    scratch.gate.as_ptr() as *const f32,
                    scratch.swiglu.as_ptr() as *const f32,
                    scratch.swiglu.as_ptr() as *mut f32,
                    ff_size,
                    device.stream(),
                )
            } else {
                Err(GpuError::UnsupportedWeightType {
                    tensor: "ffn_gate".to_string(),
                    wtype: gpu_layer.ffn_gate_meta.wtype,
                })
            }
        }
    })?;

    let mut ffn_residual_fused = false;
    if gpu_layer.ffn_down_svd.is_none() {
        ffn_residual_fused = profile_decode_stage(device, DecodeStage::FfnDown, || {
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
    }
    if !ffn_residual_fused {
        profile_decode_stage(device, DecodeStage::FfnDown, || {
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
                    device.stream(),
                )
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
                )
            } else {
                Err(GpuError::UnsupportedWeightType {
                    tensor: "ffn_down".to_string(),
                    wtype: gpu_layer.ffn_down_meta.wtype,
                })
            }
        })?;
        profile_decode_stage(device, DecodeStage::FfnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    Ok(())
}
