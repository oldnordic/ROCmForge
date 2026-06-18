use super::super::utils::{cpu_fallback_gemv_and_upload, ensure_size, residual_add_inplace};
use super::attention::gpu_attention_decode_from_state;
use super::gpu_dispatch_moe_ffn_on_stream;
use super::gpu_layer_forward_ssm_on_stream;
use super::gpu_shortconv_native_on_stream;
use crate::config::ModelConfig;
use crate::cpu::cache::{CpuForwardScratch, CpuKvCache};
use crate::cpu::forward::cpu_layer_forward;
use crate::cpu::weights::CpuLayerWeights;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::decode_profile::{profile_decode_stage, record_layer_invocation, DecodeStage};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream, kv_write_from_state_on_stream,
    kv_write_rope_from_state_on_stream,
};
use crate::gpu::kernels::gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream;
use crate::gpu::kernels::rope::rope_heads_from_state_on_stream;
use crate::gpu::kernels::{gelu_on_stream, mul_on_stream, rms_norm_batched, silu_on_stream};
use crate::gpu::ops::{
    gpu_dispatch_fused_qkv_gqa_on_stream, gpu_dispatch_fused_qkv_on_stream,
    gpu_dispatch_gate_up_raw_on_stream, gpu_dispatch_gemv_on_stream,
    gpu_dispatch_gemv_residual_on_stream, gpu_dispatch_gemv_svd_on_stream,
    gpu_dispatch_gemv_with_fallback_on_stream, gpu_dispatch_rms_norm, supports_gemv_type,
};
use crate::gpu::weights::{GpuBuffer, GpuLayerType, GpuLayerWeights};

fn layer_intermediate_dumps_enabled() -> bool {
    // Off by default: dump_gpu_f32 forces a device->host sync on every call,
    // so leaving it on in production adds one sync per dump point per layer.
    // Set ROCMFORGE_DUMP_LAYER_INTERMEDIATES=1 to re-enable for bisection.
    std::env::var("ROCMFORGE_DUMP_LAYER_INTERMEDIATES")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn dump_gpu_f32(name: &str, buf: &GpuBuffer, n: usize) {
    if !layer_intermediate_dumps_enabled() {
        return;
    }
    let elems = buf.size() / std::mem::size_of::<f32>();
    let n = n.min(elems);
    if n == 0 {
        return;
    }
    let v = buf.copy_to_host_vec().expect("dump");
    eprintln!("[GPU] {}: {:?}", name, &v[..n]);
}

fn gpu_shortconv_fallback(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    cpu_layer: Option<&CpuLayerWeights>,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    cpu_scratch: Option<&mut CpuForwardScratch>,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch) else {
        return Err(GpuError::UnsupportedOperation {
            operation: "shortconv GPU fallback".to_string(),
            reason: "CPU layer weights and scratch required for shortconv fallback".to_string(),
        });
    };

    // Download hidden state from GPU to CPU
    let mut cpu_hidden = vec![0.0f32; h];
    super::super::utils::download_f32(&scratch.hidden, &mut cpu_hidden)?;

    // Build minimal CPU KV cache for conv_state sync
    let mut cpu_kv = CpuKvCache::new(config, 1);

    // Sync conv state from GPU to CPU if present
    if let Some(ref gpu_conv_states) = kv.conv_state {
        if let Some(gpu_conv_buf) = gpu_conv_states.get(layer_idx) {
            let conv_elems = gpu_conv_buf.size() / std::mem::size_of::<f32>();
            let mut cpu_conv = vec![0.0f32; conv_elems];
            super::super::utils::download_f32(gpu_conv_buf, &mut cpu_conv)?;
            cpu_kv.conv_state[layer_idx].copy_from_slice(&cpu_conv);
        }
    }

    // Ensure CPU scratch has required sizes
    ensure_size(&mut cpu_s.normed, h);
    ensure_size(&mut cpu_s.shortconv_bcx, 3 * h);
    ensure_size(&mut cpu_s.shortconv_tmp, h);
    ensure_size(&mut cpu_s.layer_out, h);
    let ff_size = cpu_l
        .moe
        .as_ref()
        .map(|m| m.ff_size)
        .unwrap_or(config.intermediate_size);
    ensure_size(&mut cpu_s.gate, ff_size);
    ensure_size(&mut cpu_s.swiglu, ff_size);

    // Run CPU layer forward
    let rope_sin: Vec<f32> = Vec::new();
    let rope_cos: Vec<f32> = Vec::new();
    cpu_layer_forward(
        &mut cpu_hidden,
        cpu_l,
        &mut cpu_kv,
        cpu_s,
        layer_idx,
        pos,
        &rope_sin,
        &rope_cos,
        config,
        false,
    )
    .map_err(|e| super::super::utils::cpu_fallback_error("shortconv layer", e))?;

    // Upload hidden state back to GPU
    super::super::utils::upload_f32(&mut scratch.hidden, &cpu_hidden)?;

    // Sync conv state back to GPU
    if let Some(ref mut gpu_conv_states) = kv.conv_state {
        if let Some(gpu_conv_buf) = gpu_conv_states.get_mut(layer_idx) {
            let cpu_conv = &cpu_kv.conv_state[layer_idx];
            super::super::utils::upload_f32(gpu_conv_buf, cpu_conv)?;
        }
    }

    device.synchronize()?;
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
    record_layer_invocation();

    match gpu_layer.layer_type {
        GpuLayerType::Ssm => {
            // Upload decode state first so pos_ptr has correct pos
            scratch.upload_decode_state(pos, pos + 1, device.stream())?;
            // Upload positions array for RoPE and KV operations
            // For decode: seq_len = pos + 1, num_prefill = 0
            scratch.upload_positions(pos + 1, 0, config.max_seq_len, device.stream())?;
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
        GpuLayerType::Shortconv => {
            return gpu_shortconv_native_on_stream(
                device, gpu_layer, kv, scratch, layer_idx, pos, config,
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

            eprintln!("DEBUG forward_hybrid: Starting AttentionFusedQkv layer {}", layer_idx);

            // Upload decode state first so pos_ptr has correct pos
            scratch.upload_decode_state(pos, pos + 1, device.stream())?;
            // Upload positions array for RoPE and KV operations
            scratch.upload_positions(pos + 1, 0, config.max_seq_len, device.stream())?;

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

            if supports_gemv_type(wqkv_meta.wtype) {
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
                gpu_attention_decode_from_state(device, scratch, kv, layer_idx, pos, config)
            })?;

            // 7. Attention output projection
            if supports_gemv_type(gpu_layer.attn_o_meta.wtype) {
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
                if supports_gemv_type(gate_meta.wtype)
                    && supports_gemv_type(gpu_layer.ffn_up_meta.wtype)
                {
                    profile_decode_stage(device, DecodeStage::GateUp, || {
                        gpu_dispatch_gate_up_raw_on_stream(
                            device,
                            gate_buf,
                            gate_meta,
                            &gpu_layer.ffn_up,
                            &gpu_layer.ffn_up_meta,
                            scratch.normed.as_ptr() as *const f32,
                            scratch.gate.as_ptr() as *mut f32,
                            scratch.swiglu.as_ptr() as *mut f32,
                            ff_size,
                            h,
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
                    })?;
                } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
                    ensure_size(&mut cpu_s.gate, ff_size);
                    ensure_size(&mut cpu_s.swiglu, ff_size);
                    if let (Some(gate_w), Some(gate_m)) =
                        (cpu_l.ffn_gate.as_ref(), cpu_l.ffn_gate_meta.as_ref())
                    {
                        cpu_fallback_gemv_and_upload(
                            "ffn_gate",
                            gate_w,
                            gate_m,
                            &scratch.normed,
                            &mut cpu_s.normed,
                            &mut cpu_s.gate,
                            &mut scratch.gate,
                            ff_size,
                            h,
                            &mut cpu_s.q8_scratch,
                        )?;
                    }
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
                    )?;
                } else {
                    return Err(GpuError::UnsupportedWeightType {
                        tensor: "ffn_gate/up".to_string(),
                        wtype: gate_meta.wtype,
                    });
                }
            } else {
                // Standard FFN (non-SwiGLU): up -> gelu -> down
                if supports_gemv_type(gpu_layer.ffn_up_meta.wtype) {
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
                } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
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
                        tensor: "ffn_up".to_string(),
                        wtype: gpu_layer.ffn_up_meta.wtype,
                    });
                }
                gelu_on_stream(
                    scratch.swiglu.as_ptr() as *const f32,
                    scratch.swiglu.as_ptr() as *mut f32,
                    ff_size,
                    device.stream(),
                )?;
            }

            // 11. FFN output projection + residual
            if supports_gemv_type(gpu_layer.ffn_down_meta.wtype) {
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
            } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
                ensure_size(&mut cpu_s.swiglu, ff_size);
                ensure_size(&mut cpu_s.gate, h);
                cpu_fallback_gemv_and_upload(
                    "ffn_down",
                    &cpu_l.ffn_down,
                    &cpu_l.ffn_down_meta,
                    &scratch.swiglu,
                    &mut cpu_s.swiglu,
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
            profile_decode_stage(device, DecodeStage::FfnResidual, || {
                residual_add_inplace(device, &scratch.hidden, &scratch.gate, h)
            })?;

            // Fused-QKV layers are fully handled above (RMSNorm + QKV + RoPE +
            // attention + output proj + FFN), mirroring forward_graph.rs. Return
            // here so we never fall through into the separate-Q/K/V standard path
            // below, which would forward the layer a second time.
            return Ok(());
        }
        GpuLayerType::Attention => {}
    }

    let h = config.hidden_size;
    let attn_head_dim = config.head_dim;
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
    let num_q_heads = config.num_heads;
    let num_kv_heads = kv_size / attn_head_dim;
    let attn_out_size = num_q_heads * attn_head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    // Standard hybrid path
    scratch.upload_decode_state(pos, pos + 1, device.stream())?;
    // Upload positions array for RoPE and KV operations
    scratch.upload_positions(pos + 1, 0, config.max_seq_len, device.stream())?;

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
    dump_gpu_f32("attn_normed", &scratch.normed, 5);

    if supports_gemv_type(gpu_layer.attn_q_meta.wtype)
        && supports_gemv_type(gpu_layer.attn_k_meta.wtype)
        && supports_gemv_type(gpu_layer.attn_v_meta.wtype)
    {
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
    } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
        ensure_size(&mut cpu_s.q, q_size);
        ensure_size(&mut cpu_s.k, kv_size);
        ensure_size(&mut cpu_s.v, kv_size);
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
    } else {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "attn_qkv".to_string(),
            wtype: gpu_layer.attn_q_meta.wtype,
        });
    }
    dump_gpu_f32("q", &scratch.q, 5);
    dump_gpu_f32("k", &scratch.k, 5);
    dump_gpu_f32("v", &scratch.v, 5);

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

    dump_gpu_f32("q_rope", &scratch.q, 5);
    dump_gpu_f32("k_rope", &scratch.k, 5);
    dump_gpu_f32("v_raw", &scratch.v, 5);

    profile_decode_stage(device, DecodeStage::Attention, || {
        gpu_attention_decode_from_state(device, scratch, kv, layer_idx, pos, config)
    })?;
    dump_gpu_f32("attn_out", &scratch.attn_out, 5);

    if supports_gemv_type(gpu_layer.attn_o_meta.wtype) {
        profile_decode_stage(device, DecodeStage::AttnProj, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.attn_o,
                &gpu_layer.attn_o_meta,
                scratch.attn_out.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                attn_out_size,
                device.stream(),
            )
        })?;
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
        )?;
    } else {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "attn_o".to_string(),
            wtype: gpu_layer.attn_o_meta.wtype,
        });
    }
    dump_gpu_f32("attn_layer_out", &scratch.layer_out, 5);

    profile_decode_stage(device, DecodeStage::AttnResidual, || {
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
    })?;
    dump_gpu_f32("after_attn_resid", &scratch.hidden, 5);

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
    dump_gpu_f32("ffn_normed", &scratch.normed, 5);

    if gpu_dispatch_moe_ffn_on_stream(device, gpu_layer, scratch, h, ff_size, config)? {
        return Ok(());
    }

    if let (Some(gate_buf), Some(gate_meta)) = (
        gpu_layer.ffn_gate.as_ref(),
        gpu_layer.ffn_gate_meta.as_ref(),
    ) {
        if supports_gemv_type(gate_meta.wtype) && supports_gemv_type(gpu_layer.ffn_up_meta.wtype) {
            profile_decode_stage(device, DecodeStage::GateUp, || {
                gpu_dispatch_gate_up_raw_on_stream(
                    device,
                    gate_buf,
                    gate_meta,
                    &gpu_layer.ffn_up,
                    &gpu_layer.ffn_up_meta,
                    scratch.normed.as_ptr() as *const f32,
                    scratch.gate.as_ptr() as *mut f32,
                    scratch.swiglu.as_ptr() as *mut f32,
                    ff_size,
                    h,
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
            })?;
        } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
            ensure_size(&mut cpu_s.gate, ff_size);
            ensure_size(&mut cpu_s.swiglu, ff_size);
            if let (Some(gate_w), Some(gate_m)) =
                (cpu_l.ffn_gate.as_ref(), cpu_l.ffn_gate_meta.as_ref())
            {
                cpu_fallback_gemv_and_upload(
                    "ffn_gate",
                    gate_w,
                    gate_m,
                    &scratch.normed,
                    &mut cpu_s.normed,
                    &mut cpu_s.gate,
                    &mut scratch.gate,
                    ff_size,
                    h,
                    &mut cpu_s.q8_scratch,
                )?;
            }
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
            )?;
        } else {
            return Err(GpuError::UnsupportedWeightType {
                tensor: "ffn_gate/up".to_string(),
                wtype: gate_meta.wtype,
            });
        }
        dump_gpu_f32("ffn_gate_silu", &scratch.gate, 5);
        dump_gpu_f32("ffn_swiglu", &scratch.swiglu, 5);
    } else {
        // Standard FFN (non-SwiGLU): up -> gelu -> down
        if supports_gemv_type(gpu_layer.ffn_up_meta.wtype) {
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
        } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
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
                tensor: "ffn_up".to_string(),
                wtype: gpu_layer.ffn_up_meta.wtype,
            });
        }
        gelu_on_stream(
            scratch.swiglu.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            device.stream(),
        )?;
    }

    if supports_gemv_type(gpu_layer.ffn_down_meta.wtype) {
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
    } else if let (Some(cpu_l), Some(cpu_s)) = (cpu_layer, cpu_scratch.as_mut()) {
        ensure_size(&mut cpu_s.swiglu, ff_size);
        ensure_size(&mut cpu_s.gate, h);
        cpu_fallback_gemv_and_upload(
            "ffn_down",
            &cpu_l.ffn_down,
            &cpu_l.ffn_down_meta,
            &scratch.swiglu,
            &mut cpu_s.swiglu,
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
    dump_gpu_f32("ffn_down_out", &scratch.gate, 5);

    profile_decode_stage(device, DecodeStage::FfnResidual, || {
        residual_add_inplace(device, &scratch.hidden, &scratch.gate, h)
    })?;
    dump_gpu_f32("after_ffn_resid", &scratch.hidden, 5);

    Ok(())
}
