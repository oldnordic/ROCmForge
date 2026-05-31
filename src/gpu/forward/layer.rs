use crate::config::ModelConfig;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, kv_write_from_state_on_stream,
    kv_write_rope_from_state_on_stream, kv_write_rope_on_stream,
};
use crate::gpu::kernels::gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream;
use crate::gpu::kernels::rope::{rope_heads_from_state_on_stream, rope_heads_on_stream};
use crate::gpu::kernels::{
    dot_f16_f32_on_stream, mul_on_stream, rms_norm_batched, silu_on_stream, weighted_add_on_stream,
};
use crate::gpu::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_fused_qkv_gqa_on_stream,
    gpu_dispatch_fused_qkv_on_stream, gpu_dispatch_gemv_on_stream, gpu_dispatch_gemv_ptr_on_stream,
    gpu_dispatch_gemv_residual_on_stream, gpu_dispatch_gemv_svd_on_stream,
    gpu_dispatch_gemv_with_fallback_on_stream, gpu_dispatch_rms_norm,
};
use crate::gpu::weights::{GpuLayerWeights, WeightMeta};
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

pub(super) fn gpu_attention_decode_from_state(
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

const QWEN_MOE_TOP_K: usize = 8;

fn moe_expert_count(meta: &WeightMeta) -> Option<usize> {
    if meta.dims.len() == 3 {
        Some(meta.dims[2] as usize)
    } else {
        None
    }
}

fn moe_expert_stride_bytes(meta: &WeightMeta, in_dim: usize, out_dim: usize) -> Option<usize> {
    let experts = moe_expert_count(meta)?;
    if experts == 0 {
        return None;
    }
    let matrix_elements = in_dim.checked_mul(out_dim)?;
    Some(meta.wtype.bytes_for_elements(matrix_elements))
}

fn moe_matrix_meta(meta: &WeightMeta) -> WeightMeta {
    let mut matrix_meta = meta.clone();
    if matrix_meta.dims.len() == 3 {
        matrix_meta.dims.truncate(2);
    }
    matrix_meta
}

fn select_moe_topk_weights(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let k = top_k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(k);

    let max_logit = indexed
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for (_, value) in &mut indexed {
        *value = (*value - max_logit).exp();
        denom += *value;
    }
    if denom > 0.0 {
        for (_, value) in &mut indexed {
            *value /= denom;
        }
    }

    indexed
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn fwht_inplace(a: &mut [f32]) {
    let n = a.len();
    assert!(n.is_power_of_two(), "FWHT length must be a power of 2");
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in 0..h {
                let x = a[i + j];
                let y = a[i + j + h];
                a[i + j] = x + y;
                a[i + j + h] = x - y;
            }
        }
        h *= 2;
    }
}

fn fwht_inplace_normalized(a: &mut [f32]) {
    fwht_inplace(a);
    let scale = 1.0 / (a.len() as f32).sqrt();
    for x in a.iter_mut() {
        *x *= scale;
    }
}

/// Dispatch one compressed expert's GEMV: output = CSR_residual * input + U*(V*input).
///
/// Zeros `output` first (CSR kernel atomicAdds; we need a clean slate per expert).
/// When `accumulate` is true, also does `weighted_add(output, accum, weight, rows)`.
/// Set `accumulate=false` for gate/up (silu/mul still needed before accumulate),
/// `accumulate=true` for down (result goes directly into the hidden accumulator).
fn dispatch_compressed_expert(
    device: &GpuDevice,
    compressed: &crate::gpu::weights::CpuCompressedExperts,
    scratch: &mut crate::gpu::cache::GpuExpertScratch,
    expert_idx: usize,
    input: *const f32,
    output: *mut f32,
    accum: *mut f32,
    weight: f32,
    rows: usize,
    cols: usize,
    accumulate: bool,
    stream: crate::gpu::ffi::hipStream_t,
) -> GpuResult<()> {
    use crate::gpu::ffi;
    use crate::gpu::kernels::{dispatch_sparse_csr_gemv_f32, elementwise::dispatch_svd_correction};

    let u_bytes = compressed.u_bytes(expert_idx);
    let v_bytes = compressed.v_bytes(expert_idx);
    let (rp_bytes, ci_bytes, val_bytes, nnz) = compressed.csr_bytes(expert_idx);

    // Apply Fast Walsh-Hadamard Transform (FWHT) to input activation on host if required
    let active_input = if compressed.needs_fwht_input {
        let mut host_input = vec![0.0f32; cols];
        ffi::hip_memcpy_d2h(
            host_input.as_mut_ptr() as *mut u8,
            input as *const u8,
            cols * 4,
        )?;
        fwht_inplace_normalized(&mut host_input);
        ffi::hip_memcpy_h2d(
            scratch.rotated_input.as_ptr(),
            host_input.as_ptr() as *const u8,
            cols * 4,
        )?;
        scratch.rotated_input.as_ptr() as *const f32
    } else {
        input
    };

    // Upload U and V
    ffi::hip_memcpy_h2d(scratch.u.as_ptr(), u_bytes.as_ptr(), u_bytes.len())?;
    ffi::hip_memcpy_h2d(scratch.v.as_ptr(), v_bytes.as_ptr(), v_bytes.len())?;
    // Upload CSR row pointers (always present)
    ffi::hip_memcpy_h2d(
        scratch.csr_row_ptr.as_ptr(),
        rp_bytes.as_ptr(),
        rp_bytes.len(),
    )?;

    // Zero output before atomicAdd-based CSR kernel
    ffi::hip_memset(output as *mut u8, 0u8, rows * 4)?;

    // Sparse residual contribution: output += CSR * active_input
    if nnz > 0 {
        ffi::hip_memcpy_h2d(
            scratch.csr_col_idx.as_ptr(),
            ci_bytes.as_ptr(),
            ci_bytes.len(),
        )?;
        ffi::hip_memcpy_h2d(
            scratch.csr_values.as_ptr(),
            val_bytes.as_ptr(),
            val_bytes.len(),
        )?;
        dispatch_sparse_csr_gemv_f32(
            scratch.csr_values.as_ptr() as *const f32,
            scratch.csr_col_idx.as_ptr() as *const u32,
            scratch.csr_row_ptr.as_ptr() as *const u32,
            nnz,
            rows,
            cols,
            active_input,
            output,
            stream,
        )?;
    }

    // SVD correction: output += U * (V * active_input)
    dispatch_svd_correction(
        stream,
        &scratch.u,
        &scratch.v,
        scratch.k,
        active_input,
        output,
        cols,
        rows,
        scratch.temp_v.as_ptr() as *mut f32,
    )?;

    if accumulate {
        weighted_add_on_stream(output as *const f32, accum, weight, rows, stream)?;
    }

    Ok(())
}

fn gpu_dispatch_moe_ffn_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    scratch: &mut GpuForwardScratch,
    h: usize,
    ff_size: usize,
) -> GpuResult<bool> {
    let Some(moe) = gpu_layer.moe.as_ref() else {
        return Ok(false);
    };
    let Some(num_experts) = moe_expert_count(&gpu_layer.ffn_gate_meta) else {
        return Ok(false);
    };

    let stream = device.stream();

    gpu_dispatch_gemv_svd_on_stream(
        device,
        &moe.router,
        &moe.router_meta,
        moe.router_svd.as_ref(),
        scratch.normed.as_ptr() as *const f32,
        scratch.gate.as_ptr() as *mut f32,
        num_experts,
        h,
        scratch.svd_scratch.as_ptr() as *mut f32,
        stream,
    )?;
    device.synchronize()?;
    let router_logits = scratch.gate.copy_to_host_vec()?;
    let selected = select_moe_topk_weights(&router_logits[..num_experts], QWEN_MOE_TOP_K);

    let gate_meta = moe_matrix_meta(&gpu_layer.ffn_gate_meta);
    let up_meta = moe_matrix_meta(&gpu_layer.ffn_up_meta);
    let down_meta = moe_matrix_meta(&gpu_layer.ffn_down_meta);
    let gate_stride =
        moe_expert_stride_bytes(&gpu_layer.ffn_gate_meta, h, ff_size).ok_or_else(|| {
            GpuError::InvalidWeightLayout {
                tensor: "ffn_gate_exps".to_string(),
                dims: gpu_layer.ffn_gate_meta.dims.clone(),
                reason: "invalid MoE gate expert tensor shape".to_string(),
            }
        })?;
    let up_stride =
        moe_expert_stride_bytes(&gpu_layer.ffn_up_meta, h, ff_size).ok_or_else(|| {
            GpuError::InvalidWeightLayout {
                tensor: "ffn_up_exps".to_string(),
                dims: gpu_layer.ffn_up_meta.dims.clone(),
                reason: "invalid MoE up expert tensor shape".to_string(),
            }
        })?;
    let down_stride =
        moe_expert_stride_bytes(&gpu_layer.ffn_down_meta, ff_size, h).ok_or_else(|| {
            GpuError::InvalidWeightLayout {
                tensor: "ffn_down_exps".to_string(),
                dims: gpu_layer.ffn_down_meta.dims.clone(),
                reason: "invalid MoE down expert tensor shape".to_string(),
            }
        })?;

    // Check if this layer has compressed (SVD+sparse) expert weights.
    let use_compressed = gpu_layer.ffn_gate_compressed.is_some()
        && gpu_layer.ffn_up_compressed.is_some()
        && gpu_layer.ffn_down_compressed.is_some()
        && crate::gpu::safety::experimental_gpu_kernels_enabled();

    for (expert_idx, weight) in selected {
        if use_compressed {
            // Compressed path: H2D upload + CSR GEMV + SVD correction.
            // use_compressed is only true when all three are Some, so these are always inhabited.
            let (Some(gate_c), Some(up_c), Some(down_c)) = (
                gpu_layer.ffn_gate_compressed.as_ref(),
                gpu_layer.ffn_up_compressed.as_ref(),
                gpu_layer.ffn_down_compressed.as_ref(),
            ) else {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: "compressed expert fields inconsistent".to_string(),
                });
            };
            let escratch =
                scratch
                    .expert_scratch
                    .as_mut()
                    .ok_or_else(|| {
                        GpuError::HipApiError {
                    code: -1,
                    description:
                        "expert_scratch not initialised — call init_expert_scratch before decode"
                            .to_string(),
                }
                    })?;

            // gate: output = CSR * normed + U*(V*normed)  [ff_size, h]
            dispatch_compressed_expert(
                device,
                gate_c,
                escratch,
                expert_idx,
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                std::ptr::null_mut(),
                1.0,
                ff_size,
                h,
                false,
                stream,
            )?;
            // up: output = CSR * normed + U*(V*normed)  [ff_size, h]
            dispatch_compressed_expert(
                device,
                up_c,
                escratch,
                expert_idx,
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                std::ptr::null_mut(),
                1.0,
                ff_size,
                h,
                false,
                stream,
            )?;
            // SwiGLU: gate = silu(gate) * up
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
            // down: hidden += weight * (CSR * swiglu + U*(V*swiglu))  [h, ff_size]
            dispatch_compressed_expert(
                device,
                down_c,
                escratch,
                expert_idx,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                scratch.hidden.as_ptr() as *mut f32,
                weight,
                h,
                ff_size,
                true,
                stream,
            )?;
        } else {
            // Dense path (original): stride into packed 3D expert buffer.
            let gate_ptr = unsafe { gpu_layer.ffn_gate.as_ptr().add(expert_idx * gate_stride) };
            let up_ptr = unsafe { gpu_layer.ffn_up.as_ptr().add(expert_idx * up_stride) };
            let down_ptr = unsafe { gpu_layer.ffn_down.as_ptr().add(expert_idx * down_stride) };

            gpu_dispatch_gemv_ptr_on_stream(
                device,
                gate_ptr as *const u8,
                &gate_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                ff_size,
                h,
                stream,
            )?;
            gpu_dispatch_gemv_ptr_on_stream(
                device,
                up_ptr as *const u8,
                &up_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                ff_size,
                h,
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
            gpu_dispatch_gemv_ptr_on_stream(
                device,
                down_ptr as *const u8,
                &down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                ff_size,
                stream,
            )?;
            weighted_add_on_stream(
                scratch.layer_out.as_ptr() as *const f32,
                scratch.hidden.as_ptr() as *mut f32,
                weight,
                h,
                stream,
            )?;
        }
    }

    if let (
        Some(shared_gate),
        Some(shared_gate_meta),
        Some(shared_up),
        Some(shared_up_meta),
        Some(shared_down),
        Some(shared_down_meta),
    ) = (
        moe.shared_gate.as_ref(),
        moe.shared_gate_meta.as_ref(),
        moe.shared_up.as_ref(),
        moe.shared_up_meta.as_ref(),
        moe.shared_down.as_ref(),
        moe.shared_down_meta.as_ref(),
    ) {
        gpu_dispatch_gemv_svd_on_stream(
            device,
            shared_gate,
            shared_gate_meta,
            moe.shared_gate_svd.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            scratch.gate.as_ptr() as *mut f32,
            ff_size,
            h,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
        gpu_dispatch_gemv_svd_on_stream(
            device,
            shared_up,
            shared_up_meta,
            moe.shared_up_svd.as_ref(),
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
        gpu_dispatch_gemv_svd_on_stream(
            device,
            shared_down,
            shared_down_meta,
            moe.shared_down_svd.as_ref(),
            scratch.swiglu.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            ff_size,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;

        let shared_weight = if let (Some(gate_inp), Some(gate_inp_meta)) = (
            moe.shared_gate_inp.as_ref(),
            moe.shared_gate_inp_meta.as_ref(),
        ) {
            if gate_inp_meta.wtype == GgmlType::F16 {
                dot_f16_f32_on_stream(
                    gate_inp.as_ptr() as *const u8,
                    scratch.normed.as_ptr() as *const f32,
                    scratch.q.as_ptr() as *mut f32,
                    h,
                    stream,
                )?;
            } else {
                gpu_dispatch_gemv_svd_on_stream(
                    device,
                    gate_inp,
                    gate_inp_meta,
                    None,
                    scratch.normed.as_ptr() as *const f32,
                    scratch.q.as_ptr() as *mut f32,
                    1,
                    h,
                    scratch.svd_scratch.as_ptr() as *mut f32,
                    stream,
                )?;
            }
            device.synchronize()?;
            sigmoid(
                scratch
                    .q
                    .copy_to_host_vec()?
                    .first()
                    .copied()
                    .unwrap_or(0.0),
            )
        } else {
            1.0
        };

        weighted_add_on_stream(
            scratch.layer_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            shared_weight,
            h,
            stream,
        )?;
    }

    Ok(true)
}

pub(super) fn gpu_layer_forward_from_state_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    if gpu_layer.ssm.is_some() {
        return gpu_layer_forward_ssm_on_stream(device, gpu_layer, kv, scratch, layer_idx, config);
    }

    // Block layer types we can't yet handle: fused QKV input projection for non-SSM layers.
    // Per-head QK norms (attn_q_norm, attn_k_norm) are handled below.
    if gpu_layer.attn_qkv.is_some() {
        return Err(crate::gpu::error::GpuError::UnsupportedOperation {
            operation: "attention decode".to_string(),
            reason: "fused QKV input projection on non-SSM layers is not supported".to_string(),
        });
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
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    if gpu_layer.ssm.is_some() {
        // Upload decode state first so pos_ptr has correct pos
        scratch.upload_decode_state(pos, pos + 1, device.stream())?;
        return gpu_layer_forward_ssm_on_stream(device, gpu_layer, kv, scratch, layer_idx, config);
    }

    // Block layer types we can't yet handle: fused QKV input projection for non-SSM layers.
    // Per-head QK norms (attn_q_norm, attn_k_norm) are handled below.
    if gpu_layer.attn_qkv.is_some() {
        return Err(crate::gpu::error::GpuError::UnsupportedOperation {
            operation: "attention decode".to_string(),
            reason: "fused QKV input projection on non-SSM layers is not supported".to_string(),
        });
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
        })?;
        profile_decode_stage(device, DecodeStage::FfnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    Ok(())
}

fn gpu_layer_forward_ssm_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
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

    gpu_dispatch_gemv_svd_on_stream(
        device,
        wqkv,
        wqkv_meta,
        gpu_layer.attn_qkv_meta.as_ref().and_then(|meta| {
            if meta.svd_k.is_some() {
                // Return None for now as SVD correction for fused QKV is not pre-calculated
                None
            } else {
                None
            }
        }),
        scratch.normed.as_ptr() as *const f32,
        qkv_ptr,
        qkv_dim,
        h,
        scratch.svd_scratch.as_ptr() as *mut f32,
        stream,
    )?;

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

    gpu_dispatch_gemv_svd_on_stream(
        device,
        wz,
        wz_meta,
        None,
        scratch.normed.as_ptr() as *const f32,
        z_ptr,
        d_inner,
        h,
        scratch.svd_scratch.as_ptr() as *mut f32,
        stream,
    )?;

    // 4. Beta + alpha projections
    let beta_out_ptr = scratch.q.as_ptr() as *mut f32;
    let alpha_out_ptr = scratch.k.as_ptr() as *mut f32;

    // Dynamically retrieve actual SSM heads count directly from weight shape
    let ssm_heads = if ssm.beta_meta.dims[0] as usize == h {
        ssm.beta_meta.dims[1] as usize
    } else {
        ssm.beta_meta.dims[0] as usize
    }; // e.g. 32

    gpu_dispatch_gemv_svd_on_stream(
        device,
        &ssm.beta,
        &ssm.beta_meta,
        gpu_layer
            .ssm
            .as_ref()
            .and_then(|s| {
                s.beta_meta.svd_k.map(|_| {
                    // Outlier corrections for SSM are not pre-calculated, default to None
                    None
                })
            })
            .flatten(),
        scratch.normed.as_ptr() as *const f32,
        beta_out_ptr,
        ssm_heads,
        h,
        scratch.svd_scratch.as_ptr() as *mut f32,
        stream,
    )?;

    gpu_dispatch_gemv_svd_on_stream(
        device,
        &ssm.alpha,
        &ssm.alpha_meta,
        None,
        scratch.normed.as_ptr() as *const f32,
        alpha_out_ptr,
        ssm_heads,
        h,
        scratch.svd_scratch.as_ptr() as *mut f32,
        stream,
    )?;

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
        gpu_dispatch_gemv_svd_on_stream(
            device,
            &ssm.out,
            &ssm.out_meta,
            None,
            normed_out_ptr,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            q_dim,
            scratch.svd_scratch.as_ptr() as *mut f32,
            stream,
        )?;
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
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::weights::{TensorRole, WeightMeta};

    fn meta(wtype: GgmlType, dims: &[u64]) -> WeightMeta {
        WeightMeta {
            wtype,
            dims: dims.to_vec(),
            needs_transpose: false,
            role: TensorRole::Generic,
            svd_k: None,
        }
    }

    #[test]
    fn moe_expert_count_reads_third_dimension() {
        assert_eq!(
            moe_expert_count(&meta(GgmlType::Q4_K, &[2048, 512, 256])),
            Some(256)
        );
        assert_eq!(moe_expert_count(&meta(GgmlType::Q4_0, &[2048, 512])), None);
    }

    #[test]
    fn moe_expert_stride_uses_quantized_matrix_size() {
        assert_eq!(
            moe_expert_stride_bytes(&meta(GgmlType::Q4_K, &[2048, 512, 256]), 2048, 512),
            Some(589_824)
        );
        assert_eq!(
            moe_expert_stride_bytes(&meta(GgmlType::Q6_K, &[512, 2048, 256]), 512, 2048),
            Some(860_160)
        );
    }

    #[test]
    fn select_moe_topk_weights_softmaxes_selected_logits() {
        let selected = select_moe_topk_weights(&[0.0, 3.0, 1.0, 2.0], 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, 1);
        assert_eq!(selected[1].0, 3);
        assert!((selected[0].1 + selected[1].1 - 1.0).abs() < 1e-6);
        assert!(selected[0].1 > selected[1].1);
    }
}
