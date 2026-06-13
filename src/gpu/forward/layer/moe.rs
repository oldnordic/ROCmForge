use crate::config::ModelConfig;
use crate::gpu::cache::{GpuExpertScratch, GpuForwardScratch};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::kernels::{
    dispatch_mpo_apply_f32, dispatch_sparse_csr_gemv_f32, dot_f16_f32_on_stream,
    elementwise::dispatch_svd_correction, mul_on_stream, silu_on_stream, weighted_add_on_stream,
};
use crate::gpu::ops::{gpu_dispatch_gemv_ptr_on_stream, gpu_dispatch_gemv_svd_on_stream};
use crate::gpu::weights::{CpuCompressedExperts, CpuMpoExperts, GpuLayerWeights, WeightMeta};
use crate::loader::GgmlType;

const QWEN_MOE_TOP_K: usize = 8;

pub(super) fn moe_expert_count(meta: &WeightMeta) -> Option<usize> {
    if meta.dims.len() == 3 {
        Some(meta.dims[2] as usize)
    } else {
        None
    }
}

fn moe_expert_stride_bytes(meta: &WeightMeta, num_experts: usize, in_dim: usize, out_dim: usize) -> Option<usize> {
    if num_experts == 0 {
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
fn dispatch_compressed_expert(
    device: &GpuDevice,
    experts: &CpuCompressedExperts,
    scratch: &mut GpuExpertScratch,
    expert_idx: usize,
    active_input: *const f32,
    output: *mut f32,
    accum: *mut f32,
    weight: f32,
    rows: usize,
    cols: usize,
    accumulate: bool,
    stream: crate::gpu::ffi::hipStream_t,
) -> GpuResult<()> {
    let (rp_bytes, ci_bytes, val_bytes, nnz) = experts.csr_bytes(expert_idx);

    // Upload U/V for this expert
    ffi::hip_memcpy_h2d(
        scratch.u.as_ptr(),
        experts.u_bytes(expert_idx).as_ptr(),
        experts.rows * experts.k * 4,
    )?;
    ffi::hip_memcpy_h2d(
        scratch.v.as_ptr(),
        experts.v_bytes(expert_idx).as_ptr(),
        experts.cols * experts.k * 4,
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
        // Also need to upload row_ptr for this expert?
        // Actually, our CSR kernel expects full row_ptr for the whole matrix.
        // For per-expert, it might be simpler to use a single row_ptr block in scratch.
        ffi::hip_memcpy_h2d(
            scratch.csr_row_ptr.as_ptr(),
            rp_bytes.as_ptr(),
            rp_bytes.len(),
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

/// Dispatch one MPO-compressed expert: output = MPO * input.
///
/// Uploads the expert's site data from CPU-resident `CpuMpoExperts` into
/// `GpuExpertScratch`, then dispatches the MPO apply kernel.
fn dispatch_mpo_expert(
    mpo: &CpuMpoExperts,
    scratch: &mut GpuExpertScratch,
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
    let site_bytes = mpo.site_bytes(expert_idx);
    let chi = mpo.chi_max;

    // Upload site data (U_sigma + V^T concatenated)
    ffi::hip_memcpy_h2d(
        scratch.mpo_site_data.as_ptr(),
        site_bytes.as_ptr(),
        site_bytes.len(),
    )?;

    // Build and upload site_dims: [1, rows, chi, 1, chi, cols, 1, 1]
    let site_dims_host: Vec<u32> =
        vec![1, rows as u32, chi as u32, 1, chi as u32, cols as u32, 1, 1];
    let site_dims_bytes = unsafe {
        std::slice::from_raw_parts(
            site_dims_host.as_ptr() as *const u8,
            site_dims_host.len() * std::mem::size_of::<u32>(),
        )
    };
    ffi::hip_memcpy_h2d(
        scratch.mpo_site_dims.as_ptr(),
        site_dims_bytes.as_ptr(),
        site_dims_bytes.len(),
    )?;

    // Zero output (MPO kernel does direct write, not atomicAdd, but zeroing keeps parity)
    ffi::hip_memset(output as *mut u8, 0u8, rows * 4)?;

    // Dispatch MPO apply: y = MPO * x
    dispatch_mpo_apply_f32(
        scratch.mpo_site_data.as_ptr() as *const f32,
        scratch.mpo_site_dims.as_ptr() as *const u32,
        2, // n_sites
        rows,
        cols,
        input,
        output,
        stream,
    )?;

    if accumulate {
        weighted_add_on_stream(output as *const f32, accum, weight, rows, stream)?;
    }

    Ok(())
}

pub(super) fn gpu_dispatch_moe_ffn_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    scratch: &mut GpuForwardScratch,
    h: usize,
    _ff_size: usize,
    config: &ModelConfig,
) -> GpuResult<bool> {
    let Some(moe) = gpu_layer.moe.as_ref() else {
        return Ok(false);
    };
    let Some(gate_meta_orig) = gpu_layer.ffn_gate_meta.as_ref() else {
        return Ok(false);
    };
    let Some(gate_buf) = gpu_layer.ffn_gate.as_ref() else {
        return Ok(false);
    };
    let Some(num_experts) = moe_expert_count(gate_meta_orig) else {
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
    let mut router_logits = scratch.gate.copy_to_host_vec()?;
    let top_k = config.num_experts_per_tok.unwrap_or(4);
    if let Some(ref bias) = moe.router_bias {
        let bias_host = bias.copy_to_host_vec()?;
        for i in 0..num_experts {
            router_logits[i] += bias_host[i];
        }
    }
    let selected = select_moe_topk_weights(&router_logits[..num_experts], top_k);

    let gate_meta = moe_matrix_meta(gate_meta_orig);
    let up_meta = moe_matrix_meta(&gpu_layer.ffn_up_meta);
    let down_meta = moe_matrix_meta(&gpu_layer.ffn_down_meta);

    let (exp_in, exp_out) = (gate_meta.dims[0] as usize, gate_meta.dims[1] as usize);
    let (down_in, down_out) = (down_meta.dims[0] as usize, down_meta.dims[1] as usize);

    let gate_stride =
        moe_expert_stride_bytes(&gate_meta, num_experts, exp_in, exp_out).ok_or_else(|| {
            GpuError::InvalidWeightLayout {
                tensor: "ffn_gate_exps".to_string(),
                dims: gate_meta.dims.clone(),
                reason: "invalid MoE gate expert tensor shape".to_string(),
            }
        })?;
    let up_stride =
        moe_expert_stride_bytes(&up_meta, num_experts, exp_in, exp_out).ok_or_else(|| {
            GpuError::InvalidWeightLayout {
                tensor: "ffn_up_exps".to_string(),
                dims: up_meta.dims.clone(),
                reason: "invalid MoE up expert tensor shape".to_string(),
            }
        })?;
    let down_stride =
        moe_expert_stride_bytes(&down_meta, num_experts, down_in, down_out).ok_or_else(|| {
            GpuError::InvalidWeightLayout {
                tensor: "ffn_down_exps".to_string(),
                dims: down_meta.dims.clone(),
                reason: "invalid MoE down expert tensor shape".to_string(),
            }
        })?;

    // Check if this layer has compressed (SVD+sparse) expert weights.
    let use_compressed = gpu_layer.ffn_gate_compressed.is_some()
        && gpu_layer.ffn_up_compressed.is_some()
        && gpu_layer.ffn_down_compressed.is_some()
        && crate::gpu::safety::experimental_gpu_kernels_enabled();

    // Check if this layer has MPO-compressed expert weights.
    let use_mpo = gpu_layer.ffn_gate_mpo_experts.is_some()
        && gpu_layer.ffn_up_mpo_experts.is_some()
        && gpu_layer.ffn_down_mpo_experts.is_some()
        && crate::gpu::safety::experimental_gpu_kernels_enabled();

    for (expert_idx, weight) in selected {
        if use_mpo {
            let (Some(gate_mpo), Some(up_mpo), Some(down_mpo)) = (
                gpu_layer.ffn_gate_mpo_experts.as_ref(),
                gpu_layer.ffn_up_mpo_experts.as_ref(),
                gpu_layer.ffn_down_mpo_experts.as_ref(),
            ) else {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: "MPO expert fields inconsistent".to_string(),
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

            // gate: output = MPO * normed  [exp_out, h]
            dispatch_mpo_expert(
                gate_mpo,
                escratch,
                expert_idx,
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                std::ptr::null_mut(),
                1.0,
                exp_out,
                h,
                false,
                stream,
            )?;
            // up: output = MPO * normed  [exp_out, h]
            dispatch_mpo_expert(
                up_mpo,
                escratch,
                expert_idx,
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                std::ptr::null_mut(),
                1.0,
                exp_out,
                h,
                false,
                stream,
            )?;
            // SwiGLU: gate = silu(gate) * up
            silu_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                exp_out,
                stream,
            )?;
            mul_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                exp_out,
                stream,
            )?;
            // down: hidden += weight * (MPO * swiglu)  [h, exp_out]
            dispatch_mpo_expert(
                down_mpo,
                escratch,
                expert_idx,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                scratch.hidden.as_ptr() as *mut f32,
                weight,
                h,
                exp_out,
                true,
                stream,
            )?;
        } else if use_compressed {
            // Compressed path: H2D upload + CSR GEMV + SVD correction.
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

            // gate: output = CSR * normed + U*(V*normed)  [exp_out, h]
            dispatch_compressed_expert(
                device,
                gate_c,
                escratch,
                expert_idx,
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                std::ptr::null_mut(),
                1.0,
                exp_out,
                h,
                false,
                stream,
            )?;
            // up: output = CSR * normed + U*(V*normed)  [exp_out, h]
            dispatch_compressed_expert(
                device,
                up_c,
                escratch,
                expert_idx,
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                std::ptr::null_mut(),
                1.0,
                exp_out,
                h,
                false,
                stream,
            )?;
            // SwiGLU: gate = silu(gate) * up
            silu_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                exp_out,
                stream,
            )?;
            mul_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                exp_out,
                stream,
            )?;
            // down: hidden += weight * (CSR * swiglu + U*(V*swiglu))  [h, exp_out]
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
                exp_out,
                true,
                stream,
            )?;
        } else {
            // Dense path (original): stride into packed 3D expert buffer.
            let gate_ptr = unsafe { gate_buf.as_ptr().add(expert_idx * gate_stride) };
            let up_ptr = unsafe { gpu_layer.ffn_up.as_ptr().add(expert_idx * up_stride) };
            let down_ptr = unsafe { gpu_layer.ffn_down.as_ptr().add(expert_idx * down_stride) };

            gpu_dispatch_gemv_ptr_on_stream(
                device,
                gate_ptr as *const u8,
                &gate_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                exp_out,
                exp_in,
                stream,
            )?;
            gpu_dispatch_gemv_ptr_on_stream(
                device,
                up_ptr as *const u8,
                &up_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                exp_out,
                exp_in,
                stream,
            )?;
            // SwiGLU: gate = silu(gate) * up
            silu_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.gate.as_ptr() as *mut f32,
                exp_out,
                stream,
            )?;
            mul_on_stream(
                scratch.gate.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.swiglu.as_ptr() as *mut f32,
                exp_out,
                stream,
            )?;
            // down: hidden += weight * (W_down * swiglu)
            gpu_dispatch_gemv_ptr_on_stream(
                device,
                down_ptr as *const u8,
                &down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                down_out,
                down_in,
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

    Ok(true)
}
