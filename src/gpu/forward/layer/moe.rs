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
    compressed: &CpuCompressedExperts,
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
    ff_size: usize,
    config: &ModelConfig,
) -> GpuResult<bool> {
    let Some(moe) = gpu_layer.moe.as_ref() else {
        return Ok(false);
    };
    let Some(gate_meta) = gpu_layer.ffn_gate_meta.as_ref() else {
        return Ok(false);
    };
    let Some(gate_buf) = gpu_layer.ffn_gate.as_ref() else {
        return Ok(false);
    };
    let Some(num_experts) = moe_expert_count(gate_meta) else {
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

    let gate_meta = moe_matrix_meta(gate_meta);
    let up_meta = moe_matrix_meta(&gpu_layer.ffn_up_meta);
    let down_meta = moe_matrix_meta(&gpu_layer.ffn_down_meta);
    let gate_stride =
        moe_expert_stride_bytes(&gate_meta, h, ff_size).ok_or_else(|| {
            GpuError::InvalidWeightLayout {
                tensor: "ffn_gate_exps".to_string(),
                dims: gate_meta.dims.clone(),
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

            // gate: output = MPO * normed  [ff_size, h]
            dispatch_mpo_expert(
                gate_mpo,
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
            // up: output = MPO * normed  [ff_size, h]
            dispatch_mpo_expert(
                up_mpo,
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
            // down: hidden += weight * (MPO * swiglu)  [h, ff_size]
            dispatch_mpo_expert(
                down_mpo,
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
        } else if use_compressed {
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
            let gate_ptr = unsafe { gate_buf.as_ptr().add(expert_idx * gate_stride) };
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
