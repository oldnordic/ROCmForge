use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipStreamCaptureStatus, hipStream_t, hip_stream_is_capturing};
use super::super::kernels::{
    gemv_q2_k_f32_on_stream, gemv_q3_k_f32_on_stream, gemv_q4_0_f32_on_stream_unchecked,
    gemv_q4_0_f32_wave32_on_stream_unchecked, gemv_q4_1_f32_on_stream_unchecked,
    gemv_q4_1_f32_wave32_on_stream_unchecked, gemv_q4_k_f32_on_stream, gemv_q5_0_f32_on_stream,
    gemv_q5_1_f32_on_stream, gemv_q5_k_f32_on_stream, gemv_q6_k_f32_on_stream,
    gemv_q8_0_f32_lm_head_on_stream, gemv_q8_0_f32_lm_head_on_stream_variant,
    gemv_q8_0_f32_on_stream,
};
use super::super::launch_autotune::{
    lookup_lm_head_q8_variant, select_lm_head_q8_variant, VariantId,
};
use super::super::safety::{
    disable_wave32_enabled, experimental_q8_activation_fastpath_enabled, force_wave32_enabled,
    launch_autotune_enabled,
};
use super::super::weights::{
    GpuBuffer, GpuMpoWeights, GpuSparseCsrWeights, SvdCorrection, TensorRole, WeightMeta,
};
use crate::gpu::kernel_dispatch_profile::record_gemv_dispatch;
use crate::loader::GgmlType;

use super::fastpath::{q8_fastpath_ok, try_q4_0_q8_0_fastpath};
use super::{is_lm_head_role, supports_gemv_type, validate_gemv_layout};

pub(super) fn dispatch_gemv_impl(
    device: &GpuDevice,
    stream: hipStream_t,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    svd: Option<&SvdCorrection>,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    temp_vector: *mut f32,
) -> GpuResult<()> {
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                if svd.is_none()
                    && experimental_q8_activation_fastpath_enabled()
                    && q8_fastpath_ok(
                        "gemv_q4_0_q8_0",
                        try_q4_0_q8_0_fastpath(
                            device, weights, input, output, in_dim, out_dim, stream,
                        ),
                    )
                {
                    return Ok(());
                }

                let features = super::super::features::GpuFeatures::detect(device).ok();
                let is_rdna3 = features.is_some_and(|f| f.arch.starts_with("gfx11"));
                let use_wave32 = force_wave32_enabled() || (is_rdna3 && !disable_wave32_enabled());

                if use_wave32 {
                    record_gemv_dispatch("Q4_0", "wave32");
                    gemv_q4_0_f32_wave32_on_stream_unchecked(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                } else {
                    record_gemv_dispatch("Q4_0", "wave64");
                    gemv_q4_0_f32_on_stream_unchecked(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                }
            }
            GgmlType::Q4_1 => {
                let features = super::super::features::GpuFeatures::detect(device).ok();
                let is_rdna3 = features.is_some_and(|f| f.arch.starts_with("gfx11"));
                let use_wave32 = force_wave32_enabled() || (is_rdna3 && !disable_wave32_enabled());

                if use_wave32 {
                    record_gemv_dispatch("Q4_1", "wave32");
                    gemv_q4_1_f32_wave32_on_stream_unchecked(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                } else {
                    record_gemv_dispatch("Q4_1", "wave64");
                    gemv_q4_1_f32_on_stream_unchecked(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                }
            }
            GgmlType::Q8_0 => {
                record_gemv_dispatch("Q8_0", "baseline");
                if is_lm_head_role(meta.role) {
                    let capture_active = matches!(
                        hip_stream_is_capturing(stream),
                        Err(_)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
                    );

                    if launch_autotune_enabled() {
                        let variant = if capture_active {
                            lookup_lm_head_q8_variant(in_dim, out_dim)
                                .unwrap_or(VariantId::Baseline)
                        } else {
                            select_lm_head_q8_variant(in_dim, out_dim, |v| {
                                let result = gemv_q8_0_f32_lm_head_on_stream_variant(
                                    weights.as_ptr() as *const u8,
                                    input,
                                    output,
                                    in_dim,
                                    out_dim,
                                    v as i32,
                                    stream,
                                );
                                super::super::ffi::hip_stream_synchronize(stream)?;
                                result
                            })
                        };

                        gemv_q8_0_f32_lm_head_on_stream_variant(
                            weights.as_ptr() as *const u8,
                            input,
                            output,
                            in_dim,
                            out_dim,
                            variant as i32,
                            stream,
                        )?;
                    } else {
                        gemv_q8_0_f32_lm_head_on_stream(
                            weights.as_ptr() as *const u8,
                            input,
                            output,
                            in_dim,
                            out_dim,
                            stream,
                        )?;
                    }
                } else {
                    gemv_q8_0_f32_on_stream(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?
                }
            }
            GgmlType::Q4_K => {
                record_gemv_dispatch("Q4_K", "baseline");
                gemv_q4_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
            }
            GgmlType::Q5_K => {
                record_gemv_dispatch("Q5_K", "baseline");
                gemv_q5_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
            }
            GgmlType::Q6_K => {
                record_gemv_dispatch("Q6_K", "baseline");
                gemv_q6_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
            }
            GgmlType::Q5_0 => {
                record_gemv_dispatch("Q5_0", "baseline");
                gemv_q5_0_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
            }
            GgmlType::Q5_1 => {
                record_gemv_dispatch("Q5_1", "baseline");
                gemv_q5_1_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
            }
            GgmlType::Q2_K => {
                record_gemv_dispatch("Q2_K", "baseline");
                gemv_q2_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
            }
            GgmlType::Q3_K => {
                record_gemv_dispatch("Q3_K", "baseline");
                gemv_q3_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
            }
            _ => unreachable!("unsupported types return before dispatch"),
        }
    }

    if let Some(svd_corr) = svd {
        crate::gpu::kernels::elementwise::dispatch_svd_correction(
            stream,
            &svd_corr.u,
            &svd_corr.v,
            svd_corr.k,
            input,
            output,
            in_dim,
            out_dim,
            temp_vector,
        )?;
    }

    Ok(())
}

pub fn gpu_dispatch_gemv(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    validate_gemv_layout(meta, out_dim, in_dim)?;

    if meta.wtype == GgmlType::F32 {
        crate::gpu::kernels::elementwise::dispatch_gemv_f32_on_stream(
            weights.as_ptr() as *const f32,
            input,
            output,
            in_dim,
            out_dim,
            hipStream_t::null(),
        )?;
        return Ok(());
    }

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemv".to_string(),
            wtype: meta.wtype,
        });
    }

    dispatch_gemv_impl(
        device,
        hipStream_t::null(),
        weights,
        meta,
        None,
        input,
        output,
        out_dim,
        in_dim,
        std::ptr::null_mut(),
    )
}

pub fn gpu_dispatch_gemv_on_stream(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(meta, out_dim, in_dim)?;

    if meta.wtype == GgmlType::F32 {
        crate::gpu::kernels::elementwise::dispatch_gemv_f32_on_stream(
            weights.as_ptr() as *const f32,
            input,
            output,
            in_dim,
            out_dim,
            stream,
        )?;
        return Ok(());
    }

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemv_on_stream".to_string(),
            wtype: meta.wtype,
        });
    }

    dispatch_gemv_impl(
        device,
        stream,
        weights,
        meta,
        None,
        input,
        output,
        out_dim,
        in_dim,
        std::ptr::null_mut(),
    )
}

pub fn gpu_dispatch_gemv_ptr_on_stream(
    device: &GpuDevice,
    weights_ptr: *const u8,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(meta, out_dim, in_dim)?;

    if weights_ptr.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_gemv_ptr_on_stream: weights pointer must be non-null"
                .to_string(),
        });
    }

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemv_ptr_on_stream".to_string(),
            wtype: meta.wtype,
        });
    }

    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                let features = super::super::features::GpuFeatures::detect(device).ok();
                let is_rdna3 = features.is_some_and(|f| f.arch.starts_with("gfx11"));
                let use_wave32 = force_wave32_enabled() || (is_rdna3 && !disable_wave32_enabled());
                if use_wave32 {
                    gemv_q4_0_f32_wave32_on_stream_unchecked(
                        weights_ptr,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                } else {
                    gemv_q4_0_f32_on_stream_unchecked(
                        weights_ptr,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                }
            }
            GgmlType::Q4_1 => {
                let features = super::super::features::GpuFeatures::detect(device).ok();
                let is_rdna3 = features.is_some_and(|f| f.arch.starts_with("gfx11"));
                let use_wave32 = force_wave32_enabled() || (is_rdna3 && !disable_wave32_enabled());
                if use_wave32 {
                    gemv_q4_1_f32_wave32_on_stream_unchecked(
                        weights_ptr,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                } else {
                    gemv_q4_1_f32_on_stream_unchecked(
                        weights_ptr,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                }
            }
            GgmlType::Q8_0 => {
                gemv_q8_0_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            GgmlType::Q4_K => {
                gemv_q4_k_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            GgmlType::Q6_K => {
                gemv_q6_k_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            GgmlType::Q5_0 => {
                gemv_q5_0_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            GgmlType::Q5_1 => {
                gemv_q5_1_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            GgmlType::Q5_K => {
                gemv_q5_k_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            GgmlType::Q2_K => {
                gemv_q2_k_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            GgmlType::Q3_K => {
                gemv_q3_k_f32_on_stream(weights_ptr, input, output, in_dim, out_dim, stream)?;
            }
            _ => unreachable!("unsupported types return before dispatch"),
        }
    }

    Ok(())
}

pub fn gpu_dispatch_sparse_csr_gemv_on_stream(
    _device: &GpuDevice,
    weights: &GpuSparseCsrWeights,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if weights.rows != out_dim || weights.cols != in_dim {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "sparse_csr_gemv".to_string(),
            dims: vec![weights.rows as u64, weights.cols as u64],
            reason: format!(
                "sparse CSR shape {}x{} does not match GEMV dims {}x{}",
                weights.rows, weights.cols, out_dim, in_dim
            ),
        });
    }

    crate::gpu::kernels::dispatch_sparse_csr_gemv_f32(
        weights.values.as_ptr() as *const f32,
        weights.col_idx.as_ptr() as *const u32,
        weights.row_ptr.as_ptr() as *const u32,
        weights.nnz,
        weights.rows,
        weights.cols,
        input,
        output,
        stream,
    )
}

pub fn gpu_dispatch_mpo_apply_on_stream(
    _device: &GpuDevice,
    weights: &GpuMpoWeights,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if weights.site_dims.size() / std::mem::size_of::<u32>() < 6 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "mpo_apply".to_string(),
            dims: vec![weights.site_dims.size() as u64],
            reason: "MPO site_dims too short for 2-site apply".to_string(),
        });
    }

    // site_dims is now a GpuBuffer, can't index directly. Skip dim validation here
    // and let the kernel handle it.

    crate::gpu::kernels::dispatch_mpo_apply_f32(
        weights.site_data.as_ptr() as *const f32,
        weights.site_dims.as_ptr() as *const u32,
        weights.n_sites as usize,
        out_dim,
        in_dim,
        input,
        output,
        stream,
    )
}

/// Dispatch GEMV with automatic fallback to sparse CSR or MPO if available.
///
/// **Safety:** sparse CSR and MPO kernels are experimental and gated by
/// `ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1`.  Without that flag the
/// dispatcher ignores sparse/MPO weights and falls back to the dense GEMV
/// path, preventing untested kernels from running on a display-attached GPU.
pub fn gpu_dispatch_gemv_with_fallback_on_stream(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    svd: Option<&SvdCorrection>,
    sparse_weights: Option<&GpuSparseCsrWeights>,
    mpo_weights: Option<&GpuMpoWeights>,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    temp_vector: *mut f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    // Experimental kernels (sparse CSR, MPO) are opt-in only.
    // Never dispatch them on a display-attached GPU unless explicitly enabled.
    if super::super::safety::experimental_gpu_kernels_enabled() {
        if let Some(sparse) = sparse_weights {
            // Compute y = sparse_residual * x.
            gpu_dispatch_sparse_csr_gemv_on_stream(
                device, sparse, input, output, out_dim, in_dim, stream,
            )?;
            // If SVD correction is also present (SvdSparseCsr type), add U·(V·x).
            if let Some(svd_corr) = svd {
                crate::gpu::kernels::elementwise::dispatch_svd_correction(
                    stream,
                    &svd_corr.u,
                    &svd_corr.v,
                    svd_corr.k,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    temp_vector,
                )?;
            }
            return Ok(());
        }

        if let Some(mpo) = mpo_weights {
            return gpu_dispatch_mpo_apply_on_stream(
                device, mpo, input, output, out_dim, in_dim, stream,
            );
        }
    }

    gpu_dispatch_gemv_svd_on_stream(
        device,
        weights,
        meta,
        svd,
        input,
        output,
        out_dim,
        in_dim,
        temp_vector,
        stream,
    )
}

pub fn gpu_dispatch_gemv_svd_on_stream(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    svd: Option<&SvdCorrection>,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    temp_vector: *mut f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(meta, out_dim, in_dim)?;

    if meta.wtype == GgmlType::F32 {
        if let Some(svd_corr) = svd {
            crate::gpu::kernels::elementwise::dispatch_svd_correction(
                stream,
                &svd_corr.u,
                &svd_corr.v,
                svd_corr.k,
                input,
                output,
                in_dim,
                out_dim,
                temp_vector,
            )?;
        } else {
            crate::gpu::kernels::elementwise::dispatch_gemv_f32_on_stream(
                weights.as_ptr() as *const f32,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?;
        }
        return Ok(());
    }

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemv_svd_on_stream".to_string(),
            wtype: meta.wtype,
        });
    }

    dispatch_gemv_impl(
        device,
        stream,
        weights,
        meta,
        svd,
        input,
        output,
        out_dim,
        in_dim,
        temp_vector,
    )
}
