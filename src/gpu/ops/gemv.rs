use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipStreamCaptureStatus, hipStream_t, hip_stream_is_capturing};
use super::super::kernels::{
    gemv_q4_0_f32_on_stream_unchecked, gemv_q4_0_f32_wave32_on_stream_unchecked,
    gemv_q4_1_f32_on_stream_unchecked, gemv_q4_1_f32_wave32_on_stream_unchecked,
    gemv_q6_k_f32_on_stream, gemv_q8_0_f32_lm_head_on_stream,
    gemv_q8_0_f32_lm_head_on_stream_variant, gemv_q8_0_f32_on_stream,
};
use super::super::launch_autotune::{
    lookup_lm_head_q8_variant, select_lm_head_q8_variant, VariantId,
};
use super::super::safety::{experimental_q8_activation_fastpath_enabled, launch_autotune_enabled};
use super::super::weights::{GpuBuffer, TensorRole, WeightMeta};
use crate::loader::GgmlType;

use super::fastpath::{q8_fastpath_ok, try_q4_0_q8_0_fastpath};
use super::{is_lm_head_role, supports_gemv_type, validate_gemv_layout};

pub(super) fn dispatch_gemv_impl(
    device: &GpuDevice,
    stream: hipStream_t,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                if experimental_q8_activation_fastpath_enabled()
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
                let is_rdna3 = features.map_or(false, |f| f.arch.starts_with("gfx11"));

                if is_rdna3 {
                    gemv_q4_0_f32_wave32_on_stream_unchecked(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                    return Ok(());
                }

                gemv_q4_0_f32_on_stream_unchecked(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?
            }
            GgmlType::Q4_1 => {
                let features = super::super::features::GpuFeatures::detect(device).ok();
                let is_rdna3 = features.map_or(false, |f| f.arch.starts_with("gfx11"));

                if is_rdna3 {
                    gemv_q4_1_f32_wave32_on_stream_unchecked(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                } else {
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
                return Err(GpuError::UnsupportedOperation {
                    operation: format!("gpu_dispatch_gemv_on_stream for {:?}", meta.wtype),
                    reason: "Q4_K kernel not implemented".to_string(),
                });
            }
            GgmlType::Q5_K => {
                return Err(GpuError::UnsupportedOperation {
                    operation: format!("gpu_dispatch_gemv_on_stream for {:?}", meta.wtype),
                    reason: "Q5_K kernel not implemented".to_string(),
                });
            }
            GgmlType::Q6_K => {
                gemv_q6_k_f32_on_stream(
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

    if meta.needs_transpose && meta.role == TensorRole::TiedLmHead {
        return Err(GpuError::UnsupportedOperation {
            operation: "gpu_dispatch_gemv".to_string(),
            reason: "Transposed Tied LM Head is not supported on GPU".to_string(),
        });
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
        input,
        output,
        out_dim,
        in_dim,
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

    if meta.needs_transpose && meta.role == TensorRole::TiedLmHead {
        return Err(GpuError::UnsupportedOperation {
            operation: "gpu_dispatch_gemv_on_stream".to_string(),
            reason: "Transposed Tied LM Head is not supported on GPU".to_string(),
        });
    }

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemv_on_stream".to_string(),
            wtype: meta.wtype,
        });
    }

    dispatch_gemv_impl(
        device, stream, weights, meta, input, output, out_dim, in_dim,
    )
}
