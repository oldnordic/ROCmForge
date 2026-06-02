use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hip_stream_synchronize;
use super::super::ffi::{hipStreamCaptureStatus, hipStream_t, hip_stream_is_capturing};
use super::super::kernels::{
    gemv_q4_0_f32_q8_inline_residual_on_stream_variant, gemv_q4_0_f32_residual_on_stream_unchecked,
    gemv_q4_0_f32_wave32_residual_on_stream_unchecked, gemv_q4_1_f32_residual_on_stream_unchecked,
    gemv_q4_1_f32_residual_on_stream_variant_unchecked,
    gemv_q4_1_f32_wave32_residual_on_stream_unchecked,
};
use super::super::launch_autotune::{
    lookup_q4_0_q8_residual_variant, lookup_q4_1_residual_variant, select_q4_0_q8_residual_variant,
    select_q4_1_residual_variant, VariantId,
};
use super::super::safety::{
    disable_wave32_enabled, experimental_q8_activation_fastpath_enabled, force_wave32_enabled,
    launch_autotune_enabled,
};
use super::super::weights::{GpuBuffer, WeightMeta};
use crate::gpu::kernel_dispatch_profile::record_gemv_dispatch;
use crate::loader::GgmlType;

use super::fastpath::{
    q8_fastpath_ok, try_q4_0_q8_0_residual_fastpath, try_q4_0_q8_0_residual_fastpath_prequantized,
};

pub fn gpu_dispatch_gemv_residual_on_stream(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<bool> {
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                if experimental_q8_activation_fastpath_enabled() {
                    let capture_active = matches!(
                        hip_stream_is_capturing(stream),
                        Err(_)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
                    );

                    if launch_autotune_enabled() {
                        let variant = if capture_active {
                            lookup_q4_0_q8_residual_variant(in_dim, out_dim)
                                .unwrap_or(VariantId::Baseline)
                        } else {
                            select_q4_0_q8_residual_variant(in_dim, out_dim, |v| {
                                let result = match v {
                                    VariantId::Baseline | VariantId::Variant1 => {
                                        gemv_q4_0_f32_q8_inline_residual_on_stream_variant(
                                            weights.as_ptr() as *const u8,
                                            input,
                                            residual,
                                            output,
                                            in_dim,
                                            out_dim,
                                            v as i32,
                                            stream,
                                        )
                                    }
                                    VariantId::Variant2 => {
                                        gemv_q4_0_f32_q8_inline_residual_on_stream_variant(
                                            weights.as_ptr() as *const u8,
                                            input,
                                            residual,
                                            output,
                                            in_dim,
                                            out_dim,
                                            v as i32,
                                            stream,
                                        )
                                    }
                                    VariantId::Variant3 => {
                                        try_q4_0_q8_0_residual_fastpath_prequantized(
                                            device, weights, input, residual, output, in_dim,
                                            out_dim, stream,
                                        )
                                    }
                                };
                                hip_stream_synchronize(stream)?;
                                result
                            })
                        };

                        let selected_result = match variant {
                            VariantId::Baseline | VariantId::Variant1 | VariantId::Variant2 => {
                                gemv_q4_0_f32_q8_inline_residual_on_stream_variant(
                                    weights.as_ptr() as *const u8,
                                    input,
                                    residual,
                                    output,
                                    in_dim,
                                    out_dim,
                                    variant as i32,
                                    stream,
                                )
                            }
                            VariantId::Variant3 => try_q4_0_q8_0_residual_fastpath_prequantized(
                                device, weights, input, residual, output, in_dim, out_dim, stream,
                            ),
                        };

                        if q8_fastpath_ok("gemv_q4_0_f32_q8_inline_residual", selected_result) {
                            return Ok(true);
                        }
                    }

                    if q8_fastpath_ok(
                        "gemv_q4_0_f32_q8_inline_residual",
                        try_q4_0_q8_0_residual_fastpath(
                            weights, input, residual, output, in_dim, out_dim, stream,
                        ),
                    ) {
                        return Ok(true);
                    }
                }

                let features = super::super::features::GpuFeatures::detect(device).ok();
                let is_rdna3 = features.map_or(false, |f| f.arch.starts_with("gfx11"));
                let use_wave32 = force_wave32_enabled() || (is_rdna3 && !disable_wave32_enabled());

                if use_wave32 {
                    record_gemv_dispatch("Q4_0", "wave32_residual");
                    gemv_q4_0_f32_wave32_residual_on_stream_unchecked(
                        weights.as_ptr() as *const u8,
                        input,
                        residual,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?;
                    return Ok(true);
                }

                record_gemv_dispatch("Q4_0", "wave64_residual");
                gemv_q4_0_f32_residual_on_stream_unchecked(
                    weights.as_ptr() as *const u8,
                    input,
                    residual,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
                Ok(true)
            }

            GgmlType::Q4_1 => {
                let features = super::super::features::GpuFeatures::detect(device).ok();
                let is_rdna3 = features.map_or(false, |f| f.arch.starts_with("gfx11"));
                let use_wave32 = force_wave32_enabled() || (is_rdna3 && !disable_wave32_enabled());

                if use_wave32 {
                    record_gemv_dispatch("Q4_1", "wave32_residual");
                    unsafe {
                        gemv_q4_1_f32_wave32_residual_on_stream_unchecked(
                            weights.as_ptr() as *const u8,
                            input,
                            residual,
                            output,
                            in_dim,
                            out_dim,
                            stream,
                        )?;
                    }
                } else {
                    record_gemv_dispatch("Q4_1", "wave64_residual");
                    let capture_active = matches!(
                        hip_stream_is_capturing(stream),
                        Err(_)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
                    );

                    if launch_autotune_enabled() {
                        let variant = if capture_active {
                            lookup_q4_1_residual_variant(in_dim, out_dim)
                                .unwrap_or(VariantId::Baseline)
                        } else {
                            select_q4_1_residual_variant(in_dim, out_dim, |v| {
                                let result = unsafe {
                                    gemv_q4_1_f32_residual_on_stream_variant_unchecked(
                                        weights.as_ptr() as *const u8,
                                        input,
                                        residual,
                                        output,
                                        in_dim,
                                        out_dim,
                                        v as i32,
                                        stream,
                                    )
                                };
                                hip_stream_synchronize(stream)?;
                                result
                            })
                        };

                        unsafe {
                            gemv_q4_1_f32_residual_on_stream_variant_unchecked(
                                weights.as_ptr() as *const u8,
                                input,
                                residual,
                                output,
                                in_dim,
                                out_dim,
                                variant as i32,
                                stream,
                            )?;
                        }
                    } else {
                        unsafe {
                            gemv_q4_1_f32_residual_on_stream_unchecked(
                                weights.as_ptr() as *const u8,
                                input,
                                residual,
                                output,
                                in_dim,
                                out_dim,
                                stream,
                            )?;
                        }
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
