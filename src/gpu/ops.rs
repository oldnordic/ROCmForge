//! GPU math dispatch for model weights.
//!
//! This layer validates GGUF metadata before calling the raw HIP kernels.

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hip_stream_synchronize;
use super::ffi::{hipStreamCaptureStatus, hipStream_t, hip_stream_is_capturing};
use super::kernels::{
    add_on_stream, gemm_q4_0_f32, gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_k_f32, gemm_q6_k_f32,
    gemm_q8_0_f32, gemv_gate_up_q4_0_f32_on_stream, gemv_gate_up_q4_0_q8_0_on_stream,
    // DISABLED: gemv_gate_up_swiglu_q4_0_f32_on_stream not available
    // DISABLED: Experimental Q8 variants not available
    // gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream,
    // gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream,
    // gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant,
    gemv_q4_0_f32_on_stream_unchecked,
    gemv_q4_0_f32_q8_inline_residual_on_stream, gemv_q4_0_f32_q8_inline_residual_on_stream_variant,
    gemv_q4_0_f32_residual_on_stream_unchecked, gemv_q4_0_q8_0_on_stream,
    gemv_q4_0_q8_0_residual_on_stream, gemv_q4_1_f32_on_stream_unchecked,
    gemv_q4_1_f32_residual_on_stream_unchecked, gemv_q4_1_f32_residual_on_stream_variant_unchecked,
    // DISABLED: gemv_q4_k_f32_on_stream, gemv_q5_k_f32_on_stream, gemv_q6_k_f32_on_stream not available
    // gemv_q4_k_f32_on_stream, gemv_q5_k_f32_on_stream, gemv_q6_k_f32_on_stream,
    gemv_q8_0_f32_lm_head_on_stream, gemv_q8_0_f32_lm_head_on_stream_variant,
    gemv_q8_0_f32_on_stream,
    // DISABLED: gemv_qkv_q4_0_f32_on_stream not available (use fused_qkv_rope_q4_0_gqa_on_stream instead)
    // gemv_qkv_q4_0_f32_on_stream,
    // DISABLED: gemv_qkv_q4_0_f32_on_stream_variant not available
    fused_qkv_rope_q4_0_gqa_on_stream,
    mul_on_stream, q8_0_workspace_bytes, quantize_q8_0_on_stream, rms_norm_on_stream,
    rms_norm_vulkan_style, silu_on_stream,
};
use super::launch_autotune::{
    lookup_gate_up_swiglu_q8_variant, lookup_lm_head_q8_variant, lookup_q4_0_q8_residual_variant,
    lookup_q4_1_residual_variant, lookup_qkv_variant, select_gate_up_swiglu_q8_variant,
    select_lm_head_q8_variant, select_q4_0_q8_residual_variant, select_q4_1_residual_variant,
    select_qkv_variant, VariantId,
};
use super::safety::{
    disable_q8_activation_fastpath_runtime, experimental_gpu_kernels_enabled,
    experimental_q8_activation_fastpath_enabled, launch_autotune_enabled,
};
use super::weights::{GpuBuffer, TensorRole, WeightMeta};
use crate::loader::GgmlType;
fn supports_gemv_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
    )
}

fn is_lm_head_role(role: TensorRole) -> bool {
    matches!(role, TensorRole::LmHead | TensorRole::TiedLmHead)
}

fn config_num_heads(q_size: usize, h: usize) -> usize {
    // Hidden size h = n_heads * head_dim
    // For many models, head_dim is 64 or 128
    if h % 128 == 0 {
        q_size / 128
    } else {
        q_size / 64
    }
}

/// Dispatch a GPU RMS norm.
pub fn gpu_dispatch_rms_norm(
    _device: &GpuDevice,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if experimental_gpu_kernels_enabled() {
        if let Ok(()) = rms_norm_vulkan_style(x, weight, out, n, eps, stream) {
            return Ok(());
        }
    }

    rms_norm_on_stream(x, weight, out, n, eps, stream)
}

fn validate_gemv_layout(meta: &WeightMeta, out_dim: usize, in_dim: usize) -> GpuResult<()> {
    if meta.dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: "weight metadata must describe a 2D matrix".to_string(),
        });
    }

    if (meta.dims[0] as usize == in_dim && meta.dims[1] as usize == out_dim)
        || (meta.dims[0] as usize == out_dim && meta.dims[1] as usize == in_dim)
    {
        Ok(())
    } else {
        Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: format!(
                "shape mismatch: matrix is {:?}, but vector is [{}] and output is [{}]",
                meta.dims, in_dim, out_dim
            ),
        })
    }
}

fn quantize_input_q8_workspace(
    device: &GpuDevice,
    input: *const f32,
    n_rows: usize,
    stream: hipStream_t,
) -> GpuResult<*mut u8> {
    let workspace = device.q8_workspace_ptr(q8_0_workspace_bytes(n_rows))?;
    quantize_q8_0_on_stream(input, workspace, n_rows, stream)?;
    Ok(workspace)
}

fn q8_fastpath_ok(context: &str, fastpath_result: GpuResult<()>) -> bool {
    match fastpath_result {
        Ok(()) => true,
        Err(err) => {
            disable_q8_activation_fastpath_runtime(&format!("{context}: {err}"));
            false
        }
    }
}

fn try_q4_0_q8_0_fastpath(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, in_dim, stream)?;

    gemv_q4_0_q8_0_on_stream(
        weights.as_ptr() as *const u8,
        workspace as *const u8,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

fn try_q4_0_q8_0_residual_fastpath(
    weights: &GpuBuffer,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_q4_0_f32_q8_inline_residual_on_stream(
        weights.as_ptr() as *const u8,
        input,
        residual,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

fn try_q4_0_q8_0_residual_fastpath_prequantized(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, in_dim, stream)?;
    gemv_q4_0_q8_0_residual_on_stream(
        weights.as_ptr() as *const u8,
        workspace as *const u8,
        residual,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

fn try_q4_0_q8_0_gate_up_fastpath(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    gate_output: *mut f32,
    up_output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, h, stream)?;
    gemv_gate_up_q4_0_q8_0_on_stream(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        workspace as *const u8,
        gate_output,
        up_output,
        h,
        ff_size,
        stream,
    )
}

fn try_q4_0_q8_0_fused_gate_up_fastpath(
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    try_q4_0_q8_0_fused_gate_up_fastpath_variant(w_gate, w_up, input, output, h, ff_size, 0, stream)
}

fn try_q4_0_q8_0_fused_gate_up_fastpath_prequantized(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, h, stream)?;
    super::kernels::gemv_gate_up_swiglu_q4_0_q8_0_on_stream(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        workspace as *const u8,
        output,
        h,
        ff_size,
        stream,
    )
}

fn try_q4_0_q8_0_fused_gate_up_fastpath_variant(
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    variant: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    // DISABLED: Experimental Q8 variants not available
    return Err(GpuError::UnsupportedOperation {
        operation: "gpu_dispatch_fused_gate_up_with_scratch_q8_variant".to_string(),
        reason: "Experimental Q8 variants not yet implemented. Use baseline variant.".to_string(),
    });

    // TODO: Re-enable when Q8 experimental kernels are available
    // gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant(
    //     w_gate.as_ptr() as *const u8,
    //     w_up.as_ptr() as *const u8,
    //     input,
    //     output,
    //     h,
    //     ff_size,
    //     variant,
    //     stream,
    // )
}

fn dispatch_gemv_impl(
    device: &GpuDevice,
    stream: hipStream_t,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    eprintln!("[RUST] dispatch_gemv_impl called: wtype={:?}, out_dim={}, in_dim={}", meta.wtype, out_dim, in_dim);
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                eprintln!("[RUST] Q4_0 path: checking fastpaths");
                if experimental_q8_activation_fastpath_enabled()
                    && q8_fastpath_ok(
                        "gemv_q4_0_q8_0",
                        try_q4_0_q8_0_fastpath(
                            device, weights, input, output, in_dim, out_dim, stream,
                        ),
                    )
                {
                    eprintln!("[RUST] Q4_0 Q8_0 fastpath TAKEN");
                    return Ok(());
                }

                // DISABLED: vulkan_style kernels not available
                // if experimental_gpu_kernels_enabled() {
                //     eprintln!("[RUST] Q4_0 experimental vulkan_style checking");
                //     let n_waves = 8;
                //     if let Ok(()) = super::kernels::quant::gemv_q4_0_f32_vulkan_style(
                //         device,
                //         weights.as_ptr() as *const u8,
                //         input,
                //         output,
                //         in_dim,
                //         out_dim,
                //         n_waves,
                //         stream,
                //     ) {
                //         eprintln!("[RUST] Q4_0 vulkan_style TAKEN");
                //         return Ok(());
                //     }
                // }

                eprintln!("[RUST] Q4_0 falling back to gemv_q4_0_f32_on_stream_unchecked");
                gemv_q4_0_f32_on_stream_unchecked(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?
            }
            GgmlType::Q4_1 => gemv_q4_1_f32_on_stream_unchecked(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q8_0 => {
                // Check for LM-head specialization
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
                                hip_stream_synchronize(stream)?;
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
                // DISABLED: vulkan_style kernels not available
                // if experimental_gpu_kernels_enabled() {
                //     let n_waves = 8;
                //     if let Ok(()) = super::kernels::quant::gemv_q4_k_f32_vulkan_style(
                //         device,
                //         weights.as_ptr() as *const u8,
                //         input,
                //         output,
                //         in_dim,
                //         out_dim,
                //         n_waves,
                //         stream,
                //     ) {
                //         return Ok(());
                //     }
                // }

                // DISABLED: gemv_q4_k_f32_on_stream not available
                // TODO: Implement Q4_K kernel or use CPU fallback
                return Err(GpuError::UnsupportedOperation {
                    operation: format!("gpu_dispatch_gemv_on_stream for {:?}", wtype),
                    reason: "Q4_K kernel not implemented".to_string(),
                });
                /*
                gemv_q4_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?
                */
            }
            GgmlType::Q5_K => {
                // DISABLED: gemv_q5_k_f32_on_stream not available
                // TODO: Implement Q5_K kernel or use CPU fallback
                return Err(GpuError::UnsupportedOperation {
                    operation: format!("gpu_dispatch_gemv_on_stream for {:?}", wtype),
                    reason: "Q5_K kernel not implemented".to_string(),
                });
                /*
                gemv_q5_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?,
                */
            }
            GgmlType::Q6_K => {
                // DISABLED: gemv_q6_k_f32_on_stream not available
                // TODO: Implement Q6_K kernel or use CPU fallback
                return Err(GpuError::UnsupportedOperation {
                    operation: format!("gpu_dispatch_gemv_on_stream for {:?}", wtype),
                    reason: "Q6_K kernel not implemented".to_string(),
                });
                /*
                gemv_q6_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?,
                */
            }
            _ => unreachable!("unsupported types return before dispatch"),
        }
    }

    Ok(())
}

/// Dispatch a GPU GEMV for one GGUF weight tensor.
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
/// Dispatch a GPU GEMV on an explicit HIP stream.
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
                    // Check if stream capture is active - skip autotune benchmarking during capture
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
                                        // New 16-wave variant for improved parallelism
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

                        // Execute selected (or cached) fastpath variant and keep fallback behavior on failures.
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

                    // Non-autotune fastpath: try inline residual, fall back to unchecked on failure
                    if q8_fastpath_ok(
                        "gemv_q4_0_f32_q8_inline_residual",
                        try_q4_0_q8_0_residual_fastpath(
                            weights, input, residual, output, in_dim, out_dim, stream,
                        ),
                    ) {
                        return Ok(true);
                    }
                }

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
                // Check if stream capture is active - skip autotune benchmarking during capture
                let capture_active = matches!(
                    hip_stream_is_capturing(stream),
                    Err(_)
                        | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                        | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
                );

                // Reuse cached variant during capture so decode graphs keep the tuned launch.
                if launch_autotune_enabled() {
                    let variant = if capture_active {
                        lookup_q4_1_residual_variant(in_dim, out_dim).unwrap_or(VariantId::Baseline)
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

                    // Execute with selected variant
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
                    // Baseline path (backward compatible)
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
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

/// Dispatch a fused QKV GEMV with bias on an explicit HIP stream.
pub fn gpu_dispatch_fused_qkv_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q_meta.wtype == GgmlType::Q4_0
        && k_meta.wtype == GgmlType::Q4_0
        && v_meta.wtype == GgmlType::Q4_0
    {
        // DISABLED: gemv_qkv_q4_0_f32_on_stream_variant not available
        // TODO: Re-enable launch autotune when variant kernels are implemented
        // Check if stream capture is active - skip autotune benchmarking during capture
        /*
        let capture_active = matches!(
            hip_stream_is_capturing(stream),
            Err(_)
                | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
        );

        // Reuse cached variant during capture so decode graphs keep the tuned launch.
        if launch_autotune_enabled() {
            let variant = if capture_active {
                lookup_qkv_variant(h, q_size, kv_size).unwrap_or(VariantId::Baseline)
            } else {
                select_qkv_variant(h, q_size, kv_size, |v| {
                    let result = unsafe {
                        gemv_qkv_q4_0_f32_on_stream_variant(
                            w_q.as_ptr() as *const u8,
                            w_k.as_ptr() as *const u8,
                            w_v.as_ptr() as *const u8,
                            q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                            k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                            v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                            input,
                            out_q,
                            out_k,
                            out_v,
                            h,
                            q_size,
                            kv_size,
                            stream,
                            v as i32,
                        )
                    };
                    hip_stream_synchronize(stream)?;
                    result
                })
            };

            // Execute with selected variant
            unsafe {
                gemv_qkv_q4_0_f32_on_stream_variant(
                    w_q.as_ptr() as *const u8,
                    w_k.as_ptr() as *const u8,
                    w_v.as_ptr() as *const u8,
                    q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    input,
                    out_q,
                    out_k,
                    out_v,
                    h,
                    q_size,
                    kv_size,
                    stream,
                    variant as i32,
                )?;
            }
        } else {
        */
            // Baseline path (backward compatible)
            unsafe {
                gemv_qkv_q4_0_f32_on_stream(
                    w_q.as_ptr() as *const u8,
                    w_k.as_ptr() as *const u8,
                    w_v.as_ptr() as *const u8,
                    q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    input,
                    out_q,
                    out_k,
                    out_v,
                    h,
                    q_size,
                    kv_size,
                    stream,
                )?;
            }
        //} // End of disabled autotune block
        return Ok(());
    }

    // Individual dispatches are safer if we don't have a perfectly matching fused kernel
    gpu_dispatch_gemv_on_stream(device, w_q, q_meta, input, out_q, q_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(device, w_k, k_meta, input, out_k, kv_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(device, w_v, v_meta, input, out_v, kv_size, h, stream)?;

    // Add biases
    if let Some(bias) = q_bias {
        unsafe {
            add_on_stream(out_q, bias.as_ptr() as *const f32, out_q, q_size, stream)?;
        }
    }
    if let Some(bias) = k_bias {
        unsafe {
            add_on_stream(out_k, bias.as_ptr() as *const f32, out_k, kv_size, stream)?;
        }
    }
    if let Some(bias) = v_bias {
        unsafe {
            add_on_stream(out_v, bias.as_ptr() as *const f32, out_v, kv_size, stream)?;
        }
    }

    Ok(())
}

/// Dispatch GQA-aware fused QKV with RoPE on an explicit HIP stream.
///
/// This is the GQA-compatible version of fused QKV that handles grouped query attention
/// where n_kv_heads divides n_heads evenly (e.g., 14 query heads, 2 KV heads).
///
/// # GQA Compatibility
/// Works when n_kv_heads > 0 and n_heads % n_kv_heads == 0.
/// For MHA (n_heads == n_kv_heads), use gpu_dispatch_fused_qkv_on_stream instead.
pub fn gpu_dispatch_fused_qkv_gqa_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
    pos_ptr: *const i32,  // GPU pointer to decode state position
    stream: hipStream_t,
) -> GpuResult<()> {
    // Validate GQA compatibility
    if q_size % kv_size != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "GQA requires q_size divisible by kv_size ({} % {} != 0)",
                q_size, kv_size
            ),
        });
    }

    let n_heads = q_size / h;
    let n_kv_heads = kv_size / h;

    if n_heads % n_kv_heads != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "GQA requires n_heads divisible by n_kv_heads ({} % {} != 0)",
                n_heads, n_kv_heads
            ),
        });
    }

    // Check GPU features for DP4A support
    let features = super::features::GpuFeatures::detect(device)?;

    // Get RoPE parameters from model config
    let rope_theta = 10000.0f32;  // Standard for Qwen2
    let rope_neox = true;          // Qwen2 uses Neox-style RoPE

    // Convert bias Option<&GpuBuffer> to *const f32 (null if None)
    let bias_q_ptr = q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32);
    let bias_k_ptr = k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32);
    let bias_v_ptr = v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32);

    // TODO: DP4A integration requires pipeline refactoring
    // The DP4A kernel (gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream) fuses
    // RMS norm + QKV projection + RoPE + KV write in a single kernel, but the
    // current decode pipeline splits these operations:
    //   1. RMS norm (gpu_dispatch_rms_norm in forward.rs:973)
    //   2. QKV projection (this function)
    //   3. RoPE application (line 894-895 below)
    //   4. KV write (forward.rs:1056)
    //
    // To fully integrate DP4A, we need to:
    //   - Pass raw_hidden (not normed) to this function
    //   - Pass attn_norm weights and eps
    //   - Pass KV cache pointers
    //   - Skip the separate RMS norm, RoPE, and KV write steps in forward.rs
    //
    // For now, use the standard kernel path when DP4A is available.
    // Full DP4A integration is tracked in: https://github.com/your-repo/issues/XXX
    if features.has_dp4a {
        // DP4A is available but using standard path due to architectural mismatch
        // Future work: refactor decode pipeline to use fully fused DP4A kernel
    }

    // Step 1: Fused QKV projection (no RoPE - applied separately)
    // Note: pos=0 is safe here since RoPE is applied separately below
    fused_qkv_rope_q4_0_gqa_on_stream(
        device,
        w_q.as_ptr() as *const u8,
        w_k.as_ptr() as *const u8,
        w_v.as_ptr() as *const u8,
        input,
        out_q,
        out_k,
        out_v,
        0,  // pos (not used, RoPE applied separately)
        n_heads,
        n_kv_heads,
        h,  // head_dim
        rope_theta,
        rope_neox,
        stream,
    )?;

    // Step 2: Apply RoPE to Q and K separately using GPU state pointer (for graph replay compatibility)
    super::kernels::rope::rope_heads_from_state_on_stream(out_q, pos_ptr, n_heads, h, rope_theta, rope_neox, stream)?;
    super::kernels::rope::rope_heads_from_state_on_stream(out_k, pos_ptr, n_kv_heads, h, rope_theta, rope_neox, stream)?;

    Ok(())
}

/// Dispatch a fused QKV GEMV with bias on an explicit HIP stream (decode-strict).
///
/// Strictly enforces that Q/K/V weights are all Q4_0. Returns an error for any other type.
pub fn gpu_dispatch_fused_qkv_decode_strict_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q_meta.wtype != GgmlType::Q4_0
        || k_meta.wtype != GgmlType::Q4_0
        || v_meta.wtype != GgmlType::Q4_0
    {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "decode.fused_qkv".to_string(),
            wtype: q_meta.wtype,
        });
    }

    gpu_dispatch_fused_qkv_on_stream(
        device, w_q, q_meta, q_bias, w_k, k_meta, k_bias, w_v, v_meta, v_bias, input, out_q, out_k,
        out_v, q_size, kv_size, h, stream,
    )
}

/// Dispatch a fused QKV GEMV for a single row.
pub fn gpu_dispatch_fused_qkv(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
) -> GpuResult<()> {
    gpu_dispatch_fused_qkv_on_stream(
        device,
        w_q,
        q_meta,
        q_bias,
        w_k,
        k_meta,
        k_bias,
        w_v,
        v_meta,
        v_bias,
        input,
        out_q,
        out_k,
        out_v,
        q_size,
        kv_size,
        h,
        hipStream_t::null(),
    )
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row on an explicit stream.
pub(crate) fn gpu_dispatch_gate_up_raw_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    input: *const f32,
    gate_output: *mut f32,
    up_output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(gate_meta, ff_size, h)?;
    validate_gemv_layout(up_meta, ff_size, h)?;

    if gate_output.is_null() || up_output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_gate_up_raw: output pointers must be non-null".to_string(),
        });
    }
    if gate_output == up_output {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_gate_up_raw: gate and up outputs must be distinct buffers"
                .to_string(),
        });
    }

    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        if experimental_q8_activation_fastpath_enabled()
            && q8_fastpath_ok(
                "gemv_gate_up_q4_0_q8_0",
                try_q4_0_q8_0_gate_up_fastpath(
                    device,
                    w_gate,
                    w_up,
                    input,
                    gate_output,
                    up_output,
                    h,
                    ff_size,
                    stream,
                ),
            )
        {
            return Ok(());
        }

        gemv_gate_up_q4_0_f32_on_stream(
            w_gate.as_ptr() as *const u8,
            w_up.as_ptr() as *const u8,
            input,
            gate_output,
            up_output,
            h,
            ff_size,
            stream,
        )?;
        return Ok(());
    }

    gpu_dispatch_gemv_on_stream(
        device,
        w_gate,
        gate_meta,
        input,
        gate_output,
        ff_size,
        h,
        stream,
    )?;
    gpu_dispatch_gemv_on_stream(device, w_up, up_meta, input, up_output, ff_size, h, stream)?;
    Ok(())
}

pub(crate) fn gpu_dispatch_fused_gate_up_with_scratch_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    gate_scratch: *mut f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(gate_meta, ff_size, h)?;
    validate_gemv_layout(up_meta, ff_size, h)?;

    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        if experimental_q8_activation_fastpath_enabled() {
            // Check if stream capture is active - skip autotune benchmarking during capture
            let capture_active = matches!(
                hip_stream_is_capturing(stream),
                Err(_)
                    | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                    | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
            );

            // DISABLED: Experimental Q8 variants not available
            // TODO: Re-enable when experimental kernels are implemented
            /*
            if launch_autotune_enabled() {
                let variant = if capture_active {
                    lookup_gate_up_swiglu_q8_variant(h, ff_size).unwrap_or(VariantId::Baseline)
                } else {
                    select_gate_up_swiglu_q8_variant(h, ff_size, |v| {
                        let result = match v {
                            VariantId::Variant1 => {
                                if let Some(interleaved_tile4) = w_gate_up_interleaved_tile4 {
                                    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream(
                                        interleaved_tile4.as_ptr() as *const u8,
                                        input,
                                        output,
                                        h,
                                        ff_size,
                                        stream,
                                    )
                                } else {
                                    try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                        w_gate, w_up, input, output, h, ff_size, 1, stream,
                                    )
                                }
                            }
                            VariantId::Variant2 => {
                                if let Some(interleaved) = w_gate_up_interleaved {
                                    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream(
                                        interleaved.as_ptr() as *const u8,
                                        input,
                                        output,
                                        h,
                                        ff_size,
                                        stream,
                                    )
                                } else {
                                    try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                        w_gate, w_up, input, output, h, ff_size, 0, stream,
                                    )
                                }
                            }
                            _ => try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                w_gate, w_up, input, output, h, ff_size, v as i32, stream,
                            ),
                        };
                        hip_stream_synchronize(stream)?;
                        result
                    })
                };

                let selected_result = match variant {
                    VariantId::Variant1 => {
                        if let Some(interleaved_tile4) = w_gate_up_interleaved_tile4 {
                            gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream(
                                interleaved_tile4.as_ptr() as *const u8,
                                input,
                                output,
                                h,
                                ff_size,
                                stream,
                            )
                        } else {
                            try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                w_gate, w_up, input, output, h, ff_size, 1, stream,
                            )
                        }
                    }
                    VariantId::Variant2 => {
                        if let Some(interleaved) = w_gate_up_interleaved {
                            gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream(
                                interleaved.as_ptr() as *const u8,
                                input,
                                output,
                                h,
                                ff_size,
                                stream,
                            )
                        } else {
                            try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                w_gate, w_up, input, output, h, ff_size, 0, stream,
                            )
                        }
                    }
                    _ => try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                        w_gate,
                        w_up,
                        input,
                        output,
                        h,
                        ff_size,
                        variant as i32,
                        stream,
                    ),
                };

                if q8_fastpath_ok("gemv_gate_up_swiglu_q4_0_f32_q8_inline", selected_result) {
                    return Ok(());
                }
            }
            */

            if q8_fastpath_ok(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline",
                try_q4_0_q8_0_fused_gate_up_fastpath(
                    w_gate, w_up, input, output, h, ff_size, stream,
                ),
            ) {
                return Ok(());
            }
        }

        // DISABLED: gemv_gate_up_swiglu_q4_0_f32_on_stream not available
        // TODO: Implement fallback path when gate-up kernel is available
        return Err(GpuError::UnsupportedOperation {
            operation: "gpu_dispatch_fused_gate_up_q8_variant".to_string(),
            reason: "gemv_gate_up_swiglu_q4_0_f32_on_stream kernel not implemented".to_string(),
        });
        /*
        unsafe {
            gemv_gate_up_swiglu_q4_0_f32_on_stream(
                w_gate.as_ptr() as *const u8,
                w_up.as_ptr() as *const u8,
                input,
                output,
                h,
                ff_size,
                stream,
            )?;
        }
        return Ok(());
        */
    }

    if gate_scratch.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_fused_gate_up: gate scratch pointer must be non-null"
                .to_string(),
        });
    }
    if gate_scratch == output {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gpu_dispatch_fused_gate_up: gate scratch and output must be distinct buffers"
                    .to_string(),
        });
    }

    gpu_dispatch_gate_up_raw_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        input,
        gate_scratch,
        output,
        ff_size,
        h,
        stream,
    )?;
    silu_on_stream(gate_scratch, gate_scratch, ff_size, stream)?;
    mul_on_stream(gate_scratch, output, output, ff_size, stream)?;
    Ok(())
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row on an explicit stream.
pub fn gpu_dispatch_fused_gate_up_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        return gpu_dispatch_fused_gate_up_with_scratch_on_stream(
            device,
            w_gate,
            gate_meta,
            w_up,
            up_meta,
            w_gate_up_interleaved,
            w_gate_up_interleaved_tile4,
            input,
            std::ptr::null_mut(),
            output,
            ff_size,
            h,
            stream,
        );
    }

    let gate_scratch = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>())?;
    gpu_dispatch_fused_gate_up_with_scratch_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        w_gate_up_interleaved,
        w_gate_up_interleaved_tile4,
        input,
        gate_scratch.as_ptr() as *mut f32,
        output,
        ff_size,
        h,
        stream,
    )
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row.
pub fn gpu_dispatch_fused_gate_up(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
) -> GpuResult<()> {
    gpu_dispatch_fused_gate_up_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        w_gate_up_interleaved,
        w_gate_up_interleaved_tile4,
        input,
        output,
        ff_size,
        h,
        hipStream_t::null(),
    )
}

/// Dispatch a GPU GEMM for GGUF weights.
pub fn gpu_dispatch_gemm(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if seq_len == 1 && supports_gemv_type(meta.wtype) {
        return gpu_dispatch_gemv(device, weights, meta, input, output, out_dim, in_dim);
    }

    // DISABLED: GEMM kernels not available (prefill-only, not needed for decode)
    // TODO: Implement GEMM kernels or use GEMV for all cases
    return Err(GpuError::UnsupportedOperation {
        operation: format!("gpu_dispatch_gemm for seq_len={}", seq_len),
        reason: "GEMM kernels not yet implemented. Use GEMV (seq_len=1) for now.".to_string(),
    });

    // TODO: Re-enable GEMM kernels when libraries are built
    // The following code is commented out because GEMM libraries are not built
    /*
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => gemm_q4_0_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q4_1 => gemm_q4_1_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q8_0 => gemm_q8_0_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q4_K => gemm_q4_k_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q5_K => gemm_q5_k_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q6_K => gemm_q6_k_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            _ => {
                return Err(GpuError::UnsupportedWeightType {
                    tensor: "gpu_dispatch_gemm".to_string(),
                    wtype: meta.wtype,
                })
            }
        }
    }
    */
    /*
    */

    Ok(())
}
