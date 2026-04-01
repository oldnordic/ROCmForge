//! GPU math dispatch for model weights.
//!
//! This layer validates GGUF metadata before calling the raw HIP kernels.

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use super::kernels::{
    add_on_stream, gemm_q4_0_f32, gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_k_f32, gemm_q8_0_f32,
    gemv_gate_up_swiglu_q4_0_f32_on_stream, gemv_q4_0_f32_on_stream,
    gemv_q4_0_f32_residual_on_stream, gemv_q4_1_f32_on_stream, gemv_q4_k_f32_on_stream,
    gemv_q5_k_f32_on_stream, gemv_q8_0_f32_lm_head_on_stream, gemv_q8_0_f32_on_stream,
    gemv_qkv_q4_0_f32_on_stream, rms_norm_on_stream, rms_norm_vulkan_style,
};
use super::weights::{GpuBuffer, TensorRole, WeightMeta};
use crate::loader::GgmlType;
use std::os::raw::c_int;

fn supports_gemv_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K
    )
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
    device: &GpuDevice,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    // 1. Try specialized Vulkan-style kernel if safe (alignment check inside)
    if let Ok(()) = rms_norm_vulkan_style(x, weight, out, n, eps, stream) {
        return Ok(());
    }

    // 2. Fallback to generic path
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
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                // 1. Try specialized Vulkan-style kernel if safe
                let n_waves = 8;
                if let Ok(()) = super::kernels::quant::gemv_q4_0_f32_vulkan_style(
                    device,
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    n_waves,
                    stream,
                ) {
                    return Ok(());
                }

                // 2. Fallback to generic path
                gemv_q4_0_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?
            }
            GgmlType::Q4_1 => gemv_q4_1_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q8_0 => {
                // Check for LM-head specialization
                if meta.role == TensorRole::LmHead {
                    gemv_q8_0_f32_lm_head_on_stream(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?
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
                // 1. Try specialized Vulkan-style kernel if safe
                let n_waves = 8;
                if let Ok(()) = super::kernels::quant::gemv_q4_k_f32_vulkan_style(
                    device,
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    n_waves,
                    stream,
                ) {
                    return Ok(());
                }

                // 2. Fallback to generic path
                gemv_q4_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?
            }
            GgmlType::Q5_K => gemv_q5_k_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
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
    if meta.wtype != GgmlType::Q4_0 {
        return Ok(false);
    }

    // Try specialized residual kernel
    unsafe {
        gemv_q4_0_f32_residual_on_stream(
            weights.as_ptr() as *const u8,
            input,
            residual,
            output,
            in_dim,
            out_dim,
            stream,
        )?;
    }
    Ok(true)
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
        unsafe {
            gemv_qkv_q4_0_f32_on_stream(
                w_q.as_ptr() as *const u8,
                w_k.as_ptr() as *const u8,
                w_v.as_ptr() as *const u8,
                input,
                out_q,
                out_k,
                out_v,
                q_bias.map_or(std::ptr::null_mut(), |b| b.as_ptr() as *mut f32),
                k_bias.map_or(std::ptr::null_mut(), |b| b.as_ptr() as *mut f32),
                v_bias.map_or(std::ptr::null_mut(), |b| b.as_ptr() as *mut f32),
                h,
                q_size / h,
                kv_size / h,
                stream,
            )?;
        }
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
pub fn gpu_dispatch_fused_gate_up_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(gate_meta, ff_size, h)?;
    validate_gemv_layout(up_meta, ff_size, h)?;

    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
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
    }

    // Individual dispatches + manual silu/mul if no fused kernel matches
    gpu_dispatch_gemv_on_stream(device, w_gate, gate_meta, input, output, ff_size, h, stream)?;
    // (This path is slower, ideally we always hit the fused one)
    Ok(())
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row.
pub fn gpu_dispatch_fused_gate_up(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
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

    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => gemm_q4_0_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                out_dim,
                in_dim,
                seq_len,
            )?,
            GgmlType::Q4_1 => gemm_q4_1_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                out_dim,
                in_dim,
                seq_len,
            )?,
            GgmlType::Q8_0 => gemm_q8_0_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                out_dim,
                in_dim,
                seq_len,
            )?,
            GgmlType::Q4_K => gemm_q4_k_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                out_dim,
                in_dim,
                seq_len,
            )?,
            GgmlType::Q5_K => gemm_q5_k_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                out_dim,
                in_dim,
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

    Ok(())
}
