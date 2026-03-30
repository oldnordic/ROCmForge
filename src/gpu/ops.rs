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
    gemv_qkv_q4_0_f32_on_stream,
};
use super::weights::{GpuBuffer, TensorRole, WeightMeta};
use crate::loader::GgmlType;

fn supports_gemv_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K
    )
}

fn validate_gemv_layout(meta: &WeightMeta, out_dim: usize, in_dim: usize) -> GpuResult<()> {
    if meta.dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: "weight metadata must describe a 2D matrix".to_string(),
        });
    }

    let expected = [in_dim as u64, out_dim as u64];
    if meta.dims[0] != expected[0] || meta.dims[1] != expected[1] {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: format!(
                "expected GGUF dims {:?} for out_dim={} in_dim={}",
                expected, out_dim, in_dim
            ),
        });
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemvDispatchKind {
    Default,
    LmHeadQ8_0,
}

fn select_gemv_dispatch(meta: &WeightMeta) -> GemvDispatchKind {
    match (meta.role, meta.wtype) {
        // Future optimized LM-head kernels must be selected from metadata,
        // not hardcoded to any specific model family.
        (TensorRole::LmHead | TensorRole::TiedLmHead, GgmlType::Q8_0) => {
            GemvDispatchKind::LmHeadQ8_0
        }
        _ => GemvDispatchKind::Default,
    }
}

fn dispatch_gemv_impl(
    stream: hipStream_t,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    match select_gemv_dispatch(meta) {
        GemvDispatchKind::LmHeadQ8_0 => {
            gemv_q8_0_f32_lm_head_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?;
        }
        GemvDispatchKind::Default => match meta.wtype {
            GgmlType::Q4_0 => gemv_q4_0_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q4_1 => gemv_q4_1_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q8_0 => gemv_q8_0_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q4_K => gemv_q4_k_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q5_K => gemv_q5_k_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            _ => unreachable!("unsupported types return before dispatch"),
        },
    }

    Ok(())
}

/// Dispatch a GPU GEMV for one GGUF weight tensor.
pub fn gpu_dispatch_gemv(
    _device: &GpuDevice,
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

    dispatch_gemv_impl(stream, weights, meta, input, output, out_dim, in_dim)
}

pub fn gpu_dispatch_gemv_residual_on_stream(
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    stream: hipStream_t,
) -> GpuResult<bool> {
    validate_gemv_layout(meta, out_dim, in_dim)?;

    if meta.wtype != GgmlType::Q4_0 {
        return Ok(false);
    }

    gemv_q4_0_f32_residual_on_stream(
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

/// Dispatch a fused QKV GEMV for a single row.
pub fn gpu_dispatch_fused_qkv(
    _device: &GpuDevice,
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

/// Dispatch a fused QKV GEMV for a single row on an explicit HIP stream.
pub fn gpu_dispatch_fused_qkv_on_stream(
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
        gemv_qkv_q4_0_f32_on_stream(
            w_q.as_ptr() as *const u8,
            w_k.as_ptr() as *const u8,
            w_v.as_ptr() as *const u8,
            q_bias.map_or(std::ptr::null(), |bias| bias.as_ptr() as *const f32),
            k_bias.map_or(std::ptr::null(), |bias| bias.as_ptr() as *const f32),
            v_bias.map_or(std::ptr::null(), |bias| bias.as_ptr() as *const f32),
            input,
            out_q,
            out_k,
            out_v,
            h,
            q_size,
            kv_size,
            stream,
        )?;
        return Ok(());
    }

    // Fallback to individual dispatches
    gpu_dispatch_gemv_on_stream(w_q, q_meta, input, out_q, q_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(w_k, k_meta, input, out_k, kv_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(w_v, v_meta, input, out_v, kv_size, h, stream)?;
    if let Some(bias) = q_bias {
        add_on_stream(
            out_q as *const f32,
            bias.as_ptr() as *const f32,
            out_q,
            q_size,
            stream,
        )?;
    }
    if let Some(bias) = k_bias {
        add_on_stream(
            out_k as *const f32,
            bias.as_ptr() as *const f32,
            out_k,
            kv_size,
            stream,
        )?;
    }
    if let Some(bias) = v_bias {
        add_on_stream(
            out_v as *const f32,
            bias.as_ptr() as *const f32,
            out_v,
            kv_size,
            stream,
        )?;
    }

    Ok(())
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row.
pub fn gpu_dispatch_fused_gate_up(
    _device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    input: *const f32,
    out_swiglu: *mut f32,
    ff_size: usize,
    h: usize,
) -> GpuResult<()> {
    gpu_dispatch_fused_gate_up_on_stream(
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        input,
        out_swiglu,
        ff_size,
        h,
        hipStream_t::null(),
    )
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row on an explicit HIP stream.
pub fn gpu_dispatch_fused_gate_up_on_stream(
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    input: *const f32,
    out_swiglu: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        gemv_gate_up_swiglu_q4_0_f32_on_stream(
            w_gate.as_ptr() as *const u8,
            w_up.as_ptr() as *const u8,
            input,
            out_swiglu,
            h,
            ff_size,
            stream,
        )?;
        return Ok(());
    }

    // Caller must ensure Q4_0 or handle fallback.
    Err(GpuError::UnsupportedWeightType {
        tensor: "gpu_dispatch_fused_gate_up fallback not supported".to_string(),
        wtype: gate_meta.wtype,
    })
}

/// Dispatch a GPU GEMM for a batch of input rows.
pub fn gpu_dispatch_gemm(
    _device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if batch_size == 1 {
        return gpu_dispatch_gemv(_device, weights, meta, input, output, out_dim, in_dim);
    }

    validate_gemv_layout(meta, out_dim, in_dim)?;

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemm".to_string(),
            wtype: meta.wtype,
        });
    }

    match meta.wtype {
        GgmlType::Q4_0 => gemm_q4_0_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            batch_size,
        )?,
        GgmlType::Q4_1 => gemm_q4_1_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            batch_size,
        )?,
        GgmlType::Q8_0 => gemm_q8_0_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            batch_size,
        )?,
        GgmlType::Q4_K => gemm_q4_k_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            batch_size,
        )?,
        GgmlType::Q5_K => gemm_q5_k_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            batch_size,
        )?,
        _ => unreachable!("unsupported types return before dispatch"),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::TensorRole;

    #[test]
    fn layout_accepts_native_gguf_orientation() {
        let meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: vec![2048, 1024],
            needs_transpose: true,
            role: TensorRole::Generic,
        };

        validate_gemv_layout(&meta, 1024, 2048).unwrap();
    }

    #[test]
    fn layout_rejects_cpu_oriented_dimension_order() {
        let meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: vec![1024, 2048],
            needs_transpose: false,
            role: TensorRole::Generic,
        };

        let err = validate_gemv_layout(&meta, 1024, 2048).unwrap_err();
        assert!(matches!(err, GpuError::InvalidWeightLayout { .. }));
    }

    #[test]
    fn supports_gemv_type_matches_current_kernel_set() {
        assert!(supports_gemv_type(GgmlType::Q4_0));
        assert!(supports_gemv_type(GgmlType::Q4_1));
        assert!(supports_gemv_type(GgmlType::Q8_0));
        assert!(supports_gemv_type(GgmlType::Q4_K));
        assert!(supports_gemv_type(GgmlType::Q5_K));
        assert!(!supports_gemv_type(GgmlType::Q6_K));
        assert!(!supports_gemv_type(GgmlType::F32));
    }

    #[test]
    fn lm_head_specialization_is_selected_from_metadata() {
        let meta = WeightMeta {
            wtype: GgmlType::Q8_0,
            dims: vec![896, 151936],
            needs_transpose: false,
            role: TensorRole::LmHead,
        };

        assert_eq!(select_gemv_dispatch(&meta), GemvDispatchKind::LmHeadQ8_0);
    }

    #[test]
    fn generic_q8_0_keeps_default_dispatch() {
        let meta = WeightMeta {
            wtype: GgmlType::Q8_0,
            dims: vec![896, 151936],
            needs_transpose: false,
            role: TensorRole::Generic,
        };

        assert_eq!(select_gemv_dispatch(&meta), GemvDispatchKind::Default);
    }
}
