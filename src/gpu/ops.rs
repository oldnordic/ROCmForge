//! GPU math dispatch for model weights.
//!
//! This layer validates GGUF metadata before calling the raw HIP kernels.

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::kernels::{
    gemm_q4_0_f32, gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_k_f32, gemm_q8_0_f32,
    gemv_gate_up_swiglu_q4_0_f32, gemv_q4_0_f32, gemv_q4_1_f32, gemv_q4_k_f32, gemv_q5_k_f32,
    gemv_q8_0_f32, gemv_qkv_q4_0_f32,
};
use super::weights::{GpuBuffer, WeightMeta};
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

    match meta.wtype {
        GgmlType::Q4_0 => gemv_q4_0_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
        )?,
        GgmlType::Q4_1 => gemv_q4_1_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
        )?,
        GgmlType::Q8_0 => gemv_q8_0_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
        )?,
        GgmlType::Q4_K => gemv_q4_k_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
        )?,
        GgmlType::Q5_K => gemv_q5_k_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
        )?,
        _ => unreachable!("unsupported types return before dispatch"),
    }

    Ok(())
}

/// Dispatch a fused QKV GEMV for a single row.
pub fn gpu_dispatch_fused_qkv(
    _device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
) -> GpuResult<()> {
    if q_meta.wtype == GgmlType::Q4_0
        && k_meta.wtype == GgmlType::Q4_0
        && v_meta.wtype == GgmlType::Q4_0
    {
        gemv_qkv_q4_0_f32(
            w_q.as_ptr() as *const u8,
            w_k.as_ptr() as *const u8,
            w_v.as_ptr() as *const u8,
            input,
            out_q,
            out_k,
            out_v,
            h,
            q_size,
            kv_size,
        )?;
        return Ok(());
    }

    // Fallback to individual dispatches
    gpu_dispatch_gemv(_device, w_q, q_meta, input, out_q, q_size, h)?;
    gpu_dispatch_gemv(_device, w_k, k_meta, input, out_k, kv_size, h)?;
    gpu_dispatch_gemv(_device, w_v, v_meta, input, out_v, kv_size, h)?;

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
    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        gemv_gate_up_swiglu_q4_0_f32(
            w_gate.as_ptr() as *const u8,
            w_up.as_ptr() as *const u8,
            input,
            out_swiglu,
            h,
            ff_size,
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

    #[test]
    fn layout_accepts_native_gguf_orientation() {
        let meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: vec![2048, 1024],
            needs_transpose: true,
        };

        validate_gemv_layout(&meta, 1024, 2048).unwrap();
    }

    #[test]
    fn layout_rejects_cpu_oriented_dimension_order() {
        let meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: vec![1024, 2048],
            needs_transpose: false,
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
}
