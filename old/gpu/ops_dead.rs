//! Dead GPU dispatch code (quarantined 2026-04-19).
//!
//! This module contains GPU dispatch functions that have no incoming calls
//! according to Magellan analysis. These are preserved for reference but are
//! not part of the live decode spine.
//!
//! Moved from src/gpu/ops.rs during GPU decode surface cleanup.

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use super::weights::{GpuBuffer, WeightMeta};
use crate::loader::GgmlType;

/// Dispatch a fused QKV GEMV with bias on an explicit HIP stream (decode-strict).
///
/// Strictly enforces that Q/K/V weights are all Q4_0. Returns an error for any other type.
/// 
/// NOTE: No incoming calls according to Magellan analysis.
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

    super::super::ops::gpu_dispatch_fused_qkv_on_stream(
        device, w_q, q_meta, q_bias, w_k, k_meta, k_bias, w_v, v_meta, v_bias, input, out_q, out_k,
        out_v, q_size, kv_size, h, stream,
    )
}

/// Dispatch a GPU GEMM for GGUF weights.
///
/// NOTE: No incoming calls according to Magellan analysis. GEMM kernels are prefill-only,
/// not needed for the current decode hot path.
pub fn gpu_dispatch_gemm(
    _device: &GpuDevice,
    weights: &GpuBuffer,
    _meta: &WeightMeta,
    _input: *const f32,
    _output: *mut f32,
    _out_dim: usize,
    _in_dim: usize,
    _seq_len: usize,
) -> GpuResult<()> {
    // DISABLED: GEMM kernels not available (prefill-only, not needed for decode)
    // TODO: Implement GEMM kernels or use GEMV for all cases
    return Err(GpuError::UnsupportedOperation {
        operation: "gpu_dispatch_gemm".to_string(),
        reason: "GEMM kernels not yet implemented. Use GEMV (seq_len=1) for now.".to_string(),
    });
}

/// Experimental fused gate-up fastpath variant (disabled).
///
/// NOTE: Experimental tuning surface, not part of stable live decode.
fn try_q4_0_q8_0_fused_gate_up_fastpath_variant(
    _w_gate: &GpuBuffer,
    _w_up: &GpuBuffer,
    _input: *const f32,
    _output: *mut f32,
    _h: usize,
    _ff_size: usize,
    _variant: i32,
    _stream: hipStream_t,
) -> GpuResult<()> {
    // DISABLED: Experimental Q8 variants not available
    Err(GpuError::UnsupportedOperation {
        operation: "gpu_dispatch_fused_gate_up_with_scratch_q8_variant".to_string(),
        reason: "Experimental Q8 variants not yet implemented. Use baseline variant.".to_string(),
    })
}
