//! Batched GPU operations for prompt prefill.
//!
//! This module provides high-level dispatch functions for batched GPU operations
//! used during prompt prefill. It validates metadata and calls the raw HIP kernels.

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use super::kernels::quant::batched::{
    batched_fused_gate_up_q4_0_f32, batched_gemm_q4_0_f32, batched_gemm_q4_1_f32,
    wmma_matmul_q4_0_f32,
};
use super::kernels::quant::gemm_q8_0_f32_on_stream;
use super::kernels::{mul_on_stream, silu_on_stream};
use super::ops::gpu_dispatch_gemv_on_stream;
use super::weights::{GpuBuffer, WeightMeta};
use crate::loader::GgmlType;

/// Validate that weight metadata describes a valid 2D matrix for GEMM.
fn validate_gemm_layout(meta: &WeightMeta, out_dim: usize, in_dim: usize) -> GpuResult<()> {
    if meta.dims.len() != 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "batched_gemm_dispatch".to_string(),
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
            tensor: "batched_gemm_dispatch".to_string(),
            dims: meta.dims.clone(),
            reason: format!(
                "shape mismatch: matrix is {:?}, but input is [{}] and output is [{}]",
                meta.dims, in_dim, out_dim
            ),
        })
    }
}

/// Check if GgmlType is supported for batched operations.
fn supports_batched_gemm_type(wtype: GgmlType) -> bool {
    matches!(wtype, GgmlType::Q4_0 | GgmlType::Q4_1)
}

fn validate_batched_q4_0_dims(in_dim: usize, out_dim: usize, seq_len: usize) -> GpuResult<()> {
    if in_dim == 0 || out_dim == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "batched gemv: invalid dimensions in_dim={}, out_dim={}, seq_len={}",
                in_dim, out_dim, seq_len
            ),
        });
    }

    if !in_dim.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "batched q4_0 gemv: in_dim={} must be multiple of 32 for Q4_0 block layout",
                in_dim
            ),
        });
    }

    Ok(())
}

/// Dispatch a type-aware batched GEMM for prefill processing.
///
/// This function computes matrix multiplication for multiple input vectors in parallel:
/// output[seq_len][out_dim] = input[seq_len][in_dim] × weights[out_dim][in_dim]
///
/// # Arguments
/// * `_device` - GPU device reference (for future workspace allocation)
/// * `weights` - Weight buffer (Q4_0 or Q4_1 format)
/// * `meta` - Weight metadata (determines which kernel to use)
/// * `input` - GPU pointer to input activations [seq_len][in_dim] (row-major)
/// * `output` - GPU pointer to output buffer [seq_len][out_dim] (row-major)
/// * `in_dim` - Input dimension (number of columns in input, rows in weights)
/// * `out_dim` - Output dimension (number of columns in weights)
/// * `seq_len` - Number of tokens in batch (sequence length)
/// * `stream` - HIP stream for kernel execution
///
/// # Returns
/// Ok(()) on success, Err if validation or kernel launch fails
///
/// # Supported Types
/// - Q4_0: Uses batched_q4_0 kernel
/// - Q4_1: Uses batched_q4_1 kernel
/// - Other types: Not yet implemented for prefill path
pub fn gpu_dispatch_batched_gemv_batched(
    _device: &super::device::GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    // Validate weight type
    if !supports_batched_gemm_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "batched_gemv".to_string(),
            wtype: meta.wtype,
        });
    }

    // Validate layout
    validate_gemm_layout(meta, out_dim, in_dim)?;

    validate_batched_q4_0_dims(in_dim, out_dim, seq_len)?;

    // Type-aware dispatch: select kernel based on weight type
    match meta.wtype {
        GgmlType::Q4_0 => wmma_matmul_q4_0_f32(
            weights.as_ptr() as *const std::ffi::c_void,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            stream,
        ),
        GgmlType::Q4_1 => batched_gemm_q4_1_f32(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            stream,
        ),
        _ => Err(GpuError::UnsupportedWeightType {
            tensor: "batched_gemv".to_string(),
            wtype: meta.wtype,
        }),
    }
}

/// Dispatch a batched fused Gate/Up projection with SwiGLU for prefill.
///
/// Computes for all `seq_len` positions in parallel:
///   out[i] = SiLU(gate[i]) * up[i]
/// where `gate = input x W_gate` and `up = input x W_up`.
///
/// # Supported combinations
/// - **Q4_0/Q4_0** — uses an optimized fused HIP kernel (`batched_fused_gate_up_q4_0_f32`).
/// - **Q8_0/Q8_0** — uses two `gemm_q8_0_f32_on_stream` batched GEMMs plus per-row
///   element-wise `silu_on_stream` + `mul_on_stream`. A caller-provided `gate_scratch`
///   buffer of shape `[seq_len][ff_size]` is required for the intermediate gate output.
/// - **Q4_1/Q4_1, Q4_0/Q4_1, Q8_0/Q4_0, etc.** — returns `UnsupportedWeightType` so the
///   caller can fall back to the per-token loop (which already handles any type combo).
///
/// # Arguments
/// * `device`                 — GPU device reference
/// * `w_gate`                 — Gate weight buffer
/// * `gate_meta`              — Gate weight metadata
/// * `w_up`                   — Up weight buffer
/// * `up_meta`                — Up weight metadata
/// * `input`                  — Input activations `[seq_len][hidden_dim]`
/// * `output`                 — Output buffer `[seq_len][ff_size]`
/// * `gate_scratch`           — Optional scratch buffer `[seq_len][ff_size]` for `Q8_0/Q8_0`
/// * `hidden_dim`             — Input hidden dimension
/// * `ff_size`                — Feed-forward output dimension
/// * `seq_len`                — Number of tokens in batch
/// * `stream`                 — HIP stream
pub fn gpu_dispatch_batched_fused_gate_up_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    gate_scratch: *mut f32,
    hidden_dim: usize,
    ff_size: usize,
    seq_len: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    // Shared dimension validation
    if hidden_dim == 0 || ff_size == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "batched_fused_gate_up: invalid dimensions hidden_dim={}, ff_size={}, seq_len={}",
                hidden_dim, ff_size, seq_len
            ),
        });
    }

    match (gate_meta.wtype, up_meta.wtype) {
        // ── Q4_0/Q4_0: native fused batched kernel ──────────────────────────
        (GgmlType::Q4_0, GgmlType::Q4_0) => {
            validate_gemm_layout(gate_meta, ff_size, hidden_dim)?;
            validate_gemm_layout(up_meta, ff_size, hidden_dim)?;

            if !hidden_dim.is_multiple_of(32) {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: format!(
                        "batched fused gate-up q4_0: hidden_dim={} must be multiple of 32",
                        hidden_dim
                    ),
                });
            }

            batched_fused_gate_up_q4_0_f32(
                w_gate.as_ptr() as *const u8,
                w_up.as_ptr() as *const u8,
                input,
                output,
                hidden_dim,
                ff_size,
                seq_len,
                stream,
            )
        }

        // ── Q8_0/Q8_0: batched GEMM + element-wise SiLU * Up ────────────────
        (GgmlType::Q8_0, GgmlType::Q8_0) => {
            validate_gemm_layout(gate_meta, ff_size, hidden_dim)?;
            validate_gemm_layout(up_meta, ff_size, hidden_dim)?;

            if gate_scratch.is_null() {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: "batched fused gate-up Q8_0/Q8_0: gate_scratch must be non-null"
                        .to_string(),
                });
            }

            // 1. gate = input x W_gate  -> gate_scratch
            gemm_q8_0_f32_on_stream(
                w_gate.as_ptr() as *const u8,
                input,
                gate_scratch,
                hidden_dim,
                ff_size,
                seq_len,
                stream,
            )?;
            // 2. up   = input x W_up    -> output
            gemm_q8_0_f32_on_stream(
                w_up.as_ptr() as *const u8,
                input,
                output,
                hidden_dim,
                ff_size,
                seq_len,
                stream,
            )?;
            // 3. element-wise: output = SiLU(gate) * up, per row
            for i in 0..seq_len {
                let gate_row = unsafe { gate_scratch.add(i * ff_size) };
                let up_row = unsafe { output.add(i * ff_size) };
                silu_on_stream(gate_row, gate_row, ff_size, stream)?;
                mul_on_stream(gate_row, up_row, up_row, ff_size, stream)?;
            }
            Ok(())
        }

        // ── Unsupported combination: caller must fall back to per-token ─────
        _ => Err(GpuError::UnsupportedWeightType {
            tensor: "batched_fused_gate_up".to_string(),
            wtype: if gate_meta.wtype != GgmlType::Q4_0 && gate_meta.wtype != GgmlType::Q8_0 {
                gate_meta.wtype
            } else {
                up_meta.wtype
            },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_gemm_layout_accepts_correct_shapes() {
        let meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: vec![1024, 4096],
            needs_transpose: false,
            role: crate::gpu::TensorRole::Generic,
            svd_k: None,
        };

        // Test both orientations
        let result = validate_gemm_layout(&meta, 4096, 1024);
        assert!(result.is_ok());

        let result = validate_gemm_layout(&meta, 1024, 4096);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_gemm_layout_rejects_mismatched_shapes() {
        let meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: vec![1024, 4096],
            needs_transpose: false,
            role: crate::gpu::TensorRole::Generic,
            svd_k: None,
        };

        let result = validate_gemm_layout(&meta, 2048, 1024);
        assert!(result.is_err());
    }

    #[test]
    fn validate_gemm_layout_rejects_non_2d() {
        let meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: vec![1024, 4096, 8],
            needs_transpose: false,
            role: crate::gpu::TensorRole::Generic,
            svd_k: None,
        };

        let result = validate_gemm_layout(&meta, 4096, 1024);
        assert!(result.is_err());
    }

    #[test]
    fn supports_batched_gemm_type_q4_0_and_q4_1() {
        assert!(supports_batched_gemm_type(GgmlType::Q4_0));
        assert!(supports_batched_gemm_type(GgmlType::Q4_1));
        assert!(!supports_batched_gemm_type(GgmlType::Q8_0));
    }

    #[test]
    fn batched_q4_0_rejects_non_multiple_in_dim() {
        assert!(validate_batched_q4_0_dims(33, 1024, 10).is_err());
        assert!(validate_batched_q4_0_dims(32, 1024, 10).is_ok());
        assert!(validate_batched_q4_0_dims(32, 0, 10).is_err());
        assert!(validate_batched_q4_0_dims(32, 1024, 0).is_err());
    }
}
