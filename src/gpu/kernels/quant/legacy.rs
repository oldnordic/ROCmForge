//! Legacy quantization kernel wrappers.
//!
//! This file contains specialized GEMV/GEMM kernels and fusion kernels that
//! don't belong to a specific quantization type:
//! - QKV fusion kernels (gemv_qkv_q4_0_f32)
//! - Gate-up fusion kernels (gemv_gate_up_q4_0_f32)
//! - SwiGLU fusion kernels (gemv_gate_up_swiglu_q4_0_f32)
//! - DP4A-optimized kernels (gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream)
//! - GEMM kernels for all quant types
//!
//! Basic quantize/dequantize/GEMV functions have been moved to separate modules:
//! - q4_0.rs: Q4_0 quantization
//! - q4_1.rs: Q4_1 quantization
//! - q4_k.rs: Q4_K quantization
//! - q5_k.rs: Q5_K quantization
//! - q6_k.rs: Q6_K quantization
//! - q8_0.rs: Q8_0 quantization

use super::super::super::device::GpuDevice;
use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use super::super::super::safety::{decode_graph_disabled_override_requested, decode_graph_enabled};
use std::os::raw::{c_int, c_void};

// Re-export Q8_0 GEMV functions from q8_gemv module
pub use super::super::q8_gemv::{
    gemv_q8_0_f32, gemv_q8_0_f32_lm_head, gemv_q8_0_f32_lm_head_on_stream,
    gemv_q8_0_f32_lm_head_on_stream_variant, gemv_q8_0_f32_on_stream,
};

// ── Q4_K GEMV ─────────────────────────────────────────────────────────────────────

/// Q4_K × f32 GEMV: Compute output = weights @ input
///
/// Computes matrix-vector multiplication where:
/// - weights: [n_rows/256][ncols_dst][144] Q4_K quantized weight matrix (column-major)
/// - input: [n_rows] f32 input vector
/// - output: [ncols_dst] f32 output vector
///
/// # Arguments
/// * `weights_q4_k` - GPU pointer to Q4_K quantized weights
/// * `input` - GPU pointer to f32 input vector
/// * `output` - GPU pointer to f32 output vector (will be written)
/// * `n_rows` - Number of rows (input dimension, must be multiple of 256)
/// * `ncols_dst` - Number of columns (output dimension)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
// ── Q2_K GEMV ─────────────────────────────────────────────────────────────────────

pub fn gemv_q2_k_f32(
    weights_q2_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q2_k_f32_on_stream(
        weights_q2_k,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q2_k_f32_on_stream(
    weights_q2_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q2_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q2_k_f32: n_rows must be multiple of 256, got {}",
                n_rows
            ),
        });
    }

    if weights_q2_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q2_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q2_k_f32_launch(
            weights_q2_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q2_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q3_K GEMV ─────────────────────────────────────────────────────────────────────

pub fn gemv_q3_k_f32(
    weights_q3_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q3_k_f32_on_stream(
        weights_q3_k,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q3_k_f32_on_stream(
    weights_q3_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q3_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q3_k_f32: n_rows must be multiple of 256, got {}",
                n_rows
            ),
        });
    }

    if weights_q3_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q3_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q3_k_f32_launch(
            weights_q3_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q3_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q4_K GEMV ─────────────────────────────────────────────────────────────────────

pub fn gemv_q4_k_f32(
    weights_q4_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q4_k_f32_on_stream(
        weights_q4_k,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q4_k_f32_on_stream(
    weights_q4_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_k_f32: n_rows must be multiple of 256, got {}",
                n_rows
            ),
        });
    }

    if weights_q4_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q4_k_f32_launch(
            weights_q4_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q4_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q5_K GEMV ─────────────────────────────────────────────────────────────────────

pub fn gemv_q5_k_f32(
    weights_q5_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q5_k_f32_on_stream(
        weights_q5_k,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q5_k_f32_on_stream(
    weights_q5_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q5_k_f32: n_rows must be multiple of 256, got {}",
                n_rows
            ),
        });
    }

    if weights_q5_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q5_k_f32_launch(
            weights_q5_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q5_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q5_0 GEMV ─────────────────────────────────────────────────────────────────────

pub fn gemv_q5_0_f32(
    weights_q5_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q5_0_f32_on_stream(
        weights_q5_0,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q5_0_f32_on_stream(
    weights_q5_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_0_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q5_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q5_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q5_0_f32_launch(
            weights_q5_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q5_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_0_f32(
    weights_q5_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemm_q5_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q5_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_0_f32_launch(
            weights_q5_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_0_f32_on_stream(
    weights_q5_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_0_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemm_q5_0_f32_on_stream: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q5_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_0_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_0_f32_launch(
            weights_q5_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_0_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q5_1 GEMV ─────────────────────────────────────────────────────────────────────

pub fn gemv_q5_1_f32(
    weights_q5_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q5_1_f32_on_stream(
        weights_q5_1,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q5_1_f32_on_stream(
    weights_q5_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_1_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q5_1_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q5_1.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_1_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q5_1_f32_launch(
            weights_q5_1,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q5_1_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_1_f32(
    weights_q5_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_1_f32: dimensions cannot be zero".to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemm_q5_1_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q5_1.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_1_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_1_f32_launch(
            weights_q5_1,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_1_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_1_f32_on_stream(
    weights_q5_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_1_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemm_q5_1_f32_on_stream: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q5_1.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_1_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_1_f32_launch(
            weights_q5_1,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_1_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q6_K GEMV ─────────────────────────────────────────────────────────────────────

pub fn gemv_q6_k_f32(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q6_k_f32_on_stream(
        weights_q6_k,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q6_k_f32_on_stream(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q6_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q6_k_f32: n_rows must be multiple of 256, got {}",
                n_rows
            ),
        });
    }

    if weights_q6_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q6_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_q6_k_f32_launch(
            weights_q6_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q6_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q4_0 QKV Fusion ───────────────────────────────────────────────────────────────

/// Fused QKV projection for Q4_0 quantized weights.
///
/// Computes:
/// ```text
/// out_q = w_q @ input + bias_q
/// out_k = w_k @ input + bias_k
/// out_v = w_v @ input + bias_v
/// ```
///
/// # Arguments
/// * `w_q`, `w_k`, `w_v` - Q4_0 quantized Q/K/V weight matrices
/// * `bias_q`, `bias_k`, `bias_v` - Optional bias vectors
/// * `input` - Input hidden states
/// * `out_q`, `out_k`, `out_v` - Output projections (will be written)
/// * `n_rows` - Input dimension (must be multiple of 32)
/// * `n_q`, `n_kv` - Output dimensions for Q and K/V
pub fn gemv_qkv_q4_0_f32(
    w_q: *const u8,
    w_k: *const u8,
    w_v: *const u8,
    bias_q: *const f32,
    bias_k: *const f32,
    bias_v: *const f32,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    n_rows: usize,
    n_q: usize,
    n_kv: usize,
) -> GpuResult<()> {
    gemv_qkv_q4_0_f32_on_stream(
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        input,
        out_q,
        out_k,
        out_v,
        n_rows,
        n_q,
        n_kv,
        hipStream_t::null(),
    )
}

pub fn gemv_qkv_q4_0_f32_on_stream(
    w_q: *const u8,
    w_k: *const u8,
    w_v: *const u8,
    bias_q: *const f32,
    bias_k: *const f32,
    bias_v: *const f32,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    n_rows: usize,
    n_q: usize,
    n_kv: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_q == 0 || n_kv == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_qkv_q4_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if !n_rows.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_qkv_q4_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if w_q.is_null() || w_k.is_null() || w_v.is_null() || input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_qkv_q4_0_f32: required pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_qkv_q4_0_f32_launch(
            w_q,
            w_k,
            w_v,
            bias_q,
            bias_k,
            bias_v,
            input,
            out_q,
            out_k,
            out_v,
            n_rows as c_int,
            n_q as c_int,
            n_kv as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_qkv_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemv_qkv_q4_0_f32_on_stream_variant(
    w_q: *const u8,
    w_k: *const u8,
    w_v: *const u8,
    bias_q: *const f32,
    bias_k: *const f32,
    bias_v: *const f32,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    n_rows: usize,
    n_q: usize,
    n_kv: usize,
    stream: hipStream_t,
    variant: i32,
) -> GpuResult<()> {
    if n_rows == 0 || n_q == 0 || n_kv == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_qkv_q4_0_f32_variant: dimensions cannot be zero".to_string(),
        });
    }

    if !n_rows.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_qkv_q4_0_f32_variant: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if w_q.is_null() || w_k.is_null() || w_v.is_null() || input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_qkv_q4_0_f32_variant: required pointers must be non-null"
                .to_string(),
        });
    }

    let result = unsafe {
        gemv_qkv_q4_0_f32_variant_launch(
            w_q,
            w_k,
            w_v,
            bias_q,
            bias_k,
            bias_v,
            input,
            out_q,
            out_k,
            out_v,
            n_rows as c_int,
            n_q as c_int,
            n_kv as c_int,
            variant as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_qkv_q4_0_f32_variant kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q4_0 Gate-Up Fusion ───────────────────────────────────────────────────────────

/// Fused gate-up projection for Q4_0 quantized weights.
///
/// Computes:
/// ```text
/// out_gate = w_gate @ input
/// out_up = w_up @ input
/// ```
pub fn gemv_gate_up_q4_0_f32(
    w_gate: *const u8,
    w_up: *const u8,
    input: *const f32,
    out_gate: *mut f32,
    out_up: *mut f32,
    n_rows: usize,
    n_ff: usize,
) -> GpuResult<()> {
    gemv_gate_up_q4_0_f32_on_stream(
        w_gate,
        w_up,
        input,
        out_gate,
        out_up,
        n_rows,
        n_ff,
        hipStream_t::null(),
    )
}

pub fn gemv_gate_up_q4_0_f32_on_stream(
    w_gate: *const u8,
    w_up: *const u8,
    input: *const f32,
    out_gate: *mut f32,
    out_up: *mut f32,
    n_rows: usize,
    n_ff: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_q4_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if !n_rows.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_q4_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if w_gate.is_null() || w_up.is_null() || input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_q4_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_q4_0_f32_launch(
            w_gate,
            w_up,
            input,
            out_gate,
            out_up,
            n_rows as c_int,
            n_ff as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_gate_up_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q4_0 SwiGLU Gate-Up Fusion ────────────────────────────────────────────────────

/// Fused gate-up projection with SwiGLU activation for Q4_0 quantized weights.
///
/// Computes:
/// ```text
/// out_swiglu = swiglu(w_gate @ input, w_up @ input)
///            = silu(w_gate @ input) * (w_up @ input)
/// ```
pub fn gemv_gate_up_swiglu_q4_0_f32(
    w_gate: *const u8,
    w_up: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
) -> GpuResult<()> {
    gemv_gate_up_swiglu_q4_0_f32_on_stream(
        w_gate,
        w_up,
        input,
        out_swiglu,
        n_rows,
        n_ff,
        hipStream_t::null(),
    )
}

pub fn gemv_gate_up_swiglu_q4_0_f32_on_stream(
    w_gate: *const u8,
    w_up: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_q4_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if !n_rows.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_swiglu_q4_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if w_gate.is_null() || w_up.is_null() || input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_q4_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_swiglu_q4_0_f32_launch(
            w_gate,
            w_up,
            input,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_gate_up_swiglu_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Vulkan-style variant with explicit wave control
pub fn gemv_gate_up_swiglu_vulkan_q4_0_f32(
    w_gate: *const u8,
    w_up: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
    n_waves: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_ff == 0 || n_waves == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_vulkan_q4_0_f32: dimensions cannot be zero"
                .to_string(),
        });
    }

    if !n_rows.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_gate_up_swiglu_vulkan_q4_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if w_gate.is_null() || w_up.is_null() || input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_gate_up_swiglu_vulkan_q4_0_f32: pointers must be non-null"
                .to_string(),
        });
    }

    let result = unsafe {
        gemv_gate_up_swiglu_vulkan_q4_0_f32_launch(
            w_gate,
            w_up,
            input,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            n_waves as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_gate_up_swiglu_vulkan_q4_0_f32 kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

// ── GEMM Kernels ──────────────────────────────────────────────────────────────────

pub fn gemm_q4_0_f32(
    weights_q4_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_0_f32_launch(
            weights_q4_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q4_1_f32(
    weights_q4_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_1_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_1.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_1_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_1_f32_launch(
            weights_q4_1,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_1_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q8_0_f32(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q8_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q8_0_f32_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q8_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q8_0_f32_on_stream(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q8_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q8_0_f32_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q8_0_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q4_k_f32(
    weights_q4_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_k_f32_launch(
            weights_q4_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q4_k_f32_on_stream(
    weights_q4_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_k_f32_launch(
            weights_q4_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_k_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_k_f32(
    weights_q5_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q5_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_k_f32_launch(
            weights_q5_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_k_f32_on_stream(
    weights_q5_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q5_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_k_f32_launch(
            weights_q5_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_k_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q6_k_f32(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q6_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q6_k_f32_launch(
            weights_q6_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q6_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q6_k_f32_on_stream(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q6_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q6_k_f32_launch(
            weights_q6_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q6_k_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── DP4A-Optimized Q4_0 Fusion Kernel ──────────────────────────────────────────────

/// DP4A-optimized fused QKV+RoPE+KV-write kernel for Q4_0 quantized weights.
///
/// This kernel uses __builtin_amdgcn_sdot4 for 4-way int8 multiply-accumulate,
/// providing 1.5-2× speedup on RDNA2+ (gfx1030+) GPUs.
///
/// Trade-off: 0.4% noise from on-the-fly activation quantization vs 2× speedup
///
/// Computes:
/// 1. Layer norm on input hidden states
/// 2. Fused QKV projection with Q4_0 quantized weights
/// 3. RoPE rotation on Q and K
/// 4. Write K and V to KV cache
pub fn gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream(
    device: &GpuDevice,
    raw_hidden: *const f32,
    norm_weight: *const f32,
    eps: f32,
    w_q: *const u8,
    w_k: *const u8,
    w_v: *const u8,
    bias_q: Option<*const f32>,
    bias_k: Option<*const f32>,
    bias_v: Option<*const f32>,
    out_q: *mut f32,
    k_cache: *mut f32,
    v_cache: *mut f32,
    n_rows: usize,
    n_q: usize,
    n_kv: usize,
    pos_ptr: *const i32,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || n_q == 0 || n_kv == 0 || head_dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a: dimensions cannot be zero"
                .to_string(),
        });
    }

    if !n_rows.is_multiple_of(32) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if raw_hidden.is_null()
        || norm_weight.is_null()
        || w_q.is_null()
        || w_k.is_null()
        || w_v.is_null()
        || out_q.is_null()
        || k_cache.is_null()
        || v_cache.is_null()
        || pos_ptr.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a: required pointers must be non-null"
                    .to_string(),
        });
    }

    let bias_q_ptr = bias_q.unwrap_or(std::ptr::null());
    let bias_k_ptr = bias_k.unwrap_or(std::ptr::null());
    let bias_v_ptr = bias_v.unwrap_or(std::ptr::null());

    let result = unsafe {
        gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_launch(
            raw_hidden,
            norm_weight,
            eps,
            w_q,
            w_k,
            w_v,
            bias_q_ptr,
            bias_k_ptr,
            bias_v_ptr,
            out_q,
            k_cache,
            v_cache,
            n_rows as c_int,
            n_q as c_int,
            n_kv as c_int,
            pos_ptr,
            head_dim as c_int,
            theta_base,
            if neox { 1 } else { 0 },
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

// ── FFI Declarations ───────────────────────────────────────────────────────────────

unsafe extern "C" {
    // Q2_K GEMV
    fn gemv_q2_k_f32_launch(
        weights_q2_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Q3_K GEMV
    fn gemv_q3_k_f32_launch(
        weights_q3_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Q4_K GEMV
    fn gemv_q4_k_f32_launch(
        weights_q4_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Q5_K GEMV
    fn gemv_q5_k_f32_launch(
        weights_q5_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Q6_K GEMV and related
    fn gemv_q6_k_f32_launch(
        weights_q6_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q6_k_f32_launch(
        weights_q6_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn quantize_q6_k_launch(
        input: *const f32,
        output: *mut u8,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn dequantize_q6_k_launch(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn dequantize_q6_k_batched_launch(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn verify_q6_k_launch(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Q4_0 QKV fusion
    fn gemv_qkv_q4_0_f32_launch(
        w_q: *const u8,
        w_k: *const u8,
        w_v: *const u8,
        bias_q: *const f32,
        bias_k: *const f32,
        bias_v: *const f32,
        input: *const f32,
        out_q: *mut f32,
        out_k: *mut f32,
        out_v: *mut f32,
        n_rows: c_int,
        n_q: c_int,
        n_kv: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_qkv_q4_0_f32_variant_launch(
        w_q: *const u8,
        w_k: *const u8,
        w_v: *const u8,
        bias_q: *const f32,
        bias_k: *const f32,
        bias_v: *const f32,
        input: *const f32,
        out_q: *mut f32,
        out_k: *mut f32,
        out_v: *mut f32,
        n_rows: c_int,
        n_q: c_int,
        n_kv: c_int,
        variant: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Q4_0 gate-up fusion
    fn gemv_gate_up_q4_0_f32_launch(
        w_gate: *const u8,
        w_up: *const u8,
        input: *const f32,
        out_gate: *mut f32,
        out_up: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_gate_up_swiglu_q4_0_f32_launch(
        w_gate: *const u8,
        w_up: *const u8,
        input: *const f32,
        out_swiglu: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_gate_up_swiglu_vulkan_q4_0_f32_launch(
        w_gate: *const u8,
        w_up: *const u8,
        input: *const f32,
        out_swiglu: *mut f32,
        n_rows: c_int,
        n_ff: c_int,
        n_waves: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // GEMM kernels
    fn gemm_q4_0_f32_launch(
        weights_q4_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q4_1_f32_launch(
        weights_q4_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q8_0_f32_launch(
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q4_k_f32_launch(
        weights_q4_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q5_k_f32_launch(
        weights_q5_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q5_0_f32_launch(
        weights_q5_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q5_0_f32_launch(
        weights_q5_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q5_1_f32_launch(
        weights_q5_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q5_1_f32_launch(
        weights_q5_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // DP4A-optimized fusion kernel
    fn gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_launch(
        raw_hidden: *const f32,
        norm_weight: *const f32,
        eps: f32,
        w_q: *const u8,
        w_k: *const u8,
        w_v: *const u8,
        bias_q: *const f32,
        bias_k: *const f32,
        bias_v: *const f32,
        out_q: *mut f32,
        k_cache: *mut f32,
        v_cache: *mut f32,
        n_rows: c_int,
        n_q: c_int,
        n_kv: c_int,
        pos_ptr: *const c_int,
        head_dim: c_int,
        theta_base: f32,
        neox: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

// Q4_K tests moved to q4_k module
