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

unsafe extern "C" {
    fn gemv_q2_k_f32_launch(
        weights_q2_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q3_k_f32_launch(
        weights_q3_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q4_k_f32_launch(
        weights_q4_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q5_k_f32_launch(
        weights_q5_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q6_k_f32_launch(
        weights_q6_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
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
}

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

    if !n_rows.is_multiple_of(256) {
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

    if !n_rows.is_multiple_of(256) {
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

    if !n_rows.is_multiple_of(256) {
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

    if !n_rows.is_multiple_of(256) {
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

    if !n_rows.is_multiple_of(32) {
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

    if !n_rows.is_multiple_of(32) {
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

    if !n_rows.is_multiple_of(32) {
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

    if !n_rows.is_multiple_of(32) {
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

    if !n_rows.is_multiple_of(32) {
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

    if !n_rows.is_multiple_of(32) {
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

    if !n_rows.is_multiple_of(256) {
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
