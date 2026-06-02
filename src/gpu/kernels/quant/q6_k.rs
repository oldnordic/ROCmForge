//! Q6_K quantization kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

// Declare external kernels
extern "C" {
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

    fn verify_q6_k_launch(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
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

    #[link_name = "gemm_q6_k_f32_launch_dispatch"]
    fn gemm_q6_k_f32_launch_dispatch(
        weights_q6_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

/// Quantize f32 data to Q6_K format.
///
/// # Arguments
/// * `input` - GPU pointer to f32 input data [n]
/// * `output` - GPU pointer to Q6_K output data [n/256 * 210]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn quantize_q6_k(input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q6_k: n cannot be zero".to_string(),
        });
    }

    let num_blocks = n.div_ceil(256);
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { quantize_q6_k_launch(input, output, n as c_int, hipStream_t::null()) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q6_k kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dequantize Q6_K data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q6_K input data [n/256 * 210]
/// * `output` - GPU pointer to f32 output data [n]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q6_k(input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q6_k: n cannot be zero".to_string(),
        });
    }

    let num_blocks = n.div_ceil(256);
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { dequantize_q6_k_launch(input, output, n as c_int, hipStream_t::null()) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q6_k kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched dequantize Q6_K data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q6_K input data [batch_size][n/256 * 210]
/// * `output` - GPU pointer to f32 output data [batch_size][n]
/// * `n` - Number of elements per batch
/// * `batch_size` - Number of batches
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q6_k_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q6_k_batched: n and batch_size cannot be zero".to_string(),
        });
    }

    let num_blocks = n.div_ceil(256);
    if num_blocks == 0 {
        return Ok(());
    }

    // The HIP kernel handles batching internally via n * batch_size
    let total_n = n * batch_size;
    let result =
        unsafe { dequantize_q6_k_launch(input, output, total_n as c_int, hipStream_t::null()) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q6_k_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Verify Q6_K quantization accuracy.
///
/// Compares original f32 data with quantize-dequantize round-trip.
///
/// # Arguments
/// * `original` - GPU pointer to original f32 data [n]
/// * `quantized` - GPU pointer to Q6_K quantized data [n/256 * 210]
/// * `errors` - GPU pointer to error array [4] (intermediate results)
///   - errors[0]: max error
///   - errors[1]: sum of squared errors (for MSE)
///   - errors[2]: sum of original magnitudes
///   - errors[3]: sum of errors
/// * `n` - Number of elements
///
/// Must be followed by a finalize step to compute final metrics from errors.
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn verify_q6_k_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q6_k_accuracy: n cannot be zero".to_string(),
        });
    }

    let result =
        unsafe { verify_q6_k_launch(original, quantized, errors, n as c_int, hipStream_t::null()) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("verify_q6_k_accuracy kernel failed: {:?}", result),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_q6_k_rejects_zero_n() {
        let result = quantize_q6_k(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q6_k_rejects_zero_n() {
        let result = dequantize_q6_k(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q6_k_batched_rejects_zero_batch() {
        let result = dequantize_q6_k_batched(std::ptr::null(), std::ptr::null_mut(), 256, 0);
        assert!(result.is_err());
    }

    #[test]
    fn verify_q6_k_accuracy_rejects_zero_n() {
        let result =
            verify_q6_k_accuracy(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }
}
