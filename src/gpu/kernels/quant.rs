//! Q4_K and Q8_0 quantization kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipError_t;
use std::os::raw::c_int;

/// Quantize f32 data to Q4_K format.
///
/// # Arguments
/// * `input` - GPU pointer to f32 input data [n]
/// * `output` - GPU pointer to Q4_K output data [n/256 * 144]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn quantize_q4_k(input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q4_k: n cannot be zero".to_string(),
        });
    }

    // Each block processes QK_K elements
    let num_blocks = (n + 255) / 256;
    if num_blocks == 0 {
        return Ok(());
    }

    // Load and call kernel
    let result = unsafe { quantize_q4_k_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q4_k kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dequantize Q4_K data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q4_K input data [n/256 * 144]
/// * `output` - GPU pointer to f32 output data [n]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q4_k(input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_k: n cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 255) / 256;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { dequantize_q4_k_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q4_k kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched dequantize Q4_K data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q4_K input data [batch_size][n/256 * 144]
/// * `output` - GPU pointer to f32 output data [batch_size][n]
/// * `n` - Elements per batch
/// * `batch_size` - Number of batches
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q4_k_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_k_batched: n and batch_size cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 255) / 256;
    if num_blocks == 0 {
        return Ok(());
    }

    let result =
        unsafe { dequantize_q4_k_batched_kernel(input, output, n as c_int, batch_size as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q4_k_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Verify Q4_K quantization accuracy.
///
/// # Arguments
/// * `original` - GPU pointer to original f32 data [n]
/// * `quantized` - GPU pointer to Q4_K quantized data [n/256 * 144]
/// * `errors` - GPU pointer to error metrics [4] (will be written)
///   - errors[0]: max error (output)
///   - errors[1]: MSE (output)
///   - errors[2]: sum of original values (output)
///   - errors[3]: sum of errors (output)
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn verify_q4_k_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_k_accuracy: n cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 255) / 256;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { verify_q4_k_accuracy_kernel(original, quantized, errors, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("verify_q4_k_accuracy kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Finalize Q4_K accuracy metrics.
///
/// Must be called after verify_q4_k_accuracy to compute final values.
///
/// # Arguments
/// * `errors` - GPU pointer to intermediate error values [4]
/// * `metrics` - GPU pointer to final metrics [3] (will be written)
///   - metrics[0]: max error
///   - metrics[1]: MSE
///   - metrics[2]: relative error
/// * `n` - Total number of elements (for MSE normalization)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
pub fn finalize_q4_k_metrics(errors: *const f32, metrics: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_k_metrics: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { finalize_q4_k_metrics_kernel(errors, metrics, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("finalize_q4_k_metrics kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Quantize f32 data to Q5_K format.
///
/// # Arguments
/// * `input` - GPU pointer to f32 input data [n]
/// * `output` - GPU pointer to Q5_K output data [n/256 * 176]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn quantize_q5_k(input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q5_k: n cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 255) / 256;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { quantize_q5_k_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q5_k kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dequantize Q5_K data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q5_K input data [n/256 * 176]
/// * `output` - GPU pointer to f32 output data [n]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q5_k(input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q5_k: n cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 255) / 256;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { dequantize_q5_k_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q5_k kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched dequantize Q5_K data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q5_K input data [batch_size][n/256 * 176]
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
pub fn dequantize_q5_k_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q5_k_batched: n and batch_size cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 255) / 256;
    if num_blocks == 0 {
        return Ok(());
    }

    let result =
        unsafe { dequantize_q5_k_batched_kernel(input, output, n as c_int, batch_size as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q5_k_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Verify Q5_K quantization accuracy.
///
/// Compares original f32 data with quantize-dequantize round-trip.
///
/// # Arguments
/// * `original` - GPU pointer to original f32 data [n]
/// * `quantized` - GPU pointer to Q5_K quantized data [n/256 * 176]
/// * `errors` - GPU pointer to error array [4] (intermediate results)
/// * `n` - Number of elements
///
/// Must be followed by `finalize_q5_k_metrics` to get final metrics.
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn verify_q5_k_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q5_k_accuracy: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { verify_q5_k_accuracy_kernel(original, quantized, errors, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("verify_q5_k_accuracy kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Finalize Q5_K accuracy metrics.
///
/// Computes final metrics from intermediate error values.
///
/// # Arguments
/// * `errors` - GPU pointer to intermediate error array [4]
/// * `metrics` - GPU pointer to final metrics [3]: [max_error, mse, relative_error]
/// * `n` - Number of elements (for MSE normalization)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn finalize_q5_k_metrics(errors: *const f32, metrics: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q5_k_metrics: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { finalize_q5_k_metrics_kernel(errors, metrics, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("finalize_q5_k_metrics kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── FFI Declarations ─────────────────────────────────────────────────────────────

/// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn quantize_q4_k_kernel(input: *const f32, output: *mut u8, n: c_int) -> hipError_t;

    fn dequantize_q4_k_kernel(input: *const u8, output: *mut f32, n: c_int) -> hipError_t;

    fn dequantize_q4_k_batched_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        batch_size: c_int,
    ) -> hipError_t;

    fn verify_q4_k_accuracy_kernel(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
    ) -> hipError_t;

    fn finalize_q4_k_metrics_kernel(errors: *const f32, metrics: *mut f32, n: c_int) -> hipError_t;

    // Q5_K kernels
    fn quantize_q5_k_kernel(input: *const f32, output: *mut u8, n: c_int) -> hipError_t;

    fn dequantize_q5_k_kernel(input: *const u8, output: *mut f32, n: c_int) -> hipError_t;

    fn dequantize_q5_k_batched_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        batch_size: c_int,
    ) -> hipError_t;

    fn verify_q5_k_accuracy_kernel(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
    ) -> hipError_t;

    fn finalize_q5_k_metrics_kernel(errors: *const f32, metrics: *mut f32, n: c_int) -> hipError_t;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_q4_k_rejects_zero_n() {
        let result = quantize_q4_k(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q4_k_rejects_zero_n() {
        let result = dequantize_q4_k(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q4_k_batched_rejects_zero_batch() {
        let result = dequantize_q4_k_batched(std::ptr::null(), std::ptr::null_mut(), 256, 0);
        assert!(result.is_err());
    }

    #[test]
    fn verify_q4_k_accuracy_rejects_zero_n() {
        let result =
            verify_q4_k_accuracy(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn finalize_q4_k_metrics_rejects_zero_n() {
        let result = finalize_q4_k_metrics(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }
}

// ── Q8_0 Safe Wrappers ─────────────────────────────────────────────────────────────

/// Quantize f32 data to Q8_0 format.
///
/// # Arguments
/// * `input` - GPU pointer to f32 input data [n]
/// * `output` - GPU pointer to Q8_0 output data [n/32 * 34]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn quantize_q8_0(input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q8_0: n cannot be zero".to_string(),
        });
    }

    // Each block processes QK8_0 elements
    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    // Load and call kernel
    let result = unsafe { quantize_q8_0_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q8_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dequantize Q8_0 data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q8_0 input data [n/32 * 34]
/// * `output` - GPU pointer to f32 output data [n]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q8_0(input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q8_0: n cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { dequantize_q8_0_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q8_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched dequantize Q8_0 data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q8_0 input data [batch_size][n/32 * 34]
/// * `output` - GPU pointer to f32 output data [batch_size][n]
/// * `n` - Elements per batch
/// * `batch_size` - Number of batches
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q8_0_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q8_0_batched: n and batch_size cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result =
        unsafe { dequantize_q8_0_batched_kernel(input, output, n as c_int, batch_size as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q8_0_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Verify Q8_0 quantization accuracy.
///
/// # Arguments
/// * `original` - GPU pointer to original f32 data [n]
/// * `quantized` - GPU pointer to Q8_0 quantized data [n/32 * 34]
/// * `errors` - GPU pointer to error metrics [4] (will be written)
///   - errors[0]: max error (output)
///   - errors[1]: MSE (output)
///   - errors[2]: sum of original values (output)
///   - errors[3]: sum of errors (output)
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn verify_q8_0_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q8_0_accuracy: n cannot be zero".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { verify_q8_0_accuracy_kernel(original, quantized, errors, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("verify_q8_0_accuracy kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Finalize Q8_0 accuracy metrics.
///
/// Must be called after verify_q8_0_accuracy to compute final values.
///
/// # Arguments
/// * `errors` - GPU pointer to intermediate error values [4]
/// * `metrics` - GPU pointer to final metrics [3] (will be written)
///   - metrics[0]: max error
///   - metrics[1]: MSE
///   - metrics[2]: relative error
/// * `n` - Total number of elements (for MSE normalization)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
pub fn finalize_q8_0_metrics(errors: *const f32, metrics: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q8_0_metrics: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { finalize_q8_0_metrics_kernel(errors, metrics, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("finalize_q8_0_metrics kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q4_0 Safe Wrapper Functions ────────────────────────────────────────────────────────

/// Quantize f32 data to Q4_0 format.
///
/// # Arguments
/// * `input` - GPU pointer to f32 input data [n]
/// * `output` - GPU pointer to Q4_0 output data [n/32 * 18]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn quantize_q4_0(input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q4_0: n cannot be zero".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q4_0: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q4_0: output pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { quantize_q4_0_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q4_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dequantize Q4_0 data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q4_0 input data [n/32 * 18]
/// * `output` - GPU pointer to f32 output data [n]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q4_0(input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_0: n cannot be zero".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_0: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_0: output pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { dequantize_q4_0_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q4_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched dequantize Q4_0 data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q4_0 input data [batch_size][n/32 * 18]
/// * `output` - GPU pointer to f32 output data [batch_size][n]
/// * `n` - Elements per batch
/// * `batch_size` - Number of batches
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q4_0_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_0_batched: n and batch_size cannot be zero".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_0_batched: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_0_batched: output pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result =
        unsafe { dequantize_q4_0_batched_kernel(input, output, n as c_int, batch_size as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q4_0_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Verify Q4_0 quantization accuracy.
///
/// # Arguments
/// * `original` - GPU pointer to original f32 data [n]
/// * `quantized` - GPU pointer to Q4_0 quantized data [n/32 * 18]
/// * `errors` - GPU pointer to error metrics [4] (will be written)
///   - errors[0]: max error (output)
///   - errors[1]: MSE (output)
///   - errors[2]: sum of original values (output)
///   - errors[3]: sum of errors (output)
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn verify_q4_0_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_0_accuracy: n cannot be zero".to_string(),
        });
    }

    if original.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_0_accuracy: original pointer is null".to_string(),
        });
    }

    if quantized.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_0_accuracy: quantized pointer is null".to_string(),
        });
    }

    if errors.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_0_accuracy: errors pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { verify_q4_0_accuracy_kernel(original, quantized, errors, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("verify_q4_0_accuracy kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Finalize Q4_0 accuracy metrics.
///
/// Must be called after verify_q4_0_accuracy to compute final values.
///
/// # Arguments
/// * `errors` - GPU pointer to intermediate error values [4]
/// * `metrics` - GPU pointer to final metrics [3] (will be written)
///   - metrics[0]: max error
///   - metrics[1]: MSE
///   - metrics[2]: relative error
/// * `n` - Total number of elements (for MSE normalization)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
pub fn finalize_q4_0_metrics(errors: *const f32, metrics: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_0_metrics: n cannot be zero".to_string(),
        });
    }

    if errors.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_0_metrics: errors pointer is null".to_string(),
        });
    }

    if metrics.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_0_metrics: metrics pointer is null".to_string(),
        });
    }

    let result = unsafe { finalize_q4_0_metrics_kernel(errors, metrics, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("finalize_q4_0_metrics kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Q4_0 × f32 GEMV: Compute output = weights @ input
///
/// Computes matrix-vector multiplication where:
/// - weights: [n_rows/32][ncols_dst][18] Q4_0 quantized weight matrix (column-major)
/// - input: [n_rows] f32 input vector
/// - output: [ncols_dst] f32 output vector
///
/// # Arguments
/// * `weights_q4_0` - GPU pointer to Q4_0 quantized weights [n_rows/32 * ncols_dst * 18]
/// * `input` - GPU pointer to f32 input vector [n_rows]
/// * `output` - GPU pointer to f32 output vector [ncols_dst] (will be written)
/// * `n_rows` - Number of rows (input dimension, must be multiple of 32)
/// * `ncols_dst` - Number of columns (output dimension)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - n_rows must be a multiple of QK4_0 (32)
/// - Bounds are validated on CPU before kernel launch
pub fn gemv_q4_0_f32(
    weights_q4_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    // n_rows must be aligned to QK4_0
    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    // Validate pointers
    if weights_q4_0.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_f32: weights_q4_0 pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_0_f32: output pointer is null".to_string(),
        });
    }

    // Call kernel launch function
    let result = unsafe {
        gemv_q4_0_f32_launch(
            weights_q4_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            std::ptr::null_mut(), // default stream
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q4_1 Safe Wrapper Functions ────────────────────────────────────────────────────────────

/// Quantize f32 data to Q4_1 format.
///
/// # Arguments
/// * `input` - GPU pointer to f32 input data [n]
/// * `output` - GPU pointer to Q4_1 output data [n/32 * 20]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn quantize_q4_1(input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q4_1: n cannot be zero".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q4_1: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "quantize_q4_1: output pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { quantize_q4_1_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q4_1 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dequantize Q4_1 data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q4_1 input data [n/32 * 20]
/// * `output` - GPU pointer to f32 output data [n]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q4_1(input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_1: n cannot be zero".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_1: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_1: output pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { dequantize_q4_1_kernel(input, output, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q4_1 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched dequantize Q4_1 data to f32.
///
/// # Arguments
/// * `input` - GPU pointer to Q4_1 input data [batch_size][n/32 * 20]
/// * `output` - GPU pointer to f32 output data [batch_size][n]
/// * `n` - Elements per batch
/// * `batch_size` - Number of batches
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q4_1_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_1_batched: n and batch_size cannot be zero".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_1_batched: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dequantize_q4_1_batched: output pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result =
        unsafe { dequantize_q4_1_batched_kernel(input, output, n as c_int, batch_size as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q4_1_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Verify Q4_1 quantization accuracy.
///
/// # Arguments
/// * `original` - GPU pointer to original f32 data [n]
/// * `quantized` - GPU pointer to Q4_1 quantized data [n/32 * 20]
/// * `errors` - GPU pointer to error metrics [4] (will be written)
///   - errors[0]: max error (output)
///   - errors[1]: MSE (output)
///   - errors[2]: sum of original values (output)
///   - errors[3]: sum of errors (output)
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn verify_q4_1_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_1_accuracy: n cannot be zero".to_string(),
        });
    }

    if original.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_1_accuracy: original pointer is null".to_string(),
        });
    }

    if quantized.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_1_accuracy: quantized pointer is null".to_string(),
        });
    }

    if errors.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "verify_q4_1_accuracy: errors pointer is null".to_string(),
        });
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe { verify_q4_1_accuracy_kernel(original, quantized, errors, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("verify_q4_1_accuracy kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Finalize Q4_1 accuracy metrics.
///
/// Must be called after verify_q4_1_accuracy to compute final values.
///
/// # Arguments
/// * `errors` - GPU pointer to intermediate error values [4]
/// * `metrics` - GPU pointer to final metrics [3] (will be written)
///   - metrics[0]: max error
///   - metrics[1]: MSE
///   - metrics[2]: relative error
/// * `n` - Total number of elements (for MSE normalization)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
pub fn finalize_q4_1_metrics(errors: *const f32, metrics: *mut f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_1_metrics: n cannot be zero".to_string(),
        });
    }

    if errors.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_1_metrics: errors pointer is null".to_string(),
        });
    }

    if metrics.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_1_metrics: metrics pointer is null".to_string(),
        });
    }

    let result = unsafe { finalize_q4_1_metrics_kernel(errors, metrics, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("finalize_q4_1_metrics kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Q4_1 x f32 GEMV: Compute output = weights @ input
///
/// Computes matrix-vector multiplication where:
/// - weights: [n_rows/32][ncols_dst][20] Q4_1 quantized weight matrix (column-major)
/// - input: [n_rows] f32 input vector
/// - output: [ncols_dst] f32 output vector
///
/// # Arguments
/// * `weights_q4_1` - GPU pointer to Q4_1 quantized weights [n_rows/32 * ncols_dst * 20]
/// * `input` - GPU pointer to f32 input vector [n_rows]
/// * `output` - GPU pointer to f32 output vector [ncols_dst] (will be written)
/// * `n_rows` - Number of rows (input dimension, must be multiple of 32)
/// * `ncols_dst` - Number of columns (output dimension)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - n_rows must be a multiple of QK4_1 (32)
/// - Bounds are validated on CPU before kernel launch
pub fn gemv_q4_1_f32(
    weights_q4_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_1_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    // n_rows must be aligned to QK4_1
    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_1_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    // Validate pointers
    if weights_q4_1.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_1_f32: weights_q4_1 pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_1_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_1_f32: output pointer is null".to_string(),
        });
    }

    // Call kernel launch function
    let result = unsafe {
        gemv_q4_1_f32_launch(
            weights_q4_1,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            std::ptr::null_mut(), // default stream
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q4_1_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q8_0 FFI Declarations ─────────────────────────────────────────────────────────────

/// FFI declarations for Q8_0 kernels - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn quantize_q8_0_kernel(input: *const f32, output: *mut u8, n: c_int) -> hipError_t;

    fn dequantize_q8_0_kernel(input: *const u8, output: *mut f32, n: c_int) -> hipError_t;

    fn dequantize_q8_0_batched_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        batch_size: c_int,
    ) -> hipError_t;

    fn verify_q8_0_accuracy_kernel(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
    ) -> hipError_t;

    fn finalize_q8_0_metrics_kernel(errors: *const f32, metrics: *mut f32, n: c_int) -> hipError_t;

    // Q4_0 kernels - will be linked from compiled HIP kernels
    fn quantize_q4_0_kernel(input: *const f32, output: *mut u8, n: c_int) -> hipError_t;

    fn dequantize_q4_0_kernel(input: *const u8, output: *mut f32, n: c_int) -> hipError_t;

    fn dequantize_q4_0_batched_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        batch_size: c_int,
    ) -> hipError_t;

    fn verify_q4_0_accuracy_kernel(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
    ) -> hipError_t;

    fn finalize_q4_0_metrics_kernel(errors: *const f32, metrics: *mut f32, n: c_int) -> hipError_t;

    fn gemv_q4_0_f32_launch(
        weights_q4_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Q4_1 kernels
    fn quantize_q4_1_kernel(input: *const f32, output: *mut u8, n: c_int) -> hipError_t;

    fn dequantize_q4_1_kernel(input: *const u8, output: *mut f32, n: c_int) -> hipError_t;

    fn dequantize_q4_1_batched_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        batch_size: c_int,
    ) -> hipError_t;

    fn verify_q4_1_accuracy_kernel(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
    ) -> hipError_t;

    fn finalize_q4_1_metrics_kernel(errors: *const f32, metrics: *mut f32, n: c_int) -> hipError_t;

    fn gemv_q4_1_f32_launch(
        weights_q4_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

// ── Q8_0 GEMV (Matrix-Vector Multiplication) ─────────────────────────────────────────────

/// Q8_0 × f32 GEMV: Compute output = weights @ input
///
/// Computes matrix-vector multiplication where:
/// - weights: [n_rows/32][ncols_dst][34] Q8_0 quantized weight matrix (column-major)
/// - input: [n_rows] f32 input vector
/// - output: [ncols_dst] f32 output vector
///
/// # Arguments
/// * `weights_q8_0` - GPU pointer to Q8_0 quantized weights [n_rows/32 * ncols_dst * 34]
/// * `input` - GPU pointer to f32 input vector [n_rows]
/// * `output` - GPU pointer to f32 output vector [ncols_dst] (will be written)
/// * `n_rows` - Number of rows (input dimension, must be multiple of 32)
/// * `ncols_dst` - Number of columns (output dimension)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - n_rows must be a multiple of QK8_0 (32)
/// - Bounds are validated on CPU before kernel launch
pub fn gemv_q8_0_f32(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q8_0_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    // n_rows must be aligned to QK8_0
    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q8_0_f32: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    // Validate pointers
    if weights_q8_0.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q8_0_f32: weights_q8_0 pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q8_0_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q8_0_f32: output pointer is null".to_string(),
        });
    }

    // Call kernel launch function
    let result = unsafe {
        gemv_q8_0_f32_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            std::ptr::null_mut(), // default stream
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q8_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── Q8_0 GEMV FFI Declaration ───────────────────────────────────────────────────────────

/// FFI declaration for Q8_0 GEMV kernel launch
/// Uses hipStream_t for stream parameter (opaque pointer)
type hipStream_t = *mut std::os::raw::c_void;

unsafe extern "C" {
    fn gemv_q8_0_f32_launch(
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

// ── Q4_K GEMV FFI Declaration ───────────────────────────────────────────────────────────

/// FFI declaration for Q4_K GEMV kernel launch
/// Uses hipStream_t for stream parameter (opaque pointer)
unsafe extern "C" {
    fn gemv_q4_k_f32_launch(
        weights_q4_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

// ── Q5_K GEMV FFI Declaration ───────────────────────────────────────────────────────────

/// FFI declaration for Q5_K GEMV kernel launch
/// Uses hipStream_t for stream parameter (opaque pointer)
unsafe extern "C" {
    fn gemv_q5_k_f32_launch(
        weights_q5_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
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

    // Fused kernels
    fn gemv_qkv_q4_0_f32_launch(
        w_q: *const u8,
        w_k: *const u8,
        w_v: *const u8,
        input: *const f32,
        out_q: *mut f32,
        out_k: *mut f32,
        out_v: *mut f32,
        n_rows: c_int,
        n_q: c_int,
        n_kv: c_int,
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
}

/// Fused QKV GEMV for Q4_0 quantized weights.
pub fn gemv_qkv_q4_0_f32(
    w_q: *const u8,
    w_k: *const u8,
    w_v: *const u8,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    n_rows: usize,
    n_q: usize,
    n_kv: usize,
) -> GpuResult<()> {
    let result = unsafe {
        gemv_qkv_q4_0_f32_launch(
            w_q,
            w_k,
            w_v,
            input,
            out_q,
            out_k,
            out_v,
            n_rows as c_int,
            n_q as c_int,
            n_kv as c_int,
            std::ptr::null_mut(),
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

/// Fused Gate/Up + SwiGLU for Q4_0 quantized weights.
pub fn gemv_gate_up_swiglu_q4_0_f32(
    w_gate: *const u8,
    w_up: *const u8,
    input: *const f32,
    out_swiglu: *mut f32,
    n_rows: usize,
    n_ff: usize,
) -> GpuResult<()> {
    let result = unsafe {
        gemv_gate_up_swiglu_q4_0_f32_launch(
            w_gate,
            w_up,
            input,
            out_swiglu,
            n_rows as c_int,
            n_ff as c_int,
            std::ptr::null_mut(),
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

/// Batched GEMM for Q4_0 quantized weights.
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
    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("gemm_q4_0_f32: n_rows must be multiple of 32, got {}", n_rows),
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
            std::ptr::null_mut(),
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

/// Batched GEMM for Q4_1 quantized weights.
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
    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("gemm_q4_1_f32: n_rows must be multiple of 32, got {}", n_rows),
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
            std::ptr::null_mut(),
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

/// Batched GEMM for Q8_0 quantized weights.
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
    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("gemm_q8_0_f32: n_rows must be multiple of 32, got {}", n_rows),
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
            std::ptr::null_mut(),
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

/// Batched GEMM for Q4_K quantized weights.
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
    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("gemm_q4_k_f32: n_rows must be multiple of 256, got {}", n_rows),
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
            std::ptr::null_mut(),
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

/// Batched GEMM for Q5_K quantized weights.
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
    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("gemm_q5_k_f32: n_rows must be multiple of 256, got {}", n_rows),
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
            std::ptr::null_mut(),
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

/// Q4_K × f32 GEMV: quantized matrix-vector multiplication
///
/// Computes output = weights_q4_k × input where:
/// - weights_q4_k is Q4_K quantized weight matrix [n_rows × ncols_dst]
/// - input is f32 activation vector [n_rows]
/// - output is f32 result vector [ncols_dst]
///
/// # Arguments
/// * `weights_q4_k` - GPU pointer to Q4_K quantized weights
/// * `input` - GPU pointer to f32 input vector
/// * `output` - GPU pointer to f32 output vector
/// * `n_rows` - Input dimension (must be multiple of 256)
/// * `ncols_dst` - Output dimension (1-8 optimized, any for generic)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - n_rows must be aligned to QK_K (256)
/// - Bounds are validated on CPU before kernel launch
pub fn gemv_q4_k_f32(
    weights_q4_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    // n_rows must be aligned to QK_K
    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_k_f32: n_rows must be multiple of 256, got {}",
                n_rows
            ),
        });
    }

    // Validate pointers
    if weights_q4_k.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_k_f32: weights_q4_k pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_k_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_k_f32: output pointer is null".to_string(),
        });
    }

    // Call kernel launch function
    let result = unsafe {
        gemv_q4_k_f32_launch(
            weights_q4_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            std::ptr::null_mut(), // default stream
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

// ── Q5_K GEMV (Matrix-Vector Multiplication) ─────────────────────────────────────────────

/// Q5_K × f32 GEMV: Compute output = weights @ input
///
/// Computes matrix-vector multiplication with Q5_K quantized weights:
/// ```
/// output[col] = sum over rows of (dequantize_q5_k(weight[row, col]) * input[row])
/// ```
///
/// # Arguments
/// * `weights_q5_k` - GPU pointer to Q5_K quantized weights [n_rows/256 * ncols_dst * 176]
/// * `input` - GPU pointer to f32 input vector [n_rows]
/// * `output` - GPU pointer to f32 output vector [ncols_dst] (will be written)
/// * `n_rows` - Number of rows (input dimension, must be multiple of 256)
/// * `ncols_dst` - Number of columns (output dimension)
///
/// # Returns
/// Ok(()) on success
///
/// # Errors
/// - n_rows or ncols_dst is zero
/// - n_rows is not a multiple of 256
/// - Any pointer is null
/// - Kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - n_rows must be aligned to QK_K (256)
/// - Bounds are validated on CPU before kernel launch
pub fn gemv_q5_k_f32(
    weights_q5_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    // n_rows must be aligned to QK_K
    if n_rows % 256 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q5_k_f32: n_rows must be multiple of 256, got {}",
                n_rows
            ),
        });
    }

    // Validate pointers
    if weights_q5_k.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_k_f32: weights_q5_k pointer is null".to_string(),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_k_f32: input pointer is null".to_string(),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q5_k_f32: output pointer is null".to_string(),
        });
    }

    // Call kernel launch function
    let result = unsafe {
        gemv_q5_k_f32_launch(
            weights_q5_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            std::ptr::null_mut(), // default stream
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

// ── Q8_0 Unit Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod q8_0_tests {
    use super::*;

    #[test]
    fn quantize_q8_0_rejects_zero_n() {
        let result = quantize_q8_0(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q8_0_rejects_zero_n() {
        let result = dequantize_q8_0(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q8_0_batched_rejects_zero_batch() {
        let result = dequantize_q8_0_batched(std::ptr::null(), std::ptr::null_mut(), 32, 0);
        assert!(result.is_err());
    }

    #[test]
    fn verify_q8_0_accuracy_rejects_zero_n() {
        let result =
            verify_q8_0_accuracy(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn finalize_q8_0_metrics_rejects_zero_n() {
        let result = finalize_q8_0_metrics(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }
}

// ── Q4_K Unit Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod q4_k_tests {
    use super::*;

    #[test]
    fn gemv_q4_k_f32_rejects_zero_n_rows() {
        let result = gemv_q4_k_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            4,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn gemv_q4_k_f32_rejects_zero_ncols() {
        let result = gemv_q4_k_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            256,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn gemv_q4_k_f32_rejects_misaligned_n_rows() {
        let result = gemv_q4_k_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            255, // Not multiple of 256
            4,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("multiple of 256"));
    }

    #[test]
    fn gemv_q4_k_f32_rejects_null_weights() {
        let result = gemv_q4_k_f32(
            std::ptr::null(), // null weights
            std::ptr::null(),
            std::ptr::null_mut(),
            256,
            4,
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("weights_q4_k pointer is null"));
    }

    #[test]
    fn gemv_q4_k_f32_rejects_null_input() {
        let dummy_u8 = 0u8;
        let result = gemv_q4_k_f32(
            &dummy_u8 as *const u8,
            std::ptr::null(), // null input
            std::ptr::null_mut(),
            256,
            4,
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("input pointer is null"));
    }

    #[test]
    fn gemv_q4_k_f32_rejects_null_output() {
        let dummy_u8 = 0u8;
        let dummy_f32 = 0.0f32;
        let result = gemv_q4_k_f32(
            &dummy_u8 as *const u8,
            &dummy_f32 as *const f32,
            std::ptr::null_mut(), // null output
            256,
            4,
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("output pointer is null"));
    }
}

// ── Q4_0 Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod q4_0_tests {
    use super::*;

    #[test]
    fn quantize_q4_0_rejects_zero_n() {
        let result = quantize_q4_0(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn dequantize_q4_0_rejects_zero_n() {
        let result = dequantize_q4_0(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn dequantize_q4_0_batched_rejects_zero_batch() {
        let result = dequantize_q4_0_batched(std::ptr::null(), std::ptr::null_mut(), 32, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn verify_q4_0_accuracy_rejects_zero_n() {
        let result =
            verify_q4_0_accuracy(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn finalize_q4_0_metrics_rejects_zero_n() {
        let result = finalize_q4_0_metrics(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn gemv_q4_0_f32_rejects_invalid_dimensions() {
        // n_rows must be multiple of QK4_0 (32)
        let result = gemv_q4_0_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            33,
            1,
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be multiple of"));
    }
}

// ── Q4_1 Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod q4_1_tests {
    use super::*;

    #[test]
    fn quantize_q4_1_rejects_zero_n() {
        let result = quantize_q4_1(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn dequantize_q4_1_rejects_zero_n() {
        let result = dequantize_q4_1(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn dequantize_q4_1_batched_rejects_zero_batch() {
        let result = dequantize_q4_1_batched(std::ptr::null(), std::ptr::null_mut(), 32, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn verify_q4_1_accuracy_rejects_zero_n() {
        let result =
            verify_q4_1_accuracy(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn finalize_q4_1_metrics_rejects_zero_n() {
        let result = finalize_q4_1_metrics(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be zero"));
    }

    #[test]
    fn gemv_q4_1_f32_rejects_invalid_dimensions() {
        let result = gemv_q4_1_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            33,
            1,
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be multiple of"));
    }
}
