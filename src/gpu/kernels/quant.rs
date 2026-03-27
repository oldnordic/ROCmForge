//! Q4_K quantization kernel wrappers.
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
pub fn quantize_q4_k(
    input: *const f32,
    output: *mut u8,
    n: usize,
) -> GpuResult<()> {
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
    let result = unsafe {
        quantize_q4_k_kernel(input, output, n as c_int)
    };

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
pub fn dequantize_q4_k(
    input: *const u8,
    output: *mut f32,
    n: usize,
) -> GpuResult<()> {
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

    let result = unsafe {
        dequantize_q4_k_kernel(input, output, n as c_int)
    };

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

    let result = unsafe {
        dequantize_q4_k_batched_kernel(input, output, n as c_int, batch_size as c_int)
    };

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

    let result = unsafe {
        verify_q4_k_accuracy_kernel(original, quantized, errors, n as c_int)
    };

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
pub fn finalize_q4_k_metrics(
    errors: *const f32,
    metrics: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "finalize_q4_k_metrics: n cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        finalize_q4_k_metrics_kernel(errors, metrics, n as c_int)
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("finalize_q4_k_metrics kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── FFI Declarations ─────────────────────────────────────────────────────────────

/// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn quantize_q4_k_kernel(
        input: *const f32,
        output: *mut u8,
        n: c_int,
    ) -> hipError_t;

    fn dequantize_q4_k_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
    ) -> hipError_t;

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

    fn finalize_q4_k_metrics_kernel(
        errors: *const f32,
        metrics: *mut f32,
        n: c_int,
    ) -> hipError_t;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_q4_k_rejects_zero_n() {
        let result = quantize_q4_k(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q4_k_rejects_zero_n() {
        let result = dequantize_q4_k(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q4_k_batched_rejects_zero_batch() {
        let result = dequantize_q4_k_batched(
            std::ptr::null(),
            std::ptr::null_mut(),
            256,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn verify_q4_k_accuracy_rejects_zero_n() {
        let result = verify_q4_k_accuracy(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn finalize_q4_k_metrics_rejects_zero_n() {
        let result = finalize_q4_k_metrics(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }
}
