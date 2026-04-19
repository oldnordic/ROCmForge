//! Q4_1 quantization kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

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
    gemv_q4_1_f32_on_stream(
        weights_q4_1,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q4_1_f32_on_stream(
    weights_q4_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
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

    unsafe {
        gemv_q4_1_f32_on_stream_unchecked(weights_q4_1, input, output, n_rows, ncols_dst, stream)
    }
}

/// Hot-path variant used by trusted dispatch code that has already validated
/// tensor layout and pointers.
#[inline(always)]
pub unsafe fn gemv_q4_1_f32_on_stream_unchecked(
    _weights_q4_1: *const u8,
    _input: *const f32,
    _output: *mut f32,
    _n_rows: usize,
    _ncols_dst: usize,
    _stream: hipStream_t,
) -> GpuResult<()> {
    // DISABLED: gemv_q4_1_f32_launch kernel not available
    Err(GpuError::UnsupportedOperation {
        operation: "gemv_q4_1_f32_on_stream_unchecked".to_string(),
        reason: "Q4_1 kernel not implemented".to_string(),
    })
}

pub fn gemv_q4_1_f32_residual_on_stream(
    weights_q4_1: *const u8,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_1_f32_residual: n_rows and ncols_dst cannot be zero".to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_q4_1_f32_residual: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q4_1.is_null() || input.is_null() || residual.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemv_q4_1_f32_residual: kernel pointers must be non-null".to_string(),
        });
    }

    unsafe {
        gemv_q4_1_f32_residual_on_stream_unchecked(
            weights_q4_1,
            input,
            residual,
            output,
            n_rows,
            ncols_dst,
            stream,
        )
    }
}

/// Hot-path variant used by trusted dispatch code that has already validated
/// tensor layout and pointers.
#[inline(always)]
pub unsafe fn gemv_q4_1_f32_residual_on_stream_unchecked(
    _weights_q4_1: *const u8,
    _input: *const f32,
    _residual: *const f32,
    _output: *mut f32,
    _n_rows: usize,
    _ncols_dst: usize,
    _stream: hipStream_t,
) -> GpuResult<()> {
    // DISABLED: gemv_q4_1_f32_residual_launch kernel not available
    Err(GpuError::UnsupportedOperation {
        operation: "gemv_q4_1_f32_residual_on_stream_unchecked".to_string(),
        reason: "Q4_1 residual kernel not implemented".to_string(),
    })
}

/// Variant launch for autotuning Q4_1 residual kernel.
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - n_rows must be multiple of QK4_1 (32)
/// - variant: 0 = baseline (256 threads), 1 = 128 threads
#[inline(always)]
pub unsafe fn gemv_q4_1_f32_residual_on_stream_variant_unchecked(
    _weights_q4_1: *const u8,
    _input: *const f32,
    _residual: *const f32,
    _output: *mut f32,
    _n_rows: usize,
    _ncols_dst: usize,
    _variant: i32,
    _stream: hipStream_t,
) -> GpuResult<()> {
    // DISABLED: gemv_q4_1_f32_residual_variant_launch not available
    // TODO: Implement Q4_1 residual variant kernel or use CPU fallback
    Err(GpuError::UnsupportedOperation {
        operation: "gemv_q4_1_f32_residual_on_stream_variant_unchecked".to_string(),
        reason: "Q4_1 residual variant kernel not implemented".to_string(),
    })
    /*
    let result = unsafe {
        gemv_q4_1_f32_residual_variant_launch(
            weights_q4_1,
            input,
            residual,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            variant,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemv_q4_1_f32_residual_variant kernel failed: {:?}", result),
        });
    }

    Ok(())
    */
}

/// Experimental FFN-down microkernel for Q4_1 weights.
///
/// This kernel is intentionally not used by normal dispatch. It exists to
/// compare the current `materialize swiglu -> gemv_q4_1` path against a fused
/// `silu(gate) * up -> down` path on real model weights.
pub fn gemv_ffn_down_swiglu_q4_1_f32_experimental(
    weights_q4_1: *const u8,
    gate: *const f32,
    up: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_ffn_down_swiglu_q4_1_f32_experimental_on_stream(
        weights_q4_1,
        gate,
        up,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_ffn_down_swiglu_q4_1_f32_experimental_on_stream(
    weights_q4_1: *const u8,
    gate: *const f32,
    up: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_ffn_down_swiglu_q4_1_f32_experimental: n_rows and ncols_dst cannot be zero"
                    .to_string(),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gemv_ffn_down_swiglu_q4_1_f32_experimental: n_rows must be multiple of 32, got {}",
                n_rows
            ),
        });
    }

    if weights_q4_1.is_null() || gate.is_null() || up.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gemv_ffn_down_swiglu_q4_1_f32_experimental: input pointers cannot be null"
                    .to_string(),
        });
    }

    let result = unsafe {
        gemv_ffn_down_swiglu_q4_1_f32_experimental_launch(
            weights_q4_1,
            gate,
            up,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gemv_ffn_down_swiglu_q4_1_f32_experimental kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

/// Q4_1 GEMM: matrix-matrix multiplication
///
/// # Arguments
/// * `weights_q4_1` - GPU pointer to Q4_1 quantized weights
/// * `input` - GPU pointer to f32 input matrix
/// * `output` - GPU pointer to f32 output matrix
/// * `n_rows` - Number of rows in weights
/// * `ncols_dst` - Number of columns in output
/// * `batch_size` - Batch size for GEMM
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
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

// ── Q4_1 FFI Declarations ───────────────────────────────────────────────────────────

unsafe extern "C" {
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

    // DISABLED: gemv_q4_1_f32_launch not available
    // TODO: Re-enable when Q4_1 kernel is implemented
    /*
    fn gemv_q4_1_f32_launch(
        weights_q4_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
    */

    // DISABLED: gemv_q4_1_f32_residual_launch not available
    // TODO: Re-enable when Q4_1 residual kernel is implemented
    /*
    fn gemv_q4_1_f32_residual_launch(
        weights_q4_1: *const u8,
        input: *const f32,
        residual: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
    */

    // DISABLED: gemv_q4_1_f32_residual_variant_launch not available
    // TODO: Re-enable when Q4_1 residual variant kernel is implemented
    /*
    fn gemv_q4_1_f32_residual_variant_launch(
        weights_q4_1: *const u8,
        input: *const f32,
        residual: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        variant: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
    */

    fn gemv_ffn_down_swiglu_q4_1_f32_experimental_launch(
        weights_q4_1: *const u8,
        gate: *const f32,
        up: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_q4_1_rejects_zero_n() {
        let result = quantize_q4_1(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q4_1_rejects_zero_n() {
        let result = dequantize_q4_1(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q4_1_batched_rejects_zero_batch() {
        let result = dequantize_q4_1_batched(std::ptr::null(), std::ptr::null_mut(), 100, 0);
        assert!(result.is_err());
    }

    #[test]
    fn verify_q4_1_accuracy_rejects_zero_n() {
        let result =
            verify_q4_1_accuracy(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn finalize_q4_1_metrics_rejects_zero_n() {
        let result = finalize_q4_1_metrics(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn gemv_q4_1_f32_rejects_invalid_dimensions() {
        let result = gemv_q4_1_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            100,
        );
        assert!(result.is_err());

        let result = gemv_q4_1_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            100,
            0,
        );
        assert!(result.is_err());

        // n_rows must be multiple of 32
        let result = gemv_q4_1_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            100,
            100,
        );
        assert!(result.is_err());
    }

    #[test]
    fn gemv_q4_1_f32_residual_rejects_invalid_dimensions() {
        let result = gemv_q4_1_f32_residual_on_stream(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            100,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }
}
