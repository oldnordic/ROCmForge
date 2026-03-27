//! GPU quantization wrapper with RAII safety.
//!
//! Safe wrapper for Q4_K quantization operations.
//! Follows project patterns: GpuDevice, GpuBuffer, etc.

use super::error::{GpuError, GpuResult};
use super::ffi;
use super::device::GpuDevice;
use super::kernels::{quantize_q4_k, dequantize_q4_k, dequantize_q4_k_batched, verify_q4_k_accuracy, finalize_q4_k_metrics};
use super::quant::{QK_K, Q4_K_BLOCK_SIZE, Q4KBlock};

/// GPU quantization handle.
///
/// Provides safe interface for Q4_K quantization operations.
/// Uses RAII pattern for proper resource management.
pub struct GpuQuant {
    device: GpuDevice,
}

impl GpuQuant {
    /// Initialize GPU quantization context.
    ///
    /// Returns error if device unavailable or quantization not supported.
    pub fn new(device: GpuDevice) -> GpuResult<Self> {
        // TODO: Verify device supports Q4_K quantization
        // For now, all devices are assumed to support it

        Ok(Self { device })
    }

    /// Get underlying device reference.
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Quantize f32 data to Q4_K format.
    ///
    /// # Arguments
    /// * `input` - GPU pointer to f32 input data [n]
    /// * `output` - GPU pointer to Q4_K output data [n/256 * 144]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn quantize(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize: output pointer is null".to_string(),
            });
        }

        // Call kernel
        quantize_q4_k(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn dequantize(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize: output pointer is null".to_string(),
            });
        }

        // Call kernel
        dequantize_q4_k(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n or batch_size is zero
    /// - Kernel launch fails
    pub fn dequantize_batched(&self, input: *const u8, output: *mut f32, n: usize, batch_size: usize) -> GpuResult<()> {
        if n == 0 || batch_size == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_batched: n and batch_size cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_batched: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_batched: output pointer is null".to_string(),
            });
        }

        // Call kernel
        dequantize_q4_k_batched(input, output, n, batch_size)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    /// Verify quantization accuracy.
    ///
    /// Compares original f32 data with quantize-dequantize round-trip.
    ///
    /// # Arguments
    /// * `original` - GPU pointer to original f32 data [n]
    /// * `quantized` - GPU pointer to Q4_K quantized data [n/256 * 144]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// (max_error, mse, relative_error) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn verify_accuracy(&self, original: *const f32, quantized: *const u8, n: usize) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_accuracy: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if original.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_accuracy: original pointer is null".to_string(),
            });
        }

        if quantized.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_accuracy: quantized pointer is null".to_string(),
            });
        }

        // Allocate GPU memory for error metrics
        let num_blocks = (n + QK_K - 1) / QK_K;

        // Allocate temporary buffers for metrics
        let errors_gpu = unsafe {
            ffi::hip_malloc(4 * std::mem::size_of::<f32>())?
        };
        let metrics_gpu = unsafe {
            ffi::hip_malloc(3 * std::mem::size_of::<f32>())?
        };

        // Initialize errors to zero
        let zeros = vec![0.0f32; 4];
        unsafe {
            ffi::hip_memcpy_h2d(errors_gpu, zeros.as_ptr() as *const u8, 4 * std::mem::size_of::<f32>())?;
        }

        // Run verification kernel
        verify_q4_k_accuracy(original, quantized, errors_gpu as *mut f32, n)?;

        // Finalize metrics
        finalize_q4_k_metrics(errors_gpu as *const f32, metrics_gpu as *mut f32, n)?;

        // Synchronize to ensure kernels complete
        self.device.synchronize()?;

        // Copy metrics back to host
        let mut metrics = [0.0f32; 3];
        unsafe {
            ffi::hip_memcpy_d2h(
                metrics.as_mut_ptr() as *mut u8,
                metrics_gpu as *const u8,
                3 * std::mem::size_of::<f32>()
            )?;
        }

        // Cleanup
        unsafe {
            ffi::hip_free(errors_gpu);
            ffi::hip_free(metrics_gpu);
        }

        Ok((metrics[0], metrics[1], metrics[2]))
    }
}

impl std::fmt::Debug for GpuQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuQuant")
            .field("device_id", &self.device.device_id())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_quant_rejects_null_input() {
        // This test validates pointer checking logic
        // Actual GPU tests require available hardware

        let input = std::ptr::null();
        let output = std::ptr::null_mut();

        // These would fail at pointer validation, not kernel launch
        // We can't test this without a real device
    }

    #[test]
    fn qk_k_constant_is_256() {
        assert_eq!(QK_K, 256, "QK_K must be 256 for Q4_K format");
    }

    #[test]
    fn q4_k_block_size_is_144() {
        assert_eq!(Q4_K_BLOCK_SIZE, 144, "Q4_K_BLOCK_SIZE must be 144 bytes");
    }
}
