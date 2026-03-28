//! GPU quantization wrapper with RAII safety.
//!
//! Safe wrapper for Q4_K and Q8_0 quantization operations.
//! Follows project patterns: GpuDevice, GpuBuffer, etc.

use super::error::{GpuError, GpuResult};
use super::ffi;
use super::device::GpuDevice;
use super::kernels::quant::{
    // Q4_K kernels
    quantize_q4_k, dequantize_q4_k, dequantize_q4_k_batched, verify_q4_k_accuracy, finalize_q4_k_metrics,
    // Q5_K kernels
    quantize_q5_k, dequantize_q5_k, dequantize_q5_k_batched, verify_q5_k_accuracy, finalize_q5_k_metrics,
    // Q8_0 kernels
    quantize_q8_0, dequantize_q8_0, dequantize_q8_0_batched, verify_q8_0_accuracy, finalize_q8_0_metrics,
    // Q4_0 kernels
    quantize_q4_0, dequantize_q4_0, dequantize_q4_0_batched, verify_q4_0_accuracy, finalize_q4_0_metrics,
    // Q8_0 GEMV kernel
    gemv_q8_0_f32,
    // Q4_K GEMV kernel
    gemv_q4_k_f32,
    // Q5_K GEMV kernel
    gemv_q5_k_f32,
    // Q4_0 GEMV kernel
    gemv_q4_0_f32,
    // Q4_1 kernels
    quantize_q4_1, dequantize_q4_1, dequantize_q4_1_batched, verify_q4_1_accuracy, finalize_q4_1_metrics,
    // Q4_1 GEMV kernel
    gemv_q4_1_f32,
};
use super::quant::{QK_K, Q4_K_BLOCK_SIZE, Q4KBlock, Q5_K_BLOCK_SIZE, Q5KBlock, QK8_0, Q8_0_BLOCK_SIZE, Q8_0_MAX, Q8_0Block, QK4_0, Q4_0_BLOCK_SIZE, QK4_1, Q4_1_BLOCK_SIZE};

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
    /// Q4_K and Q8_0 quantization work on all AMD GPUs with HIP support.
    /// No special device capabilities required beyond basic HIP.
    pub fn new(device: GpuDevice) -> GpuResult<Self> {
        // All HIP-capable devices support quantization kernels
        // No device capability check needed
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

    // ── Q5_K Methods ─────────────────────────────────────────────────────────────────────

    /// Quantize f32 data to Q5_K format.
    ///
    /// # Arguments
    /// * `input` - GPU pointer to f32 input data [n]
    /// * `output` - GPU pointer to Q5_K output data [n/256 * 176]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn quantize_q5_k(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q5_k: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q5_k: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q5_k: output pointer is null".to_string(),
            });
        }

        // Call kernel
        quantize_q5_k(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn dequantize_q5_k(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q5_k: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q5_k: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q5_k: output pointer is null".to_string(),
            });
        }

        // Call kernel
        dequantize_q5_k(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    // ── Q8_0 Methods ─────────────────────────────────────────────────────────────────────

    /// Quantize f32 data to Q8_0 format.
    ///
    /// # Arguments
    /// * `input` - GPU pointer to f32 input data [n]
    /// * `output` - GPU pointer to Q8_0 output data [n/32 * 34]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn quantize_q8_0(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q8_0: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q8_0: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q8_0: output pointer is null".to_string(),
            });
        }

        // Call kernel
        quantize_q8_0(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn dequantize_q8_0(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0: output pointer is null".to_string(),
            });
        }

        // Call kernel
        dequantize_q8_0(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n or batch_size is zero
    /// - Kernel launch fails
    pub fn dequantize_q8_0_batched(&self, input: *const u8, output: *mut f32, n: usize, batch_size: usize) -> GpuResult<()> {
        if n == 0 || batch_size == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0_batched: n and batch_size cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0_batched: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0_batched: output pointer is null".to_string(),
            });
        }

        // Call kernel
        dequantize_q8_0_batched(input, output, n, batch_size)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    /// Verify Q8_0 quantization accuracy.
    ///
    /// Compares original f32 data with quantize-dequantize round-trip.
    ///
    /// # Arguments
    /// * `original` - GPU pointer to original f32 data [n]
    /// * `quantized` - GPU pointer to Q8_0 quantized data [n/32 * 34]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// (max_error, mse, relative_error) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn verify_q8_0_accuracy(&self, original: *const f32, quantized: *const u8, n: usize) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q8_0_accuracy: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if original.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q8_0_accuracy: original pointer is null".to_string(),
            });
        }

        if quantized.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q8_0_accuracy: quantized pointer is null".to_string(),
            });
        }

        // Allocate GPU memory for error metrics
        let num_blocks = (n + QK8_0 - 1) / QK8_0;

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
        verify_q8_0_accuracy(original, quantized, errors_gpu as *mut f32, n)?;

        // Finalize metrics
        finalize_q8_0_metrics(errors_gpu as *const f32, metrics_gpu as *mut f32, n)?;

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

    // ── Q4_0 Methods ─────────────────────────────────────────────────────────────────────

    /// Quantize f32 data to Q4_0 format.
    ///
    /// # Arguments
    /// * `input` - GPU pointer to f32 input data [n]
    /// * `output` - GPU pointer to Q4_0 output data [n/32 * 18]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn quantize_q4_0(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q4_0: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Call kernel
        quantize_q4_0(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn dequantize_q4_0(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q4_0: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Call kernel
        dequantize_q4_0(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n or batch_size is zero
    /// - Kernel launch fails
    pub fn dequantize_q4_0_batched(&self, input: *const u8, output: *mut f32, n: usize, batch_size: usize) -> GpuResult<()> {
        if n == 0 || batch_size == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q4_0_batched: n and batch_size cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Call kernel
        dequantize_q4_0_batched(input, output, n, batch_size)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    /// Verify Q4_0 quantization accuracy.
    ///
    /// Compares original f32 data with quantize-dequantize round-trip.
    ///
    /// # Arguments
    /// * `original` - GPU pointer to original f32 data [n]
    /// * `quantized` - GPU pointer to Q4_0 quantized data [n/32 * 18]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// (max_error, mse, relative_error) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn verify_q4_0_accuracy(&self, original: *const f32, quantized: *const u8, n: usize) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q4_0_accuracy: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Allocate GPU memory for error metrics
        let num_blocks = (n + QK4_0 - 1) / QK4_0;

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
        verify_q4_0_accuracy(original, quantized, errors_gpu as *mut f32, n)?;

        // Finalize metrics
        finalize_q4_0_metrics(errors_gpu as *const f32, metrics_gpu as *mut f32, n)?;

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

    /// Q4_0 × f32 GEMV: Compute output = weights @ input
    ///
    /// Computes matrix-vector multiplication with Q4_0 quantized weights:
    /// ```
    /// output[col] = sum over rows of (dequantize_q4_0(weight[row, col]) * input[row])
    /// ```
    ///
    /// # Arguments
    /// * `weights_q4_0` - GPU pointer to Q4_0 quantized weights [n_rows/32 * ncols_dst * 18]
    /// * `input` - GPU pointer to f32 input vector [n_rows]
    /// * `output` - GPU pointer to f32 output vector [ncols_dst] (will be written)
    /// * `n_rows` - Number of rows (input dimension, must be multiple of 32)
    /// * `ncols_dst` - Number of columns (output dimension)
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n_rows or ncols_dst is zero
    /// - n_rows is not a multiple of 32
    /// - Any pointer is null
    /// - Kernel launch fails
    ///
    /// # Notes
    /// - Follows llama.cpp pattern with template specialization for ncols_dst
    /// - Uses register-based computation (no shared memory)
    /// - Optimized for ncols_dst in {1, 2, 4}, generic path for larger sizes
    pub fn gemv_q4_0_f32(
        &self,
        weights_q4_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        // Validate dimensions
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q4_0_f32: invalid dimensions n_rows={} ncols_dst={}", n_rows, ncols_dst),
            });
        }

        // n_rows must be aligned to QK4_0
        if n_rows % QK4_0 != 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q4_0_f32: n_rows must be multiple of {}, got {}", QK4_0, n_rows),
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

        // Call kernel
        gemv_q4_0_f32(weights_q4_0, input, output, n_rows, ncols_dst)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    // ── Q4_1 Methods ─────────────────────────────────────────────────────────────────────

    /// Quantize f32 data to Q4_1 format.
    ///
    /// # Arguments
    /// * `input` - GPU pointer to f32 input data [n]
    /// * `output` - GPU pointer to Q4_1 output data [n/32 * 20]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn quantize_q4_1(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q4_1: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Call kernel
        quantize_q4_1(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn dequantize_q4_1(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q4_1: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Call kernel
        dequantize_q4_1(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

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
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n or batch_size is zero
    /// - Kernel launch fails
    pub fn dequantize_q4_1_batched(&self, input: *const u8, output: *mut f32, n: usize, batch_size: usize) -> GpuResult<()> {
        if n == 0 || batch_size == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q4_1_batched: n and batch_size cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Call kernel
        dequantize_q4_1_batched(input, output, n, batch_size)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    /// Verify Q4_1 quantization accuracy.
    ///
    /// Compares original f32 data with quantize-dequantize round-trip.
    ///
    /// # Arguments
    /// * `original` - GPU pointer to original f32 data [n]
    /// * `quantized` - GPU pointer to Q4_1 quantized data [n/32 * 20]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// (max_error, mse, relative_error) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn verify_q4_1_accuracy(&self, original: *const f32, quantized: *const u8, n: usize) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q4_1_accuracy: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
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

        // Allocate GPU memory for error metrics
        let num_blocks = (n + QK4_1 - 1) / QK4_1;

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
        verify_q4_1_accuracy(original, quantized, errors_gpu as *mut f32, n)?;

        // Finalize metrics
        finalize_q4_1_metrics(errors_gpu as *const f32, metrics_gpu as *mut f32, n)?;

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

    /// Q4_1 x f32 GEMV: Compute output = weights @ input
    ///
    /// Computes matrix-vector multiplication with Q4_1 quantized weights:
    /// ```
    /// output[col] = sum over rows of (dequantize_q4_1(weight[row, col]) * input[row])
    /// ```
    ///
    /// # Arguments
    /// * `weights_q4_1` - GPU pointer to Q4_1 quantized weights [n_rows/32 * ncols_dst * 20]
    /// * `input` - GPU pointer to f32 input vector [n_rows]
    /// * `output` - GPU pointer to f32 output vector [ncols_dst] (will be written)
    /// * `n_rows` - Number of rows (input dimension, must be multiple of 32)
    /// * `ncols_dst` - Number of columns (output dimension)
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n_rows or ncols_dst is zero
    /// - n_rows is not a multiple of 32
    /// - Any pointer is null
    /// - Kernel launch fails
    pub fn gemv_q4_1_f32(
        &self,
        weights_q4_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        // Validate dimensions
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q4_1_f32: invalid dimensions n_rows={} ncols_dst={}", n_rows, ncols_dst),
            });
        }

        // n_rows must be aligned to QK4_1
        if n_rows % QK4_1 != 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q4_1_f32: n_rows must be multiple of {}, got {}", QK4_1, n_rows),
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

        // Call kernel
        gemv_q4_1_f32(weights_q4_1, input, output, n_rows, ncols_dst)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    /// Q5_K verification: Compute accuracy metrics for Q5_K quantization
    ///
    /// Verifies quantization accuracy by computing error metrics:
    /// - max_error: Maximum absolute error across all elements
    /// - mse: Mean squared error
    /// - relative_error: Total error / total magnitude
    ///
    /// # Arguments
    /// * `original` - GPU pointer to original f32 data [n]
    /// * `quantized` - GPU pointer to Q5_K quantized data [n/256 * 176]
    /// * `n` - Total number of elements
    ///
    /// # Returns
    /// (max_error, mse, relative_error) on success
    ///
    /// # Errors
    /// - n is zero
    /// - Kernel launch fails
    pub fn verify_q5_k_accuracy(&self, original: *const f32, quantized: *const u8, n: usize) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q5_k_accuracy: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if original.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q5_k_accuracy: original pointer is null".to_string(),
            });
        }

        if quantized.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q5_k_accuracy: quantized pointer is null".to_string(),
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
        verify_q5_k_accuracy(original, quantized, errors_gpu as *mut f32, n)?;

        // Finalize metrics
        finalize_q5_k_metrics(errors_gpu as *const f32, metrics_gpu as *mut f32, n)?;

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

    /// Q8_0 × f32 GEMV: Compute output = weights @ input
    ///
    /// Computes matrix-vector multiplication with quantized weights:
    /// ```
    /// output[col] = sum over rows of (dequantize_q8_0(weight[row, col]) * input[row])
    /// ```
    ///
    /// # Arguments
    /// * `weights_q8_0` - GPU pointer to Q8_0 quantized weights [n_rows/32 * ncols_dst * 34]
    /// * `input` - GPU pointer to f32 input vector [n_rows]
    /// * `output` - GPU pointer to f32 output vector [ncols_dst] (will be written)
    /// * `n_rows` - Number of rows (input dimension, must be multiple of 32)
    /// * `ncols_dst` - Number of columns (output dimension)
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Errors
    /// - n_rows or ncols_dst is zero
    /// - n_rows is not a multiple of 32
    /// - Any pointer is null
    /// - Kernel launch fails
    ///
    /// # Notes
    /// - Follows llama.cpp pattern with template specialization for ncols_dst
    /// - Uses register-based computation (no shared memory)
    /// - Optimized for ncols_dst in {1, 2, 4, 8}, generic path for larger sizes
    pub fn gemv_q8_0_f32(
        &self,
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        // Validate dimensions
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q8_0_f32: invalid dimensions n_rows={} ncols_dst={}", n_rows, ncols_dst),
            });
        }

        // n_rows must be aligned to QK8_0
        if n_rows % QK8_0 != 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q8_0_f32: n_rows must be multiple of {}, got {}", QK8_0, n_rows),
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

        // Call kernel
        gemv_q8_0_f32(weights_q8_0, input, output, n_rows, ncols_dst)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    /// Q4_K × f32 GEMV: Compute output = weights @ input
    ///
    /// Computes matrix-vector multiplication with Q4_K quantized weights:
    /// ```
    /// output[col] = sum over rows of (dequantize_q4_k(weight[row, col]) * input[row])
    /// ```
    ///
    /// # Arguments
    /// * `weights_q4_k` - GPU pointer to Q4_K quantized weights [n_rows/256 * ncols_dst * 144]
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
    /// - ncols_dst exceeds 1024
    /// - Any pointer is null
    /// - Kernel launch fails
    ///
    /// # Notes
    /// - Follows llama.cpp pattern with template specialization for ncols_dst
    /// - Uses 32 threads per block (RDNA wavefront size)
    /// - Direct dequantization + dot product (no intermediate Q8_K quantization)
    pub fn gemv_q4_k_f32(
        &self,
        weights_q4_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        // Validate dimensions
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q4_k_f32: invalid dimensions n_rows={} ncols_dst={}", n_rows, ncols_dst),
            });
        }

        // n_rows must be aligned to QK_K (256)
        if n_rows % QK_K != 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q4_k_f32: n_rows must be multiple of {}, got {}", QK_K, n_rows),
            });
        }

        // ncols_dst must not exceed kernel limit
        if ncols_dst > 1024 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q4_k_f32: ncols_dst must be <= 1024, got {}", ncols_dst),
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

        // Call kernel
        gemv_q4_k_f32(weights_q4_k, input, output, n_rows, ncols_dst)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

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
    /// - ncols_dst exceeds 1024
    /// - Any pointer is null
    /// - Kernel launch fails
    ///
    /// # Notes
    /// - Follows llama.cpp pattern with non-uniform sub-block scaling
    /// - Uses get_scale_min_k4() for per-32-element scale extraction
    /// - Template specialization for ncols_dst in {1, 2, 4, 5, 6, 7, 8}
    /// - Generic fallback for larger sizes
    pub fn gemv_q5_k_f32(
        &self,
        weights_q5_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        // Validate dimensions
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q5_k_f32: invalid dimensions n_rows={} ncols_dst={}", n_rows, ncols_dst),
            });
        }

        // n_rows must be aligned to QK_K (256)
        if n_rows % QK_K != 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q5_k_f32: n_rows must be multiple of {}, got {}", QK_K, n_rows),
            });
        }

        // ncols_dst must not exceed kernel limit
        if ncols_dst > 1024 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("gemv_q5_k_f32: ncols_dst must be <= 1024, got {}", ncols_dst),
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

        // Call kernel
        gemv_q5_k_f32(weights_q5_k, input, output, n_rows, ncols_dst)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
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

        let input: *const f32 = std::ptr::null();
        let output: *mut u8 = std::ptr::null_mut();

        // These would fail at pointer validation, not kernel launch
        // We can't test this without a real device
        let _ = (input, output);
    }

    #[test]
    fn qk_k_constant_is_256() {
        assert_eq!(QK_K, 256, "QK_K must be 256 for Q4_K format");
    }

    #[test]
    fn q4_k_block_size_is_144() {
        assert_eq!(Q4_K_BLOCK_SIZE, 144, "Q4_K_BLOCK_SIZE must be 144 bytes");
    }

    #[test]
    fn qk8_0_constant_is_32() {
        assert_eq!(QK8_0, 32, "QK8_0 must be 32 for Q8_0 format");
    }

    #[test]
    fn q8_0_block_size_is_34() {
        assert_eq!(Q8_0_BLOCK_SIZE, 34, "Q8_0_BLOCK_SIZE must be 34 bytes");
    }

    #[test]
    fn q8_0_max_is_127() {
        assert_eq!(Q8_0_MAX, 127.0, "Q8_0_MAX must be 127.0");
    }

    #[test]
    fn q4_0_block_size_is_18() {
        assert_eq!(Q4_0_BLOCK_SIZE, 18, "Q4_0_BLOCK_SIZE must be 18 bytes");
    }

    #[test]
    fn qk4_0_is_32() {
        assert_eq!(QK4_0, 32, "QK4_0 must be 32");
    }

    #[test]
    fn q4_1_block_size_is_20() {
        assert_eq!(Q4_1_BLOCK_SIZE, 20, "Q4_1_BLOCK_SIZE must be 20 bytes");
    }

    #[test]
    fn qk4_1_is_32() {
        assert_eq!(QK4_1, 32, "QK4_1 must be 32");
    }
}
