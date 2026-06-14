use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::kernels::quant::{
    dequantize_q5_k, finalize_q5_k_metrics, quantize_q5_k, verify_q5_k_accuracy,
};
use crate::gpu::quant::QK_K;
use crate::gpu::weights::GpuBuffer;

use super::GpuQuant;

impl GpuQuant {
    pub fn quantize_q5_k(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q5_k: n cannot be zero".to_string(),
            });
        }

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

        quantize_q5_k(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn dequantize_q5_k(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q5_k: n cannot be zero".to_string(),
            });
        }

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

        dequantize_q5_k(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn verify_q5_k_accuracy(
        &self,
        original: *const f32,
        quantized: *const u8,
        n: usize,
    ) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q5_k_accuracy: n cannot be zero".to_string(),
            });
        }

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

        let _num_blocks = n.div_ceil(QK_K);

        let errors_buf = GpuBuffer::alloc(4 * std::mem::size_of::<f32>())?;
        let metrics_buf = GpuBuffer::alloc(3 * std::mem::size_of::<f32>())?;
        let errors_gpu = errors_buf.as_ptr() as *mut f32;
        let metrics_gpu = metrics_buf.as_ptr() as *mut f32;

        let zeros = [0.0f32; 4];
        unsafe {
            ffi::hip_memcpy_h2d(
                errors_buf.as_ptr(),
                zeros.as_ptr() as *const u8,
                4 * std::mem::size_of::<f32>(),
            )?;
        }

        verify_q5_k_accuracy(original, quantized, errors_gpu, n)?;

        finalize_q5_k_metrics(errors_gpu as *const f32, metrics_gpu, n)?;

        self.device.synchronize()?;

        let mut metrics = [0.0f32; 3];
        unsafe {
            ffi::hip_memcpy_d2h(
                metrics.as_mut_ptr() as *mut u8,
                metrics_buf.as_ptr() as *const u8,
                3 * std::mem::size_of::<f32>(),
            )?;
        }

        Ok((metrics[0], metrics[1], metrics[2]))
    }

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
                description: format!(
                    "gemv_q5_k_f32: invalid dimensions n_rows={} ncols_dst={}",
                    n_rows, ncols_dst
                ),
            });
        }

        // n_rows must be aligned to QK_K (256)
        if !n_rows.is_multiple_of(QK_K) {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q5_k_f32: n_rows must be multiple of {}, got {}",
                    QK_K, n_rows
                ),
            });
        }

        // ncols_dst must not exceed kernel limit
        if ncols_dst > 1024 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q5_k_f32: ncols_dst must be <= 1024, got {}",
                    ncols_dst
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

        // Call kernel
        crate::gpu::kernels::quant::gemv_q5_k_f32(weights_q5_k, input, output, n_rows, ncols_dst)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }
}
