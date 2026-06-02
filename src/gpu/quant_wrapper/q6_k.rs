use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::kernels::quant::{dequantize_q6_k, quantize_q6_k, verify_q6_k_accuracy};
use crate::gpu::quant::QK_K;
use crate::gpu::weights::GpuBuffer;

use super::GpuQuant;

impl GpuQuant {
    pub fn quantize_q6_k(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q6_k: n cannot be zero".to_string(),
            });
        }

        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q6_k: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q6_k: output pointer is null".to_string(),
            });
        }

        quantize_q6_k(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn dequantize_q6_k(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q6_k: n cannot be zero".to_string(),
            });
        }

        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q6_k: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q6_k: output pointer is null".to_string(),
            });
        }

        dequantize_q6_k(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn verify_q6_k_accuracy(
        &self,
        original: *const f32,
        quantized: *const u8,
        n: usize,
    ) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q6_k_accuracy: n cannot be zero".to_string(),
            });
        }

        if original.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q6_k_accuracy: original pointer is null".to_string(),
            });
        }

        if quantized.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q6_k_accuracy: quantized pointer is null".to_string(),
            });
        }

        let errors_buf = GpuBuffer::alloc(4 * std::mem::size_of::<f32>())?;
        let errors_gpu = errors_buf.as_ptr() as *mut f32;

        let zeros = [0.0f32; 4];
        unsafe {
            ffi::hip_memcpy_h2d(
                errors_buf.as_ptr(),
                zeros.as_ptr() as *const u8,
                4 * std::mem::size_of::<f32>(),
            )?;
        }

        verify_q6_k_accuracy(original, quantized, errors_gpu, n)?;

        self.device.synchronize()?;

        let mut errors = [0.0f32; 4];
        unsafe {
            ffi::hip_memcpy_d2h(
                errors.as_mut_ptr() as *mut u8,
                errors_buf.as_ptr() as *const u8,
                4 * std::mem::size_of::<f32>(),
            )?;
        }

        let max_error = errors[0];
        let mse = errors[1] / n as f32;
        let relative_error = if errors[2] > 0.0f32 {
            errors[3] / errors[2]
        } else {
            0.0f32
        };

        Ok((max_error, mse, relative_error))
    }

    pub fn gemv_q6_k_f32(
        &self,
        weights_q6_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q6_k_f32: invalid dimensions n_rows={} ncols_dst={}",
                    n_rows, ncols_dst
                ),
            });
        }

        if n_rows % QK_K != 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q6_k_f32: n_rows must be multiple of {}, got {}",
                    QK_K, n_rows
                ),
            });
        }

        if ncols_dst > 1024 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q6_k_f32: ncols_dst must be <= 1024, got {}",
                    ncols_dst
                ),
            });
        }

        if weights_q6_k.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "gemv_q6_k_f32: weights_q6_k pointer is null".to_string(),
            });
        }

        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "gemv_q6_k_f32: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "gemv_q6_k_f32: output pointer is null".to_string(),
            });
        }

        crate::gpu::kernels::quant::gemv_q6_k_f32(weights_q6_k, input, output, n_rows, ncols_dst)?;

        self.device.synchronize()?;

        Ok(())
    }
}
