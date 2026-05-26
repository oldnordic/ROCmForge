use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::kernels::quant::{
    dequantize_q4_1, dequantize_q4_1_batched, finalize_q4_1_metrics, gemv_q4_1_f32, quantize_q4_1,
    verify_q4_1_accuracy,
};
use crate::gpu::quant::QK4_1;

use super::GpuQuant;

impl GpuQuant {
    pub fn quantize_q4_1(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
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

        quantize_q4_1(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn dequantize_q4_1(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
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

        dequantize_q4_1(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn dequantize_q4_1_batched(
        &self,
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

        dequantize_q4_1_batched(input, output, n, batch_size)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn verify_q4_1_accuracy(
        &self,
        original: *const f32,
        quantized: *const u8,
        n: usize,
    ) -> GpuResult<(f32, f32, f32)> {
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

        let _num_blocks = n.div_ceil(QK4_1);

        let errors_gpu = unsafe { ffi::hip_malloc(4 * std::mem::size_of::<f32>())? };
        let metrics_gpu = unsafe { ffi::hip_malloc(3 * std::mem::size_of::<f32>())? };

        let zeros = [0.0f32; 4];
        unsafe {
            ffi::hip_memcpy_h2d(
                errors_gpu,
                zeros.as_ptr() as *const u8,
                4 * std::mem::size_of::<f32>(),
            )?;
        }

        verify_q4_1_accuracy(original, quantized, errors_gpu as *mut f32, n)?;

        finalize_q4_1_metrics(errors_gpu as *const f32, metrics_gpu as *mut f32, n)?;

        self.device.synchronize()?;

        let mut metrics = [0.0f32; 3];
        unsafe {
            ffi::hip_memcpy_d2h(
                metrics.as_mut_ptr() as *mut u8,
                metrics_gpu as *const u8,
                3 * std::mem::size_of::<f32>(),
            )?;
        }

        unsafe {
            ffi::hip_free(errors_gpu);
            ffi::hip_free(metrics_gpu);
        }

        Ok((metrics[0], metrics[1], metrics[2]))
    }

    pub fn gemv_q4_1_f32(
        &self,
        weights_q4_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q4_1_f32: invalid dimensions n_rows={} ncols_dst={}",
                    n_rows, ncols_dst
                ),
            });
        }

        if !n_rows.is_multiple_of(QK4_1) {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q4_1_f32: n_rows must be multiple of {}, got {}",
                    QK4_1, n_rows
                ),
            });
        }

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

        gemv_q4_1_f32(weights_q4_1, input, output, n_rows, ncols_dst)?;

        self.device.synchronize()?;

        Ok(())
    }
}
