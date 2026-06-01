use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::kernels::quant::{
    dequantize_q8_0, dequantize_q8_0_batched, finalize_q8_0_metrics, gemv_q8_0_f32, quantize_q8_0,
    verify_q8_0_accuracy,
};
use crate::gpu::quant::QK8_0;
use crate::gpu::weights::GpuBuffer;

use super::GpuQuant;

impl GpuQuant {
    pub fn quantize_q8_0(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q8_0: n cannot be zero".to_string(),
            });
        }

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

        quantize_q8_0(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn dequantize_q8_0(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0: n cannot be zero".to_string(),
            });
        }

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

        dequantize_q8_0(input, output, n)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn dequantize_q8_0_batched(
        &self,
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

        dequantize_q8_0_batched(input, output, n, batch_size)?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn verify_q8_0_accuracy(
        &self,
        original: *const f32,
        quantized: *const u8,
        n: usize,
    ) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_q8_0_accuracy: n cannot be zero".to_string(),
            });
        }

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

        let _num_blocks = n.div_ceil(QK8_0);

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

        verify_q8_0_accuracy(original, quantized, errors_gpu, n)?;

        finalize_q8_0_metrics(errors_gpu as *const f32, metrics_gpu, n)?;

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

    pub fn gemv_q8_0_f32(
        &self,
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: usize,
        ncols_dst: usize,
    ) -> GpuResult<()> {
        if n_rows == 0 || ncols_dst == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q8_0_f32: invalid dimensions n_rows={} ncols_dst={}",
                    n_rows, ncols_dst
                ),
            });
        }

        if !n_rows.is_multiple_of(QK8_0) {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "gemv_q8_0_f32: n_rows must be multiple of {}, got {}",
                    QK8_0, n_rows
                ),
            });
        }

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

        gemv_q8_0_f32(weights_q8_0, input, output, n_rows, ncols_dst)?;

        self.device.synchronize()?;

        Ok(())
    }
}
