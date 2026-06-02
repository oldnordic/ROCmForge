//! GPU Q5_0 quantization wrapper with RAII safety.
//!
//! Safe wrapper for Q5_0 GEMV/GEMM operations using GpuBuffer.

use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::{
    gemm_q5_0_f32, gemv_q5_0_f32, gemv_q5_0_f32_on_stream,
};
use crate::gpu::quant::{Q5_0_BLOCK_SIZE, QK5_0};
use crate::gpu::weights::GpuBuffer;

pub struct GpuQ5_0Weights {
    pub weights: GpuBuffer,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl GpuQ5_0Weights {
    pub fn gemv(
        &self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
    ) -> GpuResult<()> {
        gemv_q5_0_f32(
            self.weights.as_ptr() as *const u8,
            input.as_ptr() as *const f32,
            output.as_ptr() as *mut f32,
            self.n_rows,
            self.n_cols,
        )
    }

    pub fn gemv_on_stream(
        &self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        stream: crate::gpu::ffi::hipStream_t,
    ) -> GpuResult<()> {
        gemv_q5_0_f32_on_stream(
            self.weights.as_ptr() as *const u8,
            input.as_ptr() as *const f32,
            output.as_ptr() as *mut f32,
            self.n_rows,
            self.n_cols,
            stream,
        )
    }

    pub fn gemm(
        &self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        batch_size: usize,
    ) -> GpuResult<()> {
        gemm_q5_0_f32(
            self.weights.as_ptr() as *const u8,
            input.as_ptr() as *const f32,
            output.as_ptr() as *mut f32,
            self.n_rows,
            self.n_cols,
            batch_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q5_0_block_size_is_22() {
        assert_eq!(Q5_0_BLOCK_SIZE, 22, "Q5_0_BLOCK_SIZE must be 22 bytes");
    }

    #[test]
    fn qk5_0_is_32() {
        assert_eq!(QK5_0, 32, "QK5_0 must be 32");
    }
}
