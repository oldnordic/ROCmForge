//! GPU Q5_1 quantization wrapper with RAII safety.
//!
//! Safe wrapper for Q5_1 GEMV/GEMM operations using GpuBuffer.

use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::{
    gemm_q5_1_f32, gemv_q5_1_f32, gemv_q5_1_f32_on_stream,
};
use crate::gpu::quant::{Q5_1_BLOCK_SIZE, QK5_1};
use crate::gpu::weights::GpuBuffer;

pub struct GpuQ5_1Weights {
    pub weights: GpuBuffer,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl GpuQ5_1Weights {
    pub fn gemv(
        &self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
    ) -> GpuResult<()> {
        gemv_q5_1_f32(
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
        gemv_q5_1_f32_on_stream(
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
        gemm_q5_1_f32(
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
    fn q5_1_block_size_is_24() {
        assert_eq!(Q5_1_BLOCK_SIZE, 24, "Q5_1_BLOCK_SIZE must be 24 bytes");
    }

    #[test]
    fn qk5_1_is_32() {
        assert_eq!(QK5_1, 32, "QK5_1 must be 32");
    }
}
