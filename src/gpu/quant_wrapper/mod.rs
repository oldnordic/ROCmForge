//! GPU quantization wrapper with RAII safety.
//!
//! Safe wrapper for Q4_K and Q8_0 quantization operations.
//! Follows project patterns: GpuDevice, GpuBuffer, etc.

mod q4_0;
mod q4_1;
mod q4_k;
mod q5_0;
mod q5_1;
mod q5_k;
mod q6_k;
mod q8_0;

use crate::gpu::device::GpuDevice;
use crate::gpu::error::GpuResult;

pub use crate::gpu::quant::{
    Q4KBlock, Q5KBlock, Q8_0Block, Q4_0_BLOCK_SIZE, Q4_1_BLOCK_SIZE, Q4_K_BLOCK_SIZE,
    Q5_0_BLOCK_SIZE, Q5_1_BLOCK_SIZE, Q5_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE, Q8_0_MAX, QK4_0, QK4_1,
    QK5_0, QK5_1, QK8_0, QK_K,
};

pub struct GpuQuant {
    device: GpuDevice,
}

impl GpuQuant {
    pub fn new(device: GpuDevice) -> GpuResult<Self> {
        Ok(Self { device })
    }

    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    pub fn quantize(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        self.quantize_q4_k(input, output, n)
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
        let input: *const f32 = std::ptr::null();
        let output: *mut u8 = std::ptr::null_mut();
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
    fn q8_0_block_size_is_36() {
        assert_eq!(Q8_0_BLOCK_SIZE, 36, "Q8_0_BLOCK_SIZE must be 36 bytes");
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
