//! Common types and utilities for quantized matmul operations

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};

/// Result type for quantized operations
pub type QuantizedResult<T> = Result<T, String>;

// Constants for Q4_0 format
pub const Q4_0_BLOCK_SIZE: usize = 20;  // 4 bytes scale + 16 bytes packed data
#[allow(dead_code)] // Reserved for future Q4_0 format validation
pub const Q4_0_ELEMENTS_PER_BLOCK: usize = 32;

// Constants for Q8_0 format
pub const Q8_0_BLOCK_SIZE: usize = 36;  // 4 bytes scale + 32 bytes int8 data
#[allow(dead_code)] // Reserved for future Q8_0 format validation
pub const Q8_0_ELEMENTS_PER_BLOCK: usize = 32;

// Constants for Q4_K format
pub const Q4_K_SUPER_BLOCK_SIZE: usize = 256;  // Total bytes per super-block
pub const Q4_K_ELEMENTS_PER_BLOCK: usize = 256;  // Elements per super-block

// Constants for Q6_K format
pub const Q6_K_BLOCK_SIZE: usize = 256;  // Total bytes per block
pub const Q6_K_ELEMENTS_PER_BLOCK: usize = 256;  // Elements per block

/// Q4_0 block header (scale only)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Q4_0Block {
    /// Scale factor for this block
    pub scale: f32,
    // 16 bytes of packed 4-bit values follow in the actual data
}

/// Q8_0 block header (scale only)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Q8_0Block {
    /// Scale factor for this block
    pub scale: f32,
    // 32 bytes of int8 values follow in the actual data
}

/// Convert half-precision float (FP16) to single-precision float (FP32) on CPU
pub fn f16_to_f32_cpu(bits: u16) -> f32 {
    let sign = if (bits & 0x8000) != 0 { 1u32 << 31 } else { 0 };
    let exp = ((bits & 0x7C00) >> 10) as i32 - 15 + 127;
    let mant = ((bits & 0x03FF) << 13) as u32;

    if exp <= 0 {
        // Zero or denormal
        if bits & 0x7FFF == 0 {
            f32::from_bits(sign)
        } else {
            // Denormal - not handling for simplicity
            0.0
        }
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | mant)
    }
}
