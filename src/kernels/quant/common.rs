//! Shared utilities for quantization kernels
//!
//! Common constants, types, and helper functions used across
//! different quantization format implementations.

/// Block size for Q4_0 quantization
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Block size for Q4_K quantization
pub const Q4_K_BLOCK_SIZE: usize = 256;

/// Block size for Q6_K quantization
pub const Q6_K_BLOCK_SIZE: usize = 256;

/// Block size for Q8_0 quantization
pub const Q8_0_BLOCK_SIZE: usize = 32;

/// Minimum value for signed 4-bit integer
pub const I4_MIN: i8 = -8;

/// Maximum value for signed 4-bit integer
pub const I4_MAX: i8 = 7;

/// Minimum value for signed 8-bit integer
pub const I8_MIN: i8 = -128;

/// Maximum value for signed 8-bit integer
pub const I8_MAX: i8 = 127;

/// Dequantizes a 4-bit value to f32
///
/// # Arguments
/// * `packed` - Packed 4-bit value (0-15 range)
/// * `scale` - Scale factor for dequantization
/// * `min` - Minimum value for asymmetric quantization
#[inline]
pub fn dequant_q4_to_f32(packed: u8, scale: f32, min: f32) -> f32 {
    let signed = if packed < 8 { packed as i8 } else { (packed as i8) - 16 };
    (signed as f32) * scale + min
}

/// Dequantizes an 8-bit value to f32
///
/// # Arguments
/// * `value` - 8-bit signed value
/// * `scale` - Scale factor for dequantization
#[inline]
pub fn dequant_q8_to_f32(value: i8, scale: f32) -> f32 {
    (value as f32) * scale
}
