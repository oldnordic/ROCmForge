//! Quantization kernel implementations
//!
//! Provides CPU and GPU kernels for various quantization formats:
//! - Q4_0: 4-bit quantization with constant scale
//! - Q4_K: 4-bit K-quants with 2 sets of scales/mins
//! - Q6_K: 6-bit K-quants with 16-scale groups
//! - Q8_0: 8-bit quantization with constant scale
//! - FP16: Half-precision floating point

pub mod common;
