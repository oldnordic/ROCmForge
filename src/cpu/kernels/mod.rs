//! SIMD kernel implementations.
//!
//! This module contains optimized kernels for various operations:
//! - Q4_K_M quantization/dequantization
//! - Matrix multiplication kernels
//! - Activation function kernels

pub mod q4;

pub use q4::BlockQ4K;
