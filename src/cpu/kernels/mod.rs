//! SIMD kernel implementations.
//!
//! This module contains optimized kernels for various operations:
//! - Q4_K_M quantization/dequantization
//! - Matrix multiplication kernels
//! - Activation function kernels

pub mod q4;
pub mod q8;
pub mod q8_scalar;
pub mod gemm_q4k_q8_scalar;
pub mod gemm_q4k_q8;

pub use q4::BlockQ4K;
pub use q8::BlockQ8K;
