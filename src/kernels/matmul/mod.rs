//! Matrix multiplication kernel implementations
//!
//! Provides CPU and GPU kernels for matrix multiplication:
//! - Quantized matmul (Q4_0, Q4_K, Q6_K, Q8_0)
//! - FP16/FP32 matmul
//! - Fused operations (matmul + bias, matmul + activation)

pub mod quantized;
mod fp16;

// Re-export all quantized matmul functions
pub use quantized::*;

// Re-export FP16/FP32 matmul
pub use fp16::matmul;
