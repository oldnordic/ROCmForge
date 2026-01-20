//! Kernel implementations for CPU and GPU
//!
//! This module organizes all computational kernels by operation type:
//! - `quant`: Quantization/dequantization (Q4_0, Q4_K, Q6_K, Q8_0)
//! - `attention`: Attention operations (softmax, matmul, FlashAttention)
//! - `matmul`: Matrix multiplication (quantized, FP16)
//! - `element`: Element-wise operations (RMSNorm, SwiGLU, scale)
//! - `transpose`: GPU matrix transpose (avoids CPU round-trip)

pub mod quant;
pub mod attention;
pub mod matmul;
pub mod element;
pub mod transpose;
