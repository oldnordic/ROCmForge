//! Matrix multiplication kernel implementations
//!
//! Provides CPU and GPU kernels for matrix multiplication:
//! - Quantized matmul (Q4_0, Q4_K, Q6_K, Q8_0)
//! - FP16 matmul
//! - Fused operations (matmul + bias, matmul + activation)
