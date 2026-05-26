//! Quantization kernel modules.
//!
//! This module contains quantization and dequantization kernels for various formats:
//! - Q4_0, Q4_1: Basic 4-bit quantization
//! - Q4_K, Q5_K, Q6_K: K-quants for better compression
//! - Q8_0: 8-bit quantization
//! - GEMV/GEMM functions for each quant type
//!
//! All quantization formats have been modularized for better maintainability.

// Modularized quantization formats
mod q4_0;
mod q4_1;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_0;

// Batched kernels for prefill processing
pub mod batched;

// Legacy code - contains shared GEMV/GEMM utilities and fusion kernels
mod legacy;

// Re-export Q4_0 functions
pub use q4_0::{
    bench_dot4_hardware, bench_dot4_manual, dequantize_q4_0, dequantize_q4_0_batched,
    finalize_q4_0_metrics, gemm_q4_0_f32, gemv_q4_0_f32, gemv_q4_0_f32_on_stream,
    gemv_q4_0_f32_on_stream_unchecked, gemv_q4_0_f32_residual_on_stream,
    gemv_q4_0_f32_residual_on_stream_unchecked, gemv_q4_0_f32_wave32_on_stream_unchecked,
    gemv_q4_0_f32_wave32_residual_on_stream_unchecked, quantize_q4_0, test_dot4_hardware,
    test_dot4_manual, verify_q4_0_accuracy,
};

// Re-export Q4_1 functions
pub use q4_1::{
    dequantize_q4_1, dequantize_q4_1_batched, finalize_q4_1_metrics, gemm_q4_1_f32,
    gemv_ffn_down_swiglu_q4_1_f32_experimental,
    gemv_ffn_down_swiglu_q4_1_f32_experimental_on_stream, gemv_q4_1_f32, gemv_q4_1_f32_on_stream,
    gemv_q4_1_f32_on_stream_unchecked, gemv_q4_1_f32_residual_on_stream,
    gemv_q4_1_f32_residual_on_stream_unchecked, gemv_q4_1_f32_residual_on_stream_variant_unchecked,
    gemv_q4_1_f32_wave32_on_stream_unchecked, gemv_q4_1_f32_wave32_residual_on_stream_unchecked,
    quantize_q4_1, verify_q4_1_accuracy,
};

// Re-export Q4_K functions
pub use q4_k::{
    dequantize_q4_k, dequantize_q4_k_batched, finalize_q4_k_metrics, quantize_q4_k,
    verify_q4_k_accuracy,
};

// Re-export Q5_K functions
pub use q5_k::{
    dequantize_q5_k, dequantize_q5_k_batched, finalize_q5_k_metrics, quantize_q5_k,
    verify_q5_k_accuracy,
};

// Re-export Q6_K functions
pub use q6_k::{dequantize_q6_k, dequantize_q6_k_batched, quantize_q6_k, verify_q6_k_accuracy};

// Re-export Q8_0 functions
pub use q8_0::{
    dequantize_q8_0, dequantize_q8_0_batched, finalize_q8_0_metrics, quantize_q8_0,
    verify_q8_0_accuracy,
};

// Re-export batched functions for prefill processing
pub use batched::{batched_gemm_q4_0_f32, batched_gemm_q4_1_f32, wmma_matmul_q4_0_f32};

// Re-export legacy functions (fusion kernels, specialized GEMV variants, etc.)
pub use legacy::*;

// RFM Unpack GPU Kernels
mod rfm_unpack;
pub use rfm_unpack::{gpu_unpack_q4_fused_gate_up, gpu_unpack_q4_split};
