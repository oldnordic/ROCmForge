//! Quantized matmul operations for Q4_0, Q4_K, Q6_K, and Q8_0 formats.
//!
//! This module provides matmul operations for quantized weights using fused
//! dequantization + matmul kernels. This eliminates the intermediate FP32
//! weight buffer, providing significant memory bandwidth savings.
//!
//! # Performance Features
//!
//! - **Fused dequantization**: Eliminates intermediate FP32 buffer (~17x bandwidth reduction)
//! - **Batch processing**: Process multiple matmuls in a single kernel launch
//! - **Async kernel launch**: Overlap kernel execution with CPU operations
//! - **Profiling support**: Built-in kernel timers for performance analysis

pub mod common;
mod q4_0;
mod q4_k;
mod q6_k;
mod q8_0;

// Re-export common types
pub use common::{Q4_0Block, Q8_0Block};

// Re-export Q4_0 functions
pub use q4_0::{
    dequantize_q4_0,
    matmul_q4_0,
    matmul_q4_0_cpu_fallback,
};

#[cfg(feature = "rocm")]
pub use q4_0::matmul_q4_0_gpu;

// Re-export Q4_K functions
pub use q4_k::{
    dequantize_q4_k,
    matmul_q4_k,
};

// Re-export Q6_K functions
pub use q6_k::{
    dequantize_q6_k,
    matmul_q6_k,
};

// Re-export Q8_0 functions
pub use q8_0::{
    dequantize_q8_0,
    matmul_q8_0,
};
