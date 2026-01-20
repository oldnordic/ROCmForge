//! Re-export of quantized matmul from kernels::matmul::quantized for backward compatibility
//!
//! This module re-exports quantized matmul functions from the centralized
//! kernels::matmul::quantized module to maintain backward compatibility with existing code.
//!
//! # Usage
//! Use functions as `quantized_matmul::matmul_q4_0` or import directly:
//! `use crate::ggml::hip_backend::ops::quantized_matmul::matmul_q4_0;`

// Re-export all quantized matmul functions from the new location
pub use crate::kernels::matmul::quantized::*;
