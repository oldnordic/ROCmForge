//! Quantization kernel implementations
//!
//! Provides CPU and GPU kernels for various quantization formats:
//! - Q4_0: 4-bit quantization with constant scale
//! - Q4_K: 4-bit K-quants with 2 sets of scales/mins
//! - Q6_K: 6-bit K-quants with 16-scale groups
//! - Q8_0: 8-bit quantization with constant scale
//! - FP16: Half-precision floating point

pub mod common;
pub mod q4_0;
pub mod q4_k;
pub mod q6_k;
pub mod q8_0;

// Re-export common utilities
pub use common::*;

// Re-export Q4_0 functions
pub use q4_0::{
    dequantize_q4_0_cpu,
    Q4_0Block,
};

// Re-export Q4_K functions
pub use q4_k::{
    dequantize_q4_k_cpu,
};

// Re-export Q6_K functions
pub use q6_k::{
    dequantize_q6_k_cpu,
};

// Re-export Q8_0 functions
pub use q8_0::{
    dequantize_q8_0,
};

// GPU-only exports (require ROCm feature)
#[cfg(feature = "rocm")]
pub use q4_0::{
    dequantize_q4_0_kernel_cached,
    dequantize_q4_0_with_fallback,
    get_or_init_q4_0_dequant_cache,
    dequantize_q4_0_cpu_upload,
};

#[cfg(feature = "rocm")]
pub use q4_k::{
    dequantize_q4_k_gpu_kernel,
    dequantize_q4_k_with_fallback,
    get_or_init_q4_k_dequant_cache,
};

#[cfg(feature = "rocm")]
pub use q6_k::{
    dequantize_q6_k_gpu_kernel,
    dequantize_q6_k_with_fallback,
    get_or_init_q6_k_dequant_cache,
};
