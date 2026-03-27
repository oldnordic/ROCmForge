//! SIMD kernel implementations.
//!
//! This module contains optimized kernels for various operations:
//! - Q3_K quantization/dequantization
//! - Q4_K_M quantization/dequantization
//! - Q8_K quantization for intermediate computation
//! - Matrix multiplication kernels
//! - Activation function kernels
//!
//! # Architecture
//!
//! ## Data Formats
//!
//! - **Q3_K**: ~3.4 bits per weight, 256 weights per 110-byte block
//! - **Q4_K**: 4.5 bits per weight, 256 weights per 144-byte block
//! - **Q8_K**: 8-bit intermediate format, 256 values per 292-byte block
//!
//! ## Kernel Organization
//!
//! - **q3**: Q3_K block structure and dequantization
//! - **q4**: Q4_K block structure and dequantization
//! - **q8**: Q8_K block structure and quantization
//! - **q8_scalar**: Scalar Q8_K operations
//! - **gemm_q4k_q8_scalar**: Scalar fallback for Q4_K × Q8_K
//! - **gemm_q4k_q8**: AVX2-optimized Q4_K × Q8_K kernels with dispatch
//! - **gemm_q4k_q8_avx512**: AVX-512 kernels (uses AVX2 pattern for compatibility)
//! - **gemm_q5_0_q8**: Q5_0 × Q8 kernels
//!
//! # References
//!
//! - llama.cpp: `/home/feanor/Projects/llama.cpp`
//!   - `ggml/src/ggml-common.h` - Block structures
//!   - `ggml/src/ggml-quants.c` - Dequantization functions
//!   - `ggml/src/ggml-cpu/arch/x86/quants.c` - AVX2 kernels

pub mod q3;
pub mod q4;
pub mod q5;
pub mod q8;
pub mod q8_scalar;
pub mod gemm_q4k_q8_scalar;
pub mod gemm_q4k_q8;
pub mod gemm_q4k_q8_avx512;
pub mod gemm_q5_0_q8;

pub use q3::BlockQ3K;
pub use q4::BlockQ4K;
pub use q5::BlockQ5K;
pub use q8::BlockQ8K;
