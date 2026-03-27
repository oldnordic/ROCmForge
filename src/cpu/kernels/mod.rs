//! SIMD kernel implementations.
//!
//! This module contains optimized kernels for various operations:
//! - Q4_K_M quantization/dequantization
//! - Q8_K quantization for intermediate computation
//! - Matrix multiplication kernels
//! - Activation function kernels
//!
//! # Architecture
//!
//! ## Data Formats
//!
//! - **Q4_K**: 4.5 bits per weight, 256 weights per 144-byte block
//! - **Q8_K**: 8-bit intermediate format, 256 values per 292-byte block
//!
//! ## Kernel Organization
//!
//! - **q4**: Q4_K block structure and dequantization
//! - **q8**: Q8_K block structure and quantization
//! - **q8_scalar**: Scalar Q8_K operations
//! - **gemm_q4k_q8_scalar**: Scalar fallback for Q4_K × Q8_K
//! - **gemm_q4k_q8**: AVX2-optimized Q4_K × Q8_K kernels with dispatch
//!
//! # References
//!
//! - llama.cpp: `/home/feanor/Projects/llama.cpp`
//!   - `ggml/src/ggml-common.h` - Block structures
//!   - `ggml/src/ggml-cpu/arch/x86/quants.c` - AVX2 kernels
//! - Spec: `docs/superpowers/specs/2026-03-26-avx2-q4k-q8k-gemm-design.md`

pub mod q4;
pub mod q8;
pub mod q8_scalar;
pub mod gemm_q4k_q8_scalar;
pub mod gemm_q4k_q8;
pub mod gemm_q4k_q8_avx512;
pub mod gemm_q5_0_q8;

pub use q4::BlockQ4K;
pub use q8::BlockQ8K;
