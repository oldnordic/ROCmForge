//! CPU backend module
//!
//! Provides CPU-based implementations of tensor operations with SIMD acceleration.

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "simd")]
pub use simd::{simd_matmul_f32, simd_matmul_tiled_f32, SimdMatmulError, SimdMatmulResult};
