//! CPU backend module
//!
//! Provides CPU-based implementations of tensor operations with SIMD acceleration.

// CPU feature detection is always available (needed for runtime dispatch)
pub mod cpu_features;

#[cfg(feature = "simd")]
pub mod simd;

// Re-export SIMD types when available
#[cfg(feature = "simd")]
pub use simd::{simd_matmul_f32, simd_matmul_tiled_f32, SimdMatmulError, SimdMatmulResult};

// Re-export AVX-512 functions when avx512 feature is enabled
#[cfg(all(feature = "simd", feature = "avx512"))]
pub use simd::{avx512_simd_matmul_f32, avx512_simd_matmul_tiled_f32};

// Re-export optimized dispatch functions (always available via cpu_features)
#[cfg(feature = "simd")]
pub use simd::{matmul_optimized_f32, matmul_optimized_tiled_f32};

// CPU feature detection is always available
pub use cpu_features::{CpuArch, CpuFeatures};
