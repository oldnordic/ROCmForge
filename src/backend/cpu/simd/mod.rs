//! CPU SIMD backend for matrix operations using std::simd
//!
//! Provides SIMD-accelerated matrix multiplication using the std::simd module
//! (Rust 1.82+). Requires the portable_simd feature (enabled at crate level).
//!
//! Automatically selects optimal vector width based on architecture.

// ============================================================================
// AVX-512 support (opt-in via feature flag)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use std::simd::f32x16;

use std::simd::{f32x4, f32x8, Simd};
use std::simd::prelude::SimdFloat;

// ============================================================================
// SIMD type configuration
// ============================================================================

// Architecture detection for optimal SIMD width
#[cfg(target_arch = "x86_64")]
type SimdF32 = f32x8; // AVX2: 8 floats per vector

#[cfg(target_arch = "aarch64")]
type SimdF32 = f32x4; // NEON: 4 floats per vector

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type SimdF32 = f32x4; // Safe fallback

// Vector width in elements
#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 8;

#[cfg(target_arch = "aarch64")]
const SIMD_WIDTH: usize = 4;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const SIMD_WIDTH: usize = 4;

// ============================================================================
// AVX-512 vector width (when feature is enabled)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const AVX512_WIDTH: usize = 16;

// ============================================================================
// Module exports
// ============================================================================

pub mod error;
pub mod matmul;

// Re-exports for backward compatibility
pub use error::{SimdMatmulError, SimdMatmulResult};
pub use matmul::{
    matmul_optimized_f32, matmul_optimized_tiled_f32, scalar_matmul_f32, simd_matmul_f32,
    simd_matmul_tiled_f32,
};

// Re-export AVX-512 functions only when avx512 feature is enabled
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use matmul::{avx512_simd_matmul_f32, avx512_simd_matmul_tiled_f32};

// SIMD config and types are private to the module (used by matmul submodule)
// They are not part of the public API
