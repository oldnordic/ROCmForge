//! CPU SIMD backend for matrix operations using std::simd
//!
//! Provides SIMD-accelerated matrix multiplication using the std::simd module
//! (Rust 1.82+). Requires the portable_simd feature (enabled at crate level).
//!
//! Automatically selects optimal vector width based on architecture.

use std::simd::{f32x4, f32x8, Simd};
use std::simd::prelude::SimdFloat;

// ============================================================================
// AVX-512 support (opt-in via feature flag)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use std::simd::f32x16;

// ============================================================================
// Default SIMD types (AVX2/NEON)
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

/// Matrix multiplication error types
#[derive(Debug, thiserror::Error)]
pub enum SimdMatmulError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeError { expected: usize, actual: usize },
}

pub type SimdMatmulResult<T> = Result<T, SimdMatmulError>;

/// SIMD-accelerated matrix multiplication: C = A * B
///
/// Uses std::simd for portable SIMD acceleration. Automatically selects
/// optimal vector width based on target architecture (f32x8 for x86_64 AVX2,
/// f32x4 for aarch64 NEON).
///
/// # Arguments
///
/// * `a` - Matrix A in row-major format (m x k)
/// * `b` - Matrix B in row-major format (k x n)
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B
///
/// # Returns
///
/// * Matrix C in row-major format (m x n)
///
/// # Example
///
/// ```rust
/// use rocmforge::backend::cpu::simd::simd_matmul_f32;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
/// let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
///
/// let result = simd_matmul_f32(&a, &b, 2, 2, 2).unwrap();
/// assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn simd_matmul_f32(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> SimdMatmulResult<Vec<f32>> {
    // Validate dimensions
    if m == 0 || n == 0 || k == 0 {
        return Err(SimdMatmulError::DimensionMismatch(format!(
            "Invalid dimensions: m={}, n={}, k={} (all must be positive)",
            m, n, k
        )));
    }

    // Validate buffer sizes
    let expected_a_size = m * k;
    let expected_b_size = k * n;

    if a.len() != expected_a_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_a_size,
            actual: a.len(),
        });
    }

    if b.len() != expected_b_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_b_size,
            actual: b.len(),
        });
    }

    let mut c = vec![0.0f32; m * n];

    // Process matrix multiplication with SIMD acceleration
    // For each row i of A and column j of B, compute dot product of row i and column j
    for i in 0..m {
        for j in 0..n {
            let mut sum = SimdF32::splat(0.0);

            // Process k elements in chunks of SIMD_WIDTH
            let mut kk = 0;
            while kk + SIMD_WIDTH <= k {
                // Load SIMD vectors from A row and B column
                let a_vec: SimdF32 = Simd::from_array([
                    a[i * k + kk],
                    a[i * k + kk + 1],
                    a[i * k + kk + 2],
                    a[i * k + kk + 3],
                    #[cfg(target_arch = "x86_64")]
                    a[i * k + kk + 4],
                    #[cfg(target_arch = "x86_64")]
                    a[i * k + kk + 5],
                    #[cfg(target_arch = "x86_64")]
                    a[i * k + kk + 6],
                    #[cfg(target_arch = "x86_64")]
                    a[i * k + kk + 7],
                ]);

                let b_vec: SimdF32 = Simd::from_array([
                    b[kk * n + j],
                    b[(kk + 1) * n + j],
                    b[(kk + 2) * n + j],
                    b[(kk + 3) * n + j],
                    #[cfg(target_arch = "x86_64")]
                    b[(kk + 4) * n + j],
                    #[cfg(target_arch = "x86_64")]
                    b[(kk + 5) * n + j],
                    #[cfg(target_arch = "x86_64")]
                    b[(kk + 6) * n + j],
                    #[cfg(target_arch = "x86_64")]
                    b[(kk + 7) * n + j],
                ]);

                // FMA: multiply and accumulate
                sum += a_vec * b_vec;
                kk += SIMD_WIDTH;
            }

            // Horizontal sum to get scalar result
            let mut dot_product = sum.reduce_sum();

            // Handle remaining elements (k % SIMD_WIDTH)
            while kk < k {
                dot_product += a[i * k + kk] * b[kk * n + j];
                kk += 1;
            }

            c[i * n + j] = dot_product;
        }
    }

    Ok(c)
}

/// Scalar fallback matrix multiplication for comparison/testing
///
/// This is the reference implementation used when SIMD is not available
/// or for correctness validation.
pub fn scalar_matmul_f32(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> SimdMatmulResult<Vec<f32>> {
    // Validate dimensions
    if m == 0 || n == 0 || k == 0 {
        return Err(SimdMatmulError::DimensionMismatch(format!(
            "Invalid dimensions: m={}, n={}, k={} (all must be positive)",
            m, n, k
        )));
    }

    // Validate buffer sizes
    let expected_a_size = m * k;
    let expected_b_size = k * n;

    if a.len() != expected_a_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_a_size,
            actual: a.len(),
        });
    }

    if b.len() != expected_b_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_b_size,
            actual: b.len(),
        });
    }

    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    Ok(c)
}

/// Optimized SIMD matmul with tiling for cache efficiency
///
/// Uses blocking to improve cache utilization for larger matrices.
/// Tiles are chosen to fit in L1 cache (typically 32-64KB).
///
/// # Arguments
///
/// Same as `simd_matmul_f32`
///
/// # Performance
///
/// For matrices larger than ~64x64, this tiled version provides
/// better cache utilization and higher performance than the naive
/// SIMD version.
pub fn simd_matmul_tiled_f32(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> SimdMatmulResult<Vec<f32>> {
    const TILE_SIZE: usize = 32;

    // Validate dimensions
    if m == 0 || n == 0 || k == 0 {
        return Err(SimdMatmulError::DimensionMismatch(format!(
            "Invalid dimensions: m={}, n={}, k={} (all must be positive)",
            m, n, k
        )));
    }

    // Validate buffer sizes
    let expected_a_size = m * k;
    let expected_b_size = k * n;

    if a.len() != expected_a_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_a_size,
            actual: a.len(),
        });
    }

    if b.len() != expected_b_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_b_size,
            actual: b.len(),
        });
    }

    let mut c = vec![0.0f32; m * n];

    // Tiled matmul for cache efficiency
    for i_tile in (0..m).step_by(TILE_SIZE) {
        for j_tile in (0..n).step_by(TILE_SIZE) {
            for k_tile in (0..k).step_by(TILE_SIZE) {
                // Process tile
                let i_end = (i_tile + TILE_SIZE).min(m);
                let j_end = (j_tile + TILE_SIZE).min(n);
                let k_end = (k_tile + TILE_SIZE).min(k);

                for i in i_tile..i_end {
                    for j in j_tile..j_end {
                        let mut sum = 0.0f32;

                        // Use SIMD for inner loop when possible
                        let mut kk = k_tile;
                        while kk + SIMD_WIDTH <= k_end {
                            let a_vec: SimdF32 = Simd::from_array([
                                a[i * k + kk],
                                a[i * k + kk + 1],
                                a[i * k + kk + 2],
                                a[i * k + kk + 3],
                                #[cfg(target_arch = "x86_64")]
                                a[i * k + kk + 4],
                                #[cfg(target_arch = "x86_64")]
                                a[i * k + kk + 5],
                                #[cfg(target_arch = "x86_64")]
                                a[i * k + kk + 6],
                                #[cfg(target_arch = "x86_64")]
                                a[i * k + kk + 7],
                            ]);

                            let b_vec: SimdF32 = Simd::from_array([
                                b[kk * n + j],
                                b[(kk + 1) * n + j],
                                b[(kk + 2) * n + j],
                                b[(kk + 3) * n + j],
                                #[cfg(target_arch = "x86_64")]
                                b[(kk + 4) * n + j],
                                #[cfg(target_arch = "x86_64")]
                                b[(kk + 5) * n + j],
                                #[cfg(target_arch = "x86_64")]
                                b[(kk + 6) * n + j],
                                #[cfg(target_arch = "x86_64")]
                                b[(kk + 7) * n + j],
                            ]);

                            let product = a_vec * b_vec;
                            sum += product.reduce_sum();
                            kk += SIMD_WIDTH;
                        }

                        // Handle remaining elements
                        while kk < k_end {
                            sum += a[i * k + kk] * b[kk * n + j];
                            kk += 1;
                        }

                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }

    Ok(c)
}

// ============================================================================
// AVX-512 SIMD variants (opt-in via feature flag)
// ============================================================================

/// AVX-512-accelerated matrix multiplication: C = A * B
///
/// Uses f32x16 vectors (AVX-512: 16 floats per vector) for up to 2x speedup
/// over AVX2 on supported hardware.
///
/// # Feature Flag
///
/// This function is only available when the `avx512` feature is enabled.
/// It requires nightly Rust and AVX-512-capable CPU.
///
/// # Arguments
///
/// * `a` - Matrix A in row-major format (m x k)
/// * `b` - Matrix B in row-major format (k x n)
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B
///
/// # Returns
///
/// * Matrix C in row-major format (m x n)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn avx512_simd_matmul_f32(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> SimdMatmulResult<Vec<f32>> {
    // Validate dimensions
    if m == 0 || n == 0 || k == 0 {
        return Err(SimdMatmulError::DimensionMismatch(format!(
            "Invalid dimensions: m={}, n={}, k={} (all must be positive)",
            m, n, k
        )));
    }

    // Validate buffer sizes
    let expected_a_size = m * k;
    let expected_b_size = k * n;

    if a.len() != expected_a_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_a_size,
            actual: a.len(),
        });
    }

    if b.len() != expected_b_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_b_size,
            actual: b.len(),
        });
    }

    let mut c = vec![0.0f32; m * n];

    // Process matrix multiplication with AVX-512 SIMD acceleration
    // For each row i of A and column j of B, compute dot product of row i and column j
    for i in 0..m {
        for j in 0..n {
            let mut sum = f32x16::splat(0.0);

            // Process k elements in chunks of 16 (AVX-512 width)
            let mut kk = 0;
            while kk + AVX512_WIDTH <= k {
                // Load AVX-512 vectors from A row and B column
                let a_vec: f32x16 = Simd::from_array([
                    a[i * k + kk],
                    a[i * k + kk + 1],
                    a[i * k + kk + 2],
                    a[i * k + kk + 3],
                    a[i * k + kk + 4],
                    a[i * k + kk + 5],
                    a[i * k + kk + 6],
                    a[i * k + kk + 7],
                    a[i * k + kk + 8],
                    a[i * k + kk + 9],
                    a[i * k + kk + 10],
                    a[i * k + kk + 11],
                    a[i * k + kk + 12],
                    a[i * k + kk + 13],
                    a[i * k + kk + 14],
                    a[i * k + kk + 15],
                ]);

                let b_vec: f32x16 = Simd::from_array([
                    b[kk * n + j],
                    b[(kk + 1) * n + j],
                    b[(kk + 2) * n + j],
                    b[(kk + 3) * n + j],
                    b[(kk + 4) * n + j],
                    b[(kk + 5) * n + j],
                    b[(kk + 6) * n + j],
                    b[(kk + 7) * n + j],
                    b[(kk + 8) * n + j],
                    b[(kk + 9) * n + j],
                    b[(kk + 10) * n + j],
                    b[(kk + 11) * n + j],
                    b[(kk + 12) * n + j],
                    b[(kk + 13) * n + j],
                    b[(kk + 14) * n + j],
                    b[(kk + 15) * n + j],
                ]);

                // FMA: multiply and accumulate
                sum += a_vec * b_vec;
                kk += AVX512_WIDTH;
            }

            // Horizontal sum to get scalar result
            let mut dot_product = sum.reduce_sum();

            // Handle remaining elements (k % 16)
            while kk < k {
                dot_product += a[i * k + kk] * b[kk * n + j];
                kk += 1;
            }

            c[i * n + j] = dot_product;
        }
    }

    Ok(c)
}

/// AVX-512 tiled matmul for cache efficiency with larger matrices
///
/// Uses blocking to improve cache utilization for larger matrices.
/// Tiles are chosen to fit in L1 cache (typically 32-64KB).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn avx512_simd_matmul_tiled_f32(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> SimdMatmulResult<Vec<f32>> {
    const TILE_SIZE: usize = 32;

    // Validate dimensions
    if m == 0 || n == 0 || k == 0 {
        return Err(SimdMatmulError::DimensionMismatch(format!(
            "Invalid dimensions: m={}, n={}, k={} (all must be positive)",
            m, n, k
        )));
    }

    // Validate buffer sizes
    let expected_a_size = m * k;
    let expected_b_size = k * n;

    if a.len() != expected_a_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_a_size,
            actual: a.len(),
        });
    }

    if b.len() != expected_b_size {
        return Err(SimdMatmulError::BufferSizeError {
            expected: expected_b_size,
            actual: b.len(),
        });
    }

    let mut c = vec![0.0f32; m * n];

    // Tiled matmul for cache efficiency
    for i_tile in (0..m).step_by(TILE_SIZE) {
        for j_tile in (0..n).step_by(TILE_SIZE) {
            for k_tile in (0..k).step_by(TILE_SIZE) {
                // Process tile
                let i_end = (i_tile + TILE_SIZE).min(m);
                let j_end = (j_tile + TILE_SIZE).min(n);
                let k_end = (k_tile + TILE_SIZE).min(k);

                for i in i_tile..i_end {
                    for j in j_tile..j_end {
                        let mut sum = 0.0f32;

                        // Use AVX-512 SIMD for inner loop when possible
                        let mut kk = k_tile;
                        while kk + AVX512_WIDTH <= k_end {
                            let a_vec: f32x16 = Simd::from_array([
                                a[i * k + kk],
                                a[i * k + kk + 1],
                                a[i * k + kk + 2],
                                a[i * k + kk + 3],
                                a[i * k + kk + 4],
                                a[i * k + kk + 5],
                                a[i * k + kk + 6],
                                a[i * k + kk + 7],
                                a[i * k + kk + 8],
                                a[i * k + kk + 9],
                                a[i * k + kk + 10],
                                a[i * k + kk + 11],
                                a[i * k + kk + 12],
                                a[i * k + kk + 13],
                                a[i * k + kk + 14],
                                a[i * k + kk + 15],
                            ]);

                            let b_vec: f32x16 = Simd::from_array([
                                b[kk * n + j],
                                b[(kk + 1) * n + j],
                                b[(kk + 2) * n + j],
                                b[(kk + 3) * n + j],
                                b[(kk + 4) * n + j],
                                b[(kk + 5) * n + j],
                                b[(kk + 6) * n + j],
                                b[(kk + 7) * n + j],
                                b[(kk + 8) * n + j],
                                b[(kk + 9) * n + j],
                                b[(kk + 10) * n + j],
                                b[(kk + 11) * n + j],
                                b[(kk + 12) * n + j],
                                b[(kk + 13) * n + j],
                                b[(kk + 14) * n + j],
                                b[(kk + 15) * n + j],
                            ]);

                            let product = a_vec * b_vec;
                            sum += product.reduce_sum();
                            kk += AVX512_WIDTH;
                        }

                        // Handle remaining elements
                        while kk < k_end {
                            sum += a[i * k + kk] * b[kk * n + j];
                            kk += 1;
                        }

                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }

    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_matmul_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]

        let result = simd_matmul_f32(&a, &b, 2, 2, 2).unwrap();
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // [[19,22],[43,50]]

        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "Element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    fn test_simd_matmul_rectangular() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [[7,8],[9,10],[11,12]]

        let result = simd_matmul_f32(&a, &b, 2, 2, 3).unwrap();
        let expected = vec![58.0, 64.0, 139.0, 154.0];

        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "Element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    fn test_simd_matmul_large() {
        // Test with a larger matrix that benefits from SIMD
        let m = 64;
        let n = 64;
        let k = 64;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.01 + 1.0)).collect();

        let result = simd_matmul_f32(&a, &b, m, n, k).unwrap();
        let expected = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

        assert_eq!(result.len(), m * n);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            let abs_diff = (r - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-2 || rel_diff < 1e-4,
                "Element {} mismatch: expected {}, got {}, abs_diff={}, rel_diff={}",
                i,
                e,
                r,
                abs_diff,
                rel_diff
            );
        }
    }

    #[test]
    fn test_simd_vs_scalar_correctness() {
        // Verify SIMD produces same results as scalar
        for (m, n, k) in [(4, 4, 4), (8, 8, 8), (16, 16, 16), (17, 13, 11)] {
            let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0)).collect();

            let simd_result = simd_matmul_f32(&a, &b, m, n, k).unwrap();
            let scalar_result = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

            assert_eq!(
                simd_result.len(),
                scalar_result.len(),
                "Length mismatch for {}x{}x{}",
                m, n, k
            );

            for (i, (&s, &r)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                let abs_diff = (s - r).abs();
                let rel_diff = abs_diff / r.abs().max(1e-6);
                assert!(
                    abs_diff < 1e-3 || rel_diff < 1e-5,
                    "Mismatch at {} for {}x{}x{}: {} vs {}, abs_diff={}, rel_diff={}",
                    i, m, n, k, s, r, abs_diff, rel_diff
                );
            }
        }
    }

    #[test]
    fn test_tiled_matmul_correctness() {
        // Verify tiled version produces same results
        for (m, n, k) in [(32, 32, 32), (64, 64, 64), (128, 65, 33)] {
            let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.1).collect();

            let tiled_result = simd_matmul_tiled_f32(&a, &b, m, n, k).unwrap();
            let scalar_result = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

            assert_eq!(
                tiled_result.len(),
                scalar_result.len(),
                "Length mismatch for {}x{}x{}",
                m, n, k
            );

            for (i, (&t, &s)) in tiled_result.iter().zip(scalar_result.iter()).enumerate() {
                let abs_diff = (t - s).abs();
                let rel_diff = abs_diff / s.abs().max(1e-6);
                assert!(
                    abs_diff < 1e-2 || rel_diff < 1e-4,
                    "Mismatch at {} for {}x{}x{}: {} vs {}, abs_diff={}, rel_diff={}",
                    i, m, n, k, t, s, abs_diff, rel_diff
                );
            }
        }
    }

    #[test]
    fn test_invalid_dimensions() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // Zero dimensions should fail
        assert!(simd_matmul_f32(&a, &b, 0, 2, 2).is_err());
        assert!(simd_matmul_f32(&a, &b, 2, 0, 2).is_err());
        assert!(simd_matmul_f32(&a, &b, 2, 2, 0).is_err());

        // Buffer size mismatch should fail
        assert!(simd_matmul_f32(&a, &b, 3, 2, 2).is_err());
    }

    #[test]
    fn test_non_multiple_of_simd_width() {
        // Test dimensions not aligned to SIMD width
        let m = 5;
        let n = 7;
        let k = 6;

        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0)).collect();

        let result = simd_matmul_f32(&a, &b, m, n, k).unwrap();
        let expected = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

        assert_eq!(result.len(), m * n);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i, r, e
            );
        }
    }

    // ========================================================================
    // AVX-512 tests (only run with avx512 feature)
    // ========================================================================

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_avx512_simd_matmul_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]

        let result = avx512_simd_matmul_f32(&a, &b, 2, 2, 2).unwrap();
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // [[19,22],[43,50]]

        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "Element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_avx512_simd_matmul_large() {
        // Test with a larger matrix that benefits from AVX-512
        let m = 64;
        let n = 64;
        let k = 64;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.01 + 1.0)).collect();

        let result = avx512_simd_matmul_f32(&a, &b, m, n, k).unwrap();
        let expected = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

        assert_eq!(result.len(), m * n);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            let abs_diff = (r - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-2 || rel_diff < 1e-4,
                "Element {} mismatch: expected {}, got {}, abs_diff={}, rel_diff={}",
                i,
                e,
                r,
                abs_diff,
                rel_diff
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_avx512_simd_vs_scalar_correctness() {
        // Verify AVX-512 produces same results as scalar
        for (m, n, k) in [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 17, 19)] {
            let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0)).collect();

            let avx512_result = avx512_simd_matmul_f32(&a, &b, m, n, k).unwrap();
            let scalar_result = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

            assert_eq!(
                avx512_result.len(),
                scalar_result.len(),
                "Length mismatch for {}x{}x{}",
                m, n, k
            );

            for (i, (&a512, &s)) in avx512_result.iter().zip(scalar_result.iter()).enumerate() {
                let abs_diff = (a512 - s).abs();
                let rel_diff = abs_diff / s.abs().max(1e-6);
                assert!(
                    abs_diff < 1e-3 || rel_diff < 1e-5,
                    "Mismatch at {} for {}x{}x{}: {} vs {}, abs_diff={}, rel_diff={}",
                    i, m, n, k, a512, s, abs_diff, rel_diff
                );
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_avx512_tiled_matmul_correctness() {
        // Verify tiled AVX-512 version produces same results
        for (m, n, k) in [(32, 32, 32), (64, 64, 64), (128, 65, 33)] {
            let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.1).collect();

            let tiled_result = avx512_simd_matmul_tiled_f32(&a, &b, m, n, k).unwrap();
            let scalar_result = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

            assert_eq!(
                tiled_result.len(),
                scalar_result.len(),
                "Length mismatch for {}x{}x{}",
                m, n, k
            );

            for (i, (&t, &s)) in tiled_result.iter().zip(scalar_result.iter()).enumerate() {
                let abs_diff = (t - s).abs();
                let rel_diff = abs_diff / s.abs().max(1e-6);
                assert!(
                    abs_diff < 1e-2 || rel_diff < 1e-4,
                    "Mismatch at {} for {}x{}x{}: {} vs {}, abs_diff={}, rel_diff={}",
                    i, m, n, k, t, s, abs_diff, rel_diff
                );
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_avx512_non_multiple_of_width() {
        // Test dimensions not aligned to AVX-512 width (16)
        let m = 5;
        let n = 7;
        let k = 6;

        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0)).collect();

        let result = avx512_simd_matmul_f32(&a, &b, m, n, k).unwrap();
        let expected = scalar_matmul_f32(&a, &b, m, n, k).unwrap();

        assert_eq!(result.len(), m * n);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i, r, e
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_avx512_invalid_dimensions() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // Zero dimensions should fail
        assert!(avx512_simd_matmul_f32(&a, &b, 0, 2, 2).is_err());
        assert!(avx512_simd_matmul_f32(&a, &b, 2, 0, 2).is_err());
        assert!(avx512_simd_matmul_f32(&a, &b, 2, 2, 0).is_err());

        // Buffer size mismatch should fail
        assert!(avx512_simd_matmul_f32(&a, &b, 3, 2, 2).is_err());
    }
}
