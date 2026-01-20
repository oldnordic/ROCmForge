//! SIMD-accelerated RMSNorm (Root Mean Square Layer Normalization)
//!
//! Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
//!
//! Provides both SIMD-accelerated and scalar fallback implementations
//! for correctness validation.

use std::simd::{f32x4, f32x8, Simd};
use std::simd::prelude::SimdFloat;

use super::{SimdF32, SIMD_WIDTH};

/// SIMD-accelerated RMSNorm
///
/// Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
///
/// # Arguments
///
/// * `input` - Input tensor [hidden_size]
/// * `weight` - Learnable scale parameter [hidden_size]
/// * `eps` - Small constant for numerical stability
///
/// # Returns
///
/// * Normalized and scaled output [hidden_size]
///
/// # Example
///
/// ```rust
/// use rocmforge::backend::cpu::simd_ops::rms_norm_simd;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let weight = vec![0.5, 0.5, 0.5, 0.5];
/// let output = rms_norm_simd(&input, &weight, 1e-5);
/// ```
pub fn rms_norm_simd(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = input.len();

    // Step 1: Compute mean square: (1/n) * sum(x^2)
    let mut sum_sq = SimdF32::splat(0.0);
    let mut i = 0;

    // SIMD-accelerated sum of squares
    while i + SIMD_WIDTH <= n {
        let arr: [f32; SIMD_WIDTH] = input[i..i + SIMD_WIDTH].try_into().unwrap();
        let vec = Simd::from_array(arr);
        sum_sq += vec * vec;
        i += SIMD_WIDTH;
    }

    // Horizontal sum for SIMD portion
    let mut mean_square = sum_sq.reduce_sum();

    // Handle remaining elements
    while i < n {
        mean_square += input[i] * input[i];
        i += 1;
    }

    mean_square /= n as f32;

    // Step 2: Compute inverse scale: 1 / sqrt(ms + eps)
    let inv_scale = 1.0 / (mean_square + eps).sqrt();

    // Step 3: Normalize and scale: x * inv_scale * weight
    let scale = SimdF32::splat(inv_scale);
    let mut output = vec![0.0f32; n];
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    while i + SIMD_WIDTH <= n {
        let x_arr: [f32; SIMD_WIDTH] = input[i..i + SIMD_WIDTH].try_into().unwrap();
        let w_arr: [f32; SIMD_WIDTH] = weight[i..i + SIMD_WIDTH].try_into().unwrap();

        let x_vec = f32x8::from_array(x_arr);
        let w_vec = f32x8::from_array(w_arr);

        let result = x_vec * scale * w_vec;
        output[i..i + SIMD_WIDTH].copy_from_slice(&result.to_array());
        i += SIMD_WIDTH;
    }

    #[cfg(target_arch = "aarch64")]
    while i + SIMD_WIDTH <= n {
        let x_arr: [f32; SIMD_WIDTH] = input[i..i + SIMD_WIDTH].try_into().unwrap();
        let w_arr: [f32; SIMD_WIDTH] = weight[i..i + SIMD_WIDTH].try_into().unwrap();

        let x_vec = f32x4::from_array(x_arr);
        let w_vec = f32x4::from_array(w_arr);

        let result = x_vec * scale * w_vec;
        output[i..i + SIMD_WIDTH].copy_from_slice(&result.to_array());
        i += SIMD_WIDTH;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    while i + SIMD_WIDTH <= n {
        let x_arr: [f32; SIMD_WIDTH] = input[i..i + SIMD_WIDTH].try_into().unwrap();
        let w_arr: [f32; SIMD_WIDTH] = weight[i..i + SIMD_WIDTH].try_into().unwrap();

        let x_vec = Simd::from_array(x_arr);
        let w_vec = Simd::from_array(w_arr);

        let result = x_vec * scale * w_vec;
        output[i..i + SIMD_WIDTH].copy_from_slice(&result.to_array());
        i += SIMD_WIDTH;
    }

    // Handle remaining elements
    while i < n {
        output[i] = input[i] * inv_scale * weight[i];
        i += 1;
    }

    output
}

/// Scalar fallback RMSNorm for correctness validation
pub fn rms_norm_scalar(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = input.len();

    // Compute mean square
    let mean_square = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;

    // Compute inverse scale
    let inv_scale = 1.0 / (mean_square + eps).sqrt();

    // Normalize and scale
    input.iter()
        .zip(weight.iter())
        .map(|(&x, &w)| x * inv_scale * w)
        .collect()
}

/// RMSNorm with automatic SIMD/scalar dispatch based on CPU features
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    #[cfg(feature = "simd")]
    {
        rms_norm_simd(input, weight, eps)
    }

    #[cfg(not(feature = "simd"))]
    {
        rms_norm_scalar(input, weight, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_simd_basic() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![0.5f32, 0.5, 0.5, 0.5];

        let result = rms_norm_simd(&input, &weight, 1e-5);

        assert_eq!(result.len(), 4);
        // All values should be scaled by the same factor
        // Mean square = (1 + 4 + 9 + 16) / 4 = 7.5
        // inv_scale = 1 / sqrt(7.5 + eps)
        let mean_square = 7.5_f32;
        let expected_scale = 1.0 / (mean_square + 1e-5).sqrt();
        for (i, &val) in result.iter().enumerate() {
            let expected = input[i] * expected_scale * weight[i];
            assert!(
                (val - expected).abs() < 1e-4,
                "Element {} mismatch: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_rms_norm_simd_vs_scalar() {
        for size in [4, 8, 16, 32, 64, 100] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let weight: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.05).collect();

            let simd_result = rms_norm_simd(&input, &weight, 1e-6);
            let scalar_result = rms_norm_scalar(&input, &weight, 1e-6);

            assert_eq!(simd_result.len(), scalar_result.len());
            for (i, (&s, &r)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                assert!(
                    (s - r).abs() < 1e-4,
                    "Size {}, element {} mismatch: {} vs {}",
                    size,
                    i,
                    s,
                    r
                );
            }
        }
    }

    #[test]
    fn test_rms_norm_empty() {
        let input: Vec<f32> = vec![];
        let weight: Vec<f32> = vec![];

        let simd_result = rms_norm_simd(&input, &weight, 1e-5);
        let scalar_result = rms_norm_scalar(&input, &weight, 1e-5);

        assert!(simd_result.is_empty());
        assert!(scalar_result.is_empty());
    }

    #[test]
    fn test_rms_norm_zeros() {
        let input = vec![0.0f32; 8];
        let weight = vec![1.0f32; 8];

        let result = rms_norm_simd(&input, &weight, 1e-5);

        // All zeros should stay zero
        for &val in &result {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_rms_norm_dispatch() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![0.5f32, 0.5, 0.5, 0.5];

        let result = rms_norm(&input, &weight, 1e-5);
        let expected = rms_norm_scalar(&input, &weight, 1e-5);

        assert_eq!(result.len(), expected.len());
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Element {} mismatch: {} vs {}",
                i,
                r,
                e
            );
        }
    }
}
