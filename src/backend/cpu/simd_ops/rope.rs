//! SIMD-accelerated RoPE (Rotary Position Embedding)
//!
//! Applies rotary position embedding to input tensor by rotating pairs of
//! elements using precomputed cosine/sine values.
//!
//! Formula:
//! ```text
//! x'_{2i}   = x_{2i} * cos - x_{2i+1} * sin
//! x'_{2i+1} = x_{2i} * sin + x_{2i+1} * cos
//! ```
//!
//! Provides both SIMD-accelerated and scalar fallback implementations
//! for correctness validation.

use std::simd::{f32x4, f32x8};

use super::SIMD_WIDTH;

/// SIMD-accelerated RoPE rotation
///
/// Applies rotary position embedding to input tensor.
/// Rotates pairs of elements (x_{2i}, x_{2i+1}) using precomputed cos/sin.
///
/// # Arguments
///
/// * `input` - Input tensor (modified in place)
/// * `cos` - Precomputed cosine values [dim/2]
/// * `sin` - Precomputed sine values [dim/2]
pub fn rope_in_place_simd(input: &mut [f32], cos: &[f32], sin: &[f32]) {
    let n = input.len();
    let half_dim = cos.len();

    // Each pair of elements uses one cos/sin value
    // Process pairs (x[0], x[1]), (x[2], x[3]), ...
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    while i + SIMD_WIDTH <= half_dim {
        // Load 8 pairs (16 elements) at once for AVX2
        // Input layout: [x0, x1, x2, x3, x4, x5, x6, x7, ...] where each pair is (x0,x1), (x2,x3), etc.
        let base_idx = i * 2;

        let cos_arr: [f32; SIMD_WIDTH] = cos[i..i + SIMD_WIDTH].try_into().unwrap();
        let sin_arr: [f32; SIMD_WIDTH] = sin[i..i + SIMD_WIDTH].try_into().unwrap();

        let cos_vec = f32x8::from_array(cos_arr);
        let sin_vec = f32x8::from_array(sin_arr);

        // Load pairs: x0_arr = [x[0], x[2], x[4], x[6], ...], x1_arr = [x[1], x[3], x[5], x[7], ...]
        let mut x0_arr = [0.0f32; SIMD_WIDTH];
        let mut x1_arr = [0.0f32; SIMD_WIDTH];
        for j in 0..SIMD_WIDTH {
            x0_arr[j] = input[base_idx + j * 2];
            x1_arr[j] = input[base_idx + j * 2 + 1];
        }

        let x0_vec = f32x8::from_array(x0_arr);
        let x1_vec = f32x8::from_array(x1_arr);

        // Apply rotation: x'0 = x0*cos - x1*sin, x'1 = x0*sin + x1*cos
        let x0_new = x0_vec * cos_vec - x1_vec * sin_vec;
        let x1_new = x0_vec * sin_vec + x1_vec * cos_vec;

        // Store back in interleaved format
        let x0_result = x0_new.to_array();
        let x1_result = x1_new.to_array();
        for j in 0..SIMD_WIDTH {
            input[base_idx + j * 2] = x0_result[j];
            input[base_idx + j * 2 + 1] = x1_result[j];
        }

        i += SIMD_WIDTH;
    }

    #[cfg(target_arch = "aarch64")]
    while i + SIMD_WIDTH <= half_dim {
        // Load 4 pairs (8 elements) at once for NEON
        let base_idx = i * 2;

        let cos_arr: [f32; SIMD_WIDTH] = cos[i..i + SIMD_WIDTH].try_into().unwrap();
        let sin_arr: [f32; SIMD_WIDTH] = sin[i..i + SIMD_WIDTH].try_into().unwrap();

        let cos_vec = f32x4::from_array(cos_arr);
        let sin_vec = f32x4::from_array(sin_arr);

        // Load pairs
        let mut x0_arr = [0.0f32; SIMD_WIDTH];
        let mut x1_arr = [0.0f32; SIMD_WIDTH];
        for j in 0..SIMD_WIDTH {
            x0_arr[j] = input[base_idx + j * 2];
            x1_arr[j] = input[base_idx + j * 2 + 1];
        }

        let x0_vec = f32x4::from_array(x0_arr);
        let x1_vec = f32x4::from_array(x1_arr);

        // Apply rotation
        let x0_new = x0_vec * cos_vec - x1_vec * sin_vec;
        let x1_new = x0_vec * sin_vec + x1_vec * cos_vec;

        // Store back
        let x0_result = x0_new.to_array();
        let x1_result = x1_new.to_array();
        for j in 0..SIMD_WIDTH {
            input[base_idx + j * 2] = x0_result[j];
            input[base_idx + j * 2 + 1] = x1_result[j];
        }

        i += SIMD_WIDTH;
    }

    // Handle remaining pairs
    while i < half_dim {
        let x0_idx = i * 2;
        let x1_idx = i * 2 + 1;

        let x0 = input[x0_idx];
        let x1 = input[x1_idx];
        let c = cos[i];
        let s = sin[i];

        input[x0_idx] = x0 * c - x1 * s;
        input[x1_idx] = x0 * s + x1 * c;

        i += 1;
    }
}

/// Scalar fallback RoPE for correctness validation
pub fn rope_in_place_scalar(input: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half_dim = cos.len();

    for i in 0..half_dim {
        let x0_idx = i * 2;
        let x1_idx = i * 2 + 1;

        let x0 = input[x0_idx];
        let x1 = input[x1_idx];
        let c = cos[i];
        let s = sin[i];

        input[x0_idx] = x0 * c - x1 * s;
        input[x1_idx] = x0 * s + x1 * c;
    }
}

/// RoPE with automatic SIMD/scalar dispatch based on CPU features
pub fn rope_in_place(input: &mut [f32], cos: &[f32], sin: &[f32]) {
    #[cfg(feature = "simd")]
    {
        rope_in_place_simd(input, cos, sin);
    }

    #[cfg(not(feature = "simd"))]
    {
        rope_in_place_scalar(input, cos, sin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_simd_basic() {
        let mut input = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        // cos/sin arrays should have length equal to number of pairs (input.len() / 2)
        let cos = vec![1.0f32, 0.0, 1.0, 0.0]; // 4 pairs
        let sin = vec![0.0f32, 1.0, 0.0, 1.0];

        rope_in_place_simd(&mut input, &cos, &sin);

        // First pair: (1, 0) with (cos=1, sin=0) -> (1*1 - 0*0, 1*0 + 0*1) = (1, 0)
        // Second pair: (0, 1) with (cos=0, sin=1) -> (0*0 - 1*1, 0*1 + 1*0) = (-1, 0)
        // Third pair: (2, 0) with (cos=1, sin=0) -> (2*1 - 0*0, 2*0 + 0*1) = (2, 0)
        // Fourth pair: (0, 2) with (cos=0, sin=1) -> (0*0 - 2*1, 0*1 + 2*0) = (-2, 0)

        assert!((input[0] - 1.0).abs() < 1e-4, "input[0] = {}", input[0]);
        assert!((input[1] - 0.0).abs() < 1e-4, "input[1] = {}", input[1]);
        assert!((input[2] - (-1.0)).abs() < 1e-4, "input[2] = {}", input[2]);
        assert!((input[3] - 0.0).abs() < 1e-4, "input[3] = {}", input[3]);
        assert!((input[4] - 2.0).abs() < 1e-4, "input[4] = {}", input[4]);
        assert!((input[5] - 0.0).abs() < 1e-4, "input[5] = {}", input[5]);
        assert!((input[6] - (-2.0)).abs() < 1e-4, "input[6] = {}", input[6]);
        assert!((input[7] - 0.0).abs() < 1e-4, "input[7] = {}", input[7]);
    }

    #[test]
    fn test_rope_simd_vs_scalar() {
        for size in [4, 8, 16, 32, 64] {
            let mut input_simd: Vec<f32> = (0..size * 2).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let mut input_scalar = input_simd.clone();

            let cos: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).cos()).collect();
            let sin: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();

            rope_in_place_simd(&mut input_simd, &cos, &sin);
            rope_in_place_scalar(&mut input_scalar, &cos, &sin);

            for (i, (&s, &r)) in input_simd.iter().zip(input_scalar.iter()).enumerate() {
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
    fn test_rope_preserves_magnitude() {
        // RoPE should preserve the magnitude of each 2D vector
        let mut input = vec![3.0f32, 4.0, 5.0, 12.0, 8.0, 15.0];
        let input_copy = input.clone();

        let cos = vec![0.5f32, 0.8, 0.3];
        let sin = vec![0.8660254f32, 0.6, 0.9539392];

        rope_in_place_simd(&mut input, &cos, &sin);

        // Check magnitude preservation: x0^2 + x1^2 should be unchanged
        for i in 0..3 {
            let orig_mag_sq = input_copy[i * 2] * input_copy[i * 2] + input_copy[i * 2 + 1] * input_copy[i * 2 + 1];
            let new_mag_sq = input[i * 2] * input[i * 2] + input[i * 2 + 1] * input[i * 2 + 1];
            assert!(
                (orig_mag_sq - new_mag_sq).abs() < 1e-3,
                "Pair {}: magnitude changed from {} to {}",
                i,
                orig_mag_sq.sqrt(),
                new_mag_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_rope_dispatch() {
        let mut input_simd = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut input_scalar = input_simd.clone();
        let cos = vec![0.0f32, 1.0];
        let sin = vec![0.0f32, 1.0];

        rope_in_place(&mut input_simd, &cos, &sin);
        rope_in_place_scalar(&mut input_scalar, &cos, &sin);

        for (i, (&s, &r)) in input_simd.iter().zip(input_scalar.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-4,
                "Element {} mismatch: {} vs {}",
                i,
                s,
                r
            );
        }
    }
}
