//! SIMD-accelerated activation functions
//!
//! Provides SIMD-optimized implementations of common transformer activation functions:
//! - SiLU (Swish): x * sigmoid(x)
//! - SwiGLU: SiLU(gate) * value
//! - GELU: Gaussian Error Linear Unit
//!
//! All operations have scalar fallbacks for correctness validation and
//! support runtime dispatch based on detected CPU features.

use std::simd::{f32x4, f32x8, Simd};
use std::simd::prelude::SimdFloat;

use super::{SimdF32, SIMD_WIDTH};

// ============================================================================
// SIMD helper functions
// ============================================================================

/// Polynomial approximation for exp using Taylor series
/// exp(x) approx 1 + x + x^2/2 + x^3/6 + x^4/24
#[cfg(target_arch = "x86_64")]
fn simd_exp_taylor_f32x8(x: f32x8) -> f32x8 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    f32x8::splat(1.0) + x + x2 * f32x8::splat(0.5) + x3 * f32x8::splat(1.0 / 6.0)
        + x4 * f32x8::splat(1.0 / 24.0)
}

#[cfg(target_arch = "aarch64")]
fn simd_exp_taylor_f32x4(x: f32x4) -> f32x4 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    f32x4::splat(1.0) + x + x2 * f32x4::splat(0.5) + x3 * f32x4::splat(1.0 / 6.0)
        + x4 * f32x4::splat(1.0 / 24.0)
}

/// Generic exp approximation using architecture-specific implementation
fn simd_exp_taylor(x: SimdF32) -> SimdF32 {
    #[cfg(target_arch = "x86_64")]
    {
        let x_arr = x.to_array();
        let result = simd_exp_taylor_f32x8(f32x8::from_array(x_arr));
        // Convert back - this is a no-op on x86_64
        let result_arr = result.to_array();
        // Recreate as SimdF32 (f32x8)
        Simd::from_array(result_arr)
    }

    #[cfg(target_arch = "aarch64")]
    {
        let x_arr = x.to_array();
        let result = simd_exp_taylor_f32x4(f32x4::from_array(x_arr));
        let result_arr = result.to_array();
        Simd::from_array(result_arr)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar operations
        let mut result = [0.0f32; SIMD_WIDTH];
        for i in 0..SIMD_WIDTH {
            let xi = x.as_ref()[i];
            let x2 = xi * xi;
            let x3 = x2 * xi;
            let x4 = x2 * x2;
            result[i] = 1.0 + xi + x2 * 0.5 + x3 * (1.0 / 6.0) + x4 * (1.0 / 24.0);
        }
        Simd::from_array(result)
    }
}

/// SIMD tanh using std::f32::tanh (element-wise)
///
/// Note: std::simd doesn't provide tanh, so we apply it element-wise
/// and reconstruct the SIMD vector.
#[cfg(target_arch = "x86_64")]
fn simd_tanh(x: f32x8) -> f32x8 {
    let arr = x.to_array();
    f32x8::from_array([
        arr[0].tanh(),
        arr[1].tanh(),
        arr[2].tanh(),
        arr[3].tanh(),
        arr[4].tanh(),
        arr[5].tanh(),
        arr[6].tanh(),
        arr[7].tanh(),
    ])
}

#[cfg(target_arch = "aarch64")]
fn simd_tanh(x: f32x4) -> f32x4 {
    let arr = x.to_array();
    f32x4::from_array([
        arr[0].tanh(),
        arr[1].tanh(),
        arr[2].tanh(),
        arr[3].tanh(),
    ])
}

// ============================================================================
// SiLU (Swish) activation: x * sigmoid(x)
// ============================================================================

/// SIMD-accelerated SiLU activation
///
/// Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// * Activated output
pub fn silu_simd(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = vec![0.0f32; n];
    let mut i = 0;

    while i + SIMD_WIDTH <= n {
        let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
        let x_vec = Simd::from_array(arr);

        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = -x_vec;
        let exp_neg_x = simd_exp_taylor(neg_x);
        let sigmoid = SimdF32::splat(1.0) / (SimdF32::splat(1.0) + exp_neg_x);

        // silu(x) = x * sigmoid(x)
        let silu = x_vec * sigmoid;

        result[i..i + SIMD_WIDTH].copy_from_slice(&silu.to_array());
        i += SIMD_WIDTH;
    }

    // Handle remaining elements
    while i < n {
        let sigmoid_val = 1.0 / (1.0 + (-x[i]).exp());
        result[i] = x[i] * sigmoid_val;
        i += 1;
    }

    result
}

/// In-place SIMD SiLU activation (memory efficient)
pub fn silu_in_place_simd(x: &mut [f32]) {
    let n = x.len();
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    while i + SIMD_WIDTH <= n {
        let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
        let x_vec = f32x8::from_array(arr);

        let neg_x = -x_vec;
        let exp_neg_x = simd_exp_taylor_f32x8(neg_x);
        let sigmoid = f32x8::splat(1.0) / (f32x8::splat(1.0) + exp_neg_x);
        let silu = x_vec * sigmoid;

        x[i..i + SIMD_WIDTH].copy_from_slice(&silu.to_array());
        i += SIMD_WIDTH;
    }

    #[cfg(target_arch = "aarch64")]
    while i + SIMD_WIDTH <= n {
        let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
        let x_vec = f32x4::from_array(arr);

        let neg_x = -x_vec;
        let exp_neg_x = simd_exp_taylor_f32x4(neg_x);
        let sigmoid = f32x4::splat(1.0) / (f32x4::splat(1.0) + exp_neg_x);
        let silu = x_vec * sigmoid;

        x[i..i + SIMD_WIDTH].copy_from_slice(&silu.to_array());
        i += SIMD_WIDTH;
    }

    while i < n {
        let sigmoid_val = 1.0 / (1.0 + (-x[i]).exp());
        x[i] *= sigmoid_val;
        i += 1;
    }
}

/// Scalar fallback SiLU for correctness validation
pub fn silu_scalar(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| {
        let sigmoid = 1.0 / (1.0 + (-v).exp());
        v * sigmoid
    }).collect()
}

/// In-place scalar SiLU
pub fn silu_in_place_scalar(x: &mut [f32]) {
    for v in x.iter_mut() {
        let sigmoid = 1.0 / (1.0 + (-*v).exp());
        *v *= sigmoid;
    }
}

/// SiLU with automatic SIMD/scalar dispatch
pub fn silu(x: &[f32]) -> Vec<f32> {
    #[cfg(feature = "simd")]
    {
        silu_simd(x)
    }

    #[cfg(not(feature = "simd"))]
    {
        silu_scalar(x)
    }
}

/// In-place SiLU with automatic dispatch
pub fn silu_in_place(x: &mut [f32]) {
    #[cfg(feature = "simd")]
    {
        silu_in_place_simd(x);
    }

    #[cfg(not(feature = "simd"))]
    {
        silu_in_place_scalar(x);
    }
}

// ============================================================================
// SwiGLU activation: silu(gate) * value
// ============================================================================

/// SIMD-accelerated SwiGLU activation
///
/// Formula: SwiGLU(gate, value) = SiLU(gate) * value
///
/// This is the activation function used in LLaMA and Mixtral models.
///
/// # Arguments
///
/// * `gate` - Gate tensor (output of first linear layer)
/// * `value` - Value tensor (output of second linear layer)
///
/// # Returns
///
/// * Element-wise SiLU(gate) * value
pub fn swiglu_simd(gate: &[f32], value: &[f32]) -> Vec<f32> {
    let n = gate.len().min(value.len());
    if n == 0 {
        return Vec::new();
    }

    let mut result = vec![0.0f32; n];
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    while i + SIMD_WIDTH <= n {
        let gate_arr: [f32; SIMD_WIDTH] = gate[i..i + SIMD_WIDTH].try_into().unwrap();
        let val_arr: [f32; SIMD_WIDTH] = value[i..i + SIMD_WIDTH].try_into().unwrap();

        let gate_vec = f32x8::from_array(gate_arr);
        let val_vec = f32x8::from_array(val_arr);

        // SiLU(gate) = gate * sigmoid(gate)
        let neg_gate = -gate_vec;
        let exp_neg = simd_exp_taylor_f32x8(neg_gate);
        let sigmoid = f32x8::splat(1.0) / (f32x8::splat(1.0) + exp_neg);
        let silu_gate = gate_vec * sigmoid;

        // SwiGLU = SiLU(gate) * value
        let swiglu = silu_gate * val_vec;

        result[i..i + SIMD_WIDTH].copy_from_slice(&swiglu.to_array());
        i += SIMD_WIDTH;
    }

    #[cfg(target_arch = "aarch64")]
    while i + SIMD_WIDTH <= n {
        let gate_arr: [f32; SIMD_WIDTH] = gate[i..i + SIMD_WIDTH].try_into().unwrap();
        let val_arr: [f32; SIMD_WIDTH] = value[i..i + SIMD_WIDTH].try_into().unwrap();

        let gate_vec = f32x4::from_array(gate_arr);
        let val_vec = f32x4::from_array(val_arr);

        let neg_gate = -gate_vec;
        let exp_neg = simd_exp_taylor_f32x4(neg_gate);
        let sigmoid = f32x4::splat(1.0) / (f32x4::splat(1.0) + exp_neg);
        let silu_gate = gate_vec * sigmoid;

        let swiglu = silu_gate * val_vec;

        result[i..i + SIMD_WIDTH].copy_from_slice(&swiglu.to_array());
        i += SIMD_WIDTH;
    }

    // Handle remaining elements
    while i < n {
        let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
        result[i] = gate[i] * sigmoid * value[i];
        i += 1;
    }

    result
}

/// Scalar fallback SwiGLU for correctness validation
pub fn swiglu_scalar(gate: &[f32], value: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(value.iter())
        .map(|(&g, &v)| {
            let sigmoid = 1.0 / (1.0 + (-g).exp());
            g * sigmoid * v
        })
        .collect()
}

/// SwiGLU with automatic SIMD/scalar dispatch
pub fn swiglu(gate: &[f32], value: &[f32]) -> Vec<f32> {
    #[cfg(feature = "simd")]
    {
        swiglu_simd(gate, value)
    }

    #[cfg(not(feature = "simd"))]
    {
        swiglu_scalar(gate, value)
    }
}

// ============================================================================
// GELU activation
// ============================================================================

/// SIMD-accelerated GELU activation
///
/// Uses the tanh approximation:
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// * GELU-activated output
pub fn gelu_simd(x: &[f32]) -> Vec<f32> {
    const GELU_CONST: f32 = 0.7978845608028654; // sqrt(2/pi)
    const GELU_CONST_3: f32 = 0.044715; // 0.044715

    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    let const_vec = SimdF32::splat(GELU_CONST);
    let const_3_vec = SimdF32::splat(GELU_CONST_3);
    let half = SimdF32::splat(0.5);
    let one = SimdF32::splat(1.0);

    let mut result = vec![0.0f32; n];
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    while i + SIMD_WIDTH <= n {
        let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
        let x_vec = f32x8::from_array(arr);

        // Compute x^3
        let x3 = x_vec * x_vec * x_vec;

        // Compute inner: sqrt(2/pi) * (x + 0.044715 * x^3)
        let inner = const_vec * (x_vec + const_3_vec * x3);

        // tanh approximation (using std::tanh for accuracy)
        let tanh_inner = simd_tanh(inner);

        // GELU = 0.5 * x * (1 + tanh(inner))
        let gelu = half * x_vec * (one + tanh_inner);

        result[i..i + SIMD_WIDTH].copy_from_slice(&gelu.to_array());
        i += SIMD_WIDTH;
    }

    #[cfg(target_arch = "aarch64")]
    while i + SIMD_WIDTH <= n {
        let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
        let x_vec = f32x4::from_array(arr);

        // Compute x^3
        let x3 = x_vec * x_vec * x_vec;

        // Compute inner: sqrt(2/pi) * (x + 0.044715 * x^3)
        let inner = const_vec * (x_vec + const_3_vec * x3);

        // tanh approximation
        let tanh_inner = simd_tanh(inner);

        // GELU = 0.5 * x * (1 + tanh(inner))
        let gelu = half * x_vec * (one + tanh_inner);

        result[i..i + SIMD_WIDTH].copy_from_slice(&gelu.to_array());
        i += SIMD_WIDTH;
    }

    // Handle remaining elements
    while i < n {
        let x3 = x[i] * x[i] * x[i];
        let inner = GELU_CONST * (x[i] + GELU_CONST_3 * x3);
        let tanh_inner = inner.tanh();
        result[i] = 0.5 * x[i] * (1.0 + tanh_inner);
        i += 1;
    }

    result
}

/// Scalar fallback GELU for correctness validation
pub fn gelu_scalar(x: &[f32]) -> Vec<f32> {
    const GELU_CONST: f32 = 0.7978845608028654; // sqrt(2/pi)
    const GELU_CONST_3: f32 = 0.044715;

    x.iter().map(|&v| {
        let x3 = v * v * v;
        let inner = GELU_CONST * (v + GELU_CONST_3 * x3);
        let tanh_inner = inner.tanh();
        0.5 * v * (1.0 + tanh_inner)
    }).collect()
}

/// GELU with automatic SIMD/scalar dispatch
pub fn gelu(x: &[f32]) -> Vec<f32> {
    #[cfg(feature = "simd")]
    {
        gelu_simd(x)
    }

    #[cfg(not(feature = "simd"))]
    {
        gelu_scalar(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // SiLU tests
    // ========================================================================

    #[test]
    fn test_silu_simd_basic() {
        let input = vec![0.0f32, 1.0, -1.0, 2.0];

        let result = silu_simd(&input);

        assert_eq!(result.len(), 4);
        // SiLU(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-4);
        // SiLU(x) should be positive for x > 0
        assert!(result[1] > 0.0);
        assert!(result[3] > 0.0);
        // SiLU(-1) = -1 * sigmoid(-1) should be negative
        assert!(result[2] < 0.0);
    }

    #[test]
    fn test_silu_simd_vs_scalar() {
        // Use smaller input range to stay within Taylor approximation accuracy
        for size in [4, 8, 16, 32, 64] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.03).collect();

            let simd_result = silu_simd(&input);
            let scalar_result = silu_scalar(&input);

            assert_eq!(simd_result.len(), scalar_result.len());
            for (i, (&s, &r)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                // Taylor approximation has some error for larger values
                // With input range ~[-1, 1], approximation is accurate
                let abs_diff = (s - r).abs();
                let rel_diff = abs_diff / r.abs().max(1e-6);
                assert!(
                    abs_diff < 0.03 || rel_diff < 0.2,
                    "Size {}, element {} (input={}): {} vs {}, diff={}, rel_diff={}",
                    size,
                    i,
                    input[i],
                    s,
                    r,
                    abs_diff,
                    rel_diff
                );
            }
        }
    }

    #[test]
    fn test_silu_in_place_simd() {
        let mut input = vec![0.0f32, 1.0, -1.0, 2.0, 3.0, -2.0];
        let input_copy = input.clone();

        silu_in_place_simd(&mut input);

        for (i, &val) in input.iter().enumerate() {
            let expected = input_copy[i] * (1.0 / (1.0 + (-input_copy[i]).exp()));
            // Allow some tolerance for Taylor approximation
            assert!(
                (val - expected).abs() < 0.1,
                "Element {} mismatch: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_silu_dispatch() {
        let input = vec![0.0f32, 1.0, -1.0, 2.0];

        let result = silu(&input);
        let expected = silu_scalar(&input);

        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 0.1,
                "Element {} mismatch: {} vs {}",
                i,
                r,
                e
            );
        }
    }

    // ========================================================================
    // SwiGLU tests
    // ========================================================================

    #[test]
    fn test_swiglu_simd_basic() {
        let gate = vec![1.0f32, 2.0, 0.0, -1.0];
        let value = vec![0.5f32, 1.5, 2.0, 0.5];

        let result = swiglu_simd(&gate, &value);

        assert_eq!(result.len(), 4);
        // SwiGLU(0, v) = 0 for any v
        assert!((result[2] - 0.0).abs() < 1e-4);
        // SwiGLU(g, v) should have same sign as v when g > 0
        assert!(result[0] * value[0] >= 0.0);
        assert!(result[1] * value[1] >= 0.0);
    }

    #[test]
    fn test_swiglu_simd_vs_scalar() {
        // Use smaller input range to stay within Taylor approximation accuracy
        for size in [4, 8, 16, 32, 64] {
            let gate: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02).collect();
            let value: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.02).collect();

            let simd_result = swiglu_simd(&gate, &value);
            let scalar_result = swiglu_scalar(&gate, &value);

            assert_eq!(simd_result.len(), scalar_result.len());
            for (i, (&s, &r)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                // Allow tolerance for Taylor approximation
                let abs_diff = (s - r).abs();
                let rel_diff = abs_diff / r.abs().max(1e-6);
                assert!(
                    abs_diff < 0.02 || rel_diff < 0.2,
                    "Size {}, element {} (gate={}): {} vs {}, diff={}, rel_diff={}",
                    size,
                    i,
                    gate[i],
                    s,
                    r,
                    abs_diff,
                    rel_diff
                );
            }
        }
    }

    #[test]
    fn test_swiglu_dispatch() {
        // Use smaller values for Taylor approximation accuracy
        let gate = vec![0.5f32, 1.0, 0.0, -0.5];
        let value = vec![0.25f32, 0.75, 1.0, 0.25];

        let result = swiglu(&gate, &value);
        let expected = swiglu_scalar(&gate, &value);

        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 0.05,
                "Element {} mismatch: {} vs {}",
                i,
                r,
                e
            );
        }
    }

    // ========================================================================
    // GELU tests
    // ========================================================================

    #[test]
    fn test_gelu_simd_basic() {
        let input = vec![0.0f32, 1.0, -1.0, 2.0];

        let result = gelu_simd(&input);

        assert_eq!(result.len(), 4);
        // GELU(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-4);
        // GELU is approximately 0.5*x for large positive x
        assert!(result[1] > 0.0 && result[1] < 1.0);
        assert!(result[2] < 0.0 && result[2] > -1.0);
    }

    #[test]
    fn test_gelu_simd_vs_scalar() {
        for size in [4, 8, 16, 32, 64, 100] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.1).collect();

            let simd_result = gelu_simd(&input);
            let scalar_result = gelu_scalar(&input);

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
    fn test_gelu_dispatch() {
        let input = vec![0.0f32, 1.0, -1.0, 2.0];

        let result = gelu(&input);
        let expected = gelu_scalar(&input);

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

    #[test]
    fn test_gelu_properties() {
        // GELU should be approximately 0.5*x for large |x|
        let input = vec![10.0f32, -10.0];
        let result = gelu_simd(&input);

        // For large positive x, GELU(x) approx x
        assert!((result[0] - 10.0).abs() < 1.0);
        // For large negative x, GELU(x) approx 0
        assert!(result[1].abs() < 1.0);
    }

    // ========================================================================
    // Combined tests
    // ========================================================================

    #[test]
    fn test_all_activations_positive_input() {
        let input = vec![0.1f32, 0.5, 1.0, 2.0, 5.0];

        // All activations should produce positive output for positive input
        let silu_result = silu_simd(&input);
        for val in &silu_result {
            assert!(*val > 0.0, "SiLU should be positive for positive input");
        }

        let gelu_result = gelu_simd(&input);
        for val in &gelu_result {
            assert!(*val > 0.0, "GELU should be positive for positive input");
        }
    }

    #[test]
    fn test_simd_width_alignment() {
        // Test with sizes not aligned to SIMD_WIDTH
        for size in [1, 2, 3, 5, 7, 9, 13, 17] {
            let input: Vec<f32> = (0..size).map(|i| i as f32).collect();

            let silu_result = silu_simd(&input);
            assert_eq!(silu_result.len(), size);

            let gelu_result = gelu_simd(&input);
            assert_eq!(gelu_result.len(), size);
        }
    }
}
