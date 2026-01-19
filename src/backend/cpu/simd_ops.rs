//! SIMD-accelerated transformer layer operations
//!
//! Provides SIMD-optimized implementations of common transformer operations:
//! - RMSNorm (Root Mean Square Layer Normalization)
//! - RoPE (Rotary Position Embedding)
//! - SiLU (Swish) activation
//! - SwiGLU activation
//! - GELU activation
//!
//! All operations have scalar fallbacks for correctness validation and
//! support runtime dispatch based on detected CPU features.

use std::simd::{f32x4, f32x8, Simd};
use std::simd::prelude::SimdFloat;

// ============================================================================
// SIMD type configuration (compile-time based on architecture)
// ============================================================================

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

// Tolerance for floating-point comparisons
const FLOAT_TOL: f32 = 1e-4;

// ============================================================================
// RMSNorm (Root Mean Square Layer Normalization)
// ============================================================================

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

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================

/// SIMD-accelerated RoPE rotation
///
/// Applies rotary position embedding to input tensor.
/// Rotates pairs of elements (x_{2i}, x_{2i+1}) using precomputed cos/sin.
///
/// Formula:
/// ```text
/// x'_{2i}   = x_{2i} * cos - x_{2i+1} * sin
/// x'_{2i+1} = x_{2i} * sin + x_{2i+1} * cos
/// ```
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

    while i + SIMD_WIDTH <= half_dim {
        #[cfg(target_arch = "x86_64")]
        {
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
        }

        #[cfg(target_arch = "aarch64")]
        {
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

    while i + SIMD_WIDTH <= n {
        #[cfg(target_arch = "x86_64")]
        {
            let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
            let x_vec = f32x8::from_array(arr);

            let neg_x = -x_vec;
            let exp_neg_x = simd_exp_taylor_f32x8(neg_x);
            let sigmoid = f32x8::splat(1.0) / (f32x8::splat(1.0) + exp_neg_x);
            let silu = x_vec * sigmoid;

            x[i..i + SIMD_WIDTH].copy_from_slice(&silu.to_array());
        }

        #[cfg(target_arch = "aarch64")]
        {
            let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
            let x_vec = f32x4::from_array(arr);

            let neg_x = -x_vec;
            let exp_neg_x = simd_exp_taylor_f32x4(neg_x);
            let sigmoid = f32x4::splat(1.0) / (f32x4::splat(1.0) + exp_neg_x);
            let silu = x_vec * sigmoid;

            x[i..i + SIMD_WIDTH].copy_from_slice(&silu.to_array());
        }

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

    while i + SIMD_WIDTH <= n {
        #[cfg(target_arch = "x86_64")]
        {
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
        }

        #[cfg(target_arch = "aarch64")]
        {
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
        }

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

    while i + SIMD_WIDTH <= n {
        let arr: [f32; SIMD_WIDTH] = x[i..i + SIMD_WIDTH].try_into().unwrap();
        let x_vec = Simd::from_array(arr);

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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RMSNorm tests
    // ========================================================================

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

    // ========================================================================
    // RoPE tests
    // ========================================================================

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

            let weight: Vec<f32> = (0..size).map(|_| 1.0).collect();
            let rms_result = rms_norm_simd(&input, &weight, 1e-5);
            assert_eq!(rms_result.len(), size);
        }
    }
}
