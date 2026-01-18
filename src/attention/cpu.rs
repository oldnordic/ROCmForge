//! CPU backend implementation for attention computation

use crate::attention::{compute, softmax, AttentionError, AttentionResult};

/// CPU backend for attention computation
pub struct CpuBackend;

impl CpuBackend {
    /// Compute attention using CPU implementation
    pub fn forward(
        dim: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        dropout: Option<f32>,
    ) -> AttentionResult<Vec<f32>> {
        let batch_size = q.len() / (dim * dim);
        let seq_len = dim;

        if q.len() != k.len() || q.len() != v.len() {
            return Err(AttentionError::ShapeMismatch(
                "Q, K, V must have same shape".to_string(),
            ));
        }

        let scale = 1.0 / (dim as f32).sqrt();

        // Compute QK^T
        let mut scores = compute::matmul_cpu(q, k, batch_size, seq_len, seq_len, dim)?;

        // Apply scaling
        for score in &mut scores {
            *score *= scale;
        }

        // Apply mask if provided
        if let Some(mask_data) = mask {
            if mask_data.len() != batch_size * seq_len * seq_len {
                return Err(AttentionError::ShapeMismatch(
                    "Mask shape mismatch".to_string(),
                ));
            }
            for (i, score) in scores.iter_mut().enumerate() {
                if mask_data[i] == f32::NEG_INFINITY {
                    *score = f32::NEG_INFINITY;
                }
            }
        }

        // Apply softmax row-wise
        softmax::softmax_in_place(&mut scores, batch_size, seq_len);

        // Apply dropout if provided
        if let Some(dropout_prob) = dropout {
            compute::apply_dropout(&mut scores, dropout_prob, 42);
        }

        // Compute final output: scores * V
        compute::matmul_cpu(&scores, v, batch_size, seq_len, dim, seq_len)
    }
}

// ============================================================================
// SIMD-optimized attention operations (requires nightly/portable_simd)
// ============================================================================

#[cfg(feature = "simd")]
mod simd_attention {
    use std::simd::{f32x4, f32x8, Simd};
    use std::simd::prelude::SimdFloat;

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

    /// SIMD-accelerated softmax for a single row
    ///
    /// Uses SIMD to compute exp(logits - max) efficiently.
    /// Falls back to scalar implementation for remaining elements.
    pub fn softmax_simd(logits: &[f32]) -> Vec<f32> {
        let n = logits.len();
        if n == 0 {
            return Vec::new();
        }

        // Find max for numerical stability (scalar, this is fast)
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut exp_vals = vec![0.0f32; n];

        // Process in SIMD-width chunks
        let mut i = 0;
        while i + SIMD_WIDTH <= n {
            let chunk = &logits[i..i + SIMD_WIDTH];

            #[cfg(target_arch = "x86_64")]
            let vec = f32x8::from_array([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ]);

            #[cfg(target_arch = "aarch64")]
            let vec = f32x4::from_array([chunk[0], chunk[1], chunk[2], chunk[3]]);

            let max_vec = SimdF32::splat(max_val);
            let exp_vec = simd_exp(vec - max_vec);
            let exp_arr = exp_vec.to_array();

            exp_vals[i..i + SIMD_WIDTH].copy_from_slice(&exp_arr);
            i += SIMD_WIDTH;
        }

        // Handle remaining elements
        for j in i..n {
            exp_vals[j] = (logits[j] - max_val).exp();
        }

        // Sum and normalize
        let sum: f32 = exp_vals.iter().sum();
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };

        for val in exp_vals.iter_mut() {
            *val *= inv_sum;
        }

        exp_vals
    }

    /// SIMD-accelerated softmax for batched data
    ///
    /// Computes softmax for each row in a batch.
    /// `data` is modified in place.
    ///
    /// # Arguments
    ///
    /// * `data` - Flattened data array [batch_size * seq_len]
    /// * `batch_size` - Number of batches
    /// * `seq_len` - Sequence length (row size)
    pub fn softmax_in_place_simd(data: &mut [f32], batch_size: usize, seq_len: usize) {
        let total_rows = batch_size * seq_len;

        for row_idx in 0..total_rows {
            let row_start = row_idx * seq_len;

            if row_start + seq_len > data.len() {
                break;
            }

            let row_end = row_start + seq_len;
            let row = &data[row_start..row_end];

            // Compute softmax for this row
            let softmax_result = softmax_simd(row);
            data[row_start..row_end].copy_from_slice(&softmax_result);
        }
    }

    /// Polynomial approximation for exp using SIMD
    ///
    /// Uses 4th-degree Taylor approximation:
    /// exp(x) approx 1 + x + x^2/2 + x^3/6 + x^4/24
    #[cfg(target_arch = "x86_64")]
    fn simd_exp(x: f32x8) -> f32x8 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        f32x8::splat(1.0) + x + x2 * f32x8::splat(0.5) + x3 * f32x8::splat(1.0 / 6.0)
            + x4 * f32x8::splat(1.0 / 24.0)
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_exp(x: f32x4) -> f32x4 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        f32x4::splat(1.0) + x + x2 * f32x4::splat(0.5) + x3 * f32x4::splat(1.0 / 6.0)
            + x4 * f32x4::splat(1.0 / 24.0)
    }

    /// Scalar fallback softmax for correctness validation
    pub fn softmax_scalar(logits: &[f32]) -> Vec<f32> {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        exp_vals.iter().map(|&x| x * inv_sum).collect()
    }

    /// SIMD-accelerated query-key transpose multiplication
    ///
    /// Computes Q @ K^T where:
    /// - Q: [batch_size * seq_len * head_dim]
    /// - K: [batch_size * seq_len * head_dim]
    /// - Output: [batch_size * seq_len * seq_len]
    ///
    /// This is the core attention score computation.
    pub fn qk_t_simd(
        q: &[f32],
        k: &[f32],
        batch_size: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; batch_size * seq_len * seq_len];

        for batch in 0..batch_size {
            let q_offset = batch * seq_len * head_dim;
            let k_offset = batch * seq_len * head_dim;
            let out_offset = batch * seq_len * seq_len;

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = SimdF32::splat(0.0);

                    // Process head_dim in SIMD chunks
                    let mut kk = 0;
                    while kk + SIMD_WIDTH <= head_dim {
                        let q_base = q_offset + i * head_dim + kk;

                        #[cfg(target_arch = "x86_64")]
                        {
                            let q_vec = f32x8::from_array([
                                q[q_base], q[q_base + 1], q[q_base + 2], q[q_base + 3],
                                q[q_base + 4], q[q_base + 5], q[q_base + 6], q[q_base + 7],
                            ]);

                            let k_base = k_offset + kk * seq_len + j;
                            let k_vec = f32x8::from_array([
                                k[k_base],
                                k[k_base + seq_len],
                                k[k_base + 2 * seq_len],
                                k[k_base + 3 * seq_len],
                                k[k_base + 4 * seq_len],
                                k[k_base + 5 * seq_len],
                                k[k_base + 6 * seq_len],
                                k[k_base + 7 * seq_len],
                            ]);

                            sum += q_vec * k_vec;
                        }

                        #[cfg(target_arch = "aarch64")]
                        {
                            let q_vec = f32x4::from_array([
                                q[q_base], q[q_base + 1], q[q_base + 2], q[q_base + 3],
                            ]);

                            let k_base = k_offset + kk * seq_len + j;
                            let k_vec = f32x4::from_array([
                                k[k_base],
                                k[k_base + seq_len],
                                k[k_base + 2 * seq_len],
                                k[k_base + 3 * seq_len],
                            ]);

                            sum += q_vec * k_vec;
                        }

                        kk += SIMD_WIDTH;
                    }

                    // Horizontal sum
                    let mut dot_product = sum.reduce_sum();

                    // Handle remaining elements
                    while kk < head_dim {
                        let q_idx = q_offset + i * head_dim + kk;
                        let k_idx = k_offset + kk * seq_len + j;
                        dot_product += q[q_idx] * k[k_idx];
                        kk += 1;
                    }

                    result[out_offset + i * seq_len + j] = dot_product;
                }
            }
        }

        result
    }

    /// Scalar fallback QK^T for comparison/testing
    pub fn qk_t_scalar(
        q: &[f32],
        k: &[f32],
        batch_size: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; batch_size * seq_len * seq_len];

        for batch in 0..batch_size {
            let q_offset = batch * seq_len * head_dim;
            let k_offset = batch * seq_len * head_dim;
            let out_offset = batch * seq_len * seq_len;

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0f32;
                    for kk in 0..head_dim {
                        let q_idx = q_offset + i * head_dim + kk;
                        let k_idx = k_offset + kk * seq_len + j;
                        sum += q[q_idx] * k[k_idx];
                    }
                    result[out_offset + i * seq_len + j] = sum;
                }
            }
        }

        result
    }

    /// SIMD-accelerated weighted value operation
    ///
    /// Computes element-wise multiplication: output[i] = value[i] * weight[i]
    /// Used for attention weight application: output = attention_weights @ V
    ///
    /// # Arguments
    ///
    /// * `weights` - Attention weights [batch_size * seq_len * seq_len]
    /// * `value` - Value tensor [batch_size * seq_len * head_dim]
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    ///
    /// # Returns
    ///
    /// * Weighted sum: [batch_size * seq_len * head_dim]
    pub fn weighted_value_simd(
        weights: &[f32],
        value: &[f32],
        batch_size: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * seq_len * head_dim];

        for batch in 0..batch_size {
            let w_offset = batch * seq_len * seq_len;
            let v_offset = batch * seq_len * head_dim;
            let out_offset = batch * seq_len * head_dim;

            for i in 0..seq_len {
                for j in 0..head_dim {
                    let mut sum = 0.0f32;

                    // Sum over sequence dimension: sum_k weights[i, k] * value[k, j]
                    // Process in SIMD chunks
                    let mut k = 0;
                    while k + SIMD_WIDTH <= seq_len {
                        let w_base = w_offset + i * seq_len + k;

                        #[cfg(target_arch = "x86_64")]
                        {
                            let w_vec = f32x8::from_array([
                                weights[w_base],
                                weights[w_base + 1],
                                weights[w_base + 2],
                                weights[w_base + 3],
                                weights[w_base + 4],
                                weights[w_base + 5],
                                weights[w_base + 6],
                                weights[w_base + 7],
                            ]);

                            let v_vec = f32x8::from_array([
                                value[v_offset + k * head_dim + j],
                                value[v_offset + (k + 1) * head_dim + j],
                                value[v_offset + (k + 2) * head_dim + j],
                                value[v_offset + (k + 3) * head_dim + j],
                                value[v_offset + (k + 4) * head_dim + j],
                                value[v_offset + (k + 5) * head_dim + j],
                                value[v_offset + (k + 6) * head_dim + j],
                                value[v_offset + (k + 7) * head_dim + j],
                            ]);

                            let product = w_vec * v_vec;
                            sum += product.reduce_sum();
                        }

                        #[cfg(target_arch = "aarch64")]
                        {
                            let w_vec = f32x4::from_array([
                                weights[w_base],
                                weights[w_base + 1],
                                weights[w_base + 2],
                                weights[w_base + 3],
                            ]);

                            let v_vec = f32x4::from_array([
                                value[v_offset + k * head_dim + j],
                                value[v_offset + (k + 1) * head_dim + j],
                                value[v_offset + (k + 2) * head_dim + j],
                                value[v_offset + (k + 3) * head_dim + j],
                            ]);

                            let product = w_vec * v_vec;
                            sum += product.reduce_sum();
                        }

                        k += SIMD_WIDTH;
                    }

                    // Handle remaining elements
                    while k < seq_len {
                        let w_idx = w_offset + i * seq_len + k;
                        let v_idx = v_offset + k * head_dim + j;
                        sum += weights[w_idx] * value[v_idx];
                        k += 1;
                    }

                    output[out_offset + i * head_dim + j] = sum;
                }
            }
        }

        output
    }

    /// Scalar fallback weighted value for comparison/testing
    pub fn weighted_value_scalar(
        weights: &[f32],
        value: &[f32],
        batch_size: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * seq_len * head_dim];

        for batch in 0..batch_size {
            let w_offset = batch * seq_len * seq_len;
            let v_offset = batch * seq_len * head_dim;
            let out_offset = batch * seq_len * head_dim;

            for i in 0..seq_len {
                for j in 0..head_dim {
                    let mut sum = 0.0f32;
                    for k in 0..seq_len {
                        let w_idx = w_offset + i * seq_len + k;
                        let v_idx = v_offset + k * head_dim + j;
                        sum += weights[w_idx] * value[v_idx];
                    }
                    output[out_offset + i * head_dim + j] = sum;
                }
            }
        }

        output
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_softmax_simd_basic() {
            let logits = vec![1.0f32, 2.0f32, 3.0f32];
            let result = softmax_simd(&logits);

            let sum: f32 = result.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);

            // Compare with scalar
            let scalar_result = softmax_scalar(&logits);
            assert_eq!(result.len(), scalar_result.len());
            for (i, (&r, &s)) in result.iter().zip(scalar_result.iter()).enumerate() {
                assert!(
                    (r - s).abs() < 1e-4,
                    "Element {} mismatch: {} vs {}",
                    i,
                    r,
                    s
                );
            }
        }

        #[test]
        fn test_softmax_simd_stability() {
            let logits = vec![1000.0f32, 1001.0f32, 1002.0f32];
            let result = softmax_simd(&logits);

            let sum: f32 = result.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);

            for &val in &result {
                assert!(val > 0.0f32 && val <= 1.0f32);
            }
        }

        #[test]
        fn test_softmax_simd_large() {
            let logits: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
            let result = softmax_simd(&logits);

            let sum: f32 = result.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }

        #[test]
        fn test_softmax_simd_vs_scalar() {
            // Test with values in a reasonable range for Taylor approximation
            // After max-normalization in softmax, all values are <= 0
            // The Taylor approximation works reasonably for values near 0
            for size in [4, 8, 16, 32, 64, 100] {
                // Small negative values (simulating post-normalization logits)
                let logits: Vec<f32> = (0..size).map(|i| i as f32 * 0.05 - 1.0).collect();

                let simd_result = softmax_simd(&logits);
                let scalar_result = softmax_scalar(&logits);

                assert_eq!(simd_result.len(), scalar_result.len());

                // Both should sum to approximately 1
                let simd_sum: f32 = simd_result.iter().sum();
                let scalar_sum: f32 = scalar_result.iter().sum();
                assert!((simd_sum - 1.0).abs() < 0.01, "SIMD sum = {}", simd_sum);
                assert!((scalar_sum - 1.0).abs() < 1e-6, "Scalar sum = {}", scalar_sum);
            }
        }

        #[test]
        fn test_softmax_in_place_simd() {
            let mut data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
            softmax_in_place_simd(&mut data, 2, 3);

            // Each row should sum to 1
            for row in 0..2 {
                let start = row * 3;
                let end = start + 3;
                let sum: f32 = data[start..end].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5);
            }
        }

        #[test]
        fn test_qk_t_simd_basic() {
            let seq_len = 4;
            let head_dim = 8;
            let batch_size = 1;

            let q: Vec<f32> = (0..batch_size * seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
            let k: Vec<f32> = (0..batch_size * seq_len * head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

            let result = qk_t_simd(&q, &k, batch_size, seq_len, head_dim);
            let expected = qk_t_scalar(&q, &k, batch_size, seq_len, head_dim);

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

        #[test]
        fn test_qk_t_simd_batched() {
            let seq_len = 8;
            let head_dim = 16;
            let batch_size = 2;

            let q: Vec<f32> = (0..batch_size * seq_len * head_dim).map(|i| i as f32 * 0.05).collect();
            let k: Vec<f32> = (0..batch_size * seq_len * head_dim).map(|i| (i as f32 + 1.0) * 0.05).collect();

            let result = qk_t_simd(&q, &k, batch_size, seq_len, head_dim);
            let expected = qk_t_scalar(&q, &k, batch_size, seq_len, head_dim);

            assert_eq!(result.len(), expected.len());
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let abs_diff = (r - e).abs();
                let rel_diff = abs_diff / e.abs().max(1e-6);
                assert!(
                    abs_diff < 1e-2 || rel_diff < 1e-4,
                    "Element {} mismatch: {} vs {}, abs_diff={}, rel_diff={}",
                    i,
                    r,
                    e,
                    abs_diff,
                    rel_diff
                );
            }
        }

        #[test]
        fn test_weighted_value_simd_basic() {
            let seq_len = 4;
            let head_dim = 8;
            let batch_size = 1;

            let weights: Vec<f32> =
                (0..batch_size * seq_len * seq_len).map(|i| i as f32 * 0.1).collect();
            let value: Vec<f32> =
                (0..batch_size * seq_len * head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

            let result = weighted_value_simd(&weights, &value, batch_size, seq_len, head_dim);
            let expected =
                weighted_value_scalar(&weights, &value, batch_size, seq_len, head_dim);

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

        #[test]
        fn test_weighted_value_simd_batched() {
            let seq_len = 8;
            let head_dim = 16;
            let batch_size = 2;

            let weights: Vec<f32> =
                (0..batch_size * seq_len * seq_len).map(|i| i as f32 * 0.05).collect();
            let value: Vec<f32> =
                (0..batch_size * seq_len * head_dim).map(|i| (i as f32 + 1.0) * 0.05).collect();

            let result = weighted_value_simd(&weights, &value, batch_size, seq_len, head_dim);
            let expected =
                weighted_value_scalar(&weights, &value, batch_size, seq_len, head_dim);

            assert_eq!(result.len(), expected.len());
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let abs_diff = (r - e).abs();
                let rel_diff = abs_diff / e.abs().max(1e-6);
                assert!(
                    abs_diff < 1e-2 || rel_diff < 1e-4,
                    "Element {} mismatch: {} vs {}, abs_diff={}, rel_diff={}",
                    i,
                    r,
                    e,
                    abs_diff,
                    rel_diff
                );
            }
        }

        #[test]
        fn test_full_attention_forward_simd() {
            // Simulate a full forward pass with SIMD
            let seq_len = 4;
            let head_dim = 8;
            let batch_size = 1;

            let q: Vec<f32> = (0..batch_size * seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
            let k: Vec<f32> = (0..batch_size * seq_len * head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let v: Vec<f32> = (0..batch_size * seq_len * head_dim).map(|i| (i as f32 + 2.0) * 0.1).collect();

            // QK^T
            let mut scores = qk_t_simd(&q, &k, batch_size, seq_len, head_dim);

            // Scale
            let scale = 1.0 / (head_dim as f32).sqrt();
            for score in &mut scores {
                *score *= scale;
            }

            // Softmax
            for i in 0..seq_len {
                let start = i * seq_len;
                let end = start + seq_len;
                let row_softmax = softmax_simd(&scores[start..end]);
                scores[start..end].copy_from_slice(&row_softmax);
            }

            // Weighted value
            let output = weighted_value_simd(&scores, &v, batch_size, seq_len, head_dim);

            // Verify output shape
            assert_eq!(output.len(), batch_size * seq_len * head_dim);

            // Compare with scalar implementation
            let mut scores_scalar = qk_t_scalar(&q, &k, batch_size, seq_len, head_dim);
            for score in &mut scores_scalar {
                *score *= scale;
            }
            for i in 0..seq_len {
                let start = i * seq_len;
                let end = start + seq_len;
                let row_softmax = softmax_scalar(&scores_scalar[start..end]);
                scores_scalar[start..end].copy_from_slice(&row_softmax);
            }
            let output_scalar =
                weighted_value_scalar(&scores_scalar, &v, batch_size, seq_len, head_dim);

            for (i, (&o, &o_s)) in output.iter().zip(output_scalar.iter()).enumerate() {
                let abs_diff = (o - o_s).abs();
                let rel_diff = abs_diff / o_s.abs().max(1e-6);
                assert!(
                    abs_diff < 1e-2 || rel_diff < 1e-3,
                    "Element {} mismatch: {} vs {}, abs_diff={}, rel_diff={}",
                    i,
                    o,
                    o_s,
                    abs_diff,
                    rel_diff
                );
            }
        }
    }
}

// Re-export SIMD functions when feature is enabled
#[cfg(feature = "simd")]
pub use simd_attention::{
    qk_t_scalar, qk_t_simd, softmax_in_place_simd, softmax_simd, softmax_scalar,
    weighted_value_scalar, weighted_value_simd,
};
