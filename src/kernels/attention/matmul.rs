//! Attention matrix multiplication kernels
//!
//! Provides QK^T matmul and weighted matmul operations for attention computation.

use crate::attention::AttentionError;

// ============================================================================
// CPU QK^T Matmul
// ============================================================================

/// CPU implementation of Query-Key^T matrix multiplication
///
/// Computes attention scores: Q @ K^T
///
/// # Arguments
/// * `q` - Query tensor [batch_size, num_heads, seq_q, head_dim]
/// * `k` - Key tensor [batch_size, num_heads, head_dim, seq_k]
/// * `batch_size` - Batch size
/// * `seq_q` - Query sequence length
/// * `seq_k` - Key sequence length
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
///
/// # Returns
/// * Attention scores [batch_size, num_heads, seq_q, seq_k]
pub fn qkt_matmul_cpu(
    q: &[f32],
    k: &[f32],
    batch_size: usize,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    let expected_q = batch_size * num_heads * seq_q * head_dim;
    let expected_k = batch_size * num_heads * head_dim * seq_k;

    if q.len() != expected_q || k.len() != expected_k {
        return Err(AttentionError::ShapeMismatch(format!(
            "QK^T matmul shape mismatch: expected q={}, k={}, got q={}, k={}",
            expected_q, expected_k, q.len(), k.len()
        )));
    }

    let mut output = vec![0.0f32; batch_size * num_heads * seq_q * seq_k];

    for b in 0..batch_size {
        for h in 0..num_heads {
            let q_base = b * num_heads * seq_q * head_dim + h * seq_q * head_dim;
            let k_base = b * num_heads * head_dim * seq_k + h * head_dim * seq_k;
            let out_base = b * num_heads * seq_q * seq_k + h * seq_q * seq_k;

            for i in 0..seq_q {
                for j in 0..seq_k {
                    let mut sum = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = q_base + i * head_dim + d;
                        let k_idx = k_base + d * seq_k + j;
                        sum += q[q_idx] * k[k_idx];
                    }
                    output[out_base + i * seq_k + j] = sum;
                }
            }
        }
    }

    Ok(output)
}

// ============================================================================
// CPU Weighted Matmul (softmax weights @ V)
// ============================================================================

/// CPU implementation of weighted value multiplication
///
/// Computes: attention_weights @ V
///
/// # Arguments
/// * `weights` - Attention weights [batch_size, num_heads, seq_q, seq_k]
/// * `v` - Value tensor [batch_size, num_heads, seq_k, head_dim]
/// * `batch_size` - Batch size
/// * `seq_q` - Query sequence length
/// * `seq_k` - Key sequence length
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
///
/// # Returns
/// * Output tensor [batch_size, num_heads, seq_q, head_dim]
pub fn weighted_matmul_cpu(
    weights: &[f32],
    v: &[f32],
    batch_size: usize,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    let expected_weights = batch_size * num_heads * seq_q * seq_k;
    let expected_v = batch_size * num_heads * seq_k * head_dim;

    if weights.len() != expected_weights || v.len() != expected_v {
        return Err(AttentionError::ShapeMismatch(format!(
            "Weighted matmul shape mismatch: expected weights={}, v={}, got weights={}, v={}",
            expected_weights, expected_v, weights.len(), v.len()
        )));
    }

    let mut output = vec![0.0f32; batch_size * num_heads * seq_q * head_dim];

    for b in 0..batch_size {
        for h in 0..num_heads {
            let w_base = b * num_heads * seq_q * seq_k + h * seq_q * seq_k;
            let v_base = b * num_heads * seq_k * head_dim + h * seq_k * head_dim;
            let out_base = b * num_heads * seq_q * head_dim + h * seq_q * head_dim;

            for i in 0..seq_q {
                for j in 0..head_dim {
                    let mut sum = 0.0f32;
                    for k in 0..seq_k {
                        let w_idx = w_base + i * seq_k + k;
                        let v_idx = v_base + k * head_dim + j;
                        sum += weights[w_idx] * v[v_idx];
                    }
                    output[out_base + i * head_dim + j] = sum;
                }
            }
        }
    }

    Ok(output)
}

// ============================================================================
// GPU Kernels (re-exported from attention::kernels)
// ============================================================================

#[cfg(feature = "rocm")]
pub use crate::attention::kernels::{
    qkt_matmul_gpu_kernel,
    qkt_matmul_gpu_kernel_scaled,
    weighted_matmul_gpu_kernel,
};

// ============================================================================
// Fallback functions
// ============================================================================

/// QK^T matmul with automatic CPU fallback
#[cfg(feature = "rocm")]
pub fn qkt_matmul_with_fallback(
    q: &[f32],
    k: &[f32],
    batch_size: usize,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    // For now, use CPU implementation
    qkt_matmul_cpu(q, k, batch_size, seq_q, seq_k, num_heads, head_dim)
}

/// QK^T matmul with automatic CPU fallback (non-ROCm)
#[cfg(not(feature = "rocm"))]
pub fn qkt_matmul_with_fallback(
    q: &[f32],
    k: &[f32],
    batch_size: usize,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    qkt_matmul_cpu(q, k, batch_size, seq_q, seq_k, num_heads, head_dim)
}

/// Weighted matmul with automatic CPU fallback
#[cfg(feature = "rocm")]
pub fn weighted_matmul_with_fallback(
    weights: &[f32],
    v: &[f32],
    batch_size: usize,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    // For now, use CPU implementation
    weighted_matmul_cpu(weights, v, batch_size, seq_q, seq_k, num_heads, head_dim)
}

/// Weighted matmul with automatic CPU fallback (non-ROCm)
#[cfg(not(feature = "rocm"))]
pub fn weighted_matmul_with_fallback(
    weights: &[f32],
    v: &[f32],
    batch_size: usize,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    weighted_matmul_cpu(weights, v, batch_size, seq_q, seq_k, num_heads, head_dim)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qkt_matmul_cpu_basic() {
        // batch_size=1, seq_q=2, seq_k=3, num_heads=1, head_dim=4
        let q = vec![
            1.0, 2.0, 3.0, 4.0,  // seq_q=0
            5.0, 6.0, 7.0, 8.0,  // seq_q=1
        ];
        let k = vec![
            1.0, 0.0, 0.0, 0.0,  // seq_k=0
            0.0, 1.0, 0.0, 0.0,  // seq_k=1
            0.0, 0.0, 1.0, 0.0,  // seq_k=2
        ];

        let result = qkt_matmul_cpu(&q, &k, 1, 2, 3, 1, 4).unwrap();

        // Expected: Q @ K^T
        // [1, 2, 3, 4] @ [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]^T
        // = [1, 2, 3] for first row
        assert_eq!(result.len(), 6); // 1 * 1 * 2 * 3
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_matmul_cpu_basic() {
        // batch_size=1, seq_q=2, seq_k=3, num_heads=1, head_dim=4
        let weights = vec![
            1.0, 0.0, 0.0,  // seq_q=0 weights
            0.0, 1.0, 0.0,  // seq_q=1 weights
        ];
        let v = vec![
            1.0, 2.0, 3.0, 4.0,  // seq_k=0
            5.0, 6.0, 7.0, 8.0,  // seq_k=1
            9.0, 10.0, 11.0, 12.0,  // seq_k=2
        ];

        let result = weighted_matmul_cpu(&weights, &v, 1, 2, 3, 1, 4).unwrap();

        // First row selects only seq_k=0: [1, 2, 3, 4]
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 4.0).abs() < 1e-6);

        // Second row selects only seq_k=1: [5, 6, 7, 8]
        assert!((result[4] - 5.0).abs() < 1e-6);
        assert!((result[5] - 6.0).abs() < 1e-6);
        assert!((result[6] - 7.0).abs() < 1e-6);
        assert!((result[7] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_qkt_matmul_shape_mismatch() {
        let q = vec![1.0f32; 10];
        let k = vec![1.0f32; 15]; // Wrong size

        let result = qkt_matmul_cpu(&q, &k, 1, 2, 3, 1, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_matmul_shape_mismatch() {
        let weights = vec![1.0f32; 10];
        let v = vec![1.0f32; 15]; // Wrong size

        let result = weighted_matmul_cpu(&weights, &v, 1, 2, 3, 1, 4);
        assert!(result.is_err());
    }
}
