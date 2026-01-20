//! Masking operations for attention
//!
//! Provides causal mask generation and application.

// ============================================================================
// CPU Causal Mask
// ============================================================================

/// Create a causal mask for attention
///
/// Creates a mask where mask[i, j] = -inf if j > i (future positions)
/// This ensures each position can only attend to previous positions.
///
/// # Arguments
/// * `seq_len` - Sequence length
///
/// # Returns
/// * Flattened mask [seq_len * seq_len]
pub fn create_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Apply a mask to attention scores in-place
///
/// Adds mask values to scores (used for -inf masking)
///
/// # Arguments
/// * `scores` - Attention scores [batch_size * num_heads * seq_len * seq_len]
/// * `mask` - Mask values [seq_len * seq_len]
/// * `batch_size` - Batch size
/// * `num_heads` - Number of attention heads
/// * `seq_len` - Sequence length
pub fn apply_mask_in_place(
    scores: &mut [f32],
    mask: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
) {
    let seq_sq = seq_len * seq_len;

    for b in 0..batch_size {
        for h in 0..num_heads {
            let base = (b * num_heads + h) * seq_sq;
            for i in 0..seq_sq {
                if mask[i] == f32::NEG_INFINITY {
                    scores[base + i] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

// ============================================================================
// GPU mask kernel (re-exported from attention::kernels)
// ============================================================================

#[cfg(feature = "rocm")]
pub use crate::attention::kernels::mask_gpu_kernel;

// ============================================================================
// HIP backend mask op (re-exported for convenience)
// ============================================================================

#[cfg(feature = "rocm")]
pub use crate::ggml::hip_backend::ops::mask::mask as hip_mask_op;

// ============================================================================
// Fallback function
// ============================================================================

/// Apply mask with automatic CPU fallback
#[cfg(feature = "rocm")]
pub fn apply_mask_with_fallback(
    scores: &mut [f32],
    mask: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
) {
    // For now, use CPU implementation
    apply_mask_in_place(scores, mask, batch_size, num_heads, seq_len);
}

/// Apply mask with automatic CPU fallback (non-ROCm)
#[cfg(not(feature = "rocm"))]
pub fn apply_mask_with_fallback(
    scores: &mut [f32],
    mask: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
) {
    apply_mask_in_place(scores, mask, batch_size, num_heads, seq_len);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_causal_mask_pattern() {
        let mask = create_causal_mask(3);

        // Expected pattern:
        // [  0, -inf, -inf]
        // [  0,   0, -inf]
        // [  0,   0,   0]

        assert_eq!(mask[0], 0.0f32);
        assert_eq!(mask[1], f32::NEG_INFINITY);
        assert_eq!(mask[2], f32::NEG_INFINITY);
        assert_eq!(mask[3], 0.0f32);
        assert_eq!(mask[4], 0.0f32);
        assert_eq!(mask[5], f32::NEG_INFINITY);
        assert_eq!(mask[6], 0.0f32);
        assert_eq!(mask[7], 0.0f32);
        assert_eq!(mask[8], 0.0f32);
    }

    #[test]
    fn test_apply_mask_in_place() {
        let mut scores = vec![
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0,  // row 1
            7.0, 8.0, 9.0,  // row 2
        ];
        let mask = create_causal_mask(3);

        apply_mask_in_place(&mut scores, &mask, 1, 1, 3);

        // After masking:
        // [  1, -inf, -inf]
        // [  4,   5, -inf]
        // [  7,   8,   9]

        assert_eq!(scores[0], 1.0f32);
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        assert_eq!(scores[3], 4.0f32);
        assert_eq!(scores[4], 5.0f32);
        assert_eq!(scores[5], f32::NEG_INFINITY);
        assert_eq!(scores[6], 7.0f32);
        assert_eq!(scores[7], 8.0f32);
        assert_eq!(scores[8], 9.0f32);
    }

    #[test]
    fn test_apply_mask_batched() {
        let mut scores = vec![
            1.0, 2.0, 3.0,  // batch 0, row 0
            4.0, 5.0, 6.0,  // batch 0, row 1
            7.0, 8.0, 9.0,  // batch 1, row 0
            10.0, 11.0, 12.0,  // batch 1, row 1
        ];
        let mask = create_causal_mask(2); // For seq_len=2

        // Apply mask to batch_size=2, num_heads=1, seq_len=2
        // We need a 2x2 mask: [[0, -inf], [0, 0]]
        let mask_2x2 = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];

        apply_mask_in_place(&mut scores, &mask_2x2, 2, 1, 2);

        // First row of each batch should have -inf at position 1
        assert_eq!(scores[0], 1.0f32);
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], 4.0f32);
        assert_eq!(scores[3], 5.0f32);
        assert_eq!(scores[4], 7.0f32);
        assert_eq!(scores[5], f32::NEG_INFINITY);
        assert_eq!(scores[6], 10.0f32);
        assert_eq!(scores[7], 11.0f32);
    }
}
