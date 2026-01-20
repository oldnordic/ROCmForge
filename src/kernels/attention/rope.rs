//! Rotary Positional Embeddings (RoPE) kernels
//!
//! Applies relative position encoding to query and key tensors using rotation matrices.

// ============================================================================
// Re-exports from original rope module
// ============================================================================

// Re-export RoPE types and implementation
pub use crate::attention::rope::{
    Rope,
    RopeConfig,
};

// Re-export GPU kernel from attention::kernels
#[cfg(feature = "rocm")]
pub use crate::attention::kernels::rope_gpu_kernel;

// ============================================================================
// HIP backend RoPE op (re-exported for convenience)
// ============================================================================

#[cfg(feature = "rocm")]
pub use crate::ggml::hip_backend::ops::rope::rope as hip_rope_op;

// ============================================================================
// Fallback function
// ============================================================================

/// Apply RoPE with automatic CPU fallback
#[cfg(feature = "rocm")]
pub fn rope_with_fallback(
    x: &mut [f32],
    position_ids: &[usize],
    num_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), crate::attention::AttentionError> {
    // Create a minimal Rope config for CPU fallback
    let max_seq_len = position_ids.iter().max().copied().unwrap_or(0) + 1;
    let config = RopeConfig::new(head_dim, max_seq_len);

    // Create RoPE instance
    let rope = Rope::new(config);

    // Apply RoPE (CPU implementation)
    rope.apply_q(x, position_ids, num_heads)
}

/// Apply RoPE with automatic CPU fallback (non-ROCm)
#[cfg(not(feature = "rocm"))]
pub fn rope_with_fallback(
    x: &mut [f32],
    position_ids: &[usize],
    num_heads: usize,
    head_dim: usize,
    _cos: &[f32],
    _sin: &[f32],
) -> Result<(), crate::attention::AttentionError> {
    // Create a minimal Rope config for CPU fallback
    let max_seq_len = position_ids.iter().max().copied().unwrap_or(0) + 1;
    let config = RopeConfig::new(head_dim, max_seq_len);

    // Create RoPE instance
    let rope = Rope::new(config);

    // Apply RoPE (CPU implementation)
    rope.apply_q(x, position_ids, num_heads)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let config = RopeConfig::new(128, 1024);
        let rope = Rope::new(config);

        assert_eq!(rope.cos().len(), 1024 * 64); // max_seq_len * head_dim/2
        assert_eq!(rope.sin().len(), 1024 * 64);
    }

    #[test]
    fn test_rope_application() {
        let config = RopeConfig::new(4, 8); // Small dimensions for testing
        let rope = Rope::new(config);

        // Create test tensor: [batch=1, seq_len=2, num_heads=1, head_dim=4]
        let mut x = vec![
            1.0, 2.0, 3.0, 4.0,  // position 0 (identity - no change)
            5.0, 6.0, 7.0, 8.0,  // position 1 (should rotate)
        ];
        let position_ids = vec![0, 1];

        rope.apply_q(&mut x, &position_ids, 1).unwrap();

        // Position 0 is identity (cos=1, sin=0), so values don't change
        assert_eq!(x[0], 1.0); // Position 0: no rotation
        assert_eq!(x[1], 2.0); // Position 0: no rotation

        // Position 1 should have rotation applied
        assert_ne!(x[4], 5.0); // Position 1: rotated
        assert_ne!(x[5], 6.0); // Position 1: rotated
    }

    #[test]
    fn test_rope_odd_head_dim() {
        let config = RopeConfig::new(5, 8); // Odd dimension should fail
        let result = std::panic::catch_unwind(|| {
            Rope::new(config);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_position_id_overflow() {
        let config = RopeConfig::new(4, 4); // max_seq_len = 4
        let rope = Rope::new(config);

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let position_ids = vec![5]; // Position 5 exceeds max_seq_len

        let result = rope.apply_q(&mut x, &position_ids, 1);
        assert!(result.is_err());
    }
}
