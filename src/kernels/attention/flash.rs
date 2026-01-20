//! FlashAttention kernels
//!
//! Fused attention computation: QK^T -> scale -> (mask) -> softmax -> @V
//!
//! Provides both causal and non-causal variants of FlashAttention.

// ============================================================================
// FlashAttention Constants
// ============================================================================

/// Maximum head dimension supported by flash attention kernels
/// Due to register storage limits in the kernels
pub const MAX_FLASH_HEAD_DIM: usize = 128;

/// Maximum sequence length supported by flash attention kernels
/// Due to shared memory constraints
pub const MAX_FLASH_SEQ_LEN: usize = 2048;

// ============================================================================
// Re-exports from original flash_attention module
// ============================================================================

// Re-export FlashAttention backend
pub use crate::attention::flash_attention::FlashAttentionBackend;

// Re-export GPU kernels from attention::kernels
#[cfg(feature = "rocm")]
pub use crate::attention::kernels::{
    flash_attention_causal_gpu_kernel,
    flash_attention_gpu_kernel,
    flash_attention_nocausal_gpu_kernel,
    causal_mask_gpu_kernel,
};

// ============================================================================
// Causal mask generation (for use with FlashAttention)
// ============================================================================

/// Create a causal mask for FlashAttention
///
/// Creates a mask where mask[i, j] = -inf if j > i (future positions)
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

// ============================================================================
// Fallback functions
// ============================================================================

/// Check if FlashAttention can be used for the given configuration
pub fn can_use_flash_attention(head_dim: usize, seq_len: usize) -> bool {
    cfg!(feature = "rocm")
        && head_dim <= MAX_FLASH_HEAD_DIM
        && seq_len <= MAX_FLASH_SEQ_LEN
}

/// Check if FlashAttention supports causal masking
pub fn supports_causal_mask() -> bool {
    cfg!(feature = "rocm") // Causal kernel is available when ROCm is enabled
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_pattern() {
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
    fn test_can_use_flash_attention_valid() {
        assert!(can_use_flash_attention(64, 1024));
    }

    #[test]
    fn test_can_use_flash_attention_head_dim_too_large() {
        assert!(!can_use_flash_attention(129, 1024));
    }

    #[test]
    fn test_can_use_flash_attention_seq_len_too_large() {
        assert!(!can_use_flash_attention(64, 2049));
    }
}
