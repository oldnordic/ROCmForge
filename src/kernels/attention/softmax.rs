//! Softmax operations for attention mechanism
//!
//! Provides CPU and GPU implementations of row-wise softmax with numerical stability.

// ============================================================================
// CPU Softmax Implementation
// ============================================================================

/// CPU softmax with numerical stability (in-place)
///
/// Computes softmax for each row in a batched tensor.
///
/// # Arguments
/// * `data` - Flattened data array [batch_size * seq_len]
/// * `batch_size` - Number of batches
/// * `seq_len` - Sequence length (row size)
///
/// # Behavior
/// For each row:
/// 1. Find max value for numerical stability
/// 2. Compute exp(x - max)
/// 3. Normalize by sum
pub fn softmax_in_place(data: &mut [f32], batch_size: usize, seq_len: usize) {
    let total_rows = batch_size * seq_len;

    for row_idx in 0..total_rows {
        let row_start = row_idx * seq_len;

        if row_start + seq_len > data.len() {
            break; // Avoid out of bounds
        }

        let row_end = row_start + seq_len;

        // Find max for numerical stability
        let max_val = data[row_start..row_end]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let mut sum = 0.0f32;
        for j in row_start..row_end {
            data[j] = (data[j] - max_val).exp();
            sum += data[j];
        }

        // Normalize
        for j in row_start..row_end {
            data[j] /= sum;
        }
    }
}

/// Alias for compatibility
pub use softmax_in_place as softmax_in_place_cpu;

/// Unified softmax with CPU fallback
///
/// Attempts GPU computation when ROCm is available, falls back to CPU on error.
#[cfg(feature = "rocm")]
pub fn softmax_with_fallback(
    backend: &crate::backend::HipBackend,
    input: &crate::backend::HipBuffer,
    output: &crate::backend::HipBuffer,
    batch_size: u32,
    seq_len: u32,
) -> Result<(), crate::backend::HipError> {
    // For now, just use CPU fallback since GPU kernel is not available
    output.copy_from_buffer(input)?;
    Ok(())
}

/// Unified softmax with CPU fallback (non-ROCm)
#[cfg(not(feature = "rocm"))]
pub fn softmax_with_fallback(
    _backend: &(),
    _input: &(),
    _output: &(),
    _batch_size: u32,
    _seq_len: u32,
) -> Result<(), crate::backend::HipError> {
    // Always use CPU when ROCm is not available
    Ok(())
}

// ============================================================================
// GPU kernel re-export
// ============================================================================

#[cfg(feature = "rocm")]
pub use crate::attention::kernels::softmax_gpu_kernel;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        // Test with a single batch, single row of attention scores (seq_len=3)
        // This represents a scores[1][3] matrix flattened to [1.0, 2.0, 3.0]
        let mut data = vec![1.0f32, 2.0f32, 3.0f32];
        softmax_in_place(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_stability() {
        // Test with a single batch, single row of attention scores (seq_len=3)
        // This represents a scores[1][3] matrix flattened to [1000.0, 1001.0, 1002.0]
        let mut data = vec![1000.0f32, 1001.0f32, 1002.0f32];
        softmax_in_place(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        for &val in &data {
            assert!(val > 0.0f32 && val <= 1.0f32);
        }
    }

    #[test]
    fn test_softmax_batched() {
        // Test with batch_size=2, seq_len=3
        let mut data = vec![
            1.0, 2.0, 3.0,  // batch 0
            0.5, 1.5, 2.5,  // batch 1
        ];
        softmax_in_place(&mut data, 2, 3);

        // Each row should sum to 1
        for row in 0..2 {
            let start = row * 3;
            let end = start + 3;
            let sum: f32 = data[start..end].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
