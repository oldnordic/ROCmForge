//! Transposed GEMV kernels for quantized tensors.
//!
//! GGUF stores tensors in specific layouts. For standard GEMV (y = W * x),
//! weights should be [out_dim, in_dim] in row-major format. However:
//!
//! - Tied embeddings are stored as [hidden_size, vocab_size] = transposed layout
//! - FFN down weights are often stored as [intermediate_size, hidden_size] = transposed layout
//! - Quantized formats (Q8_0, Q4_0, Q4_1) use blocked storage
//!
//! This module provides tensor analysis to determine if transposition is needed.

use crate::config::{ModelConfig, TensorRole};
use crate::loader::TensorDesc;

/// Determine if a tensor needs transposition for GEMV.
///
/// For tied embeddings stored as [hidden_size, vocab_size], we compute logits = W^T * hidden.
/// Standard GEMV expects [vocab_size, hidden_size], so transposition is required.
///
/// Returns:
/// - `true` if tensor should be transposed before GEMV
/// - `false` if tensor is already in correct layout
pub fn needs_transposition(
    tensor: &TensorDesc,
    is_lm_head: bool,
    hidden_size: usize,
    vocab_size: usize,
) -> bool {
    // Check tensor dimensions
    if tensor.dims.len() < 2 {
        return false; // Not enough info
    }

    let (dim0, dim1) = (tensor.dims[0] as usize, tensor.dims[1] as usize);
    let expected_shape = if is_lm_head {
        // LM head: expect [vocab_size, hidden_size]
        (vocab_size, hidden_size)
    } else {
        // Other tensors: expect [in_dim, out_dim]
        (hidden_size, vocab_size)
    };

    // Check if transposition is needed
    (dim0, dim1) != expected_shape
}

/// Compute whether a weight tensor needs transposed access.
///
/// This function uses the tensor's semantic [`TensorRole`] and model configuration
/// to determine if the weight is stored in a transposed layout.
///
/// GGUF dimensions are innermost-first (column-major for 2D matrices):
/// - Standard layout: [out_dim, in_dim] - works with regular GEMV
/// - Transposed layout: [in_dim, out_dim] - needs transposed GEMV
///
/// # Arguments
/// * `role` - Semantic role of the weight tensor (e.g. `SsmConv1d`, `LmHead`)
/// * `actual_dims` - Dimensions from GGUF (innermost first)
/// * `config` - Model configuration
///
/// # Returns
/// `true` if transposed access is needed, `false` otherwise
pub fn compute_transpose_flag(
    role: TensorRole,
    actual_dims: &[u64],
    _wtype: crate::loader::GgmlType,
    config: &ModelConfig,
) -> bool {
    // Need at least 2 dimensions to determine transposition
    if actual_dims.len() < 2 {
        return false;
    }

    let dim0 = actual_dims[0] as usize;
    let dim1 = actual_dims[1] as usize;

    // GGUF stores 2D matrices with innermost dimension first
    // Standard GEMV expects: [out_dim, in_dim]
    // Transposed layout is: [in_dim, out_dim]

    match role {
        // LM head variants
        TensorRole::TiedLmHead => {
            // Tied embeddings: [hidden_size, vocab_size] where hidden_size is dim0 (innermost).
            // Since hidden_size is the contiguous dimension, we can access row-major directly
            // on CPU, so no transpose is needed.
            false
        }
        TensorRole::LmHead => {
            // Explicit LM head: standard layout, no transpose
            false
        }

        // SSM conv1d: GGUF stores [kernel_size, channels] but kernels expect
        // [channels, kernel_size]. Transpose at upload time.
        TensorRole::SsmConv1d => true,

        // Shortconv conv: same pattern as SSM conv1d
        TensorRole::ShortconvConv => true,

        // Everything else: fall back to dimension-based heuristics
        _ => {
            // Handle FFN down projection
            if dim0 == config.hidden_size && dim1 == config.intermediate_size {
                // FFN down projects from intermediate_size to hidden_size.
                // Transposed layout: dim0 = hidden_size, dim1 = intermediate_size
                return true;
            }

            // Square matrices (attention output, etc.) — transpose doesn't matter
            if dim0 == config.hidden_size && dim1 == config.hidden_size {
                return false;
            }

            // Standard layout assumed for all other roles
            false
        }
    }
}
