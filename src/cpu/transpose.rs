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

use crate::config::ModelConfig;
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
/// This function analyzes the tensor name, dimensions, and model configuration
/// to determine if the weight is stored in a transposed layout.
///
/// GGUF dimensions are innermost-first (column-major for 2D matrices):
/// - Standard layout: [out_dim, in_dim] - works with regular GEMV
/// - Transposed layout: [in_dim, out_dim] - needs transposed GEMV
///
/// # Arguments
/// * `weight_name` - Name of the weight tensor (e.g., "blk.0.ffn_down.weight")
/// * `actual_dims` - Dimensions from GGUF (innermost first)
/// * `wtype` - Quantization type
/// * `config` - Model configuration
/// * `is_lm_head` - Whether this is the language model head
/// * `is_tied` - Whether LM head is tied to embeddings
///
/// # Returns
/// `true` if transposed access is needed, `false` otherwise
pub fn compute_transpose_flag(
    weight_name: &str,
    actual_dims: &[u64],
    wtype: crate::loader::GgmlType,
    config: &ModelConfig,
    is_lm_head: bool,
    is_tied: bool,
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

    // Handle tied LM head: stored as [hidden_size, vocab_size], need transpose
    if is_lm_head && is_tied {
        // Tied embeddings: [hidden_size, vocab_size] -> need W^T * x
        // We expect to compute y = W^T * x where W is [hidden_size, vocab_size]
        // This means y[v] = sum_i(x[i] * W[i, v])
        // So we need transposed access (column-major storage)
        return true;
    }

    // Handle FFN down projection
    // Expected: [hidden_size, intermediate_size] -> no transpose
    // Actual often: [intermediate_size, hidden_size] -> needs transpose
    if weight_name.contains("ffn_down.weight") {
        // Check if dimensions are swapped
        // FFN down projects from intermediate_size to hidden_size
        // GEMV expects W [hidden_size, intermediate_size]
        // GGUF may store as [intermediate_size, hidden_size]
        let h = config.hidden_size;
        let ff = config.intermediate_size;

        // If actual is [ff, h] instead of [h, ff], need transpose
        return dim0 == ff && dim1 == h;
    }

    // Handle attention output projection
    // Expected: [hidden_size, hidden_size] (square, transpose doesn't matter)
    if weight_name.contains("attn_output.weight") {
        // Square matrix, no transpose needed
        return false;
    }

    // Handle Q/K/V projections and FFN gate/up
    // Expected: [projection_size, hidden_size]
    // These are typically in standard layout
    // Q: [num_heads * head_dim, hidden_size]
    // K/V: [num_kv_heads * head_dim, hidden_size]
    // FFN gate/up: [intermediate_size, hidden_size]
    if weight_name.contains("attn_q.weight")
        || weight_name.contains("attn_k.weight")
        || weight_name.contains("attn_v.weight")
        || weight_name.contains("ffn_gate.weight")
        || weight_name.contains("ffn_up.weight")
    {
        return false; // Standard layout
    }

    // Default: assume no transposition
    false
}
