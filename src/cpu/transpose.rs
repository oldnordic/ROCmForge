//! Transposed GEMV kernels for quantized tensors.
//!
//! GGUF stores tensors in specific layouts. For standard GEMV (y = W * x),
//! weights should be [out_dim, in_dim] in row-major format. However:
//!
//! - Tied embeddings are stored as [hidden_size, vocab_size] = transposed layout
//! - Quantized formats (Q8_0, Q4_0, Q4_1) use blocked storage
//!
//! This module provides:
//! 1. Tensor analysis to determine if transposition is needed
//! 2. Dispatch function to select appropriate kernel based on metadata

use crate::config::ModelConfig;
use crate::loader::{GgmlType, TensorDesc};

/// Determine if a tensor needs transposition for GEMV.
///
/// For tied embeddings stored as [hidden_size, vocab_size], we compute logits = W^T * hidden.
/// Standard GEMV expects [vocab_size, hidden_size], so transposition is required.
///
/// Returns:
/// - `true` if tensor should be transposed before GEMV
/// - `false` if tensor is already in correct layout
pub fn needs_transposition(tensor: &TensorDesc, is_lm_head: bool, hidden_size: usize, vocab_size: usize) -> bool {
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

/// Select appropriate GEMV kernel based on tensor properties.
///
/// Returns a function that calls the appropriate kernel.
/// For now, only Q8_0 tied embeddings need special handling.
pub fn dispatch_gemv_from_metadata(
    tensor: &TensorDesc,
    x: &[f32],
    y: &mut [f32],
    hidden_size: usize,
    vocab_size: usize,
) -> Result<(), crate::CpuError> {
    use crate::ops::{dispatch_gemv, dispatch_gemv_transposed};

    // For now, only Q8_0 tied embeddings need special handling
    if tensor.ggml_type == GgmlType::Q8_0 && needs_transposition(tensor, true, hidden_size, vocab_size) {
        // Transposed kernel for Q8_0 tied embeddings
        dispatch_gemv_transposed(tensor, x, y, vocab_size, hidden_size, true)
    } else {
        dispatch_gemv(tensor, x, y, vocab_size, hidden_size, false)
    }
}
