//! Scratch buffer manager for reusable GPU memory allocation
//! Provides preallocated buffers for attention, MLP, and layernorm operations

use crate::backend::{DeviceTensor, HipBackend, HipError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScratchError {
    #[error("Invalid buffer configuration: {0}")]
    InvalidConfiguration(String),
    #[error("GPU memory allocation failed: {0}")]
    AllocationFailed(#[from] HipError),
}

pub type ScratchResult<T> = Result<T, ScratchError>;

/// Manages preallocated scratch buffers for efficient GPU memory usage
const MAX_ATTENTION_SEQ_LEN: usize = 4096;

#[derive(Debug)]
pub struct ScratchBufferManager {
    backend: HipBackend,
    // Attention buffers
    attention_scores: DeviceTensor,
    softmax_temp: DeviceTensor,
    // MLP buffers
    mlp_intermediate: DeviceTensor,
    // Layernorm buffers
    layernorm_temp: DeviceTensor,
    // Actual sequence capacity provisioned for attention buffers
    max_seq_capacity: usize,
}

impl ScratchBufferManager {
    /// Create new scratch buffer manager with specified configuration
    pub fn new(
        backend: &HipBackend,
        num_heads: usize,
        hidden_size: usize,
        _head_dim: usize,
        max_seq_len: usize,
    ) -> ScratchResult<Self> {
        eprintln!(
            ">>> ScratchBufferManager::new ENTRY: heads={}, hidden={}, seq_len={}",
            num_heads, hidden_size, max_seq_len
        );
        eprintln!(">>> ScratchBufferManager::new: About to create attention_scores buffer...");
        // Calculate buffer sizes
        let effective_seq_len = std::cmp::max(1, std::cmp::min(max_seq_len, MAX_ATTENTION_SEQ_LEN));
        if effective_seq_len < max_seq_len {
            tracing::warn!(
                "Clamping scratch attention capacity from {} to {} tokens to avoid massive allocations",
                max_seq_len,
                effective_seq_len
            );
        }
        let _attention_scores_size = num_heads * effective_seq_len * effective_seq_len;
        let _softmax_temp_size = num_heads * effective_seq_len;
        let mlp_intermediate_size = hidden_size * 4; // SwiGLU intermediate
        let layernorm_temp_size = hidden_size;

        // Create attention scores buffer: [num_heads, max_seq_len, max_seq_len]
        let expected_size = num_heads * effective_seq_len * effective_seq_len * 2; // f16 = 2 bytes
        eprintln!(
            "ScratchBufferManager::new: Expected attention_scores buffer size: {} bytes ({} MB)",
            expected_size,
            expected_size / 1024 / 1024
        );
        eprintln!(
            "ScratchBufferManager::new: Creating attention_scores buffer [{}, {}, {}]...",
            num_heads, effective_seq_len, effective_seq_len
        );
        let attention_scores_shape = crate::loader::mmap_loader::TensorShape::from_dims(&[
            num_heads,
            effective_seq_len,
            effective_seq_len,
        ]);
        let attention_scores = DeviceTensor::empty(backend, attention_scores_shape)
            .map_err(ScratchError::AllocationFailed)?;
        let actual_size = attention_scores.size();
        eprintln!(
            "ScratchBufferManager::new: Actual attention_scores buffer size: {} bytes ({} MB)",
            actual_size,
            actual_size / 1024 / 1024
        );
        eprintln!("ScratchBufferManager::new: attention_scores created");

        // Create softmax temp buffer: [num_heads, max_seq_len]
        let softmax_temp_shape =
            crate::loader::mmap_loader::TensorShape::from_dims(&[num_heads, effective_seq_len]);
        let softmax_temp = DeviceTensor::empty(backend, softmax_temp_shape)
            .map_err(ScratchError::AllocationFailed)?;

        // Create MLP intermediate buffer: [hidden_size * 4]
        let mlp_intermediate_shape =
            crate::loader::mmap_loader::TensorShape::from_dims(&[mlp_intermediate_size]);
        let mlp_intermediate = DeviceTensor::empty(backend, mlp_intermediate_shape)
            .map_err(ScratchError::AllocationFailed)?;

        // Create layernorm temp buffer: [hidden_size]
        let layernorm_temp_shape =
            crate::loader::mmap_loader::TensorShape::from_dims(&[layernorm_temp_size]);
        let layernorm_temp = DeviceTensor::empty(backend, layernorm_temp_shape)
            .map_err(ScratchError::AllocationFailed)?;

        Ok(ScratchBufferManager {
            backend: backend.clone(),
            attention_scores,
            softmax_temp,
            mlp_intermediate,
            layernorm_temp,
            max_seq_capacity: effective_seq_len,
        })
    }

    /// Get mutable reference to attention scores buffer
    pub fn attention_scores(&mut self) -> &mut DeviceTensor {
        &mut self.attention_scores
    }

    /// Get mutable reference to softmax temporary buffer
    pub fn softmax_temp(&mut self) -> &mut DeviceTensor {
        &mut self.softmax_temp
    }

    /// Get mutable reference to MLP intermediate buffer
    pub fn mlp_intermediate(&mut self) -> &mut DeviceTensor {
        &mut self.mlp_intermediate
    }

    /// Get mutable reference to layernorm temporary buffer
    pub fn layernorm_temp(&mut self) -> &mut DeviceTensor {
        &mut self.layernorm_temp
    }

    /// Get backend reference
    pub fn backend(&self) -> &HipBackend {
        &self.backend
    }

    /// Get total memory usage in bytes
    pub fn total_memory_usage(&self) -> usize {
        self.attention_scores.size()
            + self.softmax_temp.size()
            + self.mlp_intermediate.size()
            + self.layernorm_temp.size()
    }

    /// Validate that all buffers have expected sizes
    pub fn validate_invariants(
        &self,
        num_heads: usize,
        hidden_size: usize,
        _head_dim: usize,
        max_seq_len: usize,
    ) -> ScratchResult<()> {
        let seq_capacity = std::cmp::min(self.max_seq_capacity, max_seq_len);
        let expected_attention_size = num_heads * seq_capacity * seq_capacity;
        let expected_softmax_size = num_heads * seq_capacity;
        let expected_mlp_size = hidden_size * 4;
        let expected_layernorm_size = hidden_size;

        if self.attention_scores.len() != expected_attention_size {
            return Err(ScratchError::InvalidConfiguration(format!(
                "Attention scores buffer size mismatch: expected {}, got {}",
                expected_attention_size,
                self.attention_scores.len()
            )));
        }

        if self.softmax_temp.len() != expected_softmax_size {
            return Err(ScratchError::InvalidConfiguration(format!(
                "Softmax temp buffer size mismatch: expected {}, got {}",
                expected_softmax_size,
                self.softmax_temp.len()
            )));
        }

        if self.mlp_intermediate.len() != expected_mlp_size {
            return Err(ScratchError::InvalidConfiguration(format!(
                "MLP intermediate buffer size mismatch: expected {}, got {}",
                expected_mlp_size,
                self.mlp_intermediate.len()
            )));
        }

        if self.layernorm_temp.len() != expected_layernorm_size {
            return Err(ScratchError::InvalidConfiguration(format!(
                "Layernorm temp buffer size mismatch: expected {}, got {}",
                expected_layernorm_size,
                self.layernorm_temp.len()
            )));
        }

        Ok(())
    }

    /// Return the provisioned maximum sequence length for attention buffers
    pub fn seq_capacity(&self) -> usize {
        self.max_seq_capacity
    }
}
