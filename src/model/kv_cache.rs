//! Simple KV Cache for efficient GPU memory management during inference
//!
//! This is a simple GPU-resident KV cache with preallocated memory.
//!
//! NOTE: This is a legacy/prototype implementation. For production use,
//! see the paged KV cache at `crate::kv_cache::KvCache` which has:
//! - PagedAttention support
//! - LRU eviction
//! - Block sharing between sequences
//! - Better memory management

use crate::backend::{DeviceTensor, HipBackend, HipError};
use crate::loader::mmap_loader::TensorShape;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KVCacheError {
    #[error("Cache capacity exceeded: sequence length {seq_len} exceeds max {max_seq_len}")]
    CapacityExceeded { seq_len: usize, max_seq_len: usize },
    #[error("Invalid layer index: {layer}, max layers: {max_layers}")]
    InvalidLayer { layer: usize, max_layers: usize },
    #[error("Invalid head index: {head}, max heads: {max_heads}")]
    InvalidHead { head: usize, max_heads: usize },
    #[error("Invalid sequence index: {seq_idx}, max sequence: {max_seq}")]
    InvalidSequence { seq_idx: usize, max_seq: usize },
    #[error("GPU memory allocation failed: {0}")]
    AllocationFailed(#[from] HipError),
}

pub type KVCacheResult<T> = Result<T, KVCacheError>;

/// Simple GPU-resident KV cache for transformer models
///
/// This is a legacy implementation with simple preallocated memory.
/// For production use, see `crate::kv_cache::KvCache` (paged KV cache).
#[derive(Debug)]
pub struct KVCache {
    backend: HipBackend,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    // Preallocated GPU memory for keys and values
    // Shape: [num_layers][num_heads][max_seq_len][head_dim]
    keys: Vec<DeviceTensor>,
    values: Vec<DeviceTensor>,
    // Track current sequence length for each layer
    current_seq_len: Vec<usize>,
}

impl KVCache {
    /// Create new KV cache with specified configuration
    pub fn new(
        backend: &HipBackend,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> KVCacheResult<Self> {
        eprintln!(
            "KVCache::new() called with layers={}, heads={}, head_dim={}, max_seq_len={}",
            num_layers, num_heads, head_dim, max_seq_len
        );
        // Preallocate keys and values for all layers
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        let current_seq_len = vec![0; num_layers];

        for layer in 0..num_layers {
            if layer % 8 == 0 || layer == num_layers - 1 {
                eprintln!(
                    "KVCache::new() allocating layer {}/{}",
                    layer + 1,
                    num_layers
                );
            }
            // Key/Value tensor shape: [max_seq_len, num_heads, head_dim]
            let kv_shape = TensorShape::from_dims(&[max_seq_len, num_heads, head_dim]);

            let key_tensor = DeviceTensor::empty(backend, kv_shape.clone())
                .map_err(KVCacheError::AllocationFailed)?;
            let value_tensor =
                DeviceTensor::empty(backend, kv_shape).map_err(KVCacheError::AllocationFailed)?;

            keys.push(key_tensor);
            values.push(value_tensor);
        }

        tracing::debug!(
            "KVCache::new() completed all {} layers, returning KVCache",
            num_layers
        );
        Ok(KVCache {
            backend: backend.clone(),
            num_layers,
            num_heads,
            head_dim,
            max_seq_len,
            keys,
            values,
            current_seq_len,
        })
    }

    /// Append key and value tensors to specified layer
    pub fn append(
        &mut self,
        layer: usize,
        key: &DeviceTensor,
        value: &DeviceTensor,
    ) -> KVCacheResult<()> {
        // Validate layer index
        if layer >= self.num_layers {
            return Err(KVCacheError::InvalidLayer {
                layer,
                max_layers: self.num_layers,
            });
        }

        // Check capacity
        let current_len = self.current_seq_len[layer];
        if current_len >= self.max_seq_len {
            return Err(KVCacheError::CapacityExceeded {
                seq_len: current_len + 1,
                max_seq_len: self.max_seq_len,
            });
        }

        // Validate input tensor shapes
        let key_shape = key.shape().dims();
        let value_shape = value.shape().dims();
        if key_shape.len() != 3
            || key_shape[1] != self.num_heads
            || key_shape[2] != self.head_dim
            || value_shape.len() != 3
            || value_shape[1] != self.num_heads
            || value_shape[2] != self.head_dim
            || key_shape[0] != value_shape[0]
        {
            return Err(KVCacheError::InvalidLayer {
                layer: 0,
                max_layers: 0,
            });
        }

        let seq_chunk = key_shape[0];
        if current_len + seq_chunk > self.max_seq_len {
            return Err(KVCacheError::CapacityExceeded {
                seq_len: current_len + seq_chunk,
                max_seq_len: self.max_seq_len,
            });
        }

        let chunk_elements = seq_chunk * self.num_heads * self.head_dim;
        let dst_offset = current_len * self.num_heads * self.head_dim;

        self.keys[layer].copy_from_device_region(dst_offset, key, 0, chunk_elements)?;
        self.values[layer].copy_from_device_region(dst_offset, value, 0, chunk_elements)?;

        self.current_seq_len[layer] += seq_chunk;
        Ok(())
    }

    /// Get key and value tensors for specified layer, head, and sequence index
    pub fn get(
        &self,
        layer: usize,
        head: usize,
        seq_index: usize,
    ) -> KVCacheResult<(&DeviceTensor, &DeviceTensor)> {
        // Validate indices
        if layer >= self.num_layers {
            return Err(KVCacheError::InvalidLayer {
                layer,
                max_layers: self.num_layers,
            });
        }

        if head >= self.num_heads {
            return Err(KVCacheError::InvalidHead {
                head,
                max_heads: self.num_heads,
            });
        }

        if seq_index >= self.current_seq_len[layer] {
            return Err(KVCacheError::InvalidSequence {
                seq_idx: seq_index,
                max_seq: self.current_seq_len[layer],
            });
        }

        // Return references to the cached tensors
        // Note: This returns the entire layer's KV cache
        // A more sophisticated implementation would return views into specific positions
        Ok((&self.keys[layer], &self.values[layer]))
    }

    /// Retrieve K and V tensors for attention computation
    /// Returns the full cached tensors for the layer
    pub fn retrieve(
        &self,
        layer: usize,
        _seq_len: usize,
    ) -> KVCacheResult<(DeviceTensor, DeviceTensor)> {
        if layer >= self.num_layers {
            return Err(KVCacheError::InvalidLayer {
                layer,
                max_layers: self.num_layers,
            });
        }

        let current_len = self.current_seq_len[layer];
        let key_shape = TensorShape::from_dims(&[current_len, self.num_heads, self.head_dim]);
        let value_shape = TensorShape::from_dims(&[current_len, self.num_heads, self.head_dim]);
        let mut key_copy = DeviceTensor::empty(&self.backend, key_shape)?;
        let mut value_copy = DeviceTensor::empty(&self.backend, value_shape)?;
        let elements = current_len * self.num_heads * self.head_dim;
        key_copy.copy_from_device_region(0, &self.keys[layer], 0, elements)?;
        value_copy.copy_from_device_region(0, &self.values[layer], 0, elements)?;
        Ok((key_copy, value_copy))
    }

    /// Get current sequence length for a layer
    pub fn current_seq_len(&self, layer: usize) -> KVCacheResult<usize> {
        if layer >= self.num_layers {
            return Err(KVCacheError::InvalidLayer {
                layer,
                max_layers: self.num_layers,
            });
        }
        Ok(self.current_seq_len[layer])
    }

    /// Get current sequence length for a layer (alias for current_seq_len)
    pub fn get_current_length(&self, layer: usize) -> KVCacheResult<usize> {
        self.current_seq_len(layer)
    }

    /// Get raw key/value tensors for a layer.
    pub fn layer_tensors(&self, layer: usize) -> KVCacheResult<(&DeviceTensor, &DeviceTensor)> {
        if layer >= self.num_layers {
            return Err(KVCacheError::InvalidLayer {
                layer,
                max_layers: self.num_layers,
            });
        }
        Ok((&self.keys[layer], &self.values[layer]))
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Reset cache (clear all sequence data)
    pub fn reset(&mut self) {
        for len in &mut self.current_seq_len {
            *len = 0;
        }
    }

    /// Advance current sequence length without copying new tensors.
    pub fn advance(&mut self, layer: usize, tokens: usize) -> KVCacheResult<()> {
        if layer >= self.num_layers {
            return Err(KVCacheError::InvalidLayer {
                layer,
                max_layers: self.num_layers,
            });
        }
        let current = self.current_seq_len[layer];
        if current + tokens > self.max_seq_len {
            return Err(KVCacheError::CapacityExceeded {
                seq_len: current + tokens,
                max_seq_len: self.max_seq_len,
            });
        }
        self.current_seq_len[layer] += tokens;
        Ok(())
    }

    /// Get total memory usage in bytes
    pub fn total_memory_usage(&self) -> usize {
        let key_memory = self.keys.iter().map(|k| k.size()).sum::<usize>();
        let value_memory = self.values.iter().map(|v| v.size()).sum::<usize>();
        key_memory + value_memory
    }

    /// Validate cache invariants
    pub fn validate_invariants(&self) -> KVCacheResult<()> {
        // Check that all layers have consistent sequence lengths
        for layer in 0..self.num_layers {
            if self.current_seq_len[layer] > self.max_seq_len {
                return Err(KVCacheError::CapacityExceeded {
                    seq_len: self.current_seq_len[layer],
                    max_seq_len: self.max_seq_len,
                });
            }
        }

        // Check that all tensors have correct sizes
        let expected_tensor_size = self.num_heads * self.max_seq_len * self.head_dim;
        for layer in 0..self.num_layers {
            if self.keys[layer].len() != expected_tensor_size {
                return Err(KVCacheError::InvalidLayer {
                    layer,
                    max_layers: self.num_layers,
                });
            }
            if self.values[layer].len() != expected_tensor_size {
                return Err(KVCacheError::InvalidLayer {
                    layer,
                    max_layers: self.num_layers,
                });
            }
        }

        Ok(())
    }
}
