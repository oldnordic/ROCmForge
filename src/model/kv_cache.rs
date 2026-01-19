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

use crate::backend::{DeviceTensor, HipBackend, HipError, HipBuffer};
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
    // Single large preallocated GPU memory buffers for all keys and values
    // This reduces fragmentation by using fewer hipMalloc calls
    keys_buffer: HipBuffer,
    values_buffer: HipBuffer,
    // Track current sequence length for each layer
    current_seq_len: Vec<usize>,
    // Pre-calculated sizes for sub-allocation
    layer_size_bytes: usize,
    #[allow(dead_code)] // Reserved for future head-level operations
    head_size_bytes: usize,
}

impl KVCache {
    /// Create new KV cache with specified configuration
    /// Uses single large buffer allocation to reduce fragmentation
    pub fn new(
        backend: &HipBackend,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> KVCacheResult<Self> {
        eprintln!("KVCache::new() called with layers={}, heads={}, head_dim={}, max_seq_len={}",
                 num_layers, num_heads, head_dim, max_seq_len);

        // Calculate sizes for sub-allocation
        let elements_per_layer = max_seq_len * num_heads * head_dim;
        let bytes_per_layer = elements_per_layer * std::mem::size_of::<f32>();
        let total_keys_bytes = bytes_per_layer * num_layers;
        let total_values_bytes = bytes_per_layer * num_layers;

        eprintln!("KVCache::new() allocating {} bytes for keys, {} bytes for values",
                 total_keys_bytes, total_values_bytes);

        // Allocate single large buffers for all keys and values
        // This reduces hipMalloc calls from 2*num_layers to 2, reducing fragmentation
        let keys_buffer = backend.allocate_buffer_safe(total_keys_bytes)
            .map_err(KVCacheError::AllocationFailed)?;
        let values_buffer = backend.allocate_buffer_safe(total_values_bytes)
            .map_err(KVCacheError::AllocationFailed)?;

        let current_seq_len = vec![0; num_layers];

        eprintln!("KVCache::new() completed allocation, returning KVCache");
        Ok(KVCache {
            backend: backend.clone(),
            num_layers,
            num_heads,
            head_dim,
            max_seq_len,
            keys_buffer,
            values_buffer,
            current_seq_len,
            layer_size_bytes: bytes_per_layer,
            head_size_bytes: max_seq_len * head_dim * std::mem::size_of::<f32>(),
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
        let dst_offset_elements = current_len * self.num_heads * self.head_dim;
        let dst_offset_bytes = dst_offset_elements * std::mem::size_of::<f32>();
        let layer_offset_bytes = layer * self.layer_size_bytes;

        // Copy to the appropriate layer in the large buffers
        let key_dst_offset = layer_offset_bytes + dst_offset_bytes;
        let value_dst_offset = layer_offset_bytes + dst_offset_bytes;
        let chunk_bytes = chunk_elements * std::mem::size_of::<f32>();

        // Validate offsets before creating sub-buffer views
        if key_dst_offset + chunk_bytes > self.keys_buffer.size() {
            return Err(KVCacheError::CapacityExceeded {
                seq_len: current_len + seq_chunk,
                max_seq_len: self.max_seq_len,
            });
        }
        if value_dst_offset + chunk_bytes > self.values_buffer.size() {
            return Err(KVCacheError::CapacityExceeded {
                seq_len: current_len + seq_chunk,
                max_seq_len: self.max_seq_len,
            });
        }

        // Create sub-buffer views and copy to avoid pointer arithmetic issues
        let key_dst_view = self.keys_buffer.sub_buffer_view(key_dst_offset, chunk_bytes)?;
        key_dst_view.copy_from_buffer(key.buffer())?;

        let value_dst_view = self.values_buffer.sub_buffer_view(value_dst_offset, chunk_bytes)?;
        value_dst_view.copy_from_buffer(value.buffer())?;

        self.current_seq_len[layer] += seq_chunk;
        Ok(())
    }

    /// Get key and value tensors for specified layer
    /// Returns views into the cached data for the current sequence length
    pub fn get(
        &self,
        layer: usize,
    ) -> KVCacheResult<(DeviceTensor, DeviceTensor)> {
        // Validate layer index
        if layer >= self.num_layers {
            return Err(KVCacheError::InvalidLayer {
                layer,
                max_layers: self.num_layers,
            });
        }

        let current_len = self.current_seq_len[layer];
        let layer_offset_bytes = layer * self.layer_size_bytes;

        // Create views for the current sequence data in this layer
        let key_shape = TensorShape::from_dims(&[current_len, self.num_heads, self.head_dim]);
        let value_shape = TensorShape::from_dims(&[current_len, self.num_heads, self.head_dim]);

        let key_view = self.keys_buffer.sub_buffer_view(layer_offset_bytes, current_len * self.num_heads * self.head_dim * std::mem::size_of::<f32>())?;
        let value_view = self.values_buffer.sub_buffer_view(layer_offset_bytes, current_len * self.num_heads * self.head_dim * std::mem::size_of::<f32>())?;

        let key_tensor = DeviceTensor::from_buffer(&self.backend, key_view, key_shape)?;
        let value_tensor = DeviceTensor::from_buffer(&self.backend, value_view, value_shape)?;

        Ok((key_tensor, value_tensor))
    }

    /// Retrieve K and V tensors for attention computation
    /// Returns copies of the cached tensors for the layer
    pub fn retrieve(
        &self,
        layer: usize,
        _seq_len: usize,
    ) -> KVCacheResult<(DeviceTensor, DeviceTensor)> {
        // Use get() to create views, then copy them to new tensors
        let (key_view, value_view) = self.get(layer)?;

        // Create new tensors and copy the data
        let mut key_copy = DeviceTensor::empty(&self.backend, key_view.shape().clone())?;
        let mut value_copy = DeviceTensor::empty(&self.backend, value_view.shape().clone())?;

        key_copy.copy_from_device_buffer(key_view.buffer())?;
        value_copy.copy_from_device_buffer(value_view.buffer())?;

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
        self.keys_buffer.size() + self.values_buffer.size()
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

        // Check that buffers have correct total sizes
        let expected_total_elements = self.num_layers * self.max_seq_len * self.num_heads * self.head_dim;
        let expected_total_bytes = expected_total_elements * std::mem::size_of::<f32>();

        if self.keys_buffer.size() != expected_total_bytes {
            return Err(KVCacheError::InvalidLayer {
                layer: 0,
                max_layers: self.num_layers,
            });
        }

        if self.values_buffer.size() != expected_total_bytes {
            return Err(KVCacheError::InvalidLayer {
                layer: 0,
                max_layers: self.num_layers,
            });
        }

        Ok(())
    }
}
