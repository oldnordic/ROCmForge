//! Lazy-loaded tensor with on-demand fetching
//!
//! This module provides lazy tensor handles that defer loading tensor data
//! until it's actually needed. This enables fast model initialization by
//! only loading metadata upfront.

use crate::backend::hip_backend::DeviceTensor;
use crate::loader::GgufTensorType;
use std::sync::Arc;

/// Tensor that may not be loaded yet
///
/// This enum represents a tensor handle that can be in two states:
/// - `Unloaded`: Only metadata is known, data is still on disk
/// - `Gpu`: Tensor has been loaded and uploaded to GPU memory
///
/// # Purpose
///
/// Allows fast model initialization by creating tensor handles without
/// loading the actual data. Tensors are loaded on-demand when first accessed.
///
/// # Thread Safety
///
/// `LazyTensor` is `Send + Sync` because all its fields are `Send + Sync`:
/// - `String` is `Send + Sync`
/// - `Arc<DeviceTensor>` is `Send + Sync` (DeviceTensor contains HipBuffer with Arc)
/// - `Vec<usize>` is `Send + Sync`
/// - `GgufTensorType` is `Send + Sync` (Copy type)
///
/// # Example
///
/// ```rust
/// use crate::loader::lazy_tensor::LazyTensor;
/// use crate::loader::gguf::GgufTensorType;
///
/// // Create unloaded tensor handle (fast, no I/O)
/// let tensor = LazyTensor::unloaded(
///     "blk.0.attn_q.weight".to_string(),
///     1024,
///     4096,
///     vec![4096, 1024],
///     GgufTensorType::Q8_0
/// );
///
/// // Tensor is not loaded yet
/// assert!(!tensor.is_gpu_loaded());
/// ```
#[derive(Debug, Clone)]
pub enum LazyTensor {
    /// Metadata only, data not loaded
    ///
    /// This variant stores only the tensor's metadata:
    /// - `name`: Tensor name (e.g., "blk.0.attn_q.weight")
    /// - `offset`: Byte offset in the GGUF file where tensor data starts
    /// - `size`: Size of tensor data in bytes
    /// - `shape`: Tensor dimensions (e.g., [4096, 1024])
    /// - `tensor_type`: GGUF tensor type (for dequantization)
    Unloaded {
        name: String,
        offset: u64,
        size: usize,
        shape: Vec<usize>,
        tensor_type: GgufTensorType,
    },

    /// Loaded to GPU
    ///
    /// This variant stores the fully loaded tensor:
    /// - `name`: Tensor name
    /// - `tensor`: GPU tensor with data in device memory
    Gpu {
        name: String,
        tensor: Arc<DeviceTensor>,
    },
}

// SAFETY: LazyTensor is Send+Sync because all its fields are Send+Sync.
// DeviceTensor contains HipBuffer which uses Arc for thread-safe reference counting.
unsafe impl Send for LazyTensor {}
unsafe impl Sync for LazyTensor {}

impl LazyTensor {
    /// Create unloaded tensor handle
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name (e.g., "blk.0.attn_q.weight")
    /// * `offset` - Byte offset in GGUF file
    /// * `size` - Size in bytes
    /// * `shape` - Tensor dimensions
    /// * `tensor_type` - GGUF tensor type (for dequantization)
    ///
    /// # Returns
    ///
    /// Returns a `LazyTensor::Unloaded` variant.
    ///
    /// # Performance
    ///
    /// This is O(1) - no file I/O or memory allocation occurs.
    pub fn unloaded(
        name: String,
        offset: u64,
        size: usize,
        shape: Vec<usize>,
        tensor_type: GgufTensorType,
    ) -> Self {
        Self::Unloaded {
            name,
            offset,
            size,
            shape,
            tensor_type,
        }
    }

    /// Create unloaded placeholder tensor handle (name only)
    ///
    /// # Phase 2: ExecutionPlan Integration
    ///
    /// This is a convenience method for creating LazyTensor placeholders
    /// when only the tensor name is known (not the full metadata).
    /// The GgufLoader will look up the actual metadata when loading.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name (e.g., "blk.0.attn_q.weight")
    ///
    /// # Returns
    ///
    /// Returns a `LazyTensor::Unloaded` variant with placeholder metadata.
    ///
    /// # Use Case
    ///
    /// Used by ExecutionPlan to create lazy tensor handles during
    /// initialization without querying the loader for each tensor's metadata.
    ///
    /// # Note
    ///
    /// The placeholder values (offset=0, size=0, shape=[], type=F32) will be
    /// ignored by GgufLoader::load_tensor_to_gpu(), which looks up the actual
    /// metadata by name.
    pub fn unloaded_placeholder(name: String) -> Self {
        Self::Unloaded {
            name,
            offset: 0,
            size: 0,
            shape: vec![],
            tensor_type: GgufTensorType::F32, // Placeholder, will be ignored
        }
    }

    /// Create GPU-loaded tensor handle
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    /// * `tensor` - GPU tensor
    ///
    /// # Returns
    ///
    /// Returns a `LazyTensor::Gpu` variant.
    pub fn gpu(name: String, tensor: Arc<DeviceTensor>) -> Self {
        Self::Gpu { name, tensor }
    }

    /// Get tensor name
    ///
    /// # Returns
    ///
    /// Returns the tensor's name as a string slice.
    pub fn name(&self) -> &str {
        match self {
            Self::Unloaded { name, .. } => name,
            Self::Gpu { name, .. } => name,
        }
    }

    /// Check if loaded to GPU
    ///
    /// # Returns
    ///
    /// Returns `true` if the tensor is in GPU memory, `false` if unloaded.
    pub fn is_gpu_loaded(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }

    /// Get tensor shape (if known)
    ///
    /// # Returns
    ///
    /// Returns `Some(&[usize])` with the tensor dimensions, or `None` if not available.
    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            Self::Unloaded { shape, .. } => Some(shape.as_slice()),
            Self::Gpu { tensor, .. } => {
                // Get shape from DeviceTensor
                
                Some(tensor.shape().dims())
            }
        }
    }

    /// Get tensor offset in file (only for unloaded tensors)
    ///
    /// # Returns
    ///
    /// Returns `Some(u64)` with the byte offset, or `None` if tensor is GPU-loaded.
    pub fn offset(&self) -> Option<u64> {
        match self {
            Self::Unloaded { offset, .. } => Some(*offset),
            Self::Gpu { .. } => None,
        }
    }

    /// Get tensor size in bytes (only for unloaded tensors)
    ///
    /// # Returns
    ///
    /// Returns `Some(usize)` with the size in bytes, or `None` if tensor is GPU-loaded.
    pub fn size(&self) -> Option<usize> {
        match self {
            Self::Unloaded { size, .. } => Some(*size),
            Self::Gpu { .. } => None,
        }
    }

    /// Get tensor type (for dequantization)
    ///
    /// # Returns
    ///
    /// Returns `Some(GgufTensorType)` with the tensor type, or `None` if tensor is GPU-loaded.
    pub fn tensor_type(&self) -> Option<GgufTensorType> {
        match self {
            Self::Unloaded { tensor_type, .. } => Some(*tensor_type),
            Self::Gpu { .. } => None,
        }
    }

    /// Get GPU tensor (only if loaded)
    ///
    /// # Returns
    ///
    /// Returns `Some(&Arc<DeviceTensor>)` if loaded, `None` if unloaded.
    pub fn gpu_tensor(&self) -> Option<&Arc<DeviceTensor>> {
        match self {
            Self::Unloaded { .. } => None,
            Self::Gpu { tensor, .. } => Some(tensor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::GgufTensorType;

    #[test]
    fn test_lazy_tensor_unloaded() {
        let tensor = LazyTensor::unloaded(
            "test.weight".to_string(),
            1024,
            4096,
            vec![128, 256],
            GgufTensorType::Q8_0,
        );

        assert_eq!(tensor.name(), "test.weight");
        assert!(!tensor.is_gpu_loaded());
        assert_eq!(tensor.shape(), Some(&[128, 256][..]));
        assert_eq!(tensor.offset(), Some(1024));
        assert_eq!(tensor.size(), Some(4096));
        assert_eq!(tensor.tensor_type(), Some(GgufTensorType::Q8_0));
        assert!(tensor.gpu_tensor().is_none());
    }

    #[test]
    fn test_lazy_tensor_display() {
        let tensor = LazyTensor::unloaded(
            "blk.0.attn_q.weight".to_string(),
            0,
            100,
            vec![10, 10],
            GgufTensorType::F32,
        );

        // Debug output should show tensor info
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("Unloaded"));
        assert!(debug_str.contains("blk.0.attn_q.weight"));
    }
}
