//! HIP ggml backend struct and allocator management.

use crate::backend::HipBackend;
use crate::ggml::{allocator::TensorAllocator, TensorId};
use std::collections::HashMap;
use std::sync::Arc;

/// HIP backend implementation for ggml IR execution.
///
/// Manages GPU tensor buffers and optional buffer reuse via allocator.
pub struct HipGgmlBackend {
    backend: Arc<HipBackend>,
    tensors: HashMap<TensorId, (crate::ggml::TensorDesc, crate::backend::HipBuffer)>,
    /// Optional tensor allocator for buffer reuse
    allocator: Option<TensorAllocator>,
}

impl std::fmt::Debug for HipGgmlBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipGgmlBackend")
            .field("tensor_count", &self.tensors.len())
            .field("allocator_enabled", &self.allocator.is_some())
            .finish()
    }
}

impl HipGgmlBackend {
    /// Create a new HIP ggml backend.
    pub fn new(backend: Arc<HipBackend>) -> Self {
        Self {
            backend,
            tensors: HashMap::new(),
            allocator: None,
        }
    }

    /// Get the underlying HIP backend.
    pub fn hip_backend(&self) -> &Arc<HipBackend> {
        &self.backend
    }

    /// Enable the tensor allocator for buffer reuse.
    pub fn with_allocator(mut self) -> Self {
        self.allocator = Some(TensorAllocator::new());
        self
    }

    /// Enable the tensor allocator with custom max pool size.
    pub fn with_allocator_config(mut self, max_pool_size: usize) -> Self {
        self.allocator = Some(TensorAllocator::new().with_max_pool_size(max_pool_size));
        self
    }

    /// Check if the allocator is enabled.
    pub fn has_allocator(&self) -> bool {
        self.allocator.is_some()
    }

    /// Get allocator statistics.
    pub fn allocator_stats(&self) -> Option<crate::ggml::allocator::AllocatorStats> {
        self.allocator.as_ref().map(|a| a.stats())
    }

    /// Reset the allocator, clearing all free pools.
    ///
    /// Call this between graph executions for optimal reuse.
    pub fn reset_allocator(&mut self) {
        if let Some(alloc) = &mut self.allocator {
            alloc.reset();
        }
    }

    /// Get mutable reference to tensors map (for internal use).
    pub fn tensors_mut(&mut self) -> &mut HashMap<TensorId, (crate::ggml::TensorDesc, crate::backend::HipBuffer)> {
        &mut self.tensors
    }

    /// Get reference to tensors map (for internal use).
    pub fn tensors(&self) -> &HashMap<TensorId, (crate::ggml::TensorDesc, crate::backend::HipBuffer)> {
        &self.tensors
    }

    /// Get reference to allocator (for internal use).
    pub fn allocator(&mut self) -> Option<&mut TensorAllocator> {
        self.allocator.as_mut()
    }
}
