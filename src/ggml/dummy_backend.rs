//! Dummy backend for unit testing (llama.cpp pattern).
//!
//! This backend provides host-only, no-op operations for unit testing.
//! Inspired by llama.cpp/tests/test-alloc.cpp dummy_backend.
//!
//! # Key Design Principles
//!
//! - **No GPU allocation**: Uses fake memory pointers (usize instead of actual GPU memory)
//! - **No actual execution**: All operations are no-ops
//! - **Test-friendly**: Tracks allocations for testing purposes
//! - **Follows llama.cpp**: Similar to dummy_backend with is_host=true
//!
//! # Usage
//!
//! ```rust,ignore
//! use rocmlforge::ggml::dummy_backend::DummyBackend;
//!
//! #[test]
//! fn test_something() {
//!     let mut backend = DummyBackend::new();
//!     // Test logic without touching GPU
//! }
//! ```

use crate::ggml::{GgmlBackend, GgmlResult, Op, TensorDesc, TensorId};
use std::collections::HashMap;

/// Fake buffer type (llama.cpp uses (uint8_t *) 16 as fake pointer)
///
/// This represents a "buffer" in the dummy backend's fake memory space.
/// The value is a fake offset from a fake base address.
#[derive(Debug, Clone)]
pub struct DummyBuffer {
    /// Fake offset from base (llama.cpp uses 16 as base)
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
}

impl DummyBuffer {
    /// Create a new fake buffer
    pub fn new(offset: usize, size: usize) -> Self {
        Self { offset, size }
    }

    /// Get the fake memory pointer (similar to llama.cpp's alloc_base = (uint8_t *) 16)
    pub fn as_fake_ptr(&self) -> usize {
        16 + self.offset
    }
}

/// Dummy backend for unit testing (llama.cpp pattern)
///
/// # Design
///
/// - `is_host = true` equivalent: No GPU interaction
/// - Fake memory: Uses usize offsets instead of actual GPU memory
/// - `no_alloc = true` equivalent: Tracks allocations but doesn't allocate
///
/// # Memory Safety
///
/// - No actual GPU memory is allocated
/// - All operations are no-ops
/// - Safe to run in parallel with other GPU applications
#[derive(Debug)]
pub struct DummyBackend {
    /// Fake memory allocations (TensorId -> buffer info)
    buffers: HashMap<TensorId, DummyBuffer>,
    /// Tensor descriptors
    tensors: HashMap<TensorId, TensorDesc>,
    /// Current fake offset (for tracking allocations)
    current_offset: usize,
    /// Maximum buffer size (llama.cpp pattern: limits for testing)
    max_buffer_size: usize,
    /// Alignment requirement (llama.cpp uses 8)
    alignment: usize,
    /// Statistics for testing
    stats: DummyBackendStats,
}

/// Statistics tracking for dummy backend (llama.cpp pattern)
#[derive(Debug, Default, Clone)]
pub struct DummyBackendStats {
    /// Number of alloc() calls
    pub alloc_count: usize,
    /// Number of bind() calls
    pub bind_count: usize,
    /// Number of free() calls
    pub free_count: usize,
    /// Number of execute_op() calls
    pub execute_op_count: usize,
    /// Total "allocated" bytes (fake)
    pub total_allocated_bytes: usize,
}

impl DummyBackend {
    /// Create a new dummy backend with default settings (llama.cpp pattern)
    ///
    /// Default: max_buffer_size = 64 bytes, alignment = 8
    pub fn new() -> Self {
        Self::with_config(64, 8)
    }

    /// Create a new dummy backend with custom configuration (llama.cpp pattern)
    ///
    /// # Arguments
    /// - `max_buffer_size`: Maximum fake buffer size (llama.cpp default: 64)
    /// - `alignment`: Alignment requirement (llama.cpp default: 8)
    pub fn with_config(max_buffer_size: usize, alignment: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            tensors: HashMap::new(),
            current_offset: 0,
            max_buffer_size,
            alignment,
            stats: DummyBackendStats::default(),
        }
    }

    /// Get statistics (llama.cpp pattern for tracking allocations)
    pub fn stats(&self) -> &DummyBackendStats {
        &self.stats
    }

    /// Reset statistics and clear all buffers (llama.cpp gallocr reset pattern)
    pub fn reset(&mut self) {
        self.buffers.clear();
        self.tensors.clear();
        self.current_offset = 0;
        self.stats = DummyBackendStats::default();
    }

    /// Get total "allocated" bytes (llama.cpp allocated_total() pattern)
    pub fn allocated_total(&self) -> usize {
        self.stats.total_allocated_bytes
    }

    /// Get current fake offset (for testing)
    pub fn current_offset(&self) -> usize {
        self.current_offset
    }

    /// Check if buffer exists (for testing)
    pub fn has_buffer(&self, id: TensorId) -> bool {
        self.buffers.contains_key(&id)
    }

    /// Align offset to alignment boundary (llama.cpp pattern)
    fn align_offset(&self, offset: usize) -> usize {
        ((offset + self.alignment - 1) / self.alignment) * self.alignment
    }
}

impl Default for DummyBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GgmlBackend for DummyBackend {
    type Buffer = DummyBuffer;

    /// Allocate a fake buffer (llama.cpp dummy_backend_buffer_type_alloc_buffer pattern)
    ///
    /// No actual GPU memory is allocated. This just tracks the "allocation"
    /// in fake memory space for testing purposes.
    fn alloc(&mut self, desc: &TensorDesc) -> GgmlResult<()> {
        // Calculate size (similar to llama.cpp ggml_nbytes)
        let size = desc.element_count() * std::mem::size_of::<f32>();

        // Align offset (llama.cpp pattern)
        let aligned_offset = self.align_offset(self.current_offset);

        // Check max buffer size (llama.cpp get_max_size pattern)
        if aligned_offset + size > self.max_buffer_size {
            // In llama.cpp, this would allocate a new buffer chunk
            // For dummy backend, we just extend the offset
            // (real backend would error or allocate new chunk)
        }

        // Create fake buffer (llama.cpp alloc_base + offset pattern)
        let buffer = DummyBuffer::new(aligned_offset, size);

        // Track allocation
        self.buffers.insert(desc.id, buffer.clone());
        self.tensors.insert(desc.id, desc.clone());
        self.current_offset = aligned_offset + size;

        // Update stats (llama.cpp allocated_total pattern)
        self.stats.alloc_count += 1;
        self.stats.total_allocated_bytes += size;

        Ok(())
    }

    /// Bind an existing buffer (llama.cpp dummy_backend set_tensor pattern)
    ///
    /// This is a no-op in dummy backend (llama.cpp uses empty functions)
    fn bind(&mut self, desc: &TensorDesc, buffer: Self::Buffer) -> GgmlResult<()> {
        self.buffers.insert(desc.id, buffer);
        self.tensors.insert(desc.id, desc.clone());
        self.stats.bind_count += 1;
        Ok(())
    }

    /// Free a buffer (llama.cpp dummy_backend_buffer_free_buffer pattern)
    fn free(&mut self, id: TensorId) -> GgmlResult<()> {
        self.buffers.remove(&id);
        self.tensors.remove(&id);
        self.stats.free_count += 1;
        Ok(())
    }

    /// Get tensor descriptor (llama.cpp pattern)
    fn tensor_desc(&self, id: TensorId) -> Option<&TensorDesc> {
        self.tensors.get(&id)
    }

    /// Get buffer (llama.cpp dummy_backend_buffer_get_base pattern)
    fn buffer(&self, id: TensorId) -> Option<&Self::Buffer> {
        self.buffers.get(&id)
    }

    /// Get mutable buffer
    fn buffer_mut(&mut self, id: TensorId) -> Option<&mut Self::Buffer> {
        self.buffers.get_mut(&id)
    }

    /// Execute operation (llama.cpp no-op pattern)
    ///
    /// All operations are no-ops in dummy backend.
    /// llama.cpp uses empty functions for memset_tensor, set_tensor, get_tensor.
    fn execute_op(
        &mut self,
        _op: &Op,
        _inputs: &[TensorId],
        _outputs: &[TensorId],
    ) -> GgmlResult<()> {
        // No-op (llama.cpp pattern: all buffer operations are empty functions)
        self.stats.execute_op_count += 1;
        Ok(())
    }

    /// Synchronize (llama.cpp no-op pattern)
    fn synchronize(&mut self) -> GgmlResult<()> {
        // No-op (no actual GPU to synchronize)
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{DType, Graph, Layout, TensorDesc};

    // TDD TEST: Dummy backend allocates fake memory
    #[test]
    fn test_dummy_backend_alloc() {
        let mut backend = DummyBackend::new();

        let desc = TensorDesc {
            id: TensorId(0),
            dtype: DType::F32,
            shape: vec![4], // 4 floats = 16 bytes
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        };

        backend.alloc(&desc).expect("Alloc should succeed");

        assert!(backend.has_buffer(desc.id));
        assert_eq!(backend.stats().alloc_count, 1);
        assert_eq!(backend.stats().total_allocated_bytes, 16);
    }

    // TDD TEST: Dummy backend respects alignment
    #[test]
    fn test_dummy_backend_alignment() {
        let mut backend = DummyBackend::with_config(64, 8);

        // First alloc: 5 floats = 20 bytes, aligned to 8 = 24
        let desc1 = TensorDesc {
            id: TensorId(0),
            dtype: DType::F32,
            shape: vec![5],
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        };
        backend.alloc(&desc1).expect("Alloc should succeed");

        // Second alloc should start at offset 24 (aligned)
        let desc2 = TensorDesc {
            id: TensorId(1),
            dtype: DType::F32,
            shape: vec![4],
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        };
        backend.alloc(&desc2).expect("Alloc should succeed");

        assert_eq!(backend.stats().alloc_count, 2);
        assert_eq!(backend.current_offset(), 40); // 24 + 16
    }

    // TDD TEST: Dummy backend tracks statistics
    #[test]
    fn test_dummy_backend_stats() {
        let mut backend = DummyBackend::new();

        let desc = TensorDesc {
            id: TensorId(0),
            dtype: DType::F32,
            shape: vec![8], // 32 bytes
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        };

        backend.alloc(&desc).expect("Alloc should succeed");

        let stats = backend.stats();
        assert_eq!(stats.alloc_count, 1);
        assert_eq!(stats.total_allocated_bytes, 32);
        assert_eq!(stats.bind_count, 0);
        assert_eq!(stats.free_count, 0);
    }

    // TDD TEST: Dummy backend free works
    #[test]
    fn test_dummy_backend_free() {
        let mut backend = DummyBackend::new();

        let desc = TensorDesc {
            id: TensorId(0),
            dtype: DType::F32,
            shape: vec![4],
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        };

        backend.alloc(&desc).expect("Alloc should succeed");
        assert!(backend.has_buffer(desc.id));

        backend.free(desc.id).expect("Free should succeed");
        assert!(!backend.has_buffer(desc.id));
        assert_eq!(backend.stats().free_count, 1);
    }

    // TDD TEST: Dummy backend execute_op is no-op
    #[test]
    fn test_dummy_backend_execute_op_noop() {
        let mut backend = DummyBackend::new();

        backend.execute_op(&Op::Add, &[], &[]).expect("Execute should succeed");
        backend.execute_op(&Op::Scale { factor: 1.0 }, &[], &[]).expect("Execute should succeed");

        assert_eq!(backend.stats().execute_op_count, 2);
    }

    // TDD TEST: Dummy backend reset clears everything
    #[test]
    fn test_dummy_backend_reset() {
        let mut backend = DummyBackend::new();

        let desc = TensorDesc {
            id: TensorId(0),
            dtype: DType::F32,
            shape: vec![4],
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        };

        backend.alloc(&desc).expect("Alloc should succeed");
        assert!(backend.has_buffer(desc.id));

        backend.reset();
        assert!(!backend.has_buffer(desc.id));
        assert_eq!(backend.stats().alloc_count, 0);
        assert_eq!(backend.current_offset(), 0);
    }

    // TDD TEST: Dummy backend with Graph executor
    #[test]
    fn test_dummy_backend_with_graph() {
        let mut backend = DummyBackend::new();
        let mut graph = Graph::new();

        // Create simple graph
        let input = graph.add_tensor(TensorDesc {
            id: TensorId(0),
            dtype: DType::F32,
            shape: vec![4],
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        });
        let output = graph.add_tensor(TensorDesc {
            id: TensorId(1),
            dtype: DType::F32,
            shape: vec![4],
            layout: Layout::RowMajor,
            strides: vec![1],
            byte_offset: 0,
            view_of: None,
        });
        graph.add_node(Op::Add, vec![input], vec![output]);

        // Execute with dummy backend (no GPU)
        let result = crate::ggml::executor::execute_graph_with_config(
            &mut backend,
            &mut graph,
            crate::ggml::executor::ExecuteConfig::without_optimization(),
        );

        assert!(result.is_ok());
        // Both input and output are allocated (neither is a view)
        assert_eq!(backend.stats().alloc_count, 2);
        assert_eq!(backend.stats().execute_op_count, 1);
    }
}
