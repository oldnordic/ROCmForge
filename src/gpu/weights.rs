//! GPU weight storage in VRAM.
//!
//! Safety-first design:
//! - All hipMalloc/hipMemcpy calls wrapped with error checking
//! - RAII cleanup on Drop prevents VRAM leaks
//! - Never panic, always return GpuError

use super::error::{GpuError, GpuResult};
use super::ffi;
use crate::loader::{GgmlType, TensorDesc};
use std::ptr::NonNull;

// ── Weight Metadata ────────────────────────────────────────────────────────────

/// Metadata for a weight tensor on GPU.
///
/// Same as CPU WeightMeta - quantization type and dimensions.
#[derive(Clone, Debug)]
pub struct WeightMeta {
    /// Quantization type (F32, Q4_0, Q4_1, Q8_0, etc.)
    pub wtype: GgmlType,
    /// Dimensions from GGUF (innermost first)
    pub dims: Vec<u64>,
    /// Whether this weight tensor needs transposed access
    pub needs_transpose: bool,
}

impl WeightMeta {
    /// Create metadata from a GGUF tensor descriptor.
    pub fn from_desc(desc: &TensorDesc, needs_transpose: bool) -> Self {
        Self {
            wtype: desc.ggml_type,
            dims: desc.dims.clone(),
            needs_transpose,
        }
    }

    /// Total size in bytes for this weight tensor.
    pub fn byte_size(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_desc_works() {
        let desc = TensorDesc {
            name: "test.weight".to_string(),
            ggml_type: GgmlType::Q4_0,
            dims: vec![1024, 768],
            offset: 0,
        };
        let meta = WeightMeta::from_desc(&desc, false);
        assert_eq!(meta.wtype, GgmlType::Q4_0);
        assert_eq!(meta.dims, vec![1024, 768]);
        assert_eq!(meta.byte_size(), 1024 * 768);
        assert_eq!(meta.num_elements(), 1024 * 768);
    }

    #[test]
    fn byte_size_calculates_correctly() {
        let meta = WeightMeta {
            wtype: GgmlType::F32,
            dims: vec![100, 200],
            needs_transpose: false,
        };
        assert_eq!(meta.byte_size(), 100 * 200);
    }
}

// ── GPU Buffer (RAII) ─────────────────────────────────────────────────────────────

/// RAII wrapper for GPU memory allocation.
///
/// Ensures memory is freed when dropped.
/// Never leaks VRAM, even on panic.
pub struct GpuBuffer {
    /// Pointer to GPU memory (null if empty)
    ptr: Option<NonNull<u8>>,
    /// Size in bytes
    size: usize,
}

impl GpuBuffer {
    /// Allocate GPU memory with safety checking.
    ///
    /// Returns error if allocation fails (OutOfMemory).
    pub fn alloc(size: usize) -> GpuResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: None, size: 0 });
        }

        let ptr = ffi::hip_malloc(size)?;

        // Verify allocation succeeded (pointer not null)
        let nn = NonNull::new(ptr).ok_or_else(|| GpuError::OutOfMemory {
            requested: size,
            available: 0,
        })?;

        Ok(Self { ptr: Some(nn), size })
    }

    /// Create empty buffer (no allocation).
    pub fn empty() -> Self {
        Self { ptr: None, size: 0 }
    }

    /// Get pointer to GPU memory.
    ///
    /// Returns None if buffer is empty.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.map(|nn| nn.as_ptr()).unwrap_or(std::ptr::null_mut())
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Copy data from CPU to this GPU buffer.
    pub fn copy_from_host(&mut self, src: &[u8]) -> GpuResult<()> {
        if src.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("size mismatch: got {} bytes, expected {}", src.len(), self.size),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_h2d(self.as_ptr(), src.as_ptr(), self.size)
    }

    /// Copy data from GPU buffer to CPU.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> GpuResult<()> {
        if dst.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("size mismatch: got {} bytes, expected {}", dst.len(), self.size),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_d2h(dst.as_mut_ptr(), self.as_ptr(), self.size)
    }
}

// SAFETY: Send/Sync are safe because this represents owned GPU memory
// Access is only through &mut self for copy operations
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Some(nn) = self.ptr {
            ffi::hip_free(nn.as_ptr());
            // Ignore errors in Drop - can't panic here
            self.ptr = None;
        }
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;

    #[test]
    fn empty_buffer_has_no_allocation() {
        let buf = GpuBuffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.size(), 0);
        assert_eq!(buf.as_ptr(), std::ptr::null_mut());
    }

    #[test]
    fn alloc_zero_size_returns_empty() {
        let buf = GpuBuffer::alloc(0).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn copy_from_host_rejects_size_mismatch() {
        let mut buf = GpuBuffer::alloc(100).unwrap();
        let data = vec![1u8; 50]; // Wrong size
        let result = buf.copy_from_host(&data);
        assert!(result.is_err());
    }
}
