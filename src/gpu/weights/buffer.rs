use super::super::error::{GpuError, GpuResult};
use super::super::ffi;
use super::super::ffi::hipStream_t;
use super::super::vram_budget::{
    active_or_default_device_id, query_vram_budget, DESKTOP_VRAM_RESERVATION_BYTES,
    VRAM_SAFETY_MARGIN_RATIO,
};
use std::ptr::NonNull;

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
    /// Checks available VRAM before allocation to prevent stealing memory from desktop.
    pub fn alloc(size: usize) -> GpuResult<Self> {
        Self::alloc_for_device(size, active_or_default_device_id())
    }

    /// Allocate GPU memory on a specific device with safety checking.
    pub fn alloc_for_device(size: usize, device_id: i32) -> GpuResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: None, size: 0 });
        }

        ffi::hip_set_device(device_id)?;
        let budget = query_vram_budget(device_id)?;
        if size > budget.safe_allocation_size {
            return Err(GpuError::OutOfMemory {
                requested: size,
                available: budget.safe_allocation_size,
                hint: format!(
                    "Device {} only has {} MB safely allocatable ({} MB free, {} MB reserved for desktop, {}% safety margin, {} MB total VRAM).",
                    device_id,
                    budget.safe_allocation_size / (1024 * 1024),
                    budget.free_vram / (1024 * 1024),
                    DESKTOP_VRAM_RESERVATION_BYTES / (1024 * 1024),
                    (VRAM_SAFETY_MARGIN_RATIO * 100.0) as u32,
                    budget.total_vram / (1024 * 1024)
                ),
            });
        }

        let ptr = ffi::hip_malloc(size)?;

        // Verify allocation succeeded (pointer not null)
        let nn = NonNull::new(ptr).ok_or_else(|| GpuError::OutOfMemory {
            requested: size,
            available: 0,
            hint: "hipMalloc returned null pointer".to_string(),
        })?;

        Ok(Self {
            ptr: Some(nn),
            size,
        })
    }

    /// Create empty buffer (no allocation).
    pub fn empty() -> Self {
        Self { ptr: None, size: 0 }
    }

    /// Get pointer to GPU memory.
    ///
    /// Returns None if buffer is empty.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
            .map(|nn| nn.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Set size in bytes (only for empty or manually managed buffers).
    pub fn set_size(&mut self, size: usize) {
        self.size = size;
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
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    src.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_h2d(self.as_ptr(), src.as_ptr(), self.size)
    }

    /// Copy data from CPU to this GPU buffer on an explicit HIP stream.
    pub fn copy_from_host_on_stream(&mut self, src: &[u8], stream: hipStream_t) -> GpuResult<()> {
        if src.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    src.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_h2d_async(self.as_ptr(), src.as_ptr(), self.size, stream)
    }

    /// Copy data from GPU buffer to CPU.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> GpuResult<()> {
        if dst.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    dst.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_d2h(dst.as_mut_ptr(), self.as_ptr(), self.size)
    }

    /// Helper to download GPU data directly to a Vec<f32>.
    pub fn copy_to_host_vec(&self) -> GpuResult<Vec<f32>> {
        let count = self.size / std::mem::size_of::<f32>();
        let mut dst = vec![0.0f32; count];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, self.size) };
        self.copy_to_host(bytes)?;
        Ok(dst)
    }
}

// ── GPU Pinned Buffer (RAII) ──────────────────────────────────────────────────────

/// RAII wrapper for Pinned (Page-locked) Host memory.
///
/// Allows for high-speed DMA transfers and zero-copy access from GPU.
pub struct GpuPinnedBuffer {
    ptr: Option<NonNull<u8>>,
    size: usize,
}

impl GpuPinnedBuffer {
    pub fn alloc(size: usize) -> GpuResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: None, size: 0 });
        }

        let ptr = ffi::hip_host_malloc(size)?;
        let nn = NonNull::new(ptr).ok_or_else(|| GpuError::OutOfMemory {
            requested: size,
            available: 0,
            hint: "hipHostMalloc returned null pointer".to_string(),
        })?;

        Ok(Self {
            ptr: Some(nn),
            size,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
            .map(|nn| nn.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    pub fn as_slice<T>(&self) -> &[T] {
        if self.size == 0 {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const T,
                self.size / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        if self.size == 0 {
            return &mut [];
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_ptr() as *mut T,
                self.size / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for GpuPinnedBuffer {
    fn drop(&mut self) {
        if let Some(nn) = self.ptr {
            let _ = ffi::hip_host_free(nn.as_ptr());
        }
    }
}

unsafe impl Send for GpuPinnedBuffer {}
unsafe impl Sync for GpuPinnedBuffer {}

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
