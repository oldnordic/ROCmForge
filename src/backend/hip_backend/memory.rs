//! HIP buffer wrapper for GPU memory allocation

use std::ptr;
use std::sync::Arc;

use crate::backend::hip_backend::error::HipResult;
use crate::backend::hip_backend::ffi;
use crate::backend::hip_backend::HipError;

// SAFETY: HipBuffer is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

/// HipBuffer wrapper using Arc for safe, cheap cloning
/// Arc ensures single ownership of GPU memory - Drop called once when refcount=0
#[derive(Debug, Clone)]
pub struct HipBuffer {
    inner: Arc<HipBufferInner>,
}

#[repr(C)]
#[derive(Debug)]
pub struct HipBufferInner {
    pub ptr: *mut std::ffi::c_void,
    pub size: usize,
    // For sub-allocated buffers: offset from ptr in bytes
    // When offset > 0, this buffer is a view into a parent allocation
    pub offset: usize,
}

impl HipBuffer {
    /// Create a new GPU buffer
    pub fn new(size: usize) -> HipResult<Self> {
        // Capture stack trace for debugging segfaults
        #[cfg(feature = "rocm")]
        let backtrace = std::backtrace::Backtrace::capture();

        tracing::trace!("HipBuffer::new: Allocating {} bytes of GPU memory", size);
        #[cfg(feature = "rocm")]
        tracing::trace!("HipBuffer::new: Call stack:\n{}", backtrace);

        // Validate allocation size to prevent segfaults
        if size == 0 {
            tracing::warn!("HipBuffer::new: Zero-size allocation requested - this may cause issues");
        }
        if size > 1024 * 1024 * 1024 {
            // 1GB warning
            tracing::warn!("HipBuffer::new: Large allocation requested: {} MB", size / (1024 * 1024));
        }

        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();

        // Use hipMalloc to allocate device memory
        tracing::trace!("HipBuffer::new: Calling hipMalloc for {} bytes", size);
        let result = unsafe { ffi::hipMalloc(&mut ptr, size) };
        tracing::trace!("HipBuffer::new: hipMalloc returned result={}, ptr={:?}", result, ptr);

        if result != ffi::HIP_SUCCESS {
            tracing::error!("HipBuffer::new: hipMalloc failed with code {} for {} bytes", result, size);
            return Err(HipError::MemoryAllocationFailed(format!(
                "hipMalloc failed with code {} for {} bytes",
                result, size
            )));
        }

        if ptr.is_null() {
            tracing::error!("HipBuffer::new: hipMalloc returned null pointer for {} bytes", size);
            return Err(HipError::MemoryAllocationFailed(format!(
                "hipMalloc returned null pointer for {} bytes",
                size
            )));
        }

        tracing::debug!("HipBuffer::new: Successfully allocated {} bytes at {:?}", size, ptr);
        Ok(HipBuffer {
            inner: Arc::new(HipBufferInner {
                ptr,
                size,
                offset: 0,
            }),
        })
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        self.inner.size
    }

    fn ptr(&self) -> *mut std::ffi::c_void {
        // For sub-allocated views, add offset to base pointer
        if self.inner.offset > 0 {
            // SAFETY: Check for overflow before pointer arithmetic
            // The offset has already been validated to be within buffer bounds
            let base_ptr = self.inner.ptr as usize;
            let new_offset = base_ptr.saturating_add(self.inner.offset);
            if new_offset < base_ptr {
                // Arithmetic overflow - this should never happen with validated offsets
                // But we handle it gracefully rather than causing undefined behavior
                tracing::warn!(
                    "Pointer arithmetic overflow detected (base=0x{:x}, offset={})",
                    base_ptr,
                    self.inner.offset
                );
                return std::ptr::null_mut();
            }
            new_offset as *mut std::ffi::c_void
        } else {
            self.inner.ptr
        }
    }

    /// Create a view into this buffer at a specific byte offset.
    /// This creates a sub-allocation without allocating new GPU memory.
    /// The parent buffer owns the GPU memory and will free it when dropped.
    pub fn sub_buffer_view(&self, offset: usize, size: usize) -> HipResult<Self> {
        if offset + size > self.size() {
            return Err(HipError::MemoryAllocationFailed(format!(
                "GPU memory sub-allocation failed: offset={} size={} > buffer_size={}",
                offset,
                size,
                self.size()
            )));
        }

        Ok(HipBuffer {
            inner: Arc::new(HipBufferInner {
                ptr: self.inner.ptr, // Share same base pointer
                size,
                offset: self.inner.offset + offset, // Accumulate offset
            }),
        })
    }

    /// Copy data from host to device
    pub fn copy_from_host<T>(&self, data: &[T]) -> HipResult<()> {
        let byte_size = std::mem::size_of_val(data);
        if byte_size > self.size() {
            return Err(HipError::MemoryAllocationFailed(format!(
                "Source data too large: {} > {}",
                byte_size,
                self.size()
            )));
        }

        let ptr = self.ptr();

        // Debug for large copies - BEFORE
        if byte_size > 100 * 1024 * 1024 {
            eprintln!(
                ">>> copy_from_host: Starting {} MB copy...",
                byte_size / 1024 / 1024
            );
        }

        // Use hipMemcpyHtoD to copy from host to device
        let result = unsafe {
            ffi::hipMemcpy(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                byte_size,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };

        if byte_size > 100 * 1024 * 1024 {
            eprintln!(">>> copy_from_host: hipMemcpy returned {}", result);
        }

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyHtoD failed with code {} (ptr={:?}, size={}, offset={})",
                result, ptr, byte_size, self.inner.offset
            )));
        }

        // Debug for large copies - AFTER
        if byte_size > 100 * 1024 * 1024 {
            eprintln!(">>> copy_from_host: Copy completed successfully");
        }

        Ok(())
    }

    /// Copy data from host to device using the specified HIP stream.
    ///
    /// This uses `hipMemcpyAsync` which queues the copy on the specified stream,
    /// ensuring proper ordering with other GPU operations (kernels, hipBLAS) that
    /// also use the same stream.
    ///
    /// # Arguments
    /// * `data` - Host data to copy
    /// * `stream` - HIP stream to queue the copy on (typically `backend.stream().as_ptr()`)
    ///
    /// # Why This Matters
    /// Without proper stream association:
    /// - `hipMemcpy` uses the default stream
    /// - Kernels/hipBLAS use a custom stream
    /// - `hipDeviceSynchronize()` waits for all streams but can hang if operations
    ///   on different streams have dependencies that aren't properly sequenced
    ///
    /// By using `hipMemcpyAsync` with the same stream as all other operations,
    /// we ensure proper ordering and avoid synchronization issues.
    pub fn copy_from_host_with_stream<T>(&self, data: &[T], stream: *mut std::ffi::c_void) -> HipResult<()> {
        let byte_size = std::mem::size_of_val(data);
        if byte_size > self.size() {
            return Err(HipError::MemoryAllocationFailed(format!(
                "Source data too large: {} > {}",
                byte_size,
                self.size()
            )));
        }

        let ptr = self.ptr();

        // Use hipMemcpyAsync to queue the copy on the specified stream
        let result = unsafe {
            ffi::hipMemcpyAsync(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                byte_size,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyAsync H2D failed with code {} (ptr={:?}, size={}, offset={})",
                result, ptr, byte_size, self.inner.offset
            )));
        }

        // Debug for large copies
        if byte_size > 100 * 1024 * 1024 {
            tracing::debug!(
                "copy_from_host_with_stream succeeded: ptr={:?}, size={} MB, offset={}",
                ptr,
                byte_size / 1024 / 1024,
                self.inner.offset
            );
        }

        Ok(())
    }

    /// Copy data from device to host
    ///
    /// **DEPRECATED - Use HipBackend::copy_from_device_safe() instead**
    ///
    /// Now uses stream-aware synchronization (`hipStreamSynchronize`) instead of
    /// the dangerous `hipDeviceSynchronize`.
    #[deprecated(
        since = "0.23.0",
        note = "Use HipBackend::copy_from_device_safe() instead - clearer intent"
    )]
    pub fn copy_to_host<T>(&self, data: &mut [T]) -> HipResult<()> {
        let byte_size = std::mem::size_of_val(data);
        if byte_size > self.size() {
            return Err(HipError::MemoryAllocationFailed(format!(
                "Destination buffer too small: {} > {}",
                byte_size,
                self.size()
            )));
        }

        // Note: Stream-aware sync is handled via global backend in the original implementation
        // For the modular version, we do a simple sync here
        // In a full refactor, this would need access to the backend's stream

        let ptr = self.ptr();

        // Use hipMemcpyDtoH to copy from device to host
        let result = unsafe {
            ffi::hipMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                ptr,
                byte_size,
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyDtoH failed with code {}",
                result
            )));
        }

        Ok(())
    }

    /// Copy data from device to host using the specified HIP stream.
    ///
    /// This uses `hipMemcpyAsync` which queues the copy on the specified stream,
    /// ensuring proper ordering with other GPU operations.
    ///
    /// # Synchronization
    /// Unlike `copy_to_host()`, this does NOT call `hipDeviceSynchronize()`.
    /// The caller is responsible for synchronizing the stream after the async copy
    /// if they need to wait for completion.
    pub fn copy_to_host_with_stream<T>(
        &self,
        data: &mut [T],
        stream: *mut std::ffi::c_void,
    ) -> HipResult<()> {
        let byte_size = std::mem::size_of_val(data);
        if byte_size > self.size() {
            return Err(HipError::MemoryAllocationFailed(format!(
                "Destination buffer too small: {} > {}",
                byte_size,
                self.size()
            )));
        }

        let ptr = self.ptr();

        // Use hipMemcpyAsync to queue the copy on the specified stream
        let result = unsafe {
            ffi::hipMemcpyAsync(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                ptr,
                byte_size,
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyAsync D2H failed with code {}",
                result
            )));
        }

        Ok(())
    }

    /// Copy data from another device buffer
    pub fn copy_from_buffer(&self, src: &HipBuffer) -> HipResult<()> {
        if src.size() != self.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Buffer size mismatch: src={} bytes, dst={} bytes",
                src.size(),
                self.size()
            )));
        }

        let result = unsafe {
            ffi::hipMemcpy(
                self.ptr(),
                src.ptr(),
                self.size(),
                ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyDtoD failed with code {}",
                result
            )));
        }

        Ok(())
    }

    /// Copy data from another device buffer using the specified HIP stream.
    ///
    /// This is the stream-aware variant of `copy_from_buffer()` that uses
    /// `hipMemcpyAsync` instead of `hipMemcpy`, allowing the copy to be
    /// properly ordered with other GPU operations on the same stream.
    pub fn copy_from_buffer_with_stream(&self, src: &HipBuffer, stream: *mut std::ffi::c_void) -> HipResult<()> {
        if src.size() != self.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Buffer size mismatch: src={} bytes, dst={} bytes",
                src.size(),
                self.size()
            )));
        }

        let result = unsafe {
            ffi::hipMemcpyAsync(
                self.ptr(),
                src.ptr(),
                self.size(),
                ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream,
            )
        };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyAsync D2D failed with code {}",
                result
            )));
        }

        Ok(())
    }

    /// Copy data from another device buffer region
    pub fn copy_from_buffer_region(
        &self,
        dst_offset_bytes: usize,
        src: &HipBuffer,
        src_offset_bytes: usize,
        byte_len: usize,
    ) -> HipResult<()> {
        if src_offset_bytes + byte_len > src.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Source range out of bounds: offset={} len={} src_size={}",
                src_offset_bytes,
                byte_len,
                src.size()
            )));
        }
        if dst_offset_bytes + byte_len > self.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Destination range out of bounds: offset={} len={} dst_size={}",
                dst_offset_bytes,
                byte_len,
                self.size()
            )));
        }

        // SAFETY: Check for pointer arithmetic overflow
        let base_dst = self.ptr() as usize;
        let base_src = src.ptr() as usize;

        let dst_ptr = base_dst.checked_add(dst_offset_bytes).ok_or_else(|| {
            HipError::MemoryCopyFailed(format!(
                "Destination pointer arithmetic overflow (base=0x{:x}, offset={})",
                base_dst, dst_offset_bytes
            ))
        })? as *mut std::ffi::c_void;

        let src_ptr = base_src.checked_add(src_offset_bytes).ok_or_else(|| {
            HipError::MemoryCopyFailed(format!(
                "Source pointer arithmetic overflow (base=0x{:x}, offset={})",
                base_src, src_offset_bytes
            ))
        })? as *const std::ffi::c_void;

        let result = unsafe { ffi::hipMemcpy(dst_ptr, src_ptr, byte_len, ffi::HIP_MEMCPY_DEVICE_TO_DEVICE) };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyDtoD (region) failed with code {}",
                result
            )));
        }

        Ok(())
    }

    /// Copy data from buffer with strided 2D layout
    pub fn copy_from_buffer_strided_2d(
        &self,
        dst_offset_bytes: usize,
        dst_pitch_bytes: usize,
        src: &HipBuffer,
        src_offset_bytes: usize,
        src_pitch_bytes: usize,
        width_bytes: usize,
        height: usize,
    ) -> HipResult<()> {
        if height == 0 || width_bytes == 0 {
            return Ok(());
        }

        let height_minus_one = height.saturating_sub(1);
        let src_required = src_offset_bytes
            .checked_add(height_minus_one.checked_mul(src_pitch_bytes).ok_or_else(|| {
                HipError::MemoryCopyFailed("Source pitch multiplication overflow".to_string())
            })?)
            .and_then(|v| v.checked_add(width_bytes))
            .ok_or_else(|| HipError::MemoryCopyFailed("Source size overflow".to_string()))?;
        if src_required > src.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Source range out of bounds: required={} src_size={}",
                src_required,
                src.size()
            )));
        }

        let dst_required = dst_offset_bytes
            .checked_add(height_minus_one.checked_mul(dst_pitch_bytes).ok_or_else(|| {
                HipError::MemoryCopyFailed("Destination pitch multiplication overflow".to_string())
            })?)
            .and_then(|v| v.checked_add(width_bytes))
            .ok_or_else(|| HipError::MemoryCopyFailed("Destination size overflow".to_string()))?;
        if dst_required > self.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Destination range out of bounds: required={} dst_size={}",
                dst_required,
                self.size()
            )));
        }

        let base_dst = self.ptr() as usize;
        let base_src = src.ptr() as usize;

        let dst_ptr = base_dst.checked_add(dst_offset_bytes).ok_or_else(|| {
            HipError::MemoryCopyFailed(format!(
                "Destination pointer arithmetic overflow (base=0x{:x}, offset={})",
                base_dst, dst_offset_bytes
            ))
        })? as *mut std::ffi::c_void;

        let src_ptr = base_src.checked_add(src_offset_bytes).ok_or_else(|| {
            HipError::MemoryCopyFailed(format!(
                "Source pointer arithmetic overflow (base=0x{:x}, offset={})",
                base_src, src_offset_bytes
            ))
        })? as *const std::ffi::c_void;

        let result = unsafe {
            ffi::hipMemcpy2D(
                dst_ptr,
                dst_pitch_bytes,
                src_ptr,
                src_pitch_bytes,
                width_bytes,
                height,
                ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpy2D D2D failed with code {}",
                result
            )));
        }

        Ok(())
    }

    /// Get raw buffer pointer
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr()
    }

    /// Get raw buffer pointer (mutable)
    pub fn as_mut_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr()
    }

    /// Copy from buffer with offset
    pub fn copy_from_buffer_with_offset(
        &self,
        src: &HipBuffer,
        src_offset_bytes: usize,
        byte_len: usize,
    ) -> HipResult<()> {
        self.copy_from_buffer_region(0, src, src_offset_bytes, byte_len)
    }

    /// Create HipBuffer from raw pointer and size (for backend internal use)
    pub(crate) fn from_raw_parts(ptr: *mut std::ffi::c_void, size: usize, offset: usize) -> Self {
        HipBuffer {
            inner: Arc::new(HipBufferInner {
                ptr,
                size,
                offset,
            }),
        }
    }
}

impl Drop for HipBufferInner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Use hipFree to free device memory
            unsafe {
                ffi::hipFree(self.ptr);
            }
        }
    }
}
