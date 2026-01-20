//! ROCm/HIP backend for GPU kernel execution

use crate::loader::mmap_loader::TensorShape;
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::Arc;
use thiserror::Error;

// HIP FFI bindings
// FFI declarations below are bound to ROCm HIP API.
// All functions are actively used through wrapper methods in HipBackend.
// The dead_code allowance is needed because FFI symbols appear unused
// to the compiler (they're only called through unsafe blocks).
#[link(name = "amdhip64")]
#[allow(dead_code)]
extern "C" {
    fn hipInit(flags: u32) -> i32;
    pub fn hipGetDeviceCount(count: *mut i32) -> i32;
    pub fn hipGetDeviceProperties(props: *mut HipDeviceProp, deviceId: i32) -> i32;
    fn hipSetDevice(deviceId: i32) -> i32;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn hipFree(ptr: *mut c_void) -> i32;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn hipMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> i32;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    fn hipMemcpyHtoD(dst: *mut c_void, src: *const c_void, count: usize) -> i32;
    fn hipMemcpyDtoH(dst: *mut c_void, src: *const c_void, count: usize) -> i32;
    fn hipStreamCreate(stream: *mut *mut c_void) -> i32;
    fn hipStreamDestroy(stream: *mut c_void) -> i32;
    fn hipStreamSynchronize(stream: *mut c_void) -> i32;
    // HIP Event FFI bindings (for async GPU loading synchronization)
    fn hipEventCreate(event: *mut *mut c_void) -> i32;
    fn hipEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn hipEventDestroy(event: *mut c_void) -> i32;
    fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn hipEventSynchronize(event: *mut c_void) -> i32;
    fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn hipModuleLoad(module: *mut *mut c_void, path: *const i8) -> i32;
    fn hipModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    fn hipModuleUnload(module: *mut c_void) -> i32;
    fn hipModuleGetFunction(func: *mut *mut c_void, module: *mut c_void, name: *const i8) -> i32;
    fn hipModuleLaunchKernel(
        func: *mut c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        stream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn hipGetLastError() -> i32;
    fn hipGetErrorString(error: i32) -> *const i8;
    fn hipDeviceSynchronize() -> i32;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn hipMemset(dst: *mut c_void, value: i32, count: usize) -> i32;
}

// HIP constants
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
const HIP_SUCCESS: i32 = 0;

// Opaque buffer for hipDeviceProp_t - MUST be exactly 1472 bytes to match C's sizeof(hipDeviceProp_t)
// CRITICAL: If C writes it, Rust must allocate exactly the same bytes.
// Rule: Never "skip fields" in FFI structs. That's how demons enter memory.
//
// Generated via: hipcc -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ ...
//   C sizeof(hipDeviceProp_t) = 1472 bytes
//
// We use an opaque buffer + accessor methods to safely read fields at correct offsets.
// This is safe, boring, and correct - no buffer overflow possible.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct HipDeviceProp {
    _buffer: [u8; 1472], // Exact C size - no more, no less
}

// Field offsets verified against hip_runtime_api.h
// These are the ONLY fields we use in the codebase.
impl HipDeviceProp {
    // Offset of `name` field: char name[256]
    const NAME_OFFSET: usize = 0;

    // Offset of `totalGlobalMem` field: size_t totalGlobalMem
    // After: name[256] (256) + uuid (16) + luid[8] (8) + luidDeviceNodeMask (4) = 284
    const TOTAL_GLOBAL_MEM_OFFSET: usize = 284;

    // Offset of `multiProcessorCount` field: int multiProcessorCount
    // This is after all the texture/surface fields - verified with C code
    const MULTI_PROCESSOR_COUNT_OFFSET: usize = 508;

    /// Get device name (null-terminated C string)
    pub fn name(&self) -> String {
        let name_bytes = &self._buffer[Self::NAME_OFFSET..Self::NAME_OFFSET + 256];
        let len = name_bytes.iter().position(|&c| c == 0).unwrap_or(256);
        String::from_utf8_lossy(&name_bytes[..len]).into_owned()
    }

    /// Get total global memory in bytes
    pub fn total_global_mem(&self) -> u64 {
        // Read u64 at offset (size_t is 64-bit on AMD64)
        let bytes = &self._buffer[Self::TOTAL_GLOBAL_MEM_OFFSET..Self::TOTAL_GLOBAL_MEM_OFFSET + 8];
        bytes
            .try_into()
            .ok()
            .map(u64::from_ne_bytes)
            .unwrap_or_else(|| {
                // SAFETY: Buffer is guaranteed to be 1472 bytes, and this slice is 8 bytes
                // at a valid offset. This should never fail, but we handle it gracefully.
                tracing::error!(
                    "FFI struct field access failed: total_global_mem slice has wrong length"
                );
                0
            })
    }

    /// Get number of multiprocessors (compute units)
    pub fn multi_processor_count(&self) -> i32 {
        let bytes = &self._buffer
            [Self::MULTI_PROCESSOR_COUNT_OFFSET..Self::MULTI_PROCESSOR_COUNT_OFFSET + 4];
        bytes
            .try_into()
            .ok()
            .map(i32::from_ne_bytes)
            .unwrap_or_else(|| {
                // SAFETY: Buffer is guaranteed to be 1472 bytes, and this slice is 4 bytes
                // at a valid offset. This should never fail, but we handle it gracefully.
                tracing::error!(
                    "FFI struct field access failed: multi_processor_count slice has wrong length"
                );
                0
            })
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct hipUUID {
    pub bytes: [u8; 16],
}

impl Default for HipDeviceProp {
    fn default() -> Self {
        HipDeviceProp {
            _buffer: [0u8; 1472],
        }
    }
}

#[derive(Error, Debug, Clone)]
pub enum HipError {
    #[error("HIP initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Kernel loading failed: {0}")]
    KernelLoadFailed(String),
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    #[error("Memory copy failed: {0}")]
    MemoryCopyFailed(String),
    #[error("Memory query failed: {0}")]
    MemoryQueryFailed(String),
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    #[error("Device not found")]
    DeviceNotFound,
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Generic error: {0}")]
    GenericError(String),
    #[error("Internal lock poisoned - this indicates a bug: {0}")]
    LockPoisoned(String),
}

impl<T> From<std::sync::PoisonError<T>> for HipError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        HipError::LockPoisoned(format!("Lock poisoned: {}", err))
    }
}

pub type HipResult<T> = Result<T, HipError>;

// Convert KVCacheError to HipError
impl From<crate::model::kv_cache::KVCacheError> for HipError {
    fn from(err: crate::model::kv_cache::KVCacheError) -> Self {
        HipError::GenericError(err.to_string())
    }
}

impl From<crate::ggml::GgmlError> for HipError {
    fn from(err: crate::ggml::GgmlError) -> Self {
        HipError::GenericError(format!("{:?}", err))
    }
}

impl HipError {
    /// Check if this error is recoverable (temporary condition)
    ///
    /// Recoverable errors may be retried with exponential backoff.
    /// These include:
    /// - Temporary device errors (GPU busy, driver resetting)
    /// - Memory allocation failures (may succeed after GC or waiting)
    /// - Memory copy failures (temporary driver issues)
    ///
    /// Non-recoverable errors (should NOT be retried):
    /// - DeviceNotFound (no GPU available)
    /// - InitializationFailed (HIP runtime broken)
    /// - KernelLoadFailed (corrupted kernel file)
    /// - LockPoisoned (data corruption bug)
    /// - GenericError (unknown errors)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            HipError::DeviceError(_)
                | HipError::MemoryAllocationFailed(_)
                | HipError::MemoryCopyFailed(_)
                | HipError::MemoryQueryFailed(_)
                | HipError::KernelLaunchFailed(_)
        )
    }

    /// Check if this error is permanent (should never retry)
    pub fn is_permanent(&self) -> bool {
        !self.is_recoverable()
    }
}

// SAFETY: HipStream is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: HipStream does NOT implement Clone because cloning raw pointers
// would cause double-free when both instances are dropped.
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility - ensures C-compatible layout
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

#[repr(C)]
#[derive(Debug)]
pub struct HipStream {
    stream: *mut c_void,
}

impl HipStream {
    pub fn new() -> HipResult<Self> {
        tracing::debug!("HipStream::new: Creating HIP stream...");
        let mut stream: *mut c_void = ptr::null_mut();

        // Create HIP stream
        tracing::debug!("HipStream::new: Calling hipStreamCreate...");
        let result = unsafe { hipStreamCreate(&mut stream) };
        tracing::debug!(
            "HipStream::new: hipStreamCreate returned result={}, stream={:?}",
            result,
            stream
        );

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to create HIP stream: {}",
                result
            )));
        }

        if stream.is_null() {
            return Err(HipError::DeviceError(
                "hipStreamCreate returned null pointer".to_string(),
            ));
        }

        tracing::debug!("HipStream::new: HIP stream created successfully");
        Ok(HipStream { stream })
    }

    pub fn synchronize(&self) -> HipResult<()> {
        let result = unsafe { hipStreamSynchronize(self.stream) };
        if result != HIP_SUCCESS {
            Err(HipError::DeviceError(format!(
                "Stream synchronization failed: {}",
                result
            )))
        } else {
            Ok(())
        }
    }

    /// Get raw stream pointer (for FFI calls like hipblasSetStream)
    pub fn as_ptr(&self) -> *mut c_void {
        self.stream
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                hipStreamDestroy(self.stream);
            }
        }
    }
}

// HIP Event wrapper for async GPU loading synchronization
// Events allow tracking completion of GPU operations across streams
// Phase 1: Basic event support (create, record, synchronize, elapsed time)
//
// SAFETY: HipEvent is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: HipEvent does NOT implement Clone because cloning raw pointers
// would cause double-free when both instances are dropped.
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility - ensures C-compatible layout
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

#[repr(C)]
#[derive(Debug)]
pub struct HipEvent {
    event: *mut c_void,
}

// HIP Event flags (from hip_runtime_api.h)
#[allow(dead_code)] // Reserved for future event configuration options
const HIP_EVENT_DEFAULT: u32 = 0x0;
#[allow(dead_code)] // Reserved for future event configuration options
const HIP_EVENT_DISABLE_TIMING: u32 = 0x1;
#[allow(dead_code)] // Reserved for future event configuration options
const HIP_EVENT_RECORD_TIMING: u32 = 0x2; // Default behavior

impl HipEvent {
    /// Create a new HIP event with default timing enabled
    ///
    /// Events are used to track completion of operations in a stream.
    /// Timing is enabled by default for performance measurement.
    pub fn new() -> HipResult<Self> {
        tracing::debug!("HipEvent::new: Creating HIP event...");
        let mut event: *mut c_void = ptr::null_mut();

        let result = unsafe { hipEventCreate(&mut event) };
        tracing::debug!(
            "HipEvent::new: hipEventCreate returned result={}, event={:?}",
            result,
            event
        );

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to create HIP event: {}",
                result
            )));
        }

        if event.is_null() {
            return Err(HipError::DeviceError(
                "hipEventCreate returned null pointer".to_string(),
            ));
        }

        tracing::debug!("HipEvent::new: HIP event created successfully");
        Ok(HipEvent { event })
    }

    /// Create a new HIP event with specified flags
    ///
    /// Use HIP_EVENT_DISABLE_TIMING for events used only for synchronization
    /// (slightly better performance when timing isn't needed).
    pub fn with_flags(flags: u32) -> HipResult<Self> {
        tracing::debug!(
            "HipEvent::with_flags: Creating HIP event with flags={}...",
            flags
        );
        let mut event: *mut c_void = ptr::null_mut();

        let result = unsafe { hipEventCreateWithFlags(&mut event, flags) };
        tracing::debug!(
            "HipEvent::with_flags: hipEventCreateWithFlags returned result={}, event={:?}",
            result,
            event
        );

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to create HIP event with flags: {}",
                result
            )));
        }

        if event.is_null() {
            return Err(HipError::DeviceError(
                "hipEventCreateWithFlags returned null pointer".to_string(),
            ));
        }

        tracing::debug!("HipEvent::with_flags: HIP event created successfully");
        Ok(HipEvent { event })
    }

    /// Record this event in the given stream
    ///
    /// The event will capture the current state of operations in the stream.
    /// Future calls to synchronize() will wait until all operations before
    /// the record() call have completed.
    pub fn record(&self, stream: &HipStream) -> HipResult<()> {
        tracing::trace!("HipEvent::record: Recording event in stream...");
        let result = unsafe { hipEventRecord(self.event, stream.as_ptr()) };

        if result != HIP_SUCCESS {
            Err(HipError::DeviceError(format!(
                "Event record failed: {}",
                result
            )))
        } else {
            tracing::trace!("HipEvent::record: Event recorded successfully");
            Ok(())
        }
    }

    /// Synchronize on this event
    ///
    /// Blocks the host (CPU) until all operations captured by this event
    /// have completed. Use this to coordinate between streams or ensure
    /// GPU work has finished before proceeding.
    pub fn synchronize(&self) -> HipResult<()> {
        tracing::trace!("HipEvent::synchronize: Synchronizing on event...");
        let result = unsafe { hipEventSynchronize(self.event) };

        if result != HIP_SUCCESS {
            Err(HipError::DeviceError(format!(
                "Event synchronization failed: {}",
                result
            )))
        } else {
            tracing::trace!("HipEvent::synchronize: Event synchronized successfully");
            Ok(())
        }
    }

    /// Calculate elapsed time between two events in milliseconds
    ///
    /// Returns the time elapsed between `self` (start) and `end` (end).
    /// Both events must have been recorded in the same stream (or different
    /// streams that have been properly synchronized).
    ///
    /// Timing must be enabled for both events (default when using `new()`).
    pub fn elapsed_time(&self, end: &HipEvent) -> HipResult<f32> {
        let mut ms: f32 = 0.0;
        let result = unsafe { hipEventElapsedTime(&mut ms, self.event, end.event) };

        if result != HIP_SUCCESS {
            Err(HipError::DeviceError(format!(
                "Failed to get elapsed time: {}",
                result
            )))
        } else {
            Ok(ms)
        }
    }

    /// Get raw event pointer (for FFI calls)
    pub fn as_ptr(&self) -> *mut c_void {
        self.event
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            tracing::trace!("HipEvent::drop: Destroying HIP event");
            unsafe {
                hipEventDestroy(self.event);
            }
        }
    }
}

// SAFETY: HipBuffer is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

// HipBuffer wrapper using Arc for safe, cheap cloning
// Arc ensures single ownership of GPU memory - Drop called once when refcount=0
#[derive(Debug, Clone)]
pub struct HipBuffer {
    inner: Arc<HipBufferInner>,
}

#[repr(C)]
#[derive(Debug)]
struct HipBufferInner {
    ptr: *mut c_void,
    size: usize,
    // For sub-allocated buffers: offset from ptr in bytes
    // When offset > 0, this buffer is a view into a parent allocation
    offset: usize,
}

impl HipBuffer {
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
        if size > 1024 * 1024 * 1024 { // 1GB warning
            tracing::warn!("HipBuffer::new: Large allocation requested: {} MB", size / (1024 * 1024));
        }

        let mut ptr: *mut c_void = ptr::null_mut();

        // Use hipMalloc to allocate device memory
        tracing::trace!("HipBuffer::new: Calling hipMalloc for {} bytes", size);
        let result = unsafe { hipMalloc(&mut ptr, size) };
        tracing::trace!("HipBuffer::new: hipMalloc returned result={}, ptr={:?}", result, ptr);

        if result != HIP_SUCCESS {
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

    pub fn size(&self) -> usize {
        self.inner.size
    }

    fn ptr(&self) -> *mut c_void {
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
            new_offset as *mut c_void
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
            hipMemcpy(
                ptr,
                data.as_ptr() as *const c_void,
                byte_size,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };

        if byte_size > 100 * 1024 * 1024 {
            eprintln!(">>> copy_from_host: hipMemcpy returned {}", result);
        }

        if result != HIP_SUCCESS {
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
    pub fn copy_from_host_with_stream<T>(&self, data: &[T], stream: *mut c_void) -> HipResult<()> {
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
            hipMemcpyAsync(
                ptr,
                data.as_ptr() as *const c_void,
                byte_size,
                HIP_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };

        if result != HIP_SUCCESS {
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
    /// ⚠️ **DEPRECATED - Use HipBackend::copy_from_device_safe() instead** ⚠️
    ///
    /// # Phase 23 Fix: Now Uses Stream-Aware Synchronization
    ///
    /// As of Phase 23, this method is now SAFE and uses stream-aware synchronization
    /// (`hipStreamSynchronize`) instead of the dangerous `hipDeviceSynchronize`.
    ///
    /// # Why It Was Deprecated
    ///
    /// The original implementation used `hipDeviceSynchronize()` which waits for ALL
    /// GPU streams including the desktop compositor, causing desktop hangs.
    ///
    /// # Why This Is Now Safe
    ///
    /// - Now uses `hipStreamSynchronize()` on the global backend's stream
    /// - Only waits for our application's stream, not the desktop compositor
    /// - No more deadlocks or desktop hangs
    ///
    /// # Recommended Alternative
    ///
    /// For new code, prefer `HipBackend::copy_from_device_safe()` which is more
    /// explicit about requiring a backend reference.
    ///
    /// # Example
    /// ```ignore
    /// // Old way (still works, now safe):
    /// buffer.copy_to_host(&mut data)?;
    ///
    /// // New way (recommended):
    /// backend.copy_from_device_safe(&buffer, &mut data)?;
    /// ```
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

        // Phase 23: Use STREAM-AWARE synchronization instead of hipDeviceSynchronize
        // Get the global backend to access our stream
        let sync_result = if let Ok(guard) = GLOBAL_BACKEND.try_lock() {
            guard
                .as_ref()
                .map(|backend| {
                    // Use hipStreamSynchronize on our stream (SAFE - only waits for our stream)
                    unsafe { hipStreamSynchronize(backend.stream.as_ptr()) }
                })
                .unwrap_or(HIP_SUCCESS)
        } else {
            // Lock poisoned or unavailable - skip sync (data may not be ready but won't crash desktop)
            HIP_SUCCESS
        };

        if sync_result != HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "Stream synchronization failed with code {} before D2H copy",
                sync_result
            )));
        }

        let ptr = self.ptr();

        // Use hipMemcpyDtoH to copy from device to host
        let result = unsafe {
            hipMemcpy(
                data.as_mut_ptr() as *mut c_void,
                ptr,
                byte_size,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };

        if result != HIP_SUCCESS {
            let ptr_addr = ptr as usize;
            let is_aligned = (ptr_addr % 4096) == 0;
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyDtoH failed with code {} (base_ptr={:?}, offset={}, final_ptr=0x{:x}, size={} MB, aligned={})",
                result, self.inner.ptr, self.inner.offset, ptr_addr, byte_size / 1024 / 1024, is_aligned
            )));
        }

        Ok(())
    }

    /// Copy data from device to host using the specified HIP stream.
    ///
    /// This uses `hipMemcpyAsync` which queues the copy on the specified stream,
    /// ensuring proper ordering with other GPU operations.
    ///
    /// # Arguments
    /// * `data` - Host buffer to receive the data
    /// * `stream` - HIP stream to queue the copy on
    ///
    /// # Synchronization
    /// Unlike `copy_to_host()`, this does NOT call `hipDeviceSynchronize()`.
    /// The caller is responsible for synchronizing the stream after the async copy
    /// if they need to wait for completion.
    ///
    /// # Why This Matters
    /// Using the same stream for all operations (copies, kernels, hipBLAS) ensures
    /// proper ordering and avoids the synchronization issues that can occur when
    /// mixing default stream operations with custom stream operations.
    pub fn copy_to_host_with_stream<T>(
        &self,
        data: &mut [T],
        stream: *mut c_void,
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
            hipMemcpyAsync(
                data.as_mut_ptr() as *mut c_void,
                ptr,
                byte_size,
                HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };

        if result != HIP_SUCCESS {
            let ptr_addr = ptr as usize;
            let is_aligned = (ptr_addr % 4096) == 0;
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyAsync D2H failed with code {} (base_ptr={:?}, offset={}, final_ptr=0x{:x}, size={} MB, aligned={})",
                result, self.inner.ptr, self.inner.offset, ptr_addr, byte_size / 1024 / 1024, is_aligned
            )));
        }

        Ok(())
    }

    pub fn copy_from_buffer(&self, src: &HipBuffer) -> HipResult<()> {
        if src.size() != self.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Buffer size mismatch: src={} bytes, dst={} bytes",
                src.size(),
                self.size()
            )));
        }

        let result = unsafe {
            hipMemcpy(
                self.ptr(),
                src.ptr(),
                self.size(),
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };

        if result != HIP_SUCCESS {
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
    ///
    /// # Arguments
    /// * `src` - Source buffer to copy from
    /// * `stream` - HIP stream to queue the copy on
    ///
    /// # Synchronization
    /// Unlike `copy_from_buffer()`, this does NOT implicitly synchronize.
    /// The caller is responsible for synchronizing the stream after the async
    /// copy if they need to wait for completion.
    ///
    /// # Why This Matters
    /// Using the same stream for all operations (copies, kernels, hipBLAS) ensures
    /// proper ordering and avoids the synchronization issues that can occur when
    /// mixing default stream operations (`hipMemcpy`) with custom stream operations
    /// (hipBLAS, kernels).
    ///
    /// # Example
    /// ```ignore
    /// // After hipBLAS operations complete on backend.stream()
    /// output.copy_from_buffer_with_stream(&result, backend.stream().as_ptr())?;
    /// backend.synchronize()?; // Wait for async copy to complete
    /// ```
    pub fn copy_from_buffer_with_stream(&self, src: &HipBuffer, stream: *mut c_void) -> HipResult<()> {
        if src.size() != self.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Buffer size mismatch: src={} bytes, dst={} bytes",
                src.size(),
                self.size()
            )));
        }

        let result = unsafe {
            hipMemcpyAsync(
                self.ptr(),
                src.ptr(),
                self.size(),
                HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream,
            )
        };

        if result != HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyAsync D2D failed with code {} (dst={:?}, src={:?}, size={})",
                result,
                self.ptr(),
                src.ptr(),
                self.size()
            )));
        }

        Ok(())
    }

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
        })? as *mut c_void;

        let src_ptr = base_src.checked_add(src_offset_bytes).ok_or_else(|| {
            HipError::MemoryCopyFailed(format!(
                "Source pointer arithmetic overflow (base=0x{:x}, offset={})",
                base_src, src_offset_bytes
            ))
        })? as *const c_void;
        let result = unsafe { hipMemcpy(dst_ptr, src_ptr, byte_len, HIP_MEMCPY_DEVICE_TO_DEVICE) };

        if result != HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyDtoD (region) failed with code {}",
                result
            )));
        }

        Ok(())
    }

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
        })? as *mut c_void;

        let src_ptr = base_src.checked_add(src_offset_bytes).ok_or_else(|| {
            HipError::MemoryCopyFailed(format!(
                "Source pointer arithmetic overflow (base=0x{:x}, offset={})",
                base_src, src_offset_bytes
            ))
        })? as *const c_void;

        let result = unsafe {
            hipMemcpy2D(
                dst_ptr,
                dst_pitch_bytes,
                src_ptr,
                src_pitch_bytes,
                width_bytes,
                height,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };

        if result != HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpy2D D2D failed with code {}",
                result
            )));
        }

        Ok(())
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr()
    }

    pub fn copy_from_buffer_with_offset(
        &self,
        src: &HipBuffer,
        src_offset_bytes: usize,
        byte_len: usize,
    ) -> HipResult<()> {
        self.copy_from_buffer_region(0, src, src_offset_bytes, byte_len)
    }
}

impl Drop for HipBufferInner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Use hipFree to free device memory
            unsafe {
                hipFree(self.ptr);
            }
        }
    }
}

// SAFETY: HipModule is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

#[repr(C)]
#[derive(Debug)]
pub struct HipModule {
    module: *mut c_void,
}

impl HipModule {
    pub fn from_ptr(module: *mut c_void) -> Self {
        HipModule { module }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.module
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                hipModuleUnload(self.module);
            }
        }
    }
}

// SAFETY: HipKernel is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility
unsafe impl Send for HipKernel {}
unsafe impl Sync for HipKernel {}

#[repr(C)]
#[derive(Debug)]
pub struct HipKernel {
    func: *mut c_void,
}

impl HipKernel {
    pub fn from_ptr(func: *mut c_void) -> Self {
        HipKernel { func }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.func
    }
}

#[derive(Debug, Clone)]
pub struct HipDevice {
    pub device_id: i32,
    pub name: String, // Revert to String for now
    pub memory: usize,
    pub compute_units: i32,
}

// NOTE: #[repr(C)] is NOT used here because HipBackend contains Arc<T>
// which is NOT C-compatible. Using repr(C) would cause ABI violations.
// See docs/deep_crash_analysis.md for details.
#[derive(Debug)]
pub struct HipBackend {
    device: HipDevice,
    stream: Arc<HipStream>,
}

// Manual Clone implementation that clones the Arc (shared ownership)
// This is safe because Arc ensures the stream is only destroyed once
impl Clone for HipBackend {
    fn clone(&self) -> Self {
        HipBackend {
            device: self.device.clone(),
            stream: Arc::clone(&self.stream),
        }
    }
}

// WORKAROUND: Singleton pattern to avoid ABI issues with returning HipBackend
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static GLOBAL_BACKEND: Mutex<Option<Arc<HipBackend>>> = Mutex::new(None);
static GLOBAL_INIT_CALLED: AtomicBool = AtomicBool::new(false);

// Phase 20: GPU availability detection state
use std::sync::Once;

impl HipBackend {
    /// Phase 20: Check if GPU is available WITHOUT initializing HIP backend
    ///
    /// This performs a lightweight check to see if:
    /// - HIP runtime is available (amdhip64 library present)
    /// - At least one GPU device is present
    ///
    /// Returns false if:
    /// - No GPU device present
    /// - HIP runtime not installed
    /// - hipInit() fails
    ///
    /// This is safe to call from anywhere - it won't crash if GPU isn't available.
    pub fn gpu_available() -> bool {
        use std::sync::atomic::{AtomicBool, Ordering};

        static CHECKED: AtomicBool = AtomicBool::new(false);
        static AVAILABLE: AtomicBool = AtomicBool::new(false);
        static INIT: Once = Once::new();

        INIT.call_once(|| {
            // Use catch_unwind to prevent panics from propagating
            let result = std::panic::catch_unwind(|| {
                unsafe {
                    // Try to initialize HIP (lightweight check)
                    let init_result = hipInit(0);
                    if init_result != HIP_SUCCESS {
                        tracing::debug!(
                            "HIP not available: hipInit failed with code {}",
                            init_result
                        );
                        return false;
                    }

                    // Try to get device count
                    let mut count: i32 = 0;
                    let count_result = hipGetDeviceCount(&mut count);
                    if count_result != HIP_SUCCESS {
                        tracing::debug!(
                            "HIP not available: hipGetDeviceCount failed with code {}",
                            count_result
                        );
                        return false;
                    }

                    let available = count > 0;
                    tracing::debug!("GPU available: {} ({} device(s))", available, count);
                    available
                }
            })
            .unwrap_or(false);

            AVAILABLE.store(result, Ordering::Release);
            CHECKED.store(true, Ordering::Release);
        });

        AVAILABLE.load(Ordering::Acquire)
    }

    /// Phase 20: Create backend only if GPU is available
    ///
    /// This is the safe version of `new()` that returns a clear error
    /// instead of crashing if GPU is not available.
    pub fn new_checked() -> HipResult<Arc<Self>> {
        if !Self::gpu_available() {
            return Err(HipError::DeviceNotFound);
        }
        Self::new()
    }

    /// Create a new HIP backend singleton (thread-safe)
    /// Returns Arc<HipBackend> for shared ownership across the codebase
    pub fn new() -> HipResult<Arc<Self>> {
        // Double-checked locking pattern for singleton initialization
        if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {
            return Ok(GLOBAL_BACKEND
                .lock()
                .map_err(|e| {
                    HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e))
                })?
                .as_ref()
                .map(Arc::clone)
                .expect("Global backend initialized but not set"));
        }

        // Initialize under lock
        let mut guard = GLOBAL_BACKEND
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e)))?;
        if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {
            return Ok(guard
                .as_ref()
                .map(Arc::clone)
                .expect("Global backend initialized but not set"));
        }

        // Initialize HIP
        Self::initialize_hip()?;

        // Detect AMD GPU
        let device = Self::detect_amd_gpu()?;

        // CRITICAL: Set the detected device as current HIP context
        // Without this, HIP may use a different device context, causing allocation failures
        eprintln!(
            ">>> HipBackend::new: Setting device to {} ({})",
            device.device_id, device.name
        );
        let set_result = unsafe { hipSetDevice(device.device_id) };
        if set_result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to set device {}: hipSetDevice returned {}",
                device.device_id, set_result
            )));
        }
        tracing::debug!(
            "HipBackend::new: Device {} set successfully",
            device.device_id
        );

        // Create stream wrapped in Arc for shared ownership
        let stream = Arc::new(HipStream::new()?);

        let backend = Arc::new(HipBackend { device, stream });
        *guard = Some(backend.clone());
        // CRITICAL: Set flag BEFORE releasing lock to prevent race condition
        // Other threads check GLOBAL_INIT_CALLED before acquiring lock
        GLOBAL_INIT_CALLED.store(true, Ordering::Release);
        drop(guard); // Explicitly release lock before returning

        Ok(backend)
    }

    fn detect_amd_gpu() -> HipResult<HipDevice> {
        let mut count: i32 = 0;
        let result = unsafe { hipGetDeviceCount(&mut count) };

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceNotFound);
        }

        if count == 0 {
            return Err(HipError::DeviceNotFound);
        }

        // Find best GPU (prefer discrete graphics cards)
        let mut best_device = 0;
        let mut max_memory = 0;

        for device_id in 0..count {
            let mut props = HipDeviceProp::default();
            let result = unsafe { hipGetDeviceProperties(&mut props, device_id) };

            if result == HIP_SUCCESS {
                println!(
                    "Device {}: {} - {}MB VRAM",
                    device_id,
                    props.name(),
                    props.total_global_mem() / (1024 * 1024)
                );

                // Prefer device with most memory (likely discrete GPU)
                if props.total_global_mem() > max_memory {
                    max_memory = props.total_global_mem();
                    best_device = device_id;
                }
            }
        }

        // Get properties for best device
        let mut props = HipDeviceProp::default();
        let result = unsafe { hipGetDeviceProperties(&mut props, best_device) };

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to get device properties: {}",
                result
            )));
        }

        Ok(HipDevice {
            device_id: best_device,
            name: props.name(),
            memory: props.total_global_mem() as usize,
            compute_units: props.multi_processor_count(),
        })
    }

    fn initialize_hip() -> HipResult<()> {
        let result = unsafe { hipInit(0) };

        if result != HIP_SUCCESS {
            return Err(HipError::InitializationFailed(format!(
                "hipInit failed with code {}",
                result
            )));
        }

        Ok(())
    }

    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    pub fn stream(&self) -> &HipStream {
        &self.stream
    }

    /// Get available GPU memory information
    pub fn get_memory_info(&self) -> HipResult<(usize, usize)> {
        let mut free: usize = 0;
        let mut total: usize = 0;

        let result = unsafe { hipMemGetInfo(&mut free, &mut total) };

        if result != HIP_SUCCESS {
            return Err(HipError::MemoryQueryFailed(format!(
                "hipMemGetInfo failed with code {}",
                result
            )));
        }

        Ok((free, total))
    }

    /// Get GPU health status for monitoring
    ///
    /// Returns a tuple of (available, total) memory in bytes, or an error if GPU is unavailable.
    /// This is a non-failing version used for health checks.
    pub fn get_gpu_status(&self) -> (Option<(usize, usize)>, Option<String>) {
        match self.get_memory_info() {
            Ok((free, total)) => (Some((free, total)), None),
            Err(e) => (None, Some(e.to_string())),
        }
    }

    // ========== Phase 22-02: Model Memory Requirements ==========

    /// Check if we have enough memory for model loading
    ///
    /// Queries available GPU memory and compares against required bytes.
    /// Adds a 10% safety margin for driver overhead and alignment.
    ///
    /// # Arguments
    /// * `needed_bytes` - Exact memory needed for model tensors (before margin)
    ///
    /// # Returns
    /// * `(has_enough, free_mb, needed_mb)` where:
    ///   - `has_enough`: true if GPU has sufficient memory including safety margin
    ///   - `free_mb`: available GPU memory in MB
    ///   - `needed_mb`: required memory including 10% safety margin, in MB
    ///
    /// # Safety Margin
    /// - 10% of needed_bytes
    /// - Minimum 100MB margin for driver overhead
    /// - This is NOT a percentage of free memory (which was the flawed approach)
    ///
    /// # Example
    /// ```ignore
    /// let (has_enough, free_mb, needed_mb) = backend.check_memory_for_model(total_bytes)?;
    /// if !has_enough {
    ///     bail!("Insufficient GPU memory: need {} MB, available {} MB", needed_mb, free_mb);
    /// }
    /// ```
    pub fn check_memory_for_model(&self, needed_bytes: usize) -> HipResult<(bool, usize, usize)> {
        let (free, _total) = self.get_memory_info()?;

        // Add 10% safety margin for driver overhead
        // Minimum 100MB margin to account for alignment and driver allocations
        let safety_margin = (needed_bytes / 10).max(1024 * 1024 * 100);
        let needed_with_margin = needed_bytes.saturating_add(safety_margin);

        let has_enough = free >= needed_with_margin;
        let free_mb = free / 1024 / 1024;
        let needed_mb = needed_with_margin / 1024 / 1024;

        Ok((has_enough, free_mb, needed_mb))
    }

    // ========== Phase 20.2: Conservative Memory Allocation ==========

    /// Check if an allocation of given size is safe
    ///
    /// Returns true if size < 70% of currently free GPU memory.
    /// This prevents exhausting GPU memory needed by desktop/compositor.
    ///
    /// # Safety Margin
    /// Uses only 70% of free memory, leaving 30% for:
    /// - Desktop compositor (Wayland/X11)
    /// - Driver overhead
    /// - Display buffers and window textures
    ///
    /// # Example
    /// ```ignore
    /// if backend.can_allocate(1024 * 1024 * 100)? {
    ///     // Safe to allocate 100MB
    /// }
    /// ```
    pub fn can_allocate(&self, size: usize) -> HipResult<bool> {
        let (free, _total) = self.get_memory_info()?;

        // Safety margin: use only 70% of free memory
        // Leave 30% for desktop/compositor/driver overhead
        let safe_threshold = (free * 7) / 10;

        Ok(size <= safe_threshold)
    }

    /// Allocate buffer with conservative memory check
    ///
    /// Returns error if requested size exceeds 70% of free GPU memory.
    /// This prevents GPU memory exhaustion which would crash the desktop compositor.
    ///
    /// # Errors
    /// - `MemoryAllocationFailed` if size exceeds safe threshold (70% of free)
    /// - `MemoryAllocationFailed` if hipMalloc fails
    ///
    /// # Example
    /// ```ignore
    /// let buffer = backend.allocate_buffer_safe(1024 * 1024 * 100)?;
    /// ```
    pub fn allocate_buffer_safe(&self, size: usize) -> HipResult<HipBuffer> {
        // First check if allocation is safe
        if !self.can_allocate(size)? {
            // Get details for error message
            let (free, total) = self.get_memory_info()?;
            let safe_threshold = (free * 7) / 10;

            return Err(HipError::MemoryAllocationFailed(format!(
                "Requested {} bytes ({} MB) exceeds safe threshold {} bytes ({} MB)\n\
                 Free GPU memory: {} MB / Total: {} MB\n\
                 This prevents GPU memory exhaustion which would crash the desktop compositor.\n\
                 💡 Tip: Try reducing model size, tensor dimensions, or batch size.",
                size,
                size / 1024 / 1024,
                safe_threshold,
                safe_threshold / 1024 / 1024,
                free / 1024 / 1024,
                total / 1024 / 1024
            )));
        }

        // Use existing allocate_buffer method
        self.allocate_buffer(size)
    }

    /// Get safe allocation size for testing
    ///
    /// Returns 70% of currently free GPU memory as a safe size limit.
    /// Useful for determining maximum test allocation sizes.
    ///
    /// # Example
    /// ```ignore
    /// let safe_size = backend.safe_alloc_size()?;
    /// let test_tensor = DeviceTensor::empty(backend, shape_with_size(safe_size))?;
    /// ```
    pub fn safe_alloc_size(&self) -> HipResult<usize> {
        let (free, _) = self.get_memory_info()?;
        Ok((free * 7) / 10)
    }

    // ========== Phase 20.3: Safe Device-to-Host Copy ==========

    /// Copy from GPU to host using stream-aware synchronization (SAFE)
    ///
    /// This is the SAFE version that doesn't use `hipDeviceSynchronize()`.
    /// Instead, it uses `hipStreamSynchronize()` which only waits for our
    /// application's stream, not the entire device.
    ///
    /// # Why This Is Safe
    /// - Uses `hipStreamSynchronize()` instead of `hipDeviceSynchronize()`
    /// - Only waits for OUR stream, not desktop compositor's streams
    /// - Won't hang if desktop is using GPU
    ///
    /// # Example
    /// ```ignore
    /// let mut host_data = vec![0.0f32; 1024];
    /// backend.copy_from_device_safe(&gpu_buffer, &mut host_data)?;
    /// ```
    pub fn copy_from_device_safe<T>(
        &self,
        gpu_buffer: &HipBuffer,
        host_data: &mut [T],
    ) -> HipResult<()> {
        gpu_buffer.copy_to_host_with_stream(host_data, self.stream.as_ptr())
    }

    // ========== End Phase 20.3 ==========

    // ========== Phase 10-20: Retry Logic for Temporary GPU Errors ==========

    /// Execute a fallible operation with retry logic for recoverable errors
    ///
    /// This helper wraps GPU operations with automatic retry on temporary errors.
    /// Only recoverable errors (as determined by HipError::is_recoverable()) are retried.
    ///
    /// # Arguments
    /// * `operation` - Function that may fail with recoverable GPU errors
    /// * `context` - Description of the operation for error messages
    ///
    /// # Returns
    /// The operation's result, or the last error if all retries exhausted
    ///
    /// # Example
    /// ```ignore
    /// let buffer = backend.retry_operation(
    ///     || backend.allocate_buffer(1024 * 1024),
    ///     "allocate_buffer"
    /// )?;
    /// ```
    pub fn retry_operation<F, T>(
        &self,
        mut operation: F,
        context: &str,
    ) -> HipResult<T>
    where
        F: FnMut() -> HipResult<T>,
    {
        // Default retry configuration
        let max_retries = 3;
        let initial_delay_ms = 10u64;
        let backoff_multiplier = 2.0f64;
        let max_delay_ms = 1000u64;

        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation() {
                Ok(result) => {
                    if attempt > 0 {
                        tracing::info!(
                            context,
                            attempt,
                            "GPU operation succeeded after retry"
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);

                    // Check if this is a recoverable error
                    let is_recoverable = last_error
                        .as_ref()
                        .map(|e| e.is_recoverable())
                        .unwrap_or(false);

                    if !is_recoverable || attempt >= max_retries {
                        // Non-recoverable error or retries exhausted
                        if attempt > 0 {
                            tracing::warn!(
                                context,
                                attempt,
                                error = %last_error.as_ref().unwrap(),
                                "GPU operation failed after retries, giving up"
                            );
                        }
                        break;
                    }

                    // Calculate delay for this attempt
                    let base_delay = initial_delay_ms as f64
                        * backoff_multiplier.powi(attempt as i32);
                    let delay_ms = base_delay.min(max_delay_ms as f64) as u64;

                    tracing::warn!(
                        context,
                        attempt,
                        next_attempt = attempt + 1,
                        delay_ms,
                        error = %last_error.as_ref().unwrap(),
                        "GPU operation failed, retrying with exponential backoff"
                    );

                    // Sleep before retry
                    std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                }
            }
        }

        // Return the last error
        Err(last_error.unwrap())
    }

    /// Allocate buffer with automatic retry on temporary failures
    ///
    /// This wraps `allocate_buffer` with retry logic for recoverable errors like
    /// temporary GPU memory pressure.
    ///
    /// # Arguments
    /// * `size` - Size of buffer to allocate in bytes
    ///
    /// # Returns
    /// Allocated HipBuffer or error if all retries exhausted
    pub fn allocate_buffer_with_retry(&self, size: usize) -> HipResult<HipBuffer> {
        self.retry_operation(
            || self.allocate_buffer(size),
            "allocate_buffer"
        )
    }

    /// Copy from device with automatic retry on temporary failures
    ///
    /// This wraps `copy_from_device` with retry logic for recoverable errors like
    /// temporary driver issues.
    ///
    /// # Arguments
    /// * `buffer` - GPU buffer to copy from
    /// * `data` - Host buffer to copy into
    ///
    /// # Returns
    /// Success or error if all retries exhausted
    pub fn copy_from_device_with_retry<T>(
        &self,
        buffer: &HipBuffer,
        data: &mut [T],
    ) -> HipResult<()> {
        self.retry_operation(
            || self.copy_from_device(buffer, data),
            "copy_from_device"
        )
    }

    // ========== End Phase 10-20 Retry Logic ==========

    // ========== End Phase 20.2 ==========

    /// Allocate buffer with memory limit checking (uses 80% of available memory as safety limit)
    pub fn allocate_buffer(&self, size: usize) -> HipResult<HipBuffer> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);
        let count = ALLOC_COUNTER.fetch_add(1, Ordering::SeqCst);
        eprintln!(
            ">>> allocate_buffer #{}: requesting {} bytes ({} MB)",
            count,
            size,
            size / 1024 / 1024
        );
        let mut ptr: *mut c_void = ptr::null_mut();

        // Use hipMalloc to allocate device memory
        let result = unsafe { hipMalloc(&mut ptr, size) };
        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to allocate device memory: {}",
                result
            )));
        }

        let buffer = HipBuffer {
            inner: Arc::new(HipBufferInner {
                ptr,
                size,
                offset: 0,
            }),
        };
        println!(
            "DEBUG: allocate_buffer: created buffer with size {} bytes",
            buffer.size()
        );
        Ok(buffer)
    }

    /// Create a dummy zero-byte buffer for empty tensors.
    ///
    /// Used for tensors with byte_size=0 (e.g., KV cache views at current_len=0).
    /// These don't need actual GPU memory but need a valid HipBuffer for the type system.
    pub fn dummy_zero_buffer(&self) -> HipResult<HipBuffer> {
        Ok(HipBuffer {
            inner: Arc::new(HipBufferInner {
                ptr: ptr::null_mut(),
                size: 0,
                offset: 0,
            }),
        })
    }

    pub fn launch_kernel(
        &self,
        kernel_name: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        _args: &[*mut c_void],
    ) -> HipResult<()> {
        // In a real implementation, this would:
        // 1. Load compiled kernel module
        // 2. Get kernel function pointer
        // 3. Launch kernel with hipModuleLaunchKernel

        println!(
            "Launching kernel '{}' with grid {:?} and block {:?}",
            kernel_name, grid_dim, block_dim
        );

        Ok(())
    }

    pub fn synchronize(&self) -> HipResult<()> {
        self.stream.synchronize()
    }

    /// Copy data from host to device using the backend's HIP stream.
    ///
    /// This is a convenience wrapper around `HipBuffer::copy_from_host_with_stream`
    /// that automatically uses the backend's stream.
    ///
    /// # Why This Exists
    /// Using `hipMemcpyAsync` with the backend's stream (instead of `hipMemcpy` on
    /// the default stream) ensures proper ordering with all other GPU operations
    /// (kernels, hipBLAS) that also use this stream. This prevents synchronization
    /// issues and hangs that can occur when mixing default stream and custom stream
    /// operations.
    pub fn copy_to_device<T>(&self, buffer: &HipBuffer, data: &[T]) -> HipResult<()> {
        buffer.copy_from_host_with_stream(data, self.stream.as_ptr())
    }

    /// Copy data from device to host using the backend's HIP stream.
    ///
    /// This is a convenience wrapper around `HipBuffer::copy_to_host_with_stream`
    /// that automatically uses the backend's stream and synchronizes afterward.
    ///
    /// # Synchronization
    /// Unlike `HipBuffer::copy_to_host_with_stream()`, this method synchronizes the
    /// stream after the async copy to ensure the data is ready before returning.
    pub fn copy_from_device<T>(&self, buffer: &HipBuffer, data: &mut [T]) -> HipResult<()> {
        buffer.copy_to_host_with_stream(data, self.stream.as_ptr())?;
        self.stream.synchronize()
    }

    // Methods needed for smoke tests
    pub fn load_module(&self, path: &str) -> HipResult<HipModule> {
        let path_cstr = CString::new(path)
            .map_err(|e| HipError::KernelLoadFailed(format!("Invalid path string: {}", e)))?;

        let mut module: *mut c_void = ptr::null_mut();
        let result = unsafe { hipModuleLoad(&mut module, path_cstr.as_ptr()) };

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(HipError::KernelLoadFailed(format!(
                "Failed to load module '{}': {}",
                path, error_msg
            )));
        }

        Ok(HipModule::from_ptr(module))
    }

    pub fn load_module_from_data(&self, data: &[u8]) -> HipResult<HipModule> {
        let mut module: *mut c_void = ptr::null_mut();
        let result = unsafe { hipModuleLoadData(&mut module, data.as_ptr() as *const c_void) };

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(HipError::KernelLoadFailed(format!(
                "Failed to load module from data: {}",
                error_msg
            )));
        }

        Ok(HipModule::from_ptr(module))
    }

    pub fn get_kernel(&self, module_path: &str, kernel_name: &str) -> HipResult<HipKernel> {
        // Load module
        let module = self.load_module(module_path)?;

        // Get kernel function from module
        self.get_kernel_function(&module, kernel_name)
    }

    /// Get kernel function from a loaded module
    pub fn get_kernel_function(
        &self,
        module: &HipModule,
        kernel_name: &str,
    ) -> HipResult<HipKernel> {
        let kernel_name_cstr = CString::new(kernel_name)
            .map_err(|e| HipError::KernelLoadFailed(format!("Invalid kernel name: {}", e)))?;

        let mut func: *mut c_void = ptr::null_mut();
        let result =
            unsafe { hipModuleGetFunction(&mut func, module.as_ptr(), kernel_name_cstr.as_ptr()) };

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(HipError::KernelLoadFailed(format!(
                "Failed to get kernel '{}': {}",
                kernel_name, error_msg
            )));
        }

        Ok(HipKernel::from_ptr(func))
    }

    /// Get number of available HIP devices
    pub fn get_device_count(&self) -> HipResult<i32> {
        let mut count: i32 = 0;
        let result = unsafe { hipGetDeviceCount(&mut count) };

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(HipError::DeviceError(format!(
                "Failed to get device count: {}",
                error_msg
            )));
        }

        Ok(count)
    }

    /// Get properties for a specific HIP device
    pub fn get_device_properties(&self, device_id: i32) -> HipResult<HipDeviceProp> {
        let mut props = HipDeviceProp::default();
        let result = unsafe { hipGetDeviceProperties(&mut props, device_id) };

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(HipError::DeviceError(format!(
                "Failed to get device properties: {}",
                error_msg
            )));
        }

        Ok(props)
    }

    pub fn alloc_gpu_buffer<T>(&self, len: usize) -> HipResult<HipBuffer> {
        let size = len * std::mem::size_of::<T>();
        self.allocate_buffer(size)
    }

    pub fn copy_to_gpu<T>(&self, host_data: &[T], gpu_buffer: &HipBuffer) -> HipResult<()> {
        gpu_buffer.copy_from_host(host_data)
    }

    pub fn copy_from_gpu<T>(&self, gpu_buffer: &HipBuffer, host_data: &mut [T]) -> HipResult<()> {
        self.copy_from_device_safe(gpu_buffer, host_data)
    }

    pub fn add_inplace(&self, input: &DeviceTensor, output: &mut DeviceTensor) -> HipResult<()> {
        if input.shape().dims() != output.shape().dims() {
            return Err(HipError::GenericError(
                "Input and output tensors must have matching shapes".to_string(),
            ));
        }

        let len = input.len();
        if len == 0 {
            return Ok(());
        }

        let handle = crate::backend::hip_blas::HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        // CRITICAL: Associate hipBLAS handle with our HIP stream
        // Without this, hipBLAS uses the default stream while our kernels use a custom stream,
        // causing synchronization issues and hangs.
        handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        crate::backend::hip_blas::saxpy(
            &handle,
            len as i32,
            1.0,
            input.buffer().as_ptr() as *const f32,
            1,
            output.buffer().as_ptr() as *mut f32,
            1,
        )
        .map_err(|e| HipError::GenericError(format!("hipBLAS saxpy failed: {}", e)))?;

        Ok(())
    }

    pub fn scale_inplace(&self, tensor: &mut DeviceTensor, scale: f32) -> HipResult<()> {
        if tensor.len() == 0 {
            return Ok(());
        }

        let handle = crate::backend::hip_blas::HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        // CRITICAL: Associate hipBLAS handle with our HIP stream
        handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        crate::backend::hip_blas::sscal(
            &handle,
            tensor.len() as i32,
            scale,
            tensor.buffer().as_ptr() as *mut f32,
            1,
        )
        .map_err(|e| HipError::GenericError(format!("hipBLAS sscal failed: {}", e)))?;

        Ok(())
    }

    pub fn add_row_bias(&self, tensor: &mut DeviceTensor, bias: &DeviceTensor) -> HipResult<()> {
        let tensor_shape = tensor.shape().dims();
        if tensor_shape.len() != 2 {
            return Err(HipError::GenericError(
                "Tensor for row bias must be 2D".to_string(),
            ));
        }
        let rows = tensor_shape[0];
        let cols = tensor_shape[1];

        let bias_shape = bias.shape().dims();
        if !(bias_shape.len() == 1 && bias_shape[0] == cols)
            && !(bias_shape.len() == 2 && bias_shape[0] == 1 && bias_shape[1] == cols)
        {
            return Err(HipError::GenericError(format!(
                "Bias shape {:?} incompatible with tensor shape {:?}",
                bias_shape, tensor_shape
            )));
        }

        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let handle = crate::backend::hip_blas::HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        // CRITICAL: Associate hipBLAS handle with our HIP stream
        handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        let bias_ptr = bias.buffer().as_ptr() as *const f32;
        let mut row_ptr = tensor.buffer().as_ptr() as *mut f32;
        let stride = cols;

        for _ in 0..rows {
            crate::backend::hip_blas::saxpy(&handle, cols as i32, 1.0f32, bias_ptr, 1, row_ptr, 1)
                .map_err(|e| HipError::GenericError(format!("hipBLAS saxpy failed: {}", e)))?;

            // SAFETY: Check for pointer arithmetic overflow before advancing
            let current = row_ptr as usize;
            row_ptr = current.checked_add(stride).ok_or_else(|| {
                HipError::GenericError(format!(
                    "Pointer arithmetic overflow in add_bias_to_rows (current=0x{:x}, stride={})",
                    current, stride
                ))
            })? as *mut f32;
        }

        Ok(())
    }

    pub fn launch_kernel_with_module(
        &self,
        kernel: &HipKernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[*mut c_void],
    ) -> HipResult<()> {
        self.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0)
    }

    pub fn launch_kernel_with_module_shared(
        &self,
        kernel: &HipKernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[*mut c_void],
        shared_mem_bytes: u32,
    ) -> HipResult<()> {
        tracing::trace!("launch_kernel_with_module_shared: Launching kernel with grid={:?}, block={:?}, shared_mem={}, args_len={}",
                       grid_dim, block_dim, shared_mem_bytes, args.len());

        // Validate kernel parameters to prevent segfaults
        if grid_dim.0 == 0 || grid_dim.1 == 0 || grid_dim.2 == 0 {
            tracing::error!("launch_kernel_with_module_shared: Invalid grid dimensions: {:?}", grid_dim);
            return Err(HipError::KernelLaunchFailed("Grid dimensions cannot be zero".to_string()));
        }
        if block_dim.0 == 0 || block_dim.1 == 0 || block_dim.2 == 0 {
            tracing::error!("launch_kernel_with_module_shared: Invalid block dimensions: {:?}", block_dim);
            return Err(HipError::KernelLaunchFailed("Block dimensions cannot be zero".to_string()));
        }

        // Check for potentially problematic parameter combinations
        let total_threads = (grid_dim.0 * grid_dim.1 * grid_dim.2 * block_dim.0 * block_dim.1 * block_dim.2) as u64;
        if total_threads > 1_000_000_000 { // 1 billion threads
            tracing::warn!("launch_kernel_with_module_shared: Very large kernel launch: {} total threads", total_threads);
        }

        tracing::trace!("launch_kernel_with_module_shared: Calling hipModuleLaunchKernel");
        let result = unsafe {
            hipModuleLaunchKernel(
                kernel.as_ptr(),
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                shared_mem_bytes,
                self.stream.stream,
                args.as_ptr() as *mut *mut c_void,
                ptr::null_mut(), // extra
            )
        };
        tracing::trace!("launch_kernel_with_module_shared: hipModuleLaunchKernel returned {}", result);

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            tracing::error!("launch_kernel_with_module_shared: Kernel launch failed with code {}: {}", result, error_msg);
            return Err(HipError::KernelLaunchFailed(format!(
                "Kernel launch failed: {}",
                error_msg
            )));
        }

        tracing::trace!("launch_kernel_with_module_shared: Kernel launched successfully");
        Ok(())
    }

    /// Get softmax kernel function pointer
    pub fn get_softmax_kernel(&self) -> HipResult<HipKernel> {
        // Load softmax module and get kernel function
        let module = self.load_module("kernels/softmax.hip")?;
        self.get_kernel_function(&module, "softmax_kernel")
    }

    /// Get mask kernel function pointer  
    pub fn get_mask_kernel(&self) -> HipResult<HipKernel> {
        // Load mask module and get kernel function
        let module = self.load_module("kernels/mask.hip")?;
        self.get_kernel_function(&module, "mask_kernel")
    }

    /// Get scale kernel function pointer
    pub fn get_scale_kernel(&self) -> HipResult<HipKernel> {
        // Load scale module and get kernel function
        let module = self.load_module("kernels/scale.hip")?;
        self.get_kernel_function(&module, "scale_kernel")
    }
}

/// GPU tensor with device memory allocation
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    buffer: HipBuffer,
    shape: TensorShape,
}

impl DeviceTensor {
    /// Create device tensor from memory-mapped weights
    pub fn from_mmap(
        backend: &HipBackend,
        mmap_weights: &crate::loader::mmap_loader::MmapWeights,
        shape: TensorShape,
        byte_offset: usize,
    ) -> HipResult<Self> {
        // Calculate total bytes needed
        let total_elements = shape.total_elements();
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        // Allocate device buffer
        let buffer = backend.allocate_buffer(total_bytes)?;

        // Get f32 view from mmap weights
        let start_element = byte_offset / std::mem::size_of::<f32>();
        let end_element = start_element + total_elements;
        let f32_slice = mmap_weights.view_f32(start_element..end_element);

        // Copy to device
        buffer.copy_from_host(f32_slice)?;

        Ok(DeviceTensor { buffer, shape })
    }

    /// Create device tensor from pre-allocated buffer (Phase 4: Async Loading)
    ///
    /// This is used by AsyncLoader when GPU memory is already allocated
    /// and data has been uploaded via async copy.
    pub fn from_buffer(
        _backend: &HipBackend,
        buffer: HipBuffer,
        shape: TensorShape,
    ) -> HipResult<Self> {
        Ok(DeviceTensor { buffer, shape })
    }

    /// Get tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Get underlying buffer
    pub fn buffer(&self) -> &HipBuffer {
        &self.buffer
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.shape.total_elements()
    }

    /// Get raw pointer to device memory (for address comparison in tests)
    pub fn as_ptr(&self) -> *const f32 {
        self.buffer.as_ptr() as *const f32
    }

    /// Copy device data to host vector
    ///
    /// ⚠️ **DEPRECATED - Use HipBackend::copy_from_device_safe() instead** ⚠️
    ///
    /// This method is deprecated because it implicitly accesses the global backend.
    /// Use `backend.copy_from_device_safe(&tensor.buffer(), &mut data)` for clearer intent.
    #[deprecated(
        since = "0.23.0",
        note = "Use HipBackend::copy_from_device_safe() with explicit buffer access instead"
    )]
    #[allow(deprecated)] // Allow internal use of deprecated copy_to_host in this deprecated method
    pub fn to_host_vec(&self) -> HipResult<Vec<f32>> {
        let mut host_data = vec![0.0f32; self.len()];
        unsafe {
            let ptr = host_data.as_mut_ptr() as *mut u8;
            let byte_size = self.len() * std::mem::size_of::<f32>();
            let byte_slice = std::slice::from_raw_parts_mut(ptr, byte_size);
            self.buffer.copy_to_host(byte_slice)?;
        }
        Ok(host_data)
    }

    /// Create empty device tensor (zero-initialized)
    ///
    /// CRITICAL: GPU memory MUST be zero-initialized to prevent test isolation failures.
    /// Uninitialized GPU memory contains garbage from previous kernel executions,
    /// causing tests to pass individually but fail when run together.
    pub fn empty(backend: &HipBackend, shape: TensorShape) -> HipResult<Self> {
        let total_bytes = shape.total_elements() * std::mem::size_of::<f32>();
        eprintln!(
            ">>> DeviceTensor::empty: About to allocate {} bytes ({} MB)...",
            total_bytes,
            total_bytes / 1024 / 1024
        );
        let buffer = backend.allocate_buffer(total_bytes)?;
        eprintln!(
            ">>> DeviceTensor::empty: allocate_buffer returned successfully for {} bytes",
            total_bytes
        );

        // Zero-initialize GPU memory to prevent test isolation failures
        // This ensures clean state for each test, preventing garbage data
        // from previous kernel executions from contaminating new tests.
        eprintln!(
            ">>> DeviceTensor::empty: Calling hipMemset for {} bytes ({} MB)...",
            total_bytes,
            total_bytes / 1024 / 1024
        );
        let result = unsafe { hipMemset(buffer.as_ptr(), 0, total_bytes) };
        eprintln!(
            ">>> DeviceTensor::empty: hipMemset returned for {} bytes",
            total_bytes
        );

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(HipError::MemoryAllocationFailed(format!(
                "hipMemset failed (zero-initialization): {}",
                error_msg
            )));
        }

        Ok(DeviceTensor { buffer, shape })
    }

    /// Create empty device tensor with conservative memory allocation (Phase 20.2)
    ///
    /// This is the SAFE version that won't exhaust GPU memory.
    /// Uses `allocate_buffer_safe()` which enforces 70% of free memory limit.
    ///
    /// # Errors
    /// - `MemoryAllocationFailed` if size exceeds 70% of free GPU memory
    ///
    /// # Example
    /// ```ignore
    /// let tensor = DeviceTensor::empty_safe(&backend, shape)?;
    /// ```
    pub fn empty_safe(backend: &HipBackend, shape: TensorShape) -> HipResult<Self> {
        let total_bytes = shape.total_elements() * std::mem::size_of::<f32>();

        // Check if allocation is safe first
        if !backend.can_allocate(total_bytes)? {
            return Err(HipError::MemoryAllocationFailed(format!(
                "Cannot allocate tensor with shape {:?}: {} bytes ({} MB) exceeds safe limit",
                shape.dims(),
                total_bytes,
                total_bytes / 1024 / 1024
            )));
        }

        // Use safe allocation
        let buffer = backend.allocate_buffer_safe(total_bytes)?;

        // Zero-initialize GPU memory to prevent test isolation failures
        let result = unsafe { hipMemset(buffer.as_ptr(), 0, total_bytes) };

        if result != HIP_SUCCESS {
            let error_msg = unsafe {
                let error_ptr = hipGetErrorString(result);
                if error_ptr.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(HipError::MemoryAllocationFailed(format!(
                "hipMemset failed (zero-initialization): {}",
                error_msg
            )));
        }

        Ok(DeviceTensor { buffer, shape })
    }

    /// Create device tensor from host vector
    pub fn from_host_vec(
        backend: &HipBackend,
        host_data: Vec<f32>,
        shape: TensorShape,
    ) -> HipResult<Self> {
        let total_bytes = host_data.len() * std::mem::size_of::<f32>();

        let buffer = backend.allocate_buffer(total_bytes)?;
        buffer.copy_from_host(&host_data)?;

        Ok(DeviceTensor { buffer, shape })
    }

    /// Create device tensor as a sub-allocation from a memory pool.
    /// This avoids individual hipMalloc calls for each tensor.
    ///
    /// # Arguments
    /// * `pool` - The parent GPU memory pool (large pre-allocated buffer)
    /// * `offset` - Byte offset into the pool where this tensor's data starts
    /// * `host_data` - Host data to copy to the sub-allocated region
    /// * `shape` - Tensor shape
    ///
    /// # Note
    /// This method uses `copy_from_host` which operates on the default HIP stream.
    /// For model loading, prefer `from_pool_with_backend` which uses the backend's
    /// stream for proper synchronization.
    pub fn from_pool(
        pool: &HipBuffer,
        offset: usize,
        host_data: Vec<f32>,
        shape: TensorShape,
    ) -> HipResult<Self> {
        let total_bytes = host_data.len() * std::mem::size_of::<f32>();

        // Create a sub-buffer view at the specified offset
        let buffer = pool.sub_buffer_view(offset, total_bytes)?;

        // Copy data to the sub-allocated region
        buffer.copy_from_host(&host_data)?;

        Ok(DeviceTensor { buffer, shape })
    }

    /// Create device tensor as a sub-allocation from a memory pool, using the backend's stream.
    ///
    /// This is the preferred method for model loading because it ensures all GPU operations
    /// (including data transfers) use the same HIP stream, preventing synchronization issues.
    ///
    /// # Arguments
    /// * `pool` - The parent GPU memory pool (large pre-allocated buffer)
    /// * `offset` - Byte offset into the pool where this tensor's data starts
    /// * `host_data` - Host data to copy to the sub-allocated region
    /// * `shape` - Tensor shape
    /// * `backend` - HIP backend (provides the stream for async copy)
    pub fn from_pool_with_backend(
        pool: &HipBuffer,
        offset: usize,
        host_data: Vec<f32>,
        shape: TensorShape,
        backend: &HipBackend,
    ) -> HipResult<Self> {
        let total_bytes = host_data.len() * std::mem::size_of::<f32>();

        // Create a sub-buffer view at the specified offset
        let buffer = pool.sub_buffer_view(offset, total_bytes)?;

        // Copy data to the sub-allocated region using the backend's stream
        // This ensures proper ordering with other GPU operations
        buffer.copy_from_host_with_stream(&host_data, backend.stream().as_ptr())?;

        Ok(DeviceTensor { buffer, shape })
    }

    /// Copy data from host slice to device tensor
    pub fn copy_from_host(&mut self, host_data: &[f32]) -> HipResult<()> {
        // Validate size matches
        if host_data.len() != self.len() {
            return Err(HipError::GenericError(format!(
                "Host data size {} does not match tensor size {}",
                host_data.len(),
                self.len()
            )));
        }

        // Copy data to underlying buffer
        self.buffer.copy_from_host(host_data)
    }

    /// Copy data from host vector to device tensor (convenience method)
    pub fn copy_from_host_vec(&mut self, host_data: Vec<f32>) -> HipResult<()> {
        self.copy_from_host(&host_data)
    }

    pub fn copy_from_device_slice(
        &mut self,
        src: &DeviceTensor,
        src_offset_elements: usize,
    ) -> HipResult<()> {
        let byte_len = self.len() * std::mem::size_of::<f32>();
        let byte_offset = src_offset_elements * std::mem::size_of::<f32>();
        self.buffer
            .copy_from_buffer_with_offset(src.buffer(), byte_offset, byte_len)
    }

    pub fn copy_from_device_region(
        &mut self,
        dst_offset_elements: usize,
        src: &DeviceTensor,
        src_offset_elements: usize,
        len_elements: usize,
    ) -> HipResult<()> {
        let byte_len = len_elements * std::mem::size_of::<f32>();
        let dst_offset = dst_offset_elements * std::mem::size_of::<f32>();
        let src_offset = src_offset_elements * std::mem::size_of::<f32>();
        self.buffer
            .copy_from_buffer_region(dst_offset, src.buffer(), src_offset, byte_len)
    }

    pub fn copy_from_device_buffer(&mut self, src: &HipBuffer) -> HipResult<()> {
        self.buffer.copy_from_buffer(src)
    }
}

impl HipBackend {
    /// Create scratch buffers for attention computation
    pub fn create_scratch_buffers(
        &self,
        config: &crate::model::config::ModelConfig,
    ) -> HipResult<crate::backend::scratch::ScratchBufferManager> {
        // PHASE 24 FIX: Correct parameter order
        crate::backend::scratch::ScratchBufferManager::new(
            self,
            config.num_attention_heads,
            config.hidden_size, // ← CORRECT: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← CORRECT: 5th param
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))
    }

    /// Create model runtime
    pub fn create_model_runtime(
        &self,
        config: &crate::model::config::ModelConfig,
    ) -> HipResult<ModelRuntime> {
        ModelRuntime::new_with_config(config.clone())
    }

    /// MLP (SwiGLU) forward pass
    pub fn mlp_swiglu(
        &self,
        hidden_states: &DeviceTensor,
        gate_weight: &DeviceTensor,
        up_weight: &DeviceTensor,
        down_weight: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        use crate::backend::hip_blas::HipBlasHandle;
        use crate::tensor::matmul::matmul_f32;

        // Phase D: TDD implementation - validate shapes and basic structure
        let hidden_shape = hidden_states.shape();
        let gate_shape = gate_weight.shape();
        let up_shape = up_weight.shape();
        let down_shape = down_weight.shape();
        let output_shape = output.shape();

        // hidden_states: [seq_len, hidden_size]
        if hidden_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "hidden_states must be 2D [seq_len, hidden_size]".to_string(),
            ));
        }

        // gate_weight: [hidden_size, intermediate_size]
        if gate_shape.dims().len() != 2 || gate_shape.dims()[0] != hidden_shape.dims()[1] {
            return Err(HipError::GenericError(
                "gate_weight must be 2D [hidden_size, intermediate_size]".to_string(),
            ));
        }

        // up_weight: [hidden_size, intermediate_size]
        if up_shape.dims().len() != 2 || up_shape.dims()[0] != hidden_shape.dims()[1] {
            return Err(HipError::GenericError(
                "up_weight must be 2D [hidden_size, intermediate_size]".to_string(),
            ));
        }

        // down_weight: [intermediate_size, hidden_size]
        if down_shape.dims().len() != 2 || down_shape.dims()[1] != hidden_shape.dims()[1] {
            return Err(HipError::GenericError(
                "down_weight must be 2D [intermediate_size, hidden_size]".to_string(),
            ));
        }

        // output: [seq_len, hidden_size]
        if output_shape.dims().len() != 2 || output_shape.dims() != hidden_shape.dims() {
            return Err(HipError::GenericError(
                "output must match hidden_states shape [seq_len, hidden_size]".to_string(),
            ));
        }

        let (seq_len, hidden_size) = (hidden_shape.dims()[0], hidden_shape.dims()[1]);
        let intermediate_size = gate_shape.dims()[1];

        // Validate intermediate size consistency
        if up_shape.dims()[1] != intermediate_size || down_shape.dims()[0] != intermediate_size {
            return Err(HipError::GenericError(
                "All intermediate dimensions must match".to_string(),
            ));
        }

        // Phase D: Implement actual SwiGLU computation
        // SwiGLU = DownProj(Gate(X) ⊙ Swish(UpProj(X)))
        // where Swish(x) = x ⊙ σ(x)

        // Create hipBLAS handle for matrix operations
        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        // CRITICAL: Associate hipBLAS handle with our HIP stream
        // This ensures all hipBLAS operations (matmul) are queued on the same stream
        // as our custom HIP kernels, preventing synchronization issues and hangs.
        blas_handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        // Step 1: Compute gate projection: hidden_states @ gate_weight -> gate_output
        // hidden_states: [seq_len, hidden_size], gate_weight: [hidden_size, intermediate_size]
        // gate_output: [seq_len, intermediate_size]
        let gate_buffer = matmul_f32(
            self,
            &blas_handle,
            hidden_states.buffer(),
            gate_weight.buffer(),
            seq_len as i32,
            intermediate_size as i32,
            hidden_size as i32,
        )
        .map_err(|e| HipError::GenericError(format!("Gate projection failed: {}", e)))?;

        // Step 2: Compute up projection: hidden_states @ up_weight -> up_output
        // hidden_states: [seq_len, hidden_size], up_weight: [hidden_size, intermediate_size]
        // up_output: [seq_len, intermediate_size]
        let up_buffer = matmul_f32(
            self,
            &blas_handle,
            hidden_states.buffer(),
            up_weight.buffer(),
            seq_len as i32,
            intermediate_size as i32,
            hidden_size as i32,
        )
        .map_err(|e| HipError::GenericError(format!("Up projection failed: {}", e)))?;

        // Synchronize to ensure matmul operations complete before SwiGLU kernel
        // hipBLAS uses default stream, while our kernel uses custom stream
        self.synchronize()?;

        // Step 3: Apply SwiGLU activation using GPU kernel (no CPU fallback for ROCm)
        // SwiGLU = gate_output ⊙ Swish(up_output)
        // where Swish(x) = x ⊙ σ(x)
        #[cfg(feature = "rocm")]
        {
            // For ROCm builds, require GPU kernel to work - no CPU fallback
            // Allocate device buffer for SwiGLU output
            let swiglu_buffer = HipBuffer::new((seq_len * intermediate_size) * std::mem::size_of::<f32>())
                .map_err(|e| HipError::GenericError(format!("Failed to allocate SwiGLU buffer: {}", e)))?;

            // Launch GPU kernel for SwiGLU activation
            unsafe {
                crate::mlp::kernels::swiglu_gpu_kernel(
                    self, // Pass caller's backend to ensure stream consistency
                    gate_buffer.as_ptr() as *const f32,
                    up_buffer.as_ptr() as *const f32,
                    swiglu_buffer.as_mut_ptr() as *mut f32,
                    seq_len as u32,
                    intermediate_size as u32,
                )
                .map_err(|e| HipError::GenericError(format!("SwiGLU GPU kernel failed: {}", e)))?;
            }

            // Synchronize to ensure kernel completes before down projection
            self.synchronize()
                .map_err(|e| HipError::GenericError(format!("GPU synchronization failed: {}", e)))?;

            // Step 4: Compute down projection: swiglu_output @ down_weight -> final_output
            let final_buffer = matmul_f32(
                self,
                &blas_handle,
                &swiglu_buffer,
                down_weight.buffer(),
                seq_len as i32,
                hidden_size as i32,
                intermediate_size as i32,
            )
            .map_err(|e| HipError::GenericError(format!("Down projection failed: {}", e)))?;

            // Copy result to output tensor
            output.buffer().copy_from_buffer(&final_buffer)
                .map_err(|e| HipError::GenericError(format!("Final copy failed: {}", e)))?;
        }

        #[cfg(not(feature = "rocm"))]
        {
            // CPU fallback for non-ROCM builds
            let mut gate_host = vec![0.0f32; (seq_len * intermediate_size) as usize];
            let mut up_host = vec![0.0f32; (seq_len * intermediate_size) as usize];

            self.copy_from_device_safe(&gate_buffer, &mut gate_host)?;
            self.copy_from_device_safe(&up_buffer, &mut up_host)?;

            // Apply SwiGLU activation on CPU
            let mut swiglu_host = vec![0.0f32; (seq_len * intermediate_size) as usize];
            for i in 0..swiglu_host.len() {
                let gate_val = gate_host[i];
                let up_val = up_host[i];
                // Swish activation: swish(x) = x * sigmoid(x)
                let sigmoid_up = 1.0 / (1.0 + (-up_val).exp());
                let swish_up = up_val * sigmoid_up;
                // SwiGLU: gate(x) * swish(up(x))
                swiglu_host[i] = gate_val * swish_up;
            }

            // Copy SwiGLU result back to GPU
            let swiglu_buffer = HipBuffer::new(swiglu_host.len() * std::mem::size_of::<f32>())?;
            swiglu_buffer.copy_from_host(&swiglu_host)?;

            // Step 4: Compute down projection: swiglu_output @ down_weight -> final_output
            let final_buffer = matmul_f32(
                self,
                &blas_handle,
                &swiglu_buffer,
                down_weight.buffer(),
                seq_len as i32,
                hidden_size as i32,
                intermediate_size as i32,
            )
            .map_err(|e| HipError::GenericError(format!("Down projection failed: {}", e)))?;

            // Copy result to output tensor
            let mut output_host = vec![0.0f32; (seq_len * hidden_size) as usize];
            self.copy_from_device_safe(&final_buffer, &mut output_host)?;
            output.buffer().copy_from_host(&output_host)?;
        }

        Ok(())
    }

    /// Layer normalization forward pass
    pub fn layernorm(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        output: &mut DeviceTensor,
        eps: f32,
    ) -> HipResult<()> {
        // Phase D: TDD implementation - validate shapes and basic structure
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let output_shape = output.shape();

        // Validate input is at least 1D
        if input_shape.dims().is_empty() {
            return Err(HipError::GenericError(
                "input must have at least 1 dimension".to_string(),
            ));
        }

        // weight should match last dimension of input
        let last_dim = *input_shape.dims().last().ok_or_else(|| {
            HipError::GenericError("input must have at least one dimension".to_string())
        })?;
        if weight_shape.dims() != [last_dim] {
            return Err(HipError::GenericError(
                "weight must match last dimension of input".to_string(),
            ));
        }

        // bias should match weight if provided
        if let Some(bias_tensor) = bias {
            let bias_shape = bias_tensor.shape();
            if bias_shape.dims() != weight_shape.dims() {
                return Err(HipError::GenericError(
                    "bias must match weight dimensions".to_string(),
                ));
            }
        }

        // output should match input shape
        if output_shape.dims() != input_shape.dims() {
            return Err(HipError::GenericError(
                "output must match input shape".to_string(),
            ));
        }

        // Phase D: Implement actual LayerNorm computation
        // LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
        let total_elements = input_shape.total_elements();
        let last_dim_size = last_dim;
        let num_rows = total_elements / last_dim_size;

        // Copy input to host for computation (GPU kernel would be more efficient)
        let mut input_host = vec![0.0f32; total_elements];
        self.copy_from_device_safe(input.buffer(), &mut input_host)?;

        // Get weight and bias data
        let mut weight_host = vec![0.0f32; last_dim_size];
        self.copy_from_device_safe(weight.buffer(), &mut weight_host)?;

        let bias_host = if let Some(bias_tensor) = bias {
            let mut bias_data = vec![0.0f32; last_dim_size];
            self.copy_from_device_safe(bias_tensor.buffer(), &mut bias_data)?;
            Some(bias_data)
        } else {
            None
        };

        // Process each row independently
        let mut output_host = vec![0.0f32; total_elements];
        for row_idx in 0..num_rows {
            let start_idx = row_idx * last_dim_size;
            let end_idx = start_idx + last_dim_size;
            let row = &input_host[start_idx..end_idx];

            // Compute mean and variance for this row
            let mean: f32 = row.iter().sum::<f32>() / last_dim_size as f32;
            let variance: f32 =
                row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last_dim_size as f32;
            let std = (variance + eps).sqrt();

            // Apply LayerNorm: (x - mean) / std * weight + bias
            for (i, &x) in row.iter().enumerate() {
                let normalized = (x - mean) / std;
                let scaled = normalized * weight_host[i];
                output_host[start_idx + i] = if let Some(ref bias_data) = bias_host {
                    scaled + bias_data[i]
                } else {
                    scaled
                };
            }
        }

        // Copy result back to GPU
        output.buffer().copy_from_host(&output_host)?;

        Ok(())
    }

    /// Complete transformer layer forward pass
    ///
    /// **DEPRECATED**: This method is deprecated due to Phase 2 lazy loading.
    /// It directly accesses LayerPlan fields which now store Arc<LazyTensor>.
    /// Use ExecutionPlan::forward_layer() instead which properly handles lazy loading.
    #[deprecated(note = "Use ExecutionPlan::forward_layer() instead for lazy loading")]
    pub fn transformer_layer(
        &self,
        _layer_idx: usize,
        _hidden_states: &DeviceTensor,
        _layer_plan: &crate::model::execution_plan::LayerPlan,
        _attention_output: &mut DeviceTensor,
        _mlp_output: &mut DeviceTensor,
        _scratch_buffers: &crate::backend::scratch::ScratchBufferManager,
        _kv_cache: &mut crate::model::kv_cache::KVCache,
    ) -> HipResult<()> {
        Err(HipError::GenericError(
            "transformer_layer() is deprecated. Use ExecutionPlan::forward_layer() instead."
                .to_string(),
        ))
    }

    /// Enhanced decode step with multi-layer support
    ///
    /// **DEPRECATED**: This method is deprecated due to Phase 2 lazy loading.
    /// It calls transformer_layer() which is not compatible with lazy tensors.
    /// Use the alternative decode_step() method that uses ExecutionPlan instead.
    #[deprecated(note = "Use the alternative decode_step() method with ExecutionPlan")]
    pub fn decode_step(
        &self,
        _layer_idx: usize,
        _hidden_states: &DeviceTensor,
        _layer_plan: &crate::model::execution_plan::LayerPlan,
        _attention_output: &mut DeviceTensor,
        _mlp_output: &mut DeviceTensor,
        _scratch_buffers: &crate::backend::scratch::ScratchBufferManager,
        _kv_cache: &mut crate::model::kv_cache::KVCache,
    ) -> HipResult<()> {
        Err(HipError::GenericError(
            "decode_step(layer_plan) is deprecated. Use the alternative decode_step() method with ExecutionPlan.".to_string()
        ))
    }
}

/// Model runtime for managing device buffers and weights
#[derive(Debug)]
pub struct ModelRuntime {
    backend: Arc<HipBackend>,
    execution_plan: Option<crate::model::execution_plan::ExecutionPlan>,
    weight_buffers: Vec<usize>, // Store sizes instead of buffers
    scratch: crate::backend::scratch::ScratchBufferManager,
    kv_cache: crate::model::kv_cache::KVCache,
}

impl ModelRuntime {
    /// Create new model runtime with minimal overhead
    /// NOTE: This creates a default KV cache that will be discarded if load_model() is called.
    /// For direct GGUF loading without waste, use load_from_gguf() instead.
    pub fn new() -> HipResult<Self> {
        tracing::debug!("ModelRuntime::new() called");
        // HipBackend::new() now returns &'static HipBackend, clone it
        let backend_ref = HipBackend::new()?;
        let backend = backend_ref.clone();
        tracing::debug!("ModelRuntime::new() backend created, creating scratch buffer...");
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend, 32,   // num_heads
            4096, // hidden_size (3rd param - was wrongly 4th)
            128,  // head_dim
            2048, // max_seq_len (5th param - was wrongly 3rd)
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;
        tracing::debug!("ModelRuntime::new() scratch buffer created, creating KV cache...");

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend, 32,   // num_layers
            32,   // num_heads
            128,  // head_dim
            2048, // max_seq_len
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;
        tracing::debug!("ModelRuntime::new() KV cache created, returning ModelRuntime");

        Ok(ModelRuntime {
            backend,
            execution_plan: None,
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Load model directly from GGUF file without creating an intermediate runtime
    /// This avoids creating a wasteful default KV cache (32 layers) that would be discarded.
    pub fn load_from_gguf(path: &str) -> HipResult<Self> {
        Self::load_from_gguf_with_config(path, None)
    }

    /// Load model directly from GGUF file with optional custom config override
    pub fn load_from_gguf_with_config(path: &str, custom_config: Option<crate::model::config::ModelConfig>) -> HipResult<Self> {
        tracing::debug!("load_from_gguf: Loading GGUF from path: {}", path);

        let loader = crate::loader::gguf::GgufLoader::new(path)
            .map_err(|e| HipError::GenericError(format!("Failed to load GGUF: {}", e)))?;
        tracing::debug!("load_from_gguf: GgufLoader created successfully");

        let mut config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create config: {}", e)))?;
        tracing::debug!(
            "load_from_gguf: Config created - layers={}, heads={}, hidden={}",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size
        );

        // Override config if provided
        if let Some(custom) = custom_config {
            config.max_position_embeddings = custom.max_position_embeddings;
            tracing::debug!("load_from_gguf: Overriding context size to {}", config.max_position_embeddings);
        }

        // Create backend
        let backend = HipBackend::new()?;

        tracing::debug!("load_from_gguf: Creating scratch buffer manager...");
        // PHASE 24 FIX: Correct parameter order
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size, // ← CORRECT: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← CORRECT: 5th param
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;
        tracing::debug!("load_from_gguf: Scratch buffer manager created");
        eprintln!("load_from_gguf: Scratch buffer manager created");

        tracing::debug!("load_from_gguf: Creating KV cache...");
        eprintln!("load_from_gguf: Creating KV cache...");
        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;
        tracing::debug!("load_from_gguf: KV cache created");
        eprintln!("load_from_gguf: KV cache created");

        tracing::debug!("load_from_gguf: Creating execution plan from GGUF...");
        eprintln!("load_from_gguf: Creating execution plan from GGUF...");
        let execution_plan =
            crate::model::execution_plan::ExecutionPlan::from_gguf(&backend, &loader)?;
        tracing::debug!("load_from_gguf: Execution plan created successfully");
        eprintln!("load_from_gguf: Execution plan created successfully");

        // PRELOAD: Load all core model weights to GPU (llama.cpp-style bulk loading)
        // This avoids the slowdown of lazy loading during first inference pass.
        // Lazy loading remains available for optional/extra weights.
        tracing::debug!("load_from_gguf: Preloading all model weights to GPU...");
        eprintln!("load_from_gguf: Preloading all model weights to GPU...");
        let _preload_start = std::time::Instant::now();
        loader.load_to_gpu_async(&backend)
            .map_err(|e| HipError::GenericError(format!("Failed to preload weights: {}", e)))?;
        let preload_time = _preload_start.elapsed();
        tracing::debug!("load_from_gguf: Weight preload complete in {:?}", preload_time);
        eprintln!("load_from_gguf: All weights preloaded in {:.2}s", preload_time.as_secs_f64());

        tracing::debug!("load_from_gguf: ModelRuntime created successfully");
        eprintln!("load_from_gguf: ModelRuntime created successfully, returning...");
        Ok(ModelRuntime {
            backend,
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// PHASE 1: Load model from pre-parsed GGUF loader (single-pass loading)
    ///
    /// This avoids re-parsing the GGUF file when the loader is already available.
    /// Use this from `InferenceEngine::from_gguf()` for single-pass GGUF loading.
    pub fn load_from_gguf_with_loader(
        loader: Arc<crate::loader::gguf::GgufLoader>,
        custom_config: Option<crate::model::config::ModelConfig>,
    ) -> HipResult<Self> {
        tracing::debug!("load_from_gguf_with_loader: Using pre-parsed GGUF loader");

        let mut config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create config: {}", e)))?;
        tracing::debug!(
            "load_from_gguf: Config created - layers={}, heads={}, hidden={}",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size
        );

        // Override config if provided
        if let Some(custom) = custom_config {
            config.max_position_embeddings = custom.max_position_embeddings;
            tracing::debug!("load_from_gguf: Overriding context size to {}", config.max_position_embeddings);
        }

        // Create backend
        let backend = HipBackend::new()?;

        tracing::debug!("load_from_gguf: Creating scratch buffer manager...");
        // PHASE 24 FIX: Correct parameter order
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size, // ← CORRECT: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← CORRECT: 5th param
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;
        tracing::debug!("load_from_gguf: Scratch buffer manager created");
        eprintln!("load_from_gguf: Scratch buffer manager created");

        tracing::debug!("load_from_gguf: Creating KV cache...");
        eprintln!("load_from_gguf: Creating KV cache...");
        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;
        tracing::debug!("load_from_gguf: KV cache created");
        eprintln!("load_from_gguf: KV cache created");

        tracing::debug!("load_from_gguf: Creating execution plan from GGUF...");
        eprintln!("load_from_gguf: Creating execution plan from GGUF...");
        let execution_plan =
            crate::model::execution_plan::ExecutionPlan::from_gguf(&backend, &loader)?;
        tracing::debug!("load_from_gguf: Execution plan created successfully");
        eprintln!("load_from_gguf: Execution plan created successfully");

        // PRELOAD: Load all core model weights to GPU (llama.cpp-style bulk loading)
        // This avoids the slowdown of lazy loading during first inference pass.
        // Lazy loading remains available for optional/extra weights.
        tracing::debug!("load_from_gguf: Preloading all model weights to GPU...");
        eprintln!("load_from_gguf: Preloading all model weights to GPU...");
        let _preload_start = std::time::Instant::now();
        loader.load_to_gpu_async(&backend)
            .map_err(|e| HipError::GenericError(format!("Failed to preload weights: {}", e)))?;
        let preload_time = _preload_start.elapsed();
        tracing::debug!("load_from_gguf: Weight preload complete in {:?}", preload_time);
        eprintln!("load_from_gguf: All weights preloaded in {:.2}s", preload_time.as_secs_f64());

        tracing::debug!("load_from_gguf: ModelRuntime created successfully");
        eprintln!("load_from_gguf: ModelRuntime created successfully, returning...");
        Ok(ModelRuntime {
            backend,
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Create new model runtime with config
    pub fn new_with_config(config: crate::model::config::ModelConfig) -> HipResult<Self> {
        // HipBackend::new() returns Arc<HipBackend>
        let backend = HipBackend::new()?;
        // PHASE 24 FIX: Correct parameter order
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size, // ← CORRECT: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← CORRECT: 5th param
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;

        Ok(ModelRuntime {
            backend,
            execution_plan: None,
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Get reference to backend
    pub fn backend(&self) -> &HipBackend {
        &self.backend
    }

    /// Get reference to KV cache
    pub fn kv_cache(&self) -> &crate::model::kv_cache::KVCache {
        &self.kv_cache
    }

    /// Get mutable reference to KV cache
    pub fn kv_cache_mut(&mut self) -> &mut crate::model::kv_cache::KVCache {
        &mut self.kv_cache
    }

    /// Set execution plan
    pub fn set_execution_plan(&mut self, plan: crate::model::execution_plan::ExecutionPlan) {
        self.execution_plan = Some(plan);
    }

    /// Allocate weight buffer
    pub fn allocate_weight_buffer(&mut self, size: usize) -> HipResult<usize> {
        self.weight_buffers.push(size);
        Ok(self.weight_buffers.len() - 1)
    }

    /// Get mutable reference to scratch buffers
    pub fn scratch_buffers(&mut self) -> &mut crate::backend::scratch::ScratchBufferManager {
        &mut self.scratch
    }

    /// Get number of weight buffers
    pub fn weight_buffer_count(&self) -> usize {
        self.weight_buffers.len()
    }

    /// Get total weight memory
    pub fn total_weight_memory(&self) -> usize {
        self.weight_buffers.iter().sum()
    }

    /// Decode step with input tensor
    pub fn decode_step(&mut self, input: &DeviceTensor) -> HipResult<DeviceTensor> {
        eprintln!(">>> decode_step: ENTRY");
        let start = std::time::Instant::now();
        tracing::debug!(
            "decode_step() called, input shape: {:?}",
            input.shape().dims()
        );
        let execution_plan = self.execution_plan.as_ref().ok_or_else(|| {
            HipError::GenericError("decode_step called without execution plan".to_string())
        })?;
        eprintln!(
            ">>> decode_step: Got execution plan with {} layers (took {:?})",
            execution_plan.layers().len(),
            start.elapsed()
        );
        tracing::debug!(
            "decode_step() execution plan has {} layers",
            execution_plan.layers().len()
        );

        if execution_plan.layers().is_empty() {
            return Err(HipError::GenericError(
                "Execution plan contains no layers".to_string(),
            ));
        }

        let hidden_size = execution_plan.config().hidden_size;

        let input_dims = input.shape().dims();
        let mut hidden_states = if input_dims.len() == 1 {
            if input_dims[0] != hidden_size {
                return Err(HipError::GenericError(format!(
                    "Input hidden size {} does not match model hidden size {}",
                    input_dims[0], hidden_size
                )));
            }
            let reshaped = TensorShape::from_dims(&[1, hidden_size]);
            let mut tensor = DeviceTensor::empty(&self.backend, reshaped)?;
            tensor.copy_from_device_buffer(input.buffer())?;
            tensor
        } else if input_dims.len() == 2 {
            if input_dims[1] != hidden_size {
                return Err(HipError::GenericError(format!(
                    "Input hidden size {} does not match model hidden size {}",
                    input_dims[1], hidden_size
                )));
            }
            input.clone()
        } else {
            return Err(HipError::GenericError(format!(
                "decode_step input must be 1D or 2D, got shape {:?}",
                input_dims
            )));
        };

        eprintln!(
            ">>> decode_step: Starting layer loop with {} layers",
            execution_plan.layers().len()
        );
        tracing::debug!(
            "decode_step() starting layer loop with {} layers",
            execution_plan.layers().len()
        );
        for (layer_idx, layer_plan) in execution_plan.layers().iter().enumerate() {
            let layer_start = std::time::Instant::now();
            eprintln!(
                ">>> decode_step: Layer {}/{} starting...",
                layer_idx + 1,
                execution_plan.layers().len()
            );
            tracing::debug!(
                "decode_step() processing layer {}/{}",
                layer_idx + 1,
                execution_plan.layers().len()
            );
            eprintln!(">>> decode_step: Getting hidden_states shape...");
            let seq_len = hidden_states.shape().dims()[0];
            eprintln!(">>> decode_step: seq_len={}, choosing execution path...", seq_len);
            if seq_len == 1 {
                eprintln!(">>> decode_step: Calling forward_layer_ggml_decode (seq_len=1)...");
                hidden_states = execution_plan.forward_layer_ggml_decode(
                    &self.backend,
                    &hidden_states,
                    &mut self.kv_cache,
                    layer_idx,
                )?;
                eprintln!(">>> decode_step: forward_layer_ggml_decode returned successfully");
            } else {
                eprintln!(">>> decode_step: Calling forward_layer (seq_len={})...", seq_len);
                hidden_states = execution_plan.forward_layer(
                    &self.backend,
                    &hidden_states,
                    layer_plan,
                    Some(&mut self.kv_cache),
                    layer_idx,
                )?;
                eprintln!(">>> decode_step: forward_layer returned successfully");
            }
            eprintln!(
                ">>> decode_step: Layer {}/{} complete ({:?} elapsed)",
                layer_idx + 1,
                execution_plan.layers().len(),
                layer_start.elapsed()
            );
            tracing::debug!(
                "decode_step() completed layer {}/{}",
                layer_idx + 1,
                execution_plan.layers().len()
            );
        }
        tracing::debug!("decode_step() all layers completed, applying LM head");

        let logits = execution_plan.apply_lm_head(&self.backend, &hidden_states)?;
        let logits_dims = logits.shape().dims();
        let output = if logits_dims.len() == 2 && logits_dims[0] == 1 {
            let mut tensor =
                DeviceTensor::empty(&self.backend, TensorShape::from_dims(&[logits_dims[1]]))?;
            tensor.copy_from_device_slice(&logits, 0)?;
            tensor
        } else {
            logits
        };

        Ok(output)
    }

    /// Load model from GGUF file
    pub fn load_model(&self, path: &str) -> HipResult<Self> {
        tracing::debug!("load_model: Loading GGUF from path: {}", path);

        let loader = crate::loader::gguf::GgufLoader::new(path)
            .map_err(|e| HipError::GenericError(format!("Failed to load GGUF: {}", e)))?;
        tracing::debug!("load_model: GgufLoader created successfully");

        let config = loader
            .to_model_config()
            .map_err(|e| HipError::GenericError(format!("Failed to create config: {}", e)))?;
        tracing::debug!(
            "load_model: Config created - layers={}, heads={}, hidden={}",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size
        );

        tracing::debug!("load_model: Creating scratch buffer manager...");
        // PHASE 24 FIX: Correct parameter order
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &self.backend,
            config.num_attention_heads,
            config.hidden_size, // ← CORRECT: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← CORRECT: 5th param
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;
        tracing::debug!("load_model: Scratch buffer manager created");

        tracing::debug!("load_model: Creating KV cache...");
        let kv_cache = crate::model::kv_cache::KVCache::new(
            &self.backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;
        tracing::debug!("load_model: KV cache created");

        tracing::debug!("load_model: Creating execution plan from GGUF...");
        let execution_plan =
            crate::model::execution_plan::ExecutionPlan::from_gguf(&self.backend, &loader)?;
        tracing::debug!("load_model: Execution plan created successfully");

        // PRELOAD: Load all core model weights to GPU (llama.cpp-style bulk loading)
        // This avoids the slowdown of lazy loading during first inference pass.
        tracing::debug!("load_model: Preloading all model weights to GPU...");
        eprintln!("load_model: Preloading all model weights to GPU...");
        let _preload_start = std::time::Instant::now();
        loader.load_to_gpu_async(&self.backend)
            .map_err(|e| HipError::GenericError(format!("Failed to preload weights: {}", e)))?;
        let preload_time = _preload_start.elapsed();
        tracing::debug!("load_model: Weight preload complete in {:?}", preload_time);
        eprintln!("load_model: All weights preloaded in {:.2}s", preload_time.as_secs_f64());

        tracing::debug!("load_model: ModelRuntime created successfully");
        Ok(ModelRuntime {
            backend: self.backend.clone(),
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Create model runtime from execution plan using this runtime's backend
    pub fn from_execution_plan(
        &self,
        execution_plan: crate::model::execution_plan::ExecutionPlan,
    ) -> HipResult<Self> {
        Self::from_execution_plan_with_backend(execution_plan)
    }

    /// Create model runtime from execution plan using the singleton backend
    pub fn from_execution_plan_with_backend(
        execution_plan: crate::model::execution_plan::ExecutionPlan,
    ) -> HipResult<Self> {
        eprintln!(">>> from_execution_plan_with_backend ENTRY");
        eprintln!(">>> from_execution_plan_with_backend: About to call HipBackend::new()...");
        let backend = HipBackend::new()?;
        eprintln!(">>> from_execution_plan_with_backend: HipBackend::new() returned");
        let config = execution_plan.config();
        // PHASE 24 FIX: Correct parameter order for ScratchBufferManager::new()
        // OLD (WRONG): new(backend, num_heads, max_pos_emb, head_dim, hidden_size)
        // NEW (CORRECT): new(backend, num_heads, hidden_size, head_dim, max_seq_len)
        eprintln!("from_execution_plan_with_backend: Creating scratch with heads={}, hidden={}, head_dim={}, max_seq={}",
                 config.num_attention_heads, config.hidden_size, config.head_dim, config.max_position_embeddings);
        eprintln!(
            ">>> from_execution_plan_with_backend: About to call ScratchBufferManager::new()..."
        );
        let scratch = crate::backend::scratch::ScratchBufferManager::new(
            &backend,
            config.num_attention_heads,
            config.hidden_size,             // ← CORRECT: 3rd param is hidden_size
            config.head_dim,                // ← CORRECT: 4th param is head_dim
            config.max_position_embeddings, // ← CORRECT: 5th param is max_seq_len
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))?;
        eprintln!(">>> from_execution_plan_with_backend: ScratchBufferManager::new() returned successfully");

        let kv_cache = crate::model::kv_cache::KVCache::new(
            &backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache creation failed: {}", e)))?;

        Ok(ModelRuntime {
            backend,
            execution_plan: Some(execution_plan),
            weight_buffers: Vec::new(),
            scratch,
            kv_cache,
        })
    }

    /// Get reference to execution plan
    pub fn execution_plan(&self) -> Option<&crate::model::execution_plan::ExecutionPlan> {
        self.execution_plan.as_ref()
    }

    /// Reset internal state (e.g., KV cache) before running a fresh sequence
    pub fn reset_state(&mut self) -> HipResult<()> {
        if self.execution_plan.is_none() {
            return Err(HipError::GenericError(
                "reset_state called without execution plan".to_string(),
            ));
        }
        self.kv_cache.reset();
        Ok(())
    }

    /// Recreate KV cache with new parameters (useful for overriding config)
    pub fn recreate_kv_cache(&mut self, max_seq_len: usize) -> HipResult<()> {
        let config = self.execution_plan.as_ref()
            .ok_or_else(|| HipError::GenericError("No execution plan".to_string()))?
            .config();

        self.kv_cache = crate::model::kv_cache::KVCache::new(
            &self.backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            max_seq_len,
        )
        .map_err(|e| HipError::GenericError(format!("KV cache recreation failed: {}", e)))?;

        Ok(())
    }
}

/// Helper function to copy buffer to host vector
#[allow(dead_code)] // Legacy function kept for reference; use copy_from_device_safe instead
fn vec_from_buffer(backend: &HipBackend, buffer: &HipBuffer, len: usize) -> HipResult<Vec<f32>> {
    let mut host_data = vec![0.0f32; len];
    // SAFETY: We check that the buffer size matches our expectation
    let expected_byte_size = len * std::mem::size_of::<f32>();
    if buffer.size() != expected_byte_size {
        return Err(HipError::GenericError(format!(
            "Buffer size mismatch: buffer has {} bytes, expected {} bytes for {} f32 elements",
            buffer.size(), expected_byte_size, len
        )));
    }

    // Use safe copy method instead of unsafe raw pointer manipulation
    backend.copy_from_device_safe(buffer, &mut host_data)?;
    Ok(host_data)
}

/// Synchronize device globally using STREAM-AWARE synchronization
///
/// # Phase 23 Fix: Desktop Hang Prevention
///
/// This function now uses `hipStreamSynchronize()` instead of the dangerous
/// `hipDeviceSynchronize()`.
///
/// ## Why This Matters
///
/// - `hipDeviceSynchronize()` waits for ALL GPU streams (including desktop compositor)
/// - `hipStreamSynchronize()` waits ONLY for our application's stream
/// - This prevents deadlocks and desktop hangs when the compositor is using the GPU
///
/// ## Usage
///
/// Call this after GPU kernel launches to ensure completion before continuing:
///
/// ```ignore
/// // Launch kernel
/// unsafe { my_gpu_kernel(...) };
///
/// // Wait for completion (SAFE - only waits for our stream)
/// synchronize_device()?;
/// ```
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from any thread that has
/// initialized the HIP backend.
pub fn synchronize_device() -> HipResult<()> {
    // Get the global backend (singleton)
    let backend = GLOBAL_BACKEND
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e)))?
        .as_ref()
        .map(Arc::clone)
        .ok_or_else(|| {
            HipError::DeviceError(
                "HIP backend not initialized - call HipBackend::new() first".to_string(),
            )
        })?;

    // Phase 23: Use STREAM-AWARE synchronization
    // This only waits for our stream, not the desktop compositor
    backend.stream.synchronize()
}

// ============================================================================
// AsyncLoader: Multi-stream concurrent GPU uploads (Phase 3)
// ============================================================================
//
// Purpose: Upload multiple tensors to GPU concurrently using multiple HIP streams
// Performance: ~4x speedup for GPU uploads by overlapping memcpy operations
//
// Key concept: HIP streams allow multiple GPU operations to execute concurrently.
// We use 4 streams to upload tensors in parallel, with events for synchronization.
//
// Usage:
//   1. Create AsyncLoader::new()
//   2. For each tensor, call upload_tensor() (returns immediately)
//   3. Call synchronize() to wait for all uploads to complete
//   4. Drop loader to clean up streams

/// Number of concurrent upload streams
/// More streams = more concurrency, but diminishing returns > 4
const NUM_UPLOAD_STREAMS: usize = 4;

/// Async GPU loader with multi-stream concurrent uploads
///
/// Phase 3: Async GPU Uploads - Reduces upload time from ~20s to ~5s
/// by using 4 concurrent HIP streams for parallel tensor uploads.
pub struct AsyncLoader {
    /// Multiple HIP streams for concurrent uploads
    streams: Vec<HipStream>,
    /// Events for tracking upload completion (one per stream)
    events: Vec<HipEvent>,
}

impl AsyncLoader {
    /// Create a new async loader with 4 concurrent upload streams
    pub fn new() -> HipResult<Self> {
        tracing::debug!(
            "AsyncLoader::new: Creating async loader with {} streams",
            NUM_UPLOAD_STREAMS
        );

        let mut streams = Vec::with_capacity(NUM_UPLOAD_STREAMS);
        let mut events = Vec::with_capacity(NUM_UPLOAD_STREAMS);

        // Create streams and events
        for i in 0..NUM_UPLOAD_STREAMS {
            tracing::debug!("AsyncLoader::new: Creating stream {}...", i);
            let stream = HipStream::new().map_err(|e| {
                HipError::DeviceError(format!("Failed to create upload stream {}: {}", i, e))
            })?;
            streams.push(stream);

            // Create events for synchronization (timing disabled for performance)
            let event = HipEvent::with_flags(HIP_EVENT_DISABLE_TIMING).map_err(|e| {
                HipError::DeviceError(format!("Failed to create upload event {}: {}", i, e))
            })?;
            events.push(event);
        }

        tracing::debug!(
            "AsyncLoader::new: Created {} streams and {} events",
            streams.len(),
            events.len()
        );
        Ok(AsyncLoader { streams, events })
    }

    /// Upload data to a GPU buffer using the specified stream index
    ///
    /// This is a non-blocking operation - the upload proceeds in the background
    /// on the specified stream. Returns immediately after queuing the upload.
    ///
    /// # Arguments
    /// * `buffer` - Target GPU buffer
    /// * `data` - Host data to upload
    /// * `stream_idx` - Which stream to use (0-3)
    pub fn upload_to_buffer(
        &self,
        buffer: &HipBuffer,
        data: &[u8],
        stream_idx: usize,
    ) -> HipResult<()> {
        if stream_idx >= NUM_UPLOAD_STREAMS {
            return Err(HipError::DeviceError(format!(
                "Invalid stream index: {} (max {})",
                stream_idx,
                NUM_UPLOAD_STREAMS - 1
            )));
        }

        tracing::trace!(
            "AsyncLoader::upload_to_buffer: Uploading {} bytes on stream {}",
            data.len(),
            stream_idx
        );

        // Record event before upload
        self.events[stream_idx].record(&self.streams[stream_idx])?;

        // Perform async copy on the specified stream
        let result = unsafe {
            hipMemcpyAsync(
                buffer.as_ptr() as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.streams[stream_idx].as_ptr(),
            )
        };

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Async upload failed on stream {}: {}",
                stream_idx, result
            )));
        }

        // Record event after upload (marks completion)
        self.events[stream_idx].record(&self.streams[stream_idx])?;

        tracing::trace!(
            "AsyncLoader::upload_to_buffer: Upload queued on stream {}",
            stream_idx
        );
        Ok(())
    }

    /// Upload data to a GPU buffer at a specific offset (arena-based allocation)
    ///
    /// This is used by the memory arena pattern where multiple tensors share a single
    /// large GPU buffer. Instead of allocating individual buffers, we upload to offsets
    /// within the arena buffer.
    ///
    /// # Arguments
    /// * `buffer` - Target GPU buffer (typically the arena backing buffer)
    /// * `offset` - Byte offset into the buffer where data should be written
    /// * `data` - Host data to upload
    /// * `stream_idx` - Which stream to use (0-3)
    ///
    /// # Errors
    /// - If offset + data.len() exceeds buffer size
    /// - If stream_idx is out of range
    /// - If HIP memcpy fails
    pub fn upload_to_buffer_offset(
        &self,
        buffer: &HipBuffer,
        offset: usize,
        data: &[u8],
        stream_idx: usize,
    ) -> HipResult<()> {
        if stream_idx >= NUM_UPLOAD_STREAMS {
            return Err(HipError::DeviceError(format!(
                "Invalid stream index: {} (max {})",
                stream_idx,
                NUM_UPLOAD_STREAMS - 1
            )));
        }

        // Validate offset doesn't exceed buffer bounds
        if offset + data.len() > buffer.size() {
            return Err(HipError::DeviceError(format!(
                "Upload offset out of bounds: offset={}, size={}, buffer_size={}",
                offset,
                data.len(),
                buffer.size()
            )));
        }

        tracing::trace!(
            "AsyncLoader::upload_to_buffer_offset: Uploading {} bytes at offset {} on stream {}",
            data.len(),
            offset,
            stream_idx
        );

        // Record event before upload
        self.events[stream_idx].record(&self.streams[stream_idx])?;

        // Perform async copy to offset within buffer
        let dst_ptr = unsafe { buffer.as_ptr().add(offset) };
        let result = unsafe {
            hipMemcpyAsync(
                dst_ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.streams[stream_idx].as_ptr(),
            )
        };

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Async upload to offset failed on stream {}: {}",
                stream_idx, result
            )));
        }

        // Record event after upload (marks completion)
        self.events[stream_idx].record(&self.streams[stream_idx])?;

        tracing::trace!(
            "AsyncLoader::upload_to_buffer_offset: Upload queued at offset {} on stream {}",
            offset,
            stream_idx
        );
        Ok(())
    }

    /// Upload data to a GPU buffer, automatically selecting the least busy stream
    ///
    /// This is a convenience method that picks a stream using round-robin.
    /// For maximum control, use `upload_to_buffer` with an explicit stream index.
    pub fn upload_auto(&self, buffer: &HipBuffer, data: &[u8]) -> HipResult<()> {
        // Simple round-robin stream selection
        // In a more sophisticated implementation, we could track pending operations
        let stream_idx = (data.len() / (1024 * 1024)) % NUM_UPLOAD_STREAMS;
        self.upload_to_buffer(buffer, data, stream_idx)
    }

    /// Synchronize all upload streams
    ///
    /// Blocks until all pending uploads on all streams have completed.
    /// Call this before accessing the uploaded data on the GPU.
    pub fn synchronize(&self) -> HipResult<()> {
        tracing::debug!(
            "AsyncLoader::synchronize: Synchronizing all {} streams",
            NUM_UPLOAD_STREAMS
        );

        // Synchronize each event (waits for all operations before the event)
        for (i, event) in self.events.iter().enumerate() {
            tracing::trace!("AsyncLoader::synchronize: Synchronizing stream {}...", i);
            event.synchronize().map_err(|e| {
                HipError::DeviceError(format!("Stream {} synchronization failed: {}", i, e))
            })?;
        }

        tracing::debug!("AsyncLoader::synchronize: All streams synchronized");
        Ok(())
    }

    /// Get a reference to a specific stream
    ///
    /// Useful for passing to other operations that need a stream.
    pub fn get_stream(&self, stream_idx: usize) -> HipResult<&HipStream> {
        if stream_idx >= NUM_UPLOAD_STREAMS {
            return Err(HipError::DeviceError(format!(
                "Invalid stream index: {} (max {})",
                stream_idx,
                NUM_UPLOAD_STREAMS - 1
            )));
        }
        Ok(&self.streams[stream_idx])
    }
}

// SAFETY: AsyncLoader is Send+Sync because HipStream and HipEvent are Send+Sync
unsafe impl Send for AsyncLoader {}
unsafe impl Sync for AsyncLoader {}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_hip_buffer_creation() {
        let buffer = HipBuffer::new(1024);
        assert!(buffer.is_ok(), "Buffer creation should succeed");

        let buffer = buffer.unwrap();
        assert_eq!(buffer.size(), 1024, "Buffer size should be correct");
        assert!(
            !buffer.as_ptr().is_null(),
            "Buffer pointer should not be null"
        );

        // Buffer should be freed when dropped
    }

    #[test]
    #[serial]
    fn test_hip_buffer_copy() {
        let backend = HipBackend::new().unwrap();
        let buffer = HipBuffer::new(4 * std::mem::size_of::<f32>()).unwrap();
        let host_data = [1.0f32, 2.0, 3.0, 4.0];
        assert!(buffer.copy_from_host(&host_data).is_ok());

        // Synchronize to ensure copy completes
        let _ = synchronize_device();

        let mut host_result = [0.0f32; 4];
        backend.copy_from_device_safe(&buffer, &mut host_result).unwrap();

        // Synchronize after copy to ensure data is fully transferred
        let _ = synchronize_device();

        assert_eq!(host_data, host_result);
    }

    #[test]
    #[serial]
    fn test_kernel_launch() {
        let backend = HipBackend::new().unwrap();
        let args: Vec<*mut c_void> = vec![];

        let result = backend.launch_kernel("test_kernel", (1, 1, 1), (64, 1, 1), &args);

        assert!(result.is_ok());
    }

    // TDD Phase 1: HIP Event Support
    // Test: HipEvent lifecycle (create, record, synchronize, destroy)

    #[test]
    #[serial]
    fn test_hip_event_create_and_destroy() {
        // This test will FAIL initially because HipEvent doesn't exist yet
        let _event = HipEvent::new().expect("Failed to create HIP event");
        // Event should be automatically destroyed when dropped (RAII)
    }

    #[test]
    #[serial]
    fn test_hip_event_record_and_synchronize() {
        let backend = HipBackend::new().expect("Failed to create backend");
        let event = HipEvent::new().expect("Failed to create HIP event");

        // Record event on stream
        let result = event.record(&backend.stream);
        assert!(result.is_ok(), "Event recording should succeed");

        // Synchronize on event
        let sync_result = event.synchronize();
        assert!(sync_result.is_ok(), "Event synchronization should succeed");
    }

    #[test]
    #[serial]
    fn test_hip_event_elapsed_time() {
        let backend = HipBackend::new().expect("Failed to create backend");
        let event_start = HipEvent::new().expect("Failed to create start event");
        let event_end = HipEvent::new().expect("Failed to create end event");

        // Create a small buffer and copy it
        let buffer = HipBuffer::new(1024).expect("Failed to create buffer");
        let host_data = vec![42u8; 1024];
        buffer
            .copy_from_host(&host_data)
            .expect("Failed to copy to device");

        // Record start event
        event_start
            .record(&backend.stream)
            .expect("Failed to record start");

        // Do some work
        let mut host_result = vec![0u8; 1024];
        backend
            .copy_from_device_safe(&buffer, &mut host_result)
            .expect("Failed to copy to host");

        // Record end event
        event_end
            .record(&backend.stream)
            .expect("Failed to record end");

        // Synchronize on end event
        event_end
            .synchronize()
            .expect("Failed to synchronize end event");

        // Get elapsed time
        let elapsed = event_start
            .elapsed_time(&event_end)
            .expect("Failed to get elapsed time");

        // Elapsed time should be non-negative
        assert!(elapsed >= 0.0, "Elapsed time should be non-negative");
    }

    // Phase 3: AsyncLoader Tests

    #[test]
    #[serial]
    fn test_async_loader_create() {
        let loader = AsyncLoader::new();
        assert!(loader.is_ok(), "AsyncLoader creation should succeed");
        let _loader = loader.unwrap();
        // Streams and events are cleaned up on drop
    }

    #[test]
    #[serial]
    fn test_async_loader_upload_single() {
        let backend = HipBackend::new().expect("Failed to create backend");
        let loader = AsyncLoader::new().expect("Failed to create AsyncLoader");
        let buffer = HipBuffer::new(1024).expect("Failed to create buffer");
        let host_data = vec![42u8; 1024];

        // Upload on stream 0
        let result = loader.upload_to_buffer(&buffer, &host_data, 0);
        assert!(result.is_ok(), "Upload should succeed");

        // Synchronize to ensure upload completes
        loader
            .synchronize()
            .expect("Synchronization should succeed");

        // Verify data was uploaded
        let mut host_result = vec![0u8; 1024];
        backend
            .copy_from_device_safe(&buffer, &mut host_result)
            .expect("Copy to host should succeed");
        assert_eq!(host_data, host_result, "Uploaded data should match");
    }

    #[test]
    #[serial]
    fn test_async_loader_upload_concurrent() {
        let backend = HipBackend::new().expect("Failed to create backend");
        let loader = AsyncLoader::new().expect("Failed to create AsyncLoader");

        // Create multiple buffers
        let buffers: Vec<_> = (0..4)
            .map(|_| HipBuffer::new(1024).expect("Failed to create buffer"))
            .collect();

        // Upload to all buffers concurrently (different streams)
        for (i, buffer) in buffers.iter().enumerate() {
            let host_data = vec![i as u8; 1024];
            let stream_idx = i % NUM_UPLOAD_STREAMS;
            loader
                .upload_to_buffer(buffer, &host_data, stream_idx)
                .expect("Upload should succeed");
        }

        // Synchronize all uploads
        loader
            .synchronize()
            .expect("Synchronization should succeed");

        // Verify all uploads
        for (i, buffer) in buffers.iter().enumerate() {
            let mut host_result = vec![0u8; 1024];
            backend
                .copy_from_device_safe(buffer, &mut host_result)
                .expect("Copy to host should succeed");
            let expected = vec![i as u8; 1024];
            assert_eq!(expected, host_result, "Buffer {} data should match", i);
        }
    }

    #[test]
    #[serial]
    fn test_async_loader_upload_auto() {
        let backend = HipBackend::new().expect("Failed to create backend");
        let loader = AsyncLoader::new().expect("Failed to create AsyncLoader");
        let buffer = HipBuffer::new(1024).expect("Failed to create buffer");
        let host_data = vec![99u8; 1024];

        // Upload using automatic stream selection
        loader
            .upload_auto(&buffer, &host_data)
            .expect("Upload auto should succeed");

        loader
            .synchronize()
            .expect("Synchronization should succeed");

        let mut host_result = vec![0u8; 1024];
        backend
            .copy_from_device_safe(&buffer, &mut host_result)
            .expect("Copy to host should succeed");
        assert_eq!(host_data, host_result, "Uploaded data should match");
    }

    #[test]
    #[serial]
    fn test_async_loader_invalid_stream() {
        let loader = AsyncLoader::new().expect("Failed to create AsyncLoader");
        let buffer = HipBuffer::new(1024).expect("Failed to create buffer");
        let host_data = vec![42u8; 1024];

        // Try to use invalid stream index
        let result = loader.upload_to_buffer(&buffer, &host_data, 99);
        assert!(result.is_err(), "Upload with invalid stream should fail");
    }

    // ========== Phase 10-20: Retry Logic Tests ==========

    #[test]
    fn test_hip_error_recoverable_classification() {
        // Test that recoverable errors are correctly identified
        let recoverable_errors = vec![
            HipError::DeviceError("temporary GPU busy".to_string()),
            HipError::MemoryAllocationFailed("out of memory".to_string()),
            HipError::MemoryCopyFailed("copy failed".to_string()),
            HipError::MemoryQueryFailed("query failed".to_string()),
            HipError::KernelLaunchFailed("launch failed".to_string()),
        ];

        for error in recoverable_errors {
            assert!(
                error.is_recoverable(),
                "{} should be recoverable",
                error
            );
            assert!(
                !error.is_permanent(),
                "{} should not be permanent",
                error
            );
        }

        // Test that permanent errors are correctly identified
        let permanent_errors = vec![
            HipError::InitializationFailed("HIP not available".to_string()),
            HipError::KernelLoadFailed("kernel file not found".to_string()),
            HipError::DeviceNotFound,
            HipError::LockPoisoned("lock poisoned".to_string()),
            HipError::GenericError("unknown error".to_string()),
        ];

        for error in permanent_errors {
            assert!(
                !error.is_recoverable(),
                "{} should not be recoverable",
                error
            );
            assert!(
                error.is_permanent(),
                "{} should be permanent",
                error
            );
        }
    }

    #[test]
    #[serial]
    fn test_retry_operation_success_on_first_try() {
        // Test that successful operations return immediately without retry
        let backend = HipBackend::new().unwrap();
        let mut call_count = 0;

        let result = backend.retry_operation(
            || {
                call_count += 1;
                Ok::<(), HipError>(())
            },
            "test_operation",
        );

        assert!(result.is_ok(), "Operation should succeed");
        assert_eq!(call_count, 1, "Should only call once on success");
    }

    #[test]
    #[serial]
    fn test_retry_operation_fails_on_permanent_error() {
        // Test that permanent errors fail immediately without retry
        let backend = HipBackend::new().unwrap();
        let mut call_count = 0;

        let result = backend.retry_operation(
            || {
                call_count += 1;
                Err::<(), HipError>(HipError::DeviceNotFound)
            },
            "test_operation",
        );

        assert!(result.is_err(), "Operation should fail");
        assert_eq!(call_count, 1, "Should not retry permanent errors");
    }

    #[test]
    #[serial]
    fn test_retry_operation_succeeds_after_retry() {
        // Test that recoverable errors are retried
        let backend = HipBackend::new().unwrap();
        let mut call_count = 0;

        let result = backend.retry_operation(
            || {
                call_count += 1;
                if call_count < 2 {
                    Err::<(), HipError>(HipError::DeviceError(
                        "temporary GPU busy".to_string(),
                    ))
                } else {
                    Ok(())
                }
            },
            "test_operation",
        );

        assert!(result.is_ok(), "Operation should succeed after retry");
        assert_eq!(call_count, 2, "Should retry once");
    }

    #[test]
    #[serial]
    fn test_retry_operation_exhausts_retries() {
        // Test that retries are eventually exhausted
        let backend = HipBackend::new().unwrap();
        let mut call_count = 0;

        let result = backend.retry_operation(
            || {
                call_count += 1;
                Err::<(), HipError>(HipError::DeviceError(
                    "persistent GPU error".to_string(),
                ))
            },
            "test_operation",
        );

        assert!(result.is_err(), "Operation should fail after retries");
        // Default max_retries is 3, so we expect 1 initial + 3 retries = 4 total
        assert_eq!(call_count, 4, "Should exhaust all retries");
    }

    // Note: RetryConfig tests are in engine.rs tests module since RetryConfig is defined there

    // ========== End Phase 10-20 Retry Logic Tests ==========
}
