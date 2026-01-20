//! HIP stream wrapper

use std::ptr;

use crate::backend::hip_backend::error::HipResult;
use crate::backend::hip_backend::ffi;
use crate::backend::hip_backend::HipError;

// SAFETY: HipStream is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: HipStream does NOT implement Clone because cloning raw pointers
// would cause double-free when both instances are dropped.
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility - ensures C-compatible layout
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

/// HIP stream wrapper
#[repr(C)]
#[derive(Debug)]
pub struct HipStream {
    pub(in crate::backend::hip_backend) stream: *mut std::ffi::c_void,
}

impl HipStream {
    /// Create a new HIP stream
    pub fn new() -> HipResult<Self> {
        tracing::debug!("HipStream::new: Creating HIP stream...");
        let mut stream: *mut std::ffi::c_void = ptr::null_mut();

        // Create HIP stream
        tracing::debug!("HipStream::new: Calling hipStreamCreate...");
        let result = unsafe { ffi::hipStreamCreate(&mut stream) };
        tracing::debug!(
            "HipStream::new: hipStreamCreate returned result={}, stream={:?}",
            result,
            stream
        );

        if result != ffi::HIP_SUCCESS {
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

    /// Synchronize the stream
    pub fn synchronize(&self) -> HipResult<()> {
        let result = unsafe { ffi::hipStreamSynchronize(self.stream) };
        if result != ffi::HIP_SUCCESS {
            Err(HipError::DeviceError(format!(
                "Stream synchronization failed: {}",
                result
            )))
        } else {
            Ok(())
        }
    }

    /// Get raw stream pointer (for FFI calls like hipblasSetStream)
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.stream
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                ffi::hipStreamDestroy(self.stream);
            }
        }
    }
}
