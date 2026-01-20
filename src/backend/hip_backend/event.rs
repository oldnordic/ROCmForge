//! HIP event wrapper for async GPU loading synchronization

use std::ptr;

use crate::backend::hip_backend::error::HipResult;
use crate::backend::hip_backend::ffi;
use crate::backend::hip_backend::HipError;

// SAFETY: HipEvent is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: HipEvent does NOT implement Clone because cloning raw pointers
// would cause double-free when both instances are dropped.
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility - ensures C-compatible layout
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

/// HIP event wrapper for async GPU loading synchronization
///
/// Events allow tracking completion of GPU operations across streams
#[repr(C)]
#[derive(Debug)]
pub struct HipEvent {
    event: *mut std::ffi::c_void,
}

impl HipEvent {
    /// Create a new HIP event with default timing enabled
    ///
    /// Events are used to track completion of operations in a stream.
    /// Timing is enabled by default for performance measurement.
    pub fn new() -> HipResult<Self> {
        tracing::debug!("HipEvent::new: Creating HIP event...");
        let mut event: *mut std::ffi::c_void = ptr::null_mut();

        let result = unsafe { ffi::hipEventCreate(&mut event) };
        tracing::debug!(
            "HipEvent::new: hipEventCreate returned result={}, event={:?}",
            result,
            event
        );

        if result != ffi::HIP_SUCCESS {
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
        let mut event: *mut std::ffi::c_void = ptr::null_mut();

        let result = unsafe { ffi::hipEventCreateWithFlags(&mut event, flags) };
        tracing::debug!(
            "HipEvent::with_flags: hipEventCreateWithFlags returned result={}, event={:?}",
            result,
            event
        );

        if result != ffi::HIP_SUCCESS {
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
    pub fn record(&self, stream: &super::stream::HipStream) -> HipResult<()> {
        tracing::trace!("HipEvent::record: Recording event in stream...");
        let result = unsafe { ffi::hipEventRecord(self.event, stream.as_ptr()) };

        if result != ffi::HIP_SUCCESS {
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
        let result = unsafe { ffi::hipEventSynchronize(self.event) };

        if result != ffi::HIP_SUCCESS {
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
        let result = unsafe { ffi::hipEventElapsedTime(&mut ms, self.event, end.event) };

        if result != ffi::HIP_SUCCESS {
            Err(HipError::DeviceError(format!(
                "Failed to get elapsed time: {}",
                result
            )))
        } else {
            Ok(ms)
        }
    }

    /// Get raw event pointer (for FFI calls)
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.event
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            tracing::trace!("HipEvent::drop: Destroying HIP event");
            unsafe {
                ffi::hipEventDestroy(self.event);
            }
        }
    }
}
