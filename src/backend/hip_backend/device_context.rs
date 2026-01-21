//! Thread-safe GPU device context management
//!
//! **CRITICAL:** HIP device context is PER-THREAD, not global.
//! - hipSetDevice() only affects the calling thread
//! - hipGetDevice() returns the current device for the calling thread
//! - GPU pointers allocated in one thread's context are invalid in another thread's context
//!
//! This module provides thread-local device tracking and context verification.

use std::cell::Cell;
use std::thread_local;

use crate::backend::hip_backend::device::get_error_string;
use crate::backend::hip_backend::error::{HipError, HipResult};
use crate::backend::hip_backend::ffi;

/// Thread-local current device tracking
///
/// Each thread maintains its own HIP device context.
/// We cache the device ID to avoid repeated hipGetDevice() calls.
thread_local! {
    static THREAD_DEVICE_ID: Cell<i32> = Cell::new(-1);
}

/// Get the current HIP device for this thread
///
/// Returns the cached device ID if available, otherwise queries the HIP runtime.
/// The result is cached for future calls.
///
/// # Returns
///
/// * `Ok(i32)` - The current device ID (typically 0 for single-GPU systems)
/// * `Err(HipError)` - If hipGetDevice() fails
///
/// # Example
///
/// ```rust
/// use crate::backend::hip_backend::device_context::get_thread_device;
///
/// match get_thread_device() {
///     Ok(device_id) => println!("Current thread using device {}", device_id),
///     Err(e) => eprintln!("Failed to get device: {}", e),
/// }
/// ```
pub fn get_thread_device() -> HipResult<i32> {
    THREAD_DEVICE_ID.with(|cell| {
        let cached = cell.get();
        if cached >= 0 {
            return Ok(cached);
        }

        // Query HIP for current device
        let mut device: i32 = -1;
        let result = unsafe { ffi::hipGetDevice(&mut device) };
        if result != ffi::HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "hipGetDevice failed: {}",
                get_error_string(result)
            )));
        }

        // Cache the result
        cell.set(device);
        Ok(device)
    })
}

/// Ensure current thread is using the correct device
///
/// If the current device doesn't match the expected device, calls hipSetDevice()
/// to set the correct device for this thread.
///
/// # Arguments
///
/// * `expected_device` - The device ID that should be active (typically 0)
///
/// # Returns
///
/// * `Ok(())` - Device is correct or was successfully set
/// * `Err(HipError)` - If hipSetDevice() fails
///
/// # Example
///
/// ```rust
/// use crate::backend::hip_backend::device_context::ensure_device;
///
/// // Make sure we're using device 0 before GPU operations
/// ensure_device(0)?;
/// ```
pub fn ensure_device(expected_device: i32) -> HipResult<()> {
    let current = get_thread_device()?;
    if current != expected_device {
        tracing::debug!(
            "Device mismatch: current={}, expected={}, calling hipSetDevice({})",
            current,
            expected_device,
            expected_device
        );

        let result = unsafe { ffi::hipSetDevice(expected_device) };
        if result != ffi::HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "hipSetDevice({}) failed: {}",
                expected_device,
                get_error_string(result)
            )));
        }

        // Update cached value
        THREAD_DEVICE_ID.with(|cell| cell.set(expected_device));
    }
    Ok(())
}

/// Verify device context matches (for debugging/validation)
///
/// Returns true if the current thread's device matches the expected device.
/// This does NOT call hipSetDevice - it only checks.
///
/// # Arguments
///
/// * `expected_device` - The device ID to check against
///
/// # Returns
///
/// * `true` - Current device matches expected
/// * `false` - Device mismatch (caller should call ensure_device())
pub fn verify_device(expected_device: i32) -> bool {
    match get_thread_device() {
        Ok(current) => current == expected_device,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_get_thread_device() {
        // Assuming single-GPU system, device 0 should be available
        let result = get_thread_device();
        assert!(
            result.is_ok(),
            "hipGetDevice should succeed on GPU system"
        );
        let device_id = result.unwrap();
        assert!(
            device_id >= 0,
            "Device ID should be non-negative, got {}",
            device_id
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_ensure_device_idempotent() {
        // Calling ensure_device multiple times with same ID should be safe
        assert!(ensure_device(0).is_ok(), "First ensure_device(0) should succeed");
        assert!(ensure_device(0).is_ok(), "Second ensure_device(0) should succeed");
        assert!(ensure_device(0).is_ok(), "Third ensure_device(0) should succeed");
    }

    #[test]
    #[serial_test::serial]
    fn test_verify_device() {
        // Set device to 0
        ensure_device(0).unwrap();

        // Verify should return true for device 0
        assert!(verify_device(0), "verify_device(0) should return true after ensure_device(0)");
    }
}
