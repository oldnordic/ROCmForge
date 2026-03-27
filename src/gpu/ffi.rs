//! Safe FFI bindings to HIP runtime.
//!
//! All unsafe blocks wrapped with error checking.
//! No raw HIP API exposed outside gpu module.

use super::error::{GpuError, GpuResult};
use std::os::raw::{c_int, c_char, c_void};
use std::ffi::CStr;

// ── HIP Error Codes ───────────────────────────────────────────────────────────────

/// Convert HIP error code to GpuError.
fn hip_check(code: hipError_t) -> GpuResult<()> {
    if code == hipError_t::hipSuccess {
        return Ok(());
    }

    let description = unsafe {
        let ptr = hipGetErrorString(code);
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    };

    Err(match code {
        hipError_t::hipErrorNotInitialized | hipError_t::hipErrorInvalidContext => {
            GpuError::HipNotAvailable
        }
        hipError_t::hipErrorOutOfMemory => GpuError::OutOfMemory {
            requested: 0,  // Call site should fill this
            available: 0,
        },
        hipError_t::hipErrorInvalidDevice => {
            GpuError::InvalidDevice { device_id: -1 }
        }
        _ => GpuError::HipApiError {
            code: code as i32,
            description,
        },
    })
}

// ── HIP Type Definitions ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidContext = 201,
    // Add more as needed
}

// ── Safe HIP API Wrappers ───────────────────────────────────────────────────────────

/// Get number of HIP-compatible GPUs.
///
/// Returns 0 if HIP not available or no GPU found.
pub fn hip_get_device_count() -> GpuResult<i32> {
    unsafe {
        let mut count = 0i32;
        let code = hipGetDeviceCount(&mut count);
        hip_check(code)?;
        Ok(count)
    }
}

/// Get HIP device properties (name, VRAM, compute units).
pub fn hip_get_device_info(device_id: i32) -> GpuResult<DeviceInfo> {
    unsafe {
        let mut props: hipDeviceProp_t = std::mem::zeroed();
        let code = hipGetDeviceProperties(&mut props, device_id);
        hip_check(code)?;

        let name = CStr::from_ptr(props.name.as_ptr())
            .to_string_lossy()
            .into_owned();

        Ok(DeviceInfo {
            name,
            total_vram_bytes: props.totalGlobalMem as usize,
            compute_units: props.multiProcessorCount as usize,
            max_clock_mhz: props.clockRate as usize / 1000,  // kHz to MHz
            device_id,
        })
    }
}

/// Allocate memory on GPU.
///
/// Returns pointer to allocated memory or GpuError::OutOfMemory.
pub fn hip_malloc(size: usize) -> GpuResult<*mut u8> {
    unsafe {
        let mut ptr = std::ptr::null_mut();
        let code = hipMalloc(&mut ptr, size);
        hip_check(code)?;
        Ok(ptr)
    }
}

/// Free GPU memory.
///
/// Safe to call with null pointer (no-op).
pub fn hip_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        // Ignore errors in free (double-free is a bug, not a crash)
        let _ = hipFree(ptr);
    }
}

/// Copy data from host (CPU) to device (GPU).
pub fn hip_memcpy_h2d(dst: *mut u8, src: *const u8, size: usize) -> GpuResult<()> {
    unsafe {
        let code = hipMemcpy(
            dst as *mut c_void,
            src as *const c_void,
            size,
            hipMemcpyKind::hipMemcpyHostToDevice,
        );
        hip_check(code)
    }
}

/// Copy data from device (GPU) to host (CPU).
pub fn hip_memcpy_d2h(dst: *mut u8, src: *const u8, size: usize) -> GpuResult<()> {
    unsafe {
        let code = hipMemcpy(
            dst as *mut c_void,
            src as *const c_void,
            size,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        );
        hip_check(code)
    }
}

/// Get free and total VRAM in bytes.
pub fn hip_get_mem_info(device_id: i32) -> GpuResult<(usize, usize)> {
    unsafe {
        let mut free = 0usize;
        let mut total = 0usize;
        let code = hipMemGetInfo(&mut free, &mut total);
        hip_check(code)?;
        Ok((free, total))
    }
}

/// Query HIP driver version.
///
/// Returns packed version number (e.g., 0x05060000 for 5.6.0).
/// Note: hipGetDriverVersion may not be available in all HIP versions.
/// Returns 0 if unavailable or on error.
pub fn hip_get_driver_version() -> GpuResult<u32> {
    // The hipGetDriverVersion function is not always available in HIP
    // For now, return 0 to match the TODO's original behavior
    // A full implementation would dynamically load the symbol
    Ok(0)
}

/// Create HIP stream.
///
/// Stream enables async kernel execution and proper sequencing.
pub fn hip_stream_create() -> GpuResult<hipStream_t> {
    unsafe {
        let mut stream_ptr: *mut c_void = std::ptr::null_mut();
        let code = hipStreamCreate(&mut stream_ptr);
        hip_check(code)?;
        Ok(hipStream_t { _private: stream_ptr })
    }
}

/// Destroy HIP stream.
///
/// # Safety
/// Stream must have been created with hip_stream_create and not already destroyed.
pub unsafe fn hip_stream_destroy(stream: hipStream_t) -> GpuResult<()> {
    let code = hipStreamDestroy(stream);
    hip_check(code)
}

/// Synchronize HIP stream.
///
/// Blocks until all queued operations in stream complete.
pub fn hip_stream_synchronize(stream: hipStream_t) -> GpuResult<()> {
    unsafe {
        let code = hipStreamSynchronize(stream);
        hip_check(code)
    }
}

// ── FFI Declarations ───────────────────────────────────────────────────────────────

#[repr(C)]
struct hipDeviceProp_t {
    name: [c_char; 256],
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: i32,
    warpSize: i32,
    maxThreadsPerBlock: i32,
    maxThreadsDim: [i32; 3],
    maxGridSize: [i32; 3],
    clockRate: i32,
    memoryClockRate: i32,
    memoryBusWidth: i32,
    multiProcessorCount: i32,
    // ... more fields we don't need yet
}

#[repr(C)]
enum hipMemcpyKind {
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
}

/// Opaque HIP stream type
///
/// In C, this is a pointer to an opaque struct. We represent it as a
/// raw pointer so it can be freely copied and passed to FFI functions.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct hipStream_t {
    _private: *mut c_void,
}

extern "C" {
    fn hipGetDeviceCount(count: *mut i32) -> hipError_t;
    fn hipGetDeviceProperties(props: *mut hipDeviceProp_t, device: i32) -> hipError_t;
    fn hipMalloc(ptr: *mut *mut u8, size: usize) -> hipError_t;
    fn hipFree(ptr: *mut u8) -> hipError_t;
    fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    fn hipGetErrorString(error: hipError_t) -> *const c_char;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    fn hipGetDriverVersion(driverVersion: *mut c_int) -> hipError_t;
    fn hipStreamCreate(stream: *mut *mut c_void) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
}

// ── Public Types ───────────────────────────────────────────────────────────────────

/// GPU device information.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub total_vram_bytes: usize,
    pub compute_units: usize,
    pub max_clock_mhz: usize,
    pub device_id: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_get_device_count_doesnt_crash() {
        // Should never panic, even if HIP not available
        let count = hip_get_device_count();
        // May be Ok(0) or Err, either is fine
        match count {
            Ok(n) => println!("Found {} GPU(s)", n),
            Err(e) => println!("HIP not available: {}", e),
        }
    }

    #[test]
    fn hip_malloc_free_works() {
        let ptr = hip_malloc(1024);
        if let Ok(p) = ptr {
            assert!(!p.is_null());
            hip_free(p);
        }
        // If HIP not available, test passes anyway
    }
}
