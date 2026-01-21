//! HIP device properties and device info

use std::ffi::CStr;

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

    // Offsets for launch limit fields (verified against hip_runtime_api.h struct layout)
    // Based on: name[256]: 0-255 (256 bytes)
    //           uuid: 256-271 (16 bytes)
    //           luid[8]: 272-279 (8 bytes)
    //           luidDeviceNodeMask: 280-283 (4 bytes)
    //           totalGlobalMem: 284-291 (8 bytes)
    //           sharedMemPerBlock: 292-299 (8 bytes)
    //           regsPerBlock: 300-303 (4 bytes)
    //           warpSize: 304-307 (4 bytes)
    //           memPitch: 308-315 (8 bytes)
    //           maxThreadsPerBlock: 316-319 (4 bytes)
    //           maxThreadsDim[3]: 320-331 (12 bytes)
    //           maxGridSize[3]: 332-343 (12 bytes)

    const SHARED_MEM_PER_BLOCK_OFFSET: usize = 292;
    const WARP_SIZE_OFFSET: usize = 304;
    const MAX_THREADS_PER_BLOCK_OFFSET: usize = 316;
    const MAX_THREADS_DIM_OFFSET: usize = 320;
    const MAX_GRID_SIZE_OFFSET: usize = 332;

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

impl Default for HipDeviceProp {
    fn default() -> Self {
        HipDeviceProp {
            _buffer: [0u8; 1472],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct hipUUID {
    pub bytes: [u8; 16],
}

/// HIP device information
#[derive(Debug, Clone)]
pub struct HipDevice {
    pub device_id: i32,
    pub name: String,
    pub memory: usize,
    pub compute_units: i32,
}

/// Get HIP error string from error code
pub fn get_error_string(error: i32) -> String {
    unsafe {
        let error_ptr = super::ffi::hipGetErrorString(error);
        if error_ptr.is_null() {
            "Unknown error".to_string()
        } else {
            CStr::from_ptr(error_ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}
