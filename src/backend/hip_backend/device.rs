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
    // After: name[256] (256) + uuid (16) + luid[8] (8) + luidDeviceNodeMask (4) + padding (4) = 288
    const TOTAL_GLOBAL_MEM_OFFSET: usize = 288;

    // Offset of `multiProcessorCount` field: int multiProcessorCount
    // Verified with bindgen (offset 388 for ROCm 7.x)
    const MULTI_PROCESSOR_COUNT_OFFSET: usize = 388;

    // Launch limit field offsets - verified against hip_runtime_api.h
    // IMPORTANT: C compiler inserts 4-byte padding after luidDeviceNodeMask to align
    // totalGlobalMem (size_t=8 bytes) to 8-byte boundary (offset 288).
    // Based on: name[256]: 0-255 (256 bytes)
    //           uuid: 256-271 (16 bytes)
    //           luid[8]: 272-279 (8 bytes)
    //           luidDeviceNodeMask: 280-283 (4 bytes)
    //           [PADDING]: 284-287 (4 bytes) - for alignment!
    //           totalGlobalMem: 288-295 (8 bytes)
    //           sharedMemPerBlock: 296-303 (8 bytes)
    //           regsPerBlock: 304-307 (4 bytes)
    //           warpSize: 308-311 (4 bytes)
    //           memPitch: 312-319 (8 bytes)
    //           maxThreadsPerBlock: 320-323 (4 bytes)
    //           maxThreadsDim[3]: 324-335 (12 bytes)
    //           maxGridSize[3]: 336-347 (12 bytes)
    const SHARED_MEM_PER_BLOCK_OFFSET: usize = 296;
    const WARP_SIZE_OFFSET: usize = 308;
    const MAX_THREADS_PER_BLOCK_OFFSET: usize = 320;
    const MAX_THREADS_DIM_OFFSET: usize = 324;
    const MAX_GRID_SIZE_OFFSET: usize = 336;

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

    /// Get maximum threads per block (typically 1024 for AMD GPUs)
    ///
    /// Reads the maxThreadsPerBlock field from hipDeviceProp_t.
    /// This is the total number of threads allowed in a block (product of x*y*z).
    pub fn max_threads_per_block(&self) -> i32 {
        let bytes = &self._buffer
            [Self::MAX_THREADS_PER_BLOCK_OFFSET..Self::MAX_THREADS_PER_BLOCK_OFFSET + 4];
        bytes
            .try_into()
            .ok()
            .map(i32::from_ne_bytes)
            .unwrap_or(1024)
    }

    /// Get maximum threads per dimension [x, y, z]
    ///
    /// Reads the maxThreadsDim field from hipDeviceProp_t.
    /// Each dimension has its own limit (typically 1024 per axis for AMD GPUs).
    pub fn max_threads_dim(&self) -> [i32; 3] {
        let mut result = [0i32; 3];
        for i in 0..3 {
            let offset = Self::MAX_THREADS_DIM_OFFSET + i * 4;
            let bytes = &self._buffer[offset..offset + 4];
            result[i] = bytes
                .try_into()
                .ok()
                .map(i32::from_ne_bytes)
                .unwrap_or(1024);
        }
        result
    }

    /// Get maximum grid dimensions [x, y, z]
    ///
    /// Reads the maxGridSize field from hipDeviceProp_t.
    /// For AMD GPUs: typically [2^32-1, 65535, 65535].
    pub fn max_grid_size(&self) -> [i32; 3] {
        let mut result = [0i32; 3];
        for i in 0..3 {
            let offset = Self::MAX_GRID_SIZE_OFFSET + i * 4;
            let bytes = &self._buffer[offset..offset + 4];
            result[i] = bytes
                .try_into()
                .ok()
                .map(i32::from_ne_bytes)
                .unwrap_or(65535);
        }
        result
    }

    /// Get shared memory per block in bytes
    ///
    /// Reads the sharedMemPerBlock field from hipDeviceProp_t.
    /// For AMD GPUs: typically 65536 bytes.
    pub fn shared_mem_per_block(&self) -> usize {
        let bytes = &self._buffer
            [Self::SHARED_MEM_PER_BLOCK_OFFSET..Self::SHARED_MEM_PER_BLOCK_OFFSET + 8];
        bytes
            .try_into()
            .ok()
            .map(u64::from_ne_bytes)
            .unwrap_or(65536) as usize
    }

    /// Get warp size (wavefront size)
    ///
    /// Reads the warpSize field from hipDeviceProp_t.
    /// For AMD GPUs: 32 for RDNA3, 64 for CDNA3.
    pub fn warp_size(&self) -> i32 {
        let bytes = &self._buffer[Self::WARP_SIZE_OFFSET..Self::WARP_SIZE_OFFSET + 4];
        bytes
            .try_into()
            .ok()
            .map(i32::from_ne_bytes)
            .unwrap_or(32)
    }

    /// Debug: Get raw bytes (for FFI debugging)
    #[allow(dead_code)]
    pub fn debug_bytes(&self, offset: usize, len: usize) -> &[u8] {
        &self._buffer[offset..offset + len]
    }
}

impl Default for HipDeviceProp {
    fn default() -> Self {
        HipDeviceProp {
            _buffer: [0u8; 1472],
        }
    }
}

#[cfg(test)]
mod offset_verification {
    use super::*;
    use memoffset::offset_of;

    // Include bindgen-generated bindings
    // This brings in hipDeviceProp_tR0600 struct definition
    include!(concat!(env!("OUT_DIR"), "/hip_device_bindings.rs"));

    #[test]
    fn verify_device_prop_offsets() {
        // Verify all manual offsets match bindgen's generated offsets
        // If ROCm version changes struct layout, this test will fail at compile time

        // Basic offset (name is always at 0)
        assert_eq!(
            HipDeviceProp::NAME_OFFSET,
            0,
            "NAME_OFFSET should always be 0"
        );

        // Verify totalGlobalMem offset (288 for ROCm 7.x)
        assert_eq!(
            HipDeviceProp::TOTAL_GLOBAL_MEM_OFFSET,
            offset_of!(hipDeviceProp_tR0600, totalGlobalMem),
            "TOTAL_GLOBAL_MEM_OFFSET mismatch - ROCm version may have changed struct layout"
        );

        // Verify sharedMemPerBlock offset (296 for ROCm 7.x)
        assert_eq!(
            HipDeviceProp::SHARED_MEM_PER_BLOCK_OFFSET,
            offset_of!(hipDeviceProp_tR0600, sharedMemPerBlock),
            "SHARED_MEM_PER_BLOCK_OFFSET mismatch - ROCm version may have changed struct layout"
        );

        // Verify warpSize offset (308 for ROCm 7.x)
        assert_eq!(
            HipDeviceProp::WARP_SIZE_OFFSET,
            offset_of!(hipDeviceProp_tR0600, warpSize),
            "WARP_SIZE_OFFSET mismatch - ROCm version may have changed struct layout"
        );

        // Verify maxThreadsPerBlock offset (320 for ROCm 7.x)
        assert_eq!(
            HipDeviceProp::MAX_THREADS_PER_BLOCK_OFFSET,
            offset_of!(hipDeviceProp_tR0600, maxThreadsPerBlock),
            "MAX_THREADS_PER_BLOCK_OFFSET mismatch - ROCm version may have changed struct layout"
        );

        // Verify maxThreadsDim offset (324 for ROCm 7.x)
        assert_eq!(
            HipDeviceProp::MAX_THREADS_DIM_OFFSET,
            offset_of!(hipDeviceProp_tR0600, maxThreadsDim),
            "MAX_THREADS_DIM_OFFSET mismatch - ROCm version may have changed struct layout"
        );

        // Verify maxGridSize offset (336 for ROCm 7.x)
        assert_eq!(
            HipDeviceProp::MAX_GRID_SIZE_OFFSET,
            offset_of!(hipDeviceProp_tR0600, maxGridSize),
            "MAX_GRID_SIZE_OFFSET mismatch - ROCm version may have changed struct layout"
        );

        // Verify multiProcessorCount offset
        // Note: bindgen reports this as 388, not 508 as previously coded
        assert_eq!(
            HipDeviceProp::MULTI_PROCESSOR_COUNT_OFFSET,
            offset_of!(hipDeviceProp_tR0600, multiProcessorCount),
            "MULTI_PROCESSOR_COUNT_OFFSET mismatch - ROCm version may have changed struct layout"
        );
    }

    #[test]
    fn verify_struct_size() {
        // Verify our buffer size matches C's sizeof(hipDeviceProp_t)
        // For ROCm 7.x, this should be 1472 bytes
        assert_eq!(
            std::mem::size_of::<HipDeviceProp>(),
            std::mem::size_of::<hipDeviceProp_tR0600>(),
            "HipDeviceProp buffer size mismatch with C struct - ROCm version may have changed"
        );
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
