//! GPU device handle with RAII safety.
//!
//! Ensures proper cleanup on drop.
//! Safe to use even if GPU unavailable (returns error).

use super::error::GpuResult;
use super::ffi;

/// GPU device handle.
///
/// Represents an initialized GPU device with proper resource management.
/// Uses RAII pattern to ensure cleanup on drop.
pub struct GpuDevice {
    device_id: i32,
    stream: ffi::hipStream_t,
    warp_size: usize,
    max_shared_mem_per_block: usize,
}

impl GpuDevice {
    /// Initialize GPU device with safety checks.
    ///
    /// Returns error if device ID invalid or HIP not available.
    pub fn init(device_id: i32) -> GpuResult<Self> {
        // Verify device exists
        let info = ffi::hip_get_device_info(device_id)?;

        // Create HIP stream
        let stream = ffi::hip_stream_create()?;

        Ok(Self {
            device_id,
            stream,
            warp_size: info.warp_size,
            max_shared_mem_per_block: info.max_shared_mem_per_block,
        })
    }

    /// Get device ID.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get device properties.
    pub fn get_properties(&self) -> GpuResult<super::detect::GpuCapabilities> {
        let info = ffi::hip_get_device_info(self.device_id)?;
        let (free_vram, _) = ffi::hip_get_mem_info(self.device_id)?;

        Ok(super::detect::GpuCapabilities {
            device_name: info.name,
            total_vram_bytes: info.total_vram_bytes,
            free_vram_bytes: free_vram,
            compute_units: info.compute_units,
            max_clock_mhz: info.max_clock_mhz,
            hip_driver_version: 0,
            device_id: self.device_id,
            architecture: super::arch::GpuArchitecture::from_name(&info.arch_name)
                .unwrap_or(super::arch::GpuArchitecture::Unknown(0)),
            max_shared_mem_per_block: info.max_shared_mem_per_block,
        })
    }

    /// Get the HIP stream for this device.
    ///
    /// Used for async kernel execution.
    pub fn stream(&self) -> ffi::hipStream_t {
        self.stream
    }

    /// Begin capture on the device stream.
    pub fn begin_capture(&self, mode: ffi::hipStreamCaptureMode) -> GpuResult<()> {
        ffi::hip_stream_begin_capture(self.stream, mode)
    }

    /// End capture on the device stream and return the graph template.
    pub fn end_capture(&self) -> GpuResult<ffi::hipGraph_t> {
        ffi::hip_stream_end_capture(self.stream)
    }

    /// Return the current stream capture status.
    pub fn stream_capture_status(&self) -> GpuResult<ffi::hipStreamCaptureStatus> {
        ffi::hip_stream_is_capturing(self.stream)
    }

    /// Report the device wavefront size exposed by HIP.
    pub fn warp_size(&self) -> usize {
        self.warp_size
    }

    /// Get maximum shared memory available per block in bytes.
    pub fn max_shared_mem_per_block(&self) -> usize {
        self.max_shared_mem_per_block
    }

    /// Synchronize all queued operations on this device's stream.
    ///
    /// Blocks until all previously queued operations on the stream complete.
    pub fn synchronize(&self) -> GpuResult<()> {
        ffi::hip_stream_synchronize(self.stream)
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // Ignore errors during drop (can't handle them anyway)
        // SAFETY: stream was created by hip_stream_create and not yet destroyed
        let _ = unsafe { ffi::hip_stream_destroy(self.stream) };
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device_id", &self.device_id)
            .field("stream", &self.stream)
            .field("warp_size", &self.warp_size)
            .field("max_shared_mem_per_block", &self.max_shared_mem_per_block)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_works_if_gpu_available() {
        let device = GpuDevice::init(0);
        match device {
            Ok(_) => println!("GPU device initialized"),
            Err(e) => println!("GPU init failed: {} (expected if no GPU)", e),
        }
    }

    #[test]
    fn invalid_device_returns_error() {
        let device = GpuDevice::init(999);
        assert!(device.is_err());
    }

    #[test]
    fn stream_lifecycle_works() {
        let device = GpuDevice::init(0);
        match device {
            Ok(d) => {
                // Verify stream accessor works
                let _stream = d.stream();
                // Verify sync works (no-op if no operations queued)
                let _ = d.synchronize();
                // Drop will clean up stream
            }
            Err(e) => println!("GPU init failed: {} (expected if no GPU)", e),
        }
    }
}
