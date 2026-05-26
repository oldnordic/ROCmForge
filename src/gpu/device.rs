//! GPU device handle with RAII safety.
//!
//! Ensures proper cleanup on drop.
//! Safe to use even if GPU unavailable (returns error).

use super::error::GpuResult;
use super::ffi;
use super::vram_budget::DESKTOP_VRAM_RESERVATION_BYTES;
use super::weights::GpuBuffer;
use std::sync::Mutex;

const DEFAULT_Q8_WORKSPACE_BYTES: usize = 64 * 1024;

struct Q8Workspace {
    buffer: Option<GpuBuffer>,
    size: usize,
}

/// VRAM usage statistics for a device.
///
/// Provides detailed information about current VRAM usage and safe allocation limits.
#[derive(Debug, Clone)]
pub struct VramStats {
    /// Total VRAM on the device in bytes
    pub total_vram: usize,
    /// Currently free VRAM in bytes
    pub free_vram: usize,
    /// Currently used VRAM in bytes
    pub used_vram: usize,
    /// VRAM reserved for desktop processes in bytes
    pub desktop_reserved: usize,
    /// Safely allocatable VRAM (free minus desktop reservation) in bytes
    pub safely_allocatable: usize,
}

impl VramStats {
    /// Get total VRAM in GB.
    pub fn total_vram_gb(&self) -> f64 {
        self.total_vram as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get free VRAM in GB.
    pub fn free_vram_gb(&self) -> f64 {
        self.free_vram as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get used VRAM in GB.
    pub fn used_vram_gb(&self) -> f64 {
        self.used_vram as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get safely allocatable VRAM in GB.
    pub fn safely_allocatable_gb(&self) -> f64 {
        self.safely_allocatable as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// GPU device handle.
///
/// Represents an initialized GPU device with proper resource management.
/// Uses RAII pattern to ensure cleanup on drop.
pub struct GpuDevice {
    device_id: i32,
    stream: ffi::hipStream_t,
    warp_size: usize,
    max_shared_mem_per_block: usize,
    q8_workspace: Mutex<Q8Workspace>,
}

impl GpuDevice {
    /// Initialize GPU device with safety checks.
    ///
    /// Returns error if device ID invalid or HIP not available.
    pub fn init(device_id: i32) -> GpuResult<Self> {
        ffi::hip_set_device(device_id)?;

        let info = ffi::hip_get_device_info(device_id)?;

        let (free_vram, _total_vram) = ffi::hip_get_mem_info(device_id)?;
        if free_vram < DESKTOP_VRAM_RESERVATION_BYTES {
            return Err(super::error::GpuError::DeviceInsufficientVram {
                free_vram,
                required: DESKTOP_VRAM_RESERVATION_BYTES,
            });
        }

        let stream = ffi::hip_stream_create()?;

        Ok(Self {
            device_id,
            stream,
            warp_size: info.warp_size,
            max_shared_mem_per_block: info.max_shared_mem_per_block,
            q8_workspace: Mutex::new(Q8Workspace {
                buffer: None,
                size: 0,
            }),
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
            device_name: info.name.clone(),
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

    /// Get the device name string.
    ///
    /// Returns the GPU device name as reported by HIP, or None if the query fails.
    pub fn get_name(&self) -> Option<String> {
        ffi::hip_get_device_name(self.device_id).ok()
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

    /// Get a reusable device-local Q8 workspace of at least `min_bytes`.
    ///
    /// The buffer grows on demand and is reused across decode launches to avoid
    /// per-token allocations in hot paths. Growth is synchronized so in-flight
    /// kernels never observe a freed workspace.
    pub fn q8_workspace_ptr(&self, min_bytes: usize) -> GpuResult<*mut u8> {
        if min_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let mut workspace = self
            .q8_workspace
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        if workspace.size < min_bytes {
            if workspace.size != 0 {
                self.synchronize()?;
            }

            let target_bytes = min_bytes.max(DEFAULT_Q8_WORKSPACE_BYTES);
            workspace.buffer = Some(GpuBuffer::alloc_for_device(target_bytes, self.device_id)?);
            workspace.size = target_bytes;
        }

        Ok(workspace
            .buffer
            .as_ref()
            .map(|buf| buf.as_ptr())
            .unwrap_or(std::ptr::null_mut()))
    }

    /// Reserve a reusable Q8 workspace before entering a capture-sensitive path.
    ///
    /// This keeps lazy `hipMalloc` growth outside HIP graph capture so later
    /// decode launches can safely reuse the same buffer.
    pub fn reserve_q8_workspace(&self, min_bytes: usize) -> GpuResult<()> {
        let _ = self.q8_workspace_ptr(min_bytes)?;
        Ok(())
    }

    /// Check if allocation of given size is safe for this device.
    ///
    /// Returns error if allocation would exceed safe VRAM limits.
    /// Accounts for desktop VRAM reservation and safety margin.
    pub fn can_allocate(&self, size: usize) -> GpuResult<()> {
        let (free_vram, _) = ffi::hip_get_mem_info(self.device_id)?;
        let usable_vram = free_vram.saturating_sub(DESKTOP_VRAM_RESERVATION_BYTES);
        let safe_size = (usable_vram as f64 * 0.9) as usize; // 90% of usable VRAM

        if size > safe_size {
            return Err(super::error::GpuError::OutOfMemory {
                requested: size,
                available: safe_size,
                hint: format!(
                    "Allocation would exceed safe VRAM limit on device {} ({} MB requested, {} MB safely allocatable)",
                    self.device_id,
                    size / (1024 * 1024),
                    safe_size / (1024 * 1024)
                ),
            });
        }

        Ok(())
    }

    /// Get current VRAM usage statistics for this device.
    ///
    /// Returns detailed information about VRAM usage including safe allocation limits.
    pub fn vram_stats(&self) -> GpuResult<VramStats> {
        let (free_vram, total_vram) = ffi::hip_get_mem_info(self.device_id)?;
        let used_vram = total_vram - free_vram;

        Ok(VramStats {
            total_vram,
            free_vram,
            used_vram,
            desktop_reserved: DESKTOP_VRAM_RESERVATION_BYTES,
            safely_allocatable: free_vram.saturating_sub(DESKTOP_VRAM_RESERVATION_BYTES),
        })
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
            .field(
                "q8_workspace_bytes",
                &self
                    .q8_workspace
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .size,
            )
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
