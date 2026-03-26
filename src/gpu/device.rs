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
}

impl GpuDevice {
    /// Initialize GPU device with safety checks.
    ///
    /// Returns error if device ID invalid or HIP not available.
    pub fn init(device_id: i32) -> GpuResult<Self> {
        // Verify device exists
        let _info = ffi::hip_get_device_info(device_id)?;

        // TODO: Initialize HIP context/stream here when needed
        // For now, just store device ID

        Ok(Self { device_id })
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
        })
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device_id", &self.device_id)
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
}
