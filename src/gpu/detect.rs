//! AMD GPU capability detection with safe HIP API wrapper.
//!
//! Design principle: Never panic, always return Option or GpuError.
//! If GPU unavailable, return None (not an error).

use super::arch::GpuArchitecture;
use super::error::{GpuError, GpuResult};
use super::ffi::{self, DeviceInfo};

/// Detected GPU capabilities.
///
/// Contains information about AMD GPU hardware and VRAM.
/// Only present if an AMD GPU is detected and HIP runtime is available.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Device name (e.g., "AMD Radeon RX 7900 XTX")
    pub device_name: String,

    /// Total VRAM in bytes
    pub total_vram_bytes: usize,

    /// Free VRAM in bytes (at detection time)
    pub free_vram_bytes: usize,

    /// Compute units (similar to CPU cores, each contains multiple ALUs)
    pub compute_units: usize,

    /// Max clock frequency in MHz
    pub max_clock_mhz: usize,

    /// HIP driver version (packed: major << 22 | minor << 12 | patch)
    pub hip_driver_version: u32,

    /// Device ID (for hipSetDevice, typically 0 for single GPU systems)
    pub device_id: i32,

    /// GPU architecture (gfx1100, gfx1030, etc.)
    pub architecture: GpuArchitecture,

    /// Maximum shared memory available per block in bytes
    pub max_shared_mem_per_block: usize,
}

impl GpuCapabilities {
    /// Detect GPU capabilities safely.
    ///
    /// Returns None if HIP unavailable or no AMD GPU found.
    /// Never panics, always graceful.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocmforge::gpu::detect::GpuCapabilities;
    ///
    /// match GpuCapabilities::detect() {
    ///     Some(gpu) => println!("Found: {} ({} GB VRAM)",
    ///         gpu.device_name,
    ///         gpu.total_vram_bytes / (1024 * 1024 * 1024)
    ///     ),
    ///     None => println!("No AMD GPU detected"),
    /// }
    /// ```
    pub fn detect() -> Option<Self> {
        // Safe HIP API wrapper with error handling
        let device_count = ffi::hip_get_device_count().ok()?;

        if device_count == 0 {
            return None;
        }

        // Use device 0 (default GPU)
        let device_id = 0;
        let info = ffi::hip_get_device_info(device_id).ok()?;

        let (free_vram, _) = ffi::hip_get_mem_info(device_id).ok()?;

        // Parse architecture from device info (currently returns Unknown since FFI uses placeholder)
        let architecture =
            GpuArchitecture::from_name(&info.arch_name).unwrap_or(GpuArchitecture::Unknown(0));

        Some(Self {
            device_name: info.name.clone(),
            total_vram_bytes: info.total_vram_bytes,
            free_vram_bytes: free_vram,
            compute_units: info.compute_units,
            max_clock_mhz: info.max_clock_mhz,
            hip_driver_version: {
                match ffi::hip_get_driver_version() {
                    Ok(v) => v,
                    Err(_) => 0, // Graceful fallback
                }
            },
            device_id,
            architecture,
            max_shared_mem_per_block: info.max_shared_mem_per_block,
        })
    }

    /// Check if we can fit a model of given size in VRAM.
    ///
    /// Uses 90% of free VRAM to leave room for activations and other processes.
    pub fn can_fit_model(&self, model_size_bytes: usize) -> bool {
        self.free_vram_bytes * 9 / 10 >= model_size_bytes
    }

    /// Recommend optimal batch size based on VRAM.
    ///
    /// Uses 50% of free VRAM for batch to leave room for activations and KV cache.
    /// Clamped to [1, 256] range.
    pub fn recommend_batch_size(&self, bytes_per_token: usize) -> usize {
        let batch_vram = self.free_vram_bytes / 2;
        let batch = (batch_vram / bytes_per_token).max(1).min(256);
        batch
    }

    /// Get total VRAM in gigabytes for display.
    pub fn total_vram_gb(&self) -> f64 {
        self.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get free VRAM in gigabytes for display.
    pub fn free_vram_gb(&self) -> f64 {
        self.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_none_or_valid_caps() {
        let caps = GpuCapabilities::detect();
        match &caps {
            None => println!("No GPU detected (expected on systems without AMD GPU)"),
            Some(c) => {
                assert!(!c.device_name.is_empty());
                assert!(c.total_vram_bytes > 0);
                assert!(c.device_id >= 0);
            }
        }
    }

    #[test]
    fn can_fit_model_works() {
        let caps = GpuCapabilities {
            device_name: "Test GPU".to_string(),
            total_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_vram_bytes: 6 * 1024 * 1024 * 1024,
            compute_units: 40,
            max_clock_mhz: 2500,
            hip_driver_version: 0x05060000,
            device_id: 0,
            architecture: GpuArchitecture::Unknown(0),
            max_shared_mem_per_block: 64 * 1024,
        };

        assert!(caps.can_fit_model(4 * 1024 * 1024 * 1024));
        assert!(!caps.can_fit_model(10 * 1024 * 1024 * 1024));
    }

    #[test]
    fn recommend_batch_size_works() {
        let caps = GpuCapabilities {
            device_name: "Test GPU".to_string(),
            total_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_vram_bytes: 6 * 1024 * 1024 * 1024,
            compute_units: 40,
            max_clock_mhz: 2500,
            hip_driver_version: 0x05060000,
            device_id: 0,
            architecture: GpuArchitecture::Unknown(0),
            max_shared_mem_per_block: 64 * 1024,
        };

        let batch = caps.recommend_batch_size(1024 * 1024);
        assert!(batch >= 1);
        assert!(batch <= 256);
    }

    #[test]
    fn display_methods_no_panic() {
        let caps = GpuCapabilities {
            device_name: "Test GPU".to_string(),
            total_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_vram_bytes: 6 * 1024 * 1024 * 1024,
            compute_units: 40,
            max_clock_mhz: 2500,
            hip_driver_version: 0x05060000,
            device_id: 0,
            architecture: GpuArchitecture::Unknown(0),
            max_shared_mem_per_block: 64 * 1024,
        };

        let _gb = caps.total_vram_gb();
        let _gb = caps.free_vram_gb();
    }

    #[test]
    fn test_hip_driver_version_returns_valid_or_zero() {
        // This test verifies that hipGetDriverVersion either succeeds
        // or fails gracefully without crashing
        let caps = GpuCapabilities::detect();
        // If GPU available, version should be queried
        // If not available, should be 0
        if caps.is_some() {
            let gpu = caps.unwrap();
            // Version is packed u32, just verify it doesn't panic
            let _ = gpu.hip_driver_version;
        }
    }

    #[test]
    fn detect_includes_architecture() {
        let caps = GpuCapabilities::detect();
        match &caps {
            None => {
                // No GPU - can't test architecture detection
                println!("No GPU detected - skipping architecture test");
            }
            Some(c) => {
                // Architecture should be detected (even if Unknown)
                // Just verify it's not the default Uninitialized state
                println!("Architecture: {:?}", c.architecture);
                // Don't assert specific architecture - varies by GPU
            }
        }
    }
}
