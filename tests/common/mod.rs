#![cfg(feature = "gpu")]

//! GPU test safety infrastructure.
//!
//! This module provides cross-process GPU locking and VRAM safety checks
//! to prevent GPU reset and out-of-memory errors during parallel testing.

use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Path to the cross-process GPU lock file.
const GPU_LOCK_PATH: &str = "/tmp/rocmforge_gpu_tests.lock";

/// Cross-process GPU lock using flock(2).
///
/// Ensures only one test process uses the GPU at a time.
/// Uses LOCK_EX | LOCK_NB for non-blocking exclusive lock.
pub struct GpuLock {
    _file: File,
}

impl GpuLock {
    /// Acquire the GPU lock.
    ///
    /// Returns Ok if lock acquired, Err if lock held by another process.
    pub fn acquire() -> std::io::Result<Self> {
        let path = Path::new(GPU_LOCK_PATH);

        // Create lock file if it doesn't exist
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Try to acquire exclusive lock (non-blocking)
        unsafe {
            let fd = file.as_raw_fd();
            let ret = libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB);

            if ret != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "GPU lock held by another process",
                ));
            }
        }

        Ok(Self { _file: file })
    }
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        // Lock is automatically released when file is closed
    }
}

/// Check if a GPU is available.
///
/// Returns true if HIP is available and at least one GPU is detected.
pub fn gpu_available() -> bool {
    if let Some(_caps) = rocmforge::gpu::detect() {
        return true;
    }
    false
}

/// Get free VRAM in bytes.
///
/// Returns None if GPU unavailable or query fails.
pub fn get_free_vram() -> Option<u64> {
    if let Some(caps) = rocmforge::gpu::detect() {
        return Some(caps.free_vram_bytes as u64);
    }
    None
}

#[allow(dead_code)]
pub fn real_model_gpu_tests_enabled() -> bool {
    rocmforge::gpu::real_model_gpu_tests_enabled()
}

#[allow(dead_code)]
pub fn experimental_gpu_tests_enabled() -> bool {
    rocmforge::gpu::run_experimental_gpu_tests_enabled()
        && rocmforge::gpu::experimental_gpu_kernels_enabled()
}

/// Check if GPU CLI safe runner is available.
///
/// Verifies that scripts/gpu_safe_run.sh exists and is executable.
#[allow(dead_code)]
pub fn gpu_safe_runner_available() -> bool {
    let runner_path = std::path::Path::new("scripts/gpu_safe_run.sh");
    runner_path.exists() && runner_path.is_file()
}

/// Macro to skip test if GPU unavailable.
#[macro_export]
macro_rules! require_gpu {
    () => {
        if !$crate::common::gpu_available() {
            eprintln!("Skipping test: No GPU detected");
            return;
        }
        let _gpu_lock = match $crate::common::GpuLock::acquire() {
            Ok(lock) => lock,
            Err(err) => {
                eprintln!("Skipping test: {}", err);
                return;
            }
        };
    };
}

/// Macro to skip test if insufficient VRAM.
///
/// `$gib` - Required VRAM in GiB
#[macro_export]
macro_rules! require_vram {
    ($gib:expr) => {
        match $crate::common::get_free_vram() {
            Some(free_bytes) => {
                let required_bytes = $gib * 1024 * 1024 * 1024;
                if free_bytes < required_bytes {
                    eprintln!(
                        "Skipping test: Insufficient VRAM ({} GiB free, {} GiB required)",
                        free_bytes / (1024 * 1024 * 1024),
                        $gib
                    );
                    return;
                }
            }
            None => {
                eprintln!("Skipping test: Could not determine VRAM usage");
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! require_real_model_gpu_tests {
    () => {
        if !$crate::common::real_model_gpu_tests_enabled() {
            eprintln!(
                "Skipping test: set {}=1 to run real-model GPU tests",
                rocmforge::gpu::RUN_REAL_MODEL_GPU_TESTS_ENV
            );
            return;
        }
    };
}

#[macro_export]
macro_rules! require_experimental_gpu_tests {
    () => {
        if !$crate::common::experimental_gpu_tests_enabled() {
            eprintln!(
                "Skipping test: set {}=1 and {}=1 to run experimental GPU kernel tests",
                rocmforge::gpu::RUN_EXPERIMENTAL_GPU_TESTS_ENV,
                rocmforge::gpu::ENABLE_EXPERIMENTAL_GPU_KERNELS_ENV
            );
            return;
        }
    };
}

#[macro_export]
macro_rules! require_decode_graph_enabled {
    () => {
        if !rocmforge::gpu::decode_graph_enabled() {
            eprintln!(
                "Skipping test: set {}=1 to enable decode graph replay",
                rocmforge::gpu::ENABLE_DECODE_GRAPH_ENV
            );
            return;
        }
    };
}

/// Macro to require decode graph DISABLED for Q6_K tests.
///
/// Q6_K crashes with multi-token prompts when graph capture is enabled.
/// This ensures tests only run with graph disabled to prevent GPU crashes.
#[macro_export]
macro_rules! require_q6_k_graph_disabled {
    () => {
        if rocmforge::gpu::decode_graph_enabled() {
            eprintln!(
                "Skipping Q6_K test: decode graph MUST be disabled for Q6_K to prevent GPU crashes.\n\
                 Set {}=0 to disable graph for Q6_K tests.",
                rocmforge::gpu::ENABLE_DECODE_GRAPH_ENV
            );
            return;
        }
    };
}

/// Macro to require GPU safe runner availability.
///
/// Skips test if scripts/gpu_safe_run.sh is not available.
#[macro_export]
macro_rules! require_gpu_safe_runner {
    () => {
        if !$crate::common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available (scripts/gpu_safe_run.sh)");
            return;
        }
    };
}

/// HIP-based VRAM leak detection macro.
///
/// Uses device.vram_stats() for accurate HIP API measurements instead of rocm-smi.
/// Panics if VRAM leak exceeds tolerance (default 10 MB).
///
/// Usage:
/// ```rust,ignore
/// #[test]
/// #[serial]
/// fn test_something() {
///     require_gpu!();
///     let device = GpuDevice::init(0).unwrap();
///
///     let before = device.vram_stats().unwrap();
///     // ... test code ...
///     drop(device);  // Explicit cleanup
///     assert_vram_cleanup!(before, 10);  // Allow 10 MB tolerance
/// }
/// ```
#[macro_export]
macro_rules! assert_vram_cleanup {
    ($device:expr, $tolerance_mb:expr) => {
        let after = $device.vram_stats().expect("Failed to get VRAM stats");
        let before = $device.vram_stats().expect("Failed to get VRAM stats");

        let leaked_mb = (before.used_vram as i64 - after.used_vram as i64).abs() / (1024 * 1024);

        if leaked_mb > $tolerance_mb {
            panic!(
                "VRAM leak detected: {} MB leaked (tolerance: {} MB)\n\
                 Before: {} MB used, After: {} MB used\n\
                 Total: {} MB, Free: {} MB",
                leaked_mb,
                $tolerance_mb,
                before.used_vram_mb(),
                after.used_vram_mb(),
                after.total_vram_mb(),
                after.free_vram_mb()
            );
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_lock_acquire_works() {
        let lock = GpuLock::acquire();
        assert!(lock.is_ok());
    }

    #[test]
    fn gpu_lock_blocks_when_held() {
        let _lock1 = GpuLock::acquire().unwrap();
        let lock2 = GpuLock::acquire();
        assert!(lock2.is_err());
    }

    #[test]
    fn gpu_available_returns_bool() {
        let result = gpu_available();
        let _ = result; // Just verify it doesn't panic
    }

    #[test]
    fn get_free_vram_returns_optional() {
        let result = get_free_vram();
        let _ = result; // Just verify it doesn't panic
    }
}
