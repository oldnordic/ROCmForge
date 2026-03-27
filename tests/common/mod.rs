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

/// Macro to skip test if GPU unavailable.
#[macro_export]
macro_rules! require_gpu {
    () => {
        if !crate::common::gpu_available() {
            eprintln!("Skipping test: No GPU detected");
            return;
        }
    };
}

/// Macro to skip test if insufficient VRAM.
///
/// `$gib` - Required VRAM in GiB
#[macro_export]
macro_rules! require_vram {
    ($gib:expr) => {
        match crate::common::get_free_vram() {
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
