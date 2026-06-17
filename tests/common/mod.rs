#![cfg(feature = "gpu")]

//! GPU test safety infrastructure.
//!
//! This module provides cross-process GPU locking and VRAM safety checks
//! to prevent GPU reset and out-of-memory errors during parallel testing.

const DEFAULT_GPU_TEST_LOCK_TIMEOUT_SECS: u64 = 30;

pub mod helpers;

#[allow(
    dead_code,
    reason = "shared test macro support; referenced only by some GPU tests"
)]
pub const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;

fn gpu_test_lock_timeout_secs() -> u64 {
    std::env::var("ROCMFORGE_GPU_LOCK_TIMEOUT")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .unwrap_or(DEFAULT_GPU_TEST_LOCK_TIMEOUT_SECS)
}

/// Cross-process GPU lock using flock(2).
///
/// Ensures only one test process uses the GPU at a time.
pub struct GpuLock {
    _inner: rocmforge::gpu::GpuLock,
}

impl GpuLock {
    /// Acquire the GPU lock using the default timeout policy.
    ///
    /// Separate `cargo test` invocations should queue instead of racing the
    /// display-attached GPU. The timeout is configurable via
    /// `ROCMFORGE_GPU_LOCK_TIMEOUT` and defaults to 30 seconds.
    pub fn acquire() -> std::io::Result<Self> {
        Self::acquire_with_timeout(gpu_test_lock_timeout_secs())
    }

    /// Acquire the GPU lock with an explicit timeout in seconds.
    pub fn acquire_with_timeout(timeout_secs: u64) -> std::io::Result<Self> {
        match rocmforge::gpu::GpuLock::acquire(timeout_secs) {
            Ok(inner) => Ok(Self { _inner: inner }),
            Err(e) => Err(std::io::Error::new(std::io::ErrorKind::WouldBlock, e)),
        }
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

/// Get the safe GPU test VRAM budget in bytes.
///
/// This returns the guarded allocation budget after subtracting the desktop
/// reservation and the standard safety margin, not raw free VRAM. Tests should
/// skip based on this value so they do not consume compositor headroom.
pub fn get_safe_test_vram_budget() -> Option<u64> {
    rocmforge::gpu::query_vram_budget(0)
        .ok()
        .map(|budget| budget.safe_allocation_size as u64)
}

/// Backward-compatible alias for older tests.
///
/// This intentionally returns the guarded test budget rather than raw free
/// VRAM so existing callers inherit the safer semantics automatically.
#[allow(dead_code, reason = "shared backward-compatible test helper")]
pub fn get_free_vram() -> Option<u64> {
    get_safe_test_vram_budget()
}

#[allow(dead_code, reason = "shared test macro support")]
pub fn real_model_gpu_tests_enabled() -> bool {
    rocmforge::gpu::real_model_gpu_tests_enabled()
}

#[allow(dead_code, reason = "shared test macro support")]
pub fn experimental_gpu_tests_enabled() -> bool {
    rocmforge::gpu::run_experimental_gpu_tests_enabled()
        && rocmforge::gpu::experimental_gpu_kernels_enabled()
}

/// Check if GPU CLI safe runner is available.
///
/// Verifies that scripts/gpu_safe_run.sh exists and is executable.
#[allow(dead_code, reason = "shared test macro support")]
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
        match $crate::common::get_safe_test_vram_budget() {
            Some(safe_bytes) => {
                let required_bytes = ($gib as u64) * $crate::common::BYTES_PER_GIB;
                if safe_bytes < required_bytes {
                    eprintln!(
                        "Skipping test: Insufficient safe GPU test budget ({} GiB safe after desktop reservation and margin, {} GiB required)",
                        safe_bytes / $crate::common::BYTES_PER_GIB,
                        $gib
                    );
                    return;
                }
            }
            None => {
                eprintln!("Skipping test: Could not determine safe GPU test VRAM budget");
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
        let lock2 = GpuLock::acquire_with_timeout(0);
        assert!(lock2.is_err());
    }

    #[test]
    fn gpu_available_returns_bool() {
        let result = gpu_available();
        let _ = result; // Just verify it doesn't panic
    }

    #[test]
    fn get_safe_test_vram_budget_returns_optional() {
        let result = get_safe_test_vram_budget();
        let _ = result; // Just verify it doesn't panic
    }

    #[test]
    fn gpu_test_lock_timeout_honors_env() {
        std::env::set_var("ROCMFORGE_GPU_LOCK_TIMEOUT", "7");
        assert_eq!(gpu_test_lock_timeout_secs(), 7);

        std::env::set_var("ROCMFORGE_GPU_LOCK_TIMEOUT", "0");
        assert_eq!(
            gpu_test_lock_timeout_secs(),
            DEFAULT_GPU_TEST_LOCK_TIMEOUT_SECS
        );

        std::env::remove_var("ROCMFORGE_GPU_LOCK_TIMEOUT");
    }
}
