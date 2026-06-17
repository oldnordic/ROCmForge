//! Safety policy for optional GPU fast paths.
//!
//! Experimental Vulkan-style kernels remain opt-in.
//! HIP decode graph replay is enabled by default to reduce per-token launch
//! overhead, but callers can force the conservative path with
//! `ROCMFORGE_DISABLE_DECODE_GRAPH=1` or `ROCMFORGE_GPU_SAFE_MODE=1`.
//! A bad launch on a display-attached GPU can wedge the desktop hard enough
//! to trigger a driver reset, so runtime guards still auto-disable unstable
//! paths for the current process.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

pub const ENABLE_DECODE_GRAPH_ENV: &str = "ROCMFORGE_ENABLE_DECODE_GRAPH";
pub const DISABLE_DECODE_GRAPH_ENV: &str = "ROCMFORGE_DISABLE_DECODE_GRAPH";
pub const ENABLE_EXPERIMENTAL_GPU_KERNELS_ENV: &str = "ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS";
pub const ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV: &str = "ROCMFORGE_ENABLE_EXPERIMENTAL_FFN_FASTPATH";
pub const ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV: &str =
    "ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH";
pub const ENABLE_LAUNCH_AUTOTUNE_ENV: &str = "ROCMFORGE_ENABLE_LAUNCH_AUTOTUNE";
pub const USE_DP4A_ENV: &str = "ROCMFORGE_USE_DP4A";
pub const FORCE_WAVE32_ENV: &str = "ROCMFORGE_FORCE_WAVE32";
pub const DISABLE_WAVE32_ENV: &str = "ROCMFORGE_DISABLE_WAVE32";
pub const GPU_SAFE_MODE_ENV: &str = "ROCMFORGE_GPU_SAFE_MODE";
pub const RUN_REAL_MODEL_GPU_TESTS_ENV: &str = "ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS";
pub const RUN_EXPERIMENTAL_GPU_TESTS_ENV: &str = "ROCMFORGE_RUN_EXPERIMENTAL_GPU_TESTS";
pub const RUN_GPU_BENCHES_ENV: &str = "ROCMFORGE_RUN_GPU_BENCHES";

const ENV_UNKNOWN: u8 = 0;
const ENV_DISABLED: u8 = 1;
const ENV_ENABLED: u8 = 2;

struct CachedEnvFlag {
    name: &'static str,
    default: bool,
    cached: AtomicU8,
}

impl CachedEnvFlag {
    const fn new(name: &'static str, default: bool) -> Self {
        Self {
            name,
            default,
            cached: AtomicU8::new(ENV_UNKNOWN),
        }
    }

    fn enabled(&self) -> bool {
        match self.cached.load(Ordering::Relaxed) {
            ENV_DISABLED => false,
            ENV_ENABLED => true,
            _ => {
                let enabled = parse_env_flag(std::env::var(self.name).ok(), self.default);
                self.cached.store(
                    if enabled { ENV_ENABLED } else { ENV_DISABLED },
                    Ordering::Relaxed,
                );
                enabled
            }
        }
    }

    fn reset(&self) {
        self.cached.store(ENV_UNKNOWN, Ordering::Relaxed);
    }
}

static ENABLE_DECODE_GRAPH_FLAG: CachedEnvFlag = CachedEnvFlag::new(ENABLE_DECODE_GRAPH_ENV, true);
static DISABLE_DECODE_GRAPH_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(DISABLE_DECODE_GRAPH_ENV, false);
static ENABLE_EXPERIMENTAL_GPU_KERNELS_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(ENABLE_EXPERIMENTAL_GPU_KERNELS_ENV, false);
static ENABLE_EXPERIMENTAL_FFN_FASTPATH_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV, true);
static ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, true);
static ENABLE_LAUNCH_AUTOTUNE_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(ENABLE_LAUNCH_AUTOTUNE_ENV, true);
static USE_DP4A_FLAG: CachedEnvFlag = CachedEnvFlag::new(USE_DP4A_ENV, true);
static FORCE_WAVE32_FLAG: CachedEnvFlag = CachedEnvFlag::new(FORCE_WAVE32_ENV, false);
static DISABLE_WAVE32_FLAG: CachedEnvFlag = CachedEnvFlag::new(DISABLE_WAVE32_ENV, false);
static GPU_SAFE_MODE_FLAG: CachedEnvFlag = CachedEnvFlag::new(GPU_SAFE_MODE_ENV, false);
static RUN_REAL_MODEL_GPU_TESTS_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(RUN_REAL_MODEL_GPU_TESTS_ENV, false);
static RUN_EXPERIMENTAL_GPU_TESTS_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(RUN_EXPERIMENTAL_GPU_TESTS_ENV, false);
static RUN_GPU_BENCHES_FLAG: CachedEnvFlag = CachedEnvFlag::new(RUN_GPU_BENCHES_ENV, false);
static DECODE_GRAPH_RUNTIME_DISABLED: AtomicBool = AtomicBool::new(false);
static Q8_ACTIVATION_FASTPATH_RUNTIME_DISABLED: AtomicBool = AtomicBool::new(false);
static DECODE_GRAPH_RUNTIME_DISABLE_LOGGED: AtomicBool = AtomicBool::new(false);
static Q8_FASTPATH_RUNTIME_DISABLE_LOGGED: AtomicBool = AtomicBool::new(false);

fn parse_env_flag(value: Option<String>, default: bool) -> bool {
    match value.map(|value| value.trim().to_ascii_lowercase()) {
        Some(value) => matches!(value.as_str(), "1" | "true" | "yes" | "on"),
        None => default,
    }
}

/// Refresh cached runtime env flags.
///
/// Decode dispatch reads these flags frequently enough that live `std::env`
/// lookups show up in profiles. The cache is process-local and callers that
/// intentionally mutate GPU feature flags at runtime, such as integration
/// tests, should call this after changing the environment.
pub fn refresh_runtime_env_flags() {
    ENABLE_DECODE_GRAPH_FLAG.reset();
    DISABLE_DECODE_GRAPH_FLAG.reset();
    ENABLE_EXPERIMENTAL_GPU_KERNELS_FLAG.reset();
    ENABLE_EXPERIMENTAL_FFN_FASTPATH_FLAG.reset();
    ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_FLAG.reset();
    ENABLE_LAUNCH_AUTOTUNE_FLAG.reset();
    USE_DP4A_FLAG.reset();
    FORCE_WAVE32_FLAG.reset();
    DISABLE_WAVE32_FLAG.reset();
    GPU_SAFE_MODE_FLAG.reset();
    RUN_REAL_MODEL_GPU_TESTS_FLAG.reset();
    RUN_EXPERIMENTAL_GPU_TESTS_FLAG.reset();
    RUN_GPU_BENCHES_FLAG.reset();
    DECODE_GRAPH_RUNTIME_DISABLED.store(false, Ordering::Relaxed);
    Q8_ACTIVATION_FASTPATH_RUNTIME_DISABLED.store(false, Ordering::Relaxed);
    DECODE_GRAPH_RUNTIME_DISABLE_LOGGED.store(false, Ordering::Relaxed);
    Q8_FASTPATH_RUNTIME_DISABLE_LOGGED.store(false, Ordering::Relaxed);
    super::decode_profile::refresh_decode_profile_env_flag();
}

pub fn decode_graph_enabled() -> bool {
    !gpu_safe_mode_enabled()
        && ENABLE_DECODE_GRAPH_FLAG.enabled()
        && !decode_graph_runtime_disabled()
}

pub fn decode_graph_disabled_override_requested() -> bool {
    DISABLE_DECODE_GRAPH_FLAG.enabled()
}

pub fn experimental_gpu_kernels_enabled() -> bool {
    !gpu_safe_mode_enabled() && ENABLE_EXPERIMENTAL_GPU_KERNELS_FLAG.enabled()
}

/// Enables the decode FFN fast path without turning on the broader
/// Vulkan-style prototype bundle gated by `ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS`.
///
/// This path defaults off (opt-in) due to measured perf regressions on some
/// workloads. Set `ROCMFORGE_ENABLE_EXPERIMENTAL_FFN_FASTPATH=1` to enable it.
pub fn experimental_ffn_fastpath_enabled() -> bool {
    !gpu_safe_mode_enabled() && ENABLE_EXPERIMENTAL_FFN_FASTPATH_FLAG.enabled()
}

pub fn experimental_q8_activation_fastpath_enabled() -> bool {
    !gpu_safe_mode_enabled()
        && ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_FLAG.enabled()
        && !q8_activation_fastpath_runtime_disabled()
}

/// Enables launch-policy autotune for decode hot paths.
///
/// This is opt-in (default off) to maintain backward compatibility.
/// Set `ROCMFORGE_ENABLE_LAUNCH_AUTOTUNE=1` to enable shape-class
/// keyed autotuning for QKV, gate_up, LM-head, and residual launches.
pub fn launch_autotune_enabled() -> bool {
    !gpu_safe_mode_enabled() && ENABLE_LAUNCH_AUTOTUNE_FLAG.enabled()
}

/// Force DP4A-optimized kernels even when feature detection would skip them.
///
/// Default: enabled (follow feature detection). Set `ROCMFORGE_USE_DP4A=0` to
/// force scalar fallback on hardware that supports DP4A.
pub fn use_dp4a_enabled() -> bool {
    !gpu_safe_mode_enabled() && USE_DP4A_FLAG.enabled()
}

/// Force wave32 kernels on all architectures (may crash on wave64-only hardware).
///
/// Default: off. Set `ROCMFORGE_FORCE_WAVE32=1` to opt in.
pub fn force_wave32_enabled() -> bool {
    !gpu_safe_mode_enabled() && FORCE_WAVE32_FLAG.enabled()
}

/// Disable wave32 kernels and force wave64 even on wave32-capable hardware.
///
/// Default: off. Set `ROCMFORGE_DISABLE_WAVE32=1` to opt in.
pub fn disable_wave32_enabled() -> bool {
    DISABLE_WAVE32_FLAG.enabled()
}

pub fn gpu_safe_mode_enabled() -> bool {
    GPU_SAFE_MODE_FLAG.enabled()
}

pub fn real_model_gpu_tests_enabled() -> bool {
    RUN_REAL_MODEL_GPU_TESTS_FLAG.enabled()
}

pub fn run_experimental_gpu_tests_enabled() -> bool {
    RUN_EXPERIMENTAL_GPU_TESTS_FLAG.enabled()
}

pub fn run_gpu_benches_enabled() -> bool {
    RUN_GPU_BENCHES_FLAG.enabled()
}

pub fn decode_graph_runtime_disabled() -> bool {
    DECODE_GRAPH_RUNTIME_DISABLED.load(Ordering::Relaxed)
}

pub fn q8_activation_fastpath_runtime_disabled() -> bool {
    Q8_ACTIVATION_FASTPATH_RUNTIME_DISABLED.load(Ordering::Relaxed)
}

pub fn disable_decode_graph_runtime(reason: &str) {
    DECODE_GRAPH_RUNTIME_DISABLED.store(true, Ordering::Relaxed);
    if !DECODE_GRAPH_RUNTIME_DISABLE_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[rocmforge][gpu safety] decode graph auto-disabled for this process: {}",
            reason
        );
    }
}

pub fn disable_q8_activation_fastpath_runtime(reason: &str) {
    Q8_ACTIVATION_FASTPATH_RUNTIME_DISABLED.store(true, Ordering::Relaxed);
    if !Q8_FASTPATH_RUNTIME_DISABLE_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[rocmforge][gpu safety] q8 activation fastpath auto-disabled for this process: {}",
            reason
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        disable_decode_graph_runtime, disable_q8_activation_fastpath_runtime, parse_env_flag,
        refresh_runtime_env_flags, ENABLE_DECODE_GRAPH_ENV, ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV,
        ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, FORCE_WAVE32_ENV, GPU_SAFE_MODE_ENV,
        USE_DP4A_ENV,
    };

    #[test]
    fn parse_env_flag_uses_default_when_missing() {
        assert!(!parse_env_flag(None, false));
        assert!(parse_env_flag(None, true));
    }

    #[test]
    fn parse_env_flag_recognizes_truthy_values() {
        assert!(parse_env_flag(Some("1".to_string()), false));
        assert!(parse_env_flag(Some("true".to_string()), false));
        assert!(parse_env_flag(Some("On".to_string()), false));
    }

    #[test]
    fn parse_env_flag_treats_non_truthy_values_as_false() {
        assert!(!parse_env_flag(Some("0".to_string()), true));
        assert!(!parse_env_flag(Some("false".to_string()), true));
        assert!(!parse_env_flag(Some("no".to_string()), true));
    }

    use std::sync::Mutex;
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn refresh_runtime_env_flags_reloads_cached_defaults() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        unsafe {
            std::env::set_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV, "0");
        }
        refresh_runtime_env_flags();
        assert!(!super::experimental_ffn_fastpath_enabled());

        unsafe {
            std::env::remove_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV);
        }
        refresh_runtime_env_flags();
        assert!(super::experimental_ffn_fastpath_enabled());

        unsafe {
            std::env::set_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, "0");
        }
        refresh_runtime_env_flags();
        assert!(!super::experimental_q8_activation_fastpath_enabled());

        unsafe {
            std::env::remove_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV);
        }
        refresh_runtime_env_flags();
        assert!(super::experimental_q8_activation_fastpath_enabled());
    }

    #[test]
    fn runtime_disable_decode_graph_is_process_local_until_refresh() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        unsafe {
            std::env::set_var(ENABLE_DECODE_GRAPH_ENV, "1");
        }
        refresh_runtime_env_flags();
        assert!(super::decode_graph_enabled());

        disable_decode_graph_runtime("unit test");
        assert!(!super::decode_graph_enabled());

        refresh_runtime_env_flags();
        assert!(super::decode_graph_enabled());

        unsafe {
            std::env::remove_var(ENABLE_DECODE_GRAPH_ENV);
        }
        refresh_runtime_env_flags();
        assert!(super::decode_graph_enabled());
    }

    #[test]
    fn runtime_disable_q8_fastpath_is_process_local_until_refresh() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        unsafe {
            std::env::set_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, "1");
        }
        refresh_runtime_env_flags();
        assert!(super::experimental_q8_activation_fastpath_enabled());

        disable_q8_activation_fastpath_runtime("unit test");
        assert!(!super::experimental_q8_activation_fastpath_enabled());

        refresh_runtime_env_flags();
        assert!(super::experimental_q8_activation_fastpath_enabled());

        unsafe {
            std::env::remove_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV);
        }
        refresh_runtime_env_flags();
        assert!(super::experimental_q8_activation_fastpath_enabled());
    }

    #[test]
    fn gpu_safe_mode_forces_conservative_feature_set() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        unsafe {
            std::env::set_var(ENABLE_DECODE_GRAPH_ENV, "1");
            std::env::set_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV, "1");
            std::env::set_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, "1");
            std::env::set_var(USE_DP4A_ENV, "1");
            std::env::set_var(FORCE_WAVE32_ENV, "1");
            std::env::set_var(GPU_SAFE_MODE_ENV, "1");
        }
        refresh_runtime_env_flags();

        assert!(super::gpu_safe_mode_enabled());
        assert!(!super::decode_graph_enabled());
        assert!(!super::experimental_ffn_fastpath_enabled());
        assert!(!super::experimental_q8_activation_fastpath_enabled());
        assert!(!super::use_dp4a_enabled());
        assert!(!super::force_wave32_enabled());

        unsafe {
            std::env::remove_var(ENABLE_DECODE_GRAPH_ENV);
            std::env::remove_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV);
            std::env::remove_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV);
            std::env::remove_var(USE_DP4A_ENV);
            std::env::remove_var(FORCE_WAVE32_ENV);
            std::env::remove_var(GPU_SAFE_MODE_ENV);
        }
        refresh_runtime_env_flags();
    }

    #[test]
    fn use_dp4a_env_flag_respected() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        unsafe {
            std::env::set_var(USE_DP4A_ENV, "0");
        }
        refresh_runtime_env_flags();
        assert!(!super::use_dp4a_enabled());

        unsafe {
            std::env::set_var(USE_DP4A_ENV, "1");
        }
        refresh_runtime_env_flags();
        assert!(super::use_dp4a_enabled());

        unsafe {
            std::env::remove_var(USE_DP4A_ENV);
        }
        refresh_runtime_env_flags();
    }

    #[test]
    fn force_wave32_env_flag_respected() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        unsafe {
            std::env::set_var(FORCE_WAVE32_ENV, "1");
        }
        refresh_runtime_env_flags();
        assert!(super::force_wave32_enabled());

        unsafe {
            std::env::set_var(FORCE_WAVE32_ENV, "0");
        }
        refresh_runtime_env_flags();
        assert!(!super::force_wave32_enabled());

        unsafe {
            std::env::remove_var(FORCE_WAVE32_ENV);
        }
        refresh_runtime_env_flags();
    }

    #[test]
    fn disable_wave32_env_flag_respected() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        unsafe {
            std::env::set_var(super::DISABLE_WAVE32_ENV, "1");
        }
        refresh_runtime_env_flags();
        assert!(super::disable_wave32_enabled());

        unsafe {
            std::env::set_var(super::DISABLE_WAVE32_ENV, "0");
        }
        refresh_runtime_env_flags();
        assert!(!super::disable_wave32_enabled());

        unsafe {
            std::env::remove_var(super::DISABLE_WAVE32_ENV);
        }
        refresh_runtime_env_flags();
    }

    #[test]
    fn test_gpu_lock_and_preflight_native() {
        if let Some(_) = crate::gpu::detect() {
            let lock = super::GpuLock::acquire(5).expect("Should acquire lock");
            super::gpu_safety_preflight().expect("Preflight should pass");
            let lock2 = super::GpuLock::acquire(1);
            assert!(lock2.is_err());
            drop(lock);
            let _lock3 = super::GpuLock::acquire(1).expect("Should acquire lock after drop");
        }
    }
}

// ── Native Safety and Lock Infrastructure ─────────────────────────────────────────

use crate::gpu::kernels;
use crate::gpu::{detect, GpuBuffer};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Path to the cross-process GPU lock file.
const GPU_LOCK_PATH: &str = "/tmp/rocmforge_gpu_tests.lock";

/// Cross-process GPU lock using flock(2).
pub struct GpuLock {
    _file: File,
}

impl GpuLock {
    /// Acquire the GPU lock, waiting up to `timeout_secs` if it is held by another process.
    pub fn acquire(timeout_secs: u64) -> Result<Self, String> {
        let path = Path::new(GPU_LOCK_PATH);
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("Failed to open lock file: {}", e))?;

        let start = std::time::Instant::now();
        loop {
            unsafe {
                let fd = file.as_raw_fd();
                let ret = libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB);
                if ret == 0 {
                    return Ok(Self { _file: file });
                }
            }

            if start.elapsed().as_secs() >= timeout_secs {
                return Err(format!(
                    "Timeout after {}s waiting for GPU lock",
                    timeout_secs
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

/// Run staged GPU preflight checks.
/// Returns Ok(()) if all checks pass, Err(description) otherwise.
pub fn gpu_safety_preflight() -> crate::error::RocmForgeResult<()> {
    // 1. Render node detection
    let mut render_node_found = false;
    if std::path::Path::new("/dev/dri").exists() {
        if let Ok(entries) = std::fs::read_dir("/dev/dri") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with("renderD") {
                    render_node_found = true;
                    break;
                }
            }
        }
    }
    if !render_node_found {
        return Err("No render node found in /dev/dri".into());
    }

    // 2. HIP runtime device visibility
    let _caps = detect().ok_or_else(|| {
        crate::error::RocmForgeError::Generic(
            "No AMD GPU detected via HIP/ROCm runtime".to_string(),
        )
    })?;

    // 3. Memory round-trip
    let size = 1024;
    let mut h_in1 = vec![1.0f32; size];
    let h_in2 = vec![2.0f32; size];

    let mut d_in1 = GpuBuffer::alloc(size * std::mem::size_of::<f32>())
        .map_err(|e| format!("hipMalloc (in1) failed: {:?}", e))?;
    let mut d_in2 = GpuBuffer::alloc(size * std::mem::size_of::<f32>())
        .map_err(|e| format!("hipMalloc (in2) failed: {:?}", e))?;
    let mut d_out = GpuBuffer::alloc(size * std::mem::size_of::<f32>())
        .map_err(|e| format!("hipMalloc (out) failed: {:?}", e))?;

    d_in1
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(h_in1.as_ptr() as *const u8, h_in1.len() * 4)
        })
        .map_err(|e| format!("hipMemcpy H2D (in1) failed: {:?}", e))?;

    d_in2
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(h_in2.as_ptr() as *const u8, h_in2.len() * 4)
        })
        .map_err(|e| format!("hipMemcpy H2D (in2) failed: {:?}", e))?;

    let mut h_verify = vec![0.0f32; size];
    d_in1
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(h_verify.as_mut_ptr() as *mut u8, h_verify.len() * 4)
        })
        .map_err(|e| format!("hipMemcpy D2H (in1) failed: {:?}", e))?;

    for i in 0..size {
        if h_verify[i] != h_in1[i] {
            return Err("Memory roundtrip verification failed".into());
        }
    }

    // 4. Trivial kernel launch
    kernels::add(
        d_in1.as_ptr() as *const f32,
        d_in2.as_ptr() as *const f32,
        d_out.as_ptr() as *mut f32,
        size,
    )
    .map_err(|e| format!("Elementwise add kernel launch failed: {:?}", e))?;

    crate::gpu::ffi::hip_device_synchronize()
        .map_err(|e| format!("hipDeviceSynchronize failed: {:?}", e))?;

    let mut h_out = vec![0.0f32; size];
    d_out
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(h_out.as_mut_ptr() as *mut u8, h_out.len() * 4)
        })
        .map_err(|e| format!("hipMemcpy D2H (out) failed: {:?}", e))?;

    for i in 0..size {
        if (h_out[i] - 3.0f32).abs() > 1e-5f32 {
            return Err(format!(
                "Kernel execution verification failed: got {}, expected 3.0",
                h_out[i]
            )
            .into());
        }
    }

    Ok(())
}
