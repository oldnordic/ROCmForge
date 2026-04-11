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
    CachedEnvFlag::new(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV, false);
static ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, true);
static ENABLE_LAUNCH_AUTOTUNE_FLAG: CachedEnvFlag =
    CachedEnvFlag::new(ENABLE_LAUNCH_AUTOTUNE_ENV, false);
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
        ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, GPU_SAFE_MODE_ENV,
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

    #[test]
    fn refresh_runtime_env_flags_reloads_cached_defaults() {
        unsafe {
            std::env::set_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV, "0");
        }
        refresh_runtime_env_flags();
        assert!(!super::experimental_ffn_fastpath_enabled());

        unsafe {
            std::env::remove_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV);
        }
        refresh_runtime_env_flags();
        assert!(!super::experimental_ffn_fastpath_enabled());

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
        unsafe {
            std::env::set_var(ENABLE_DECODE_GRAPH_ENV, "1");
            std::env::set_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV, "1");
            std::env::set_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, "1");
            std::env::set_var(GPU_SAFE_MODE_ENV, "1");
        }
        refresh_runtime_env_flags();

        assert!(super::gpu_safe_mode_enabled());
        assert!(!super::decode_graph_enabled());
        assert!(!super::experimental_ffn_fastpath_enabled());
        assert!(!super::experimental_q8_activation_fastpath_enabled());

        unsafe {
            std::env::remove_var(ENABLE_DECODE_GRAPH_ENV);
            std::env::remove_var(ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV);
            std::env::remove_var(ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV);
            std::env::remove_var(GPU_SAFE_MODE_ENV);
        }
        refresh_runtime_env_flags();
    }
}
