//! Launch-policy autotune scaffold for GPU decode hot paths.
//!
//! Provides conservative micro-benchmarking (1 warmup + 2 timed repeats)
//! to select optimal kernel launch variants keyed by shape class.
//!
//! Environment:
//! - ROCMFORGE_ENABLE_LAUNCH_AUTOTUNE=1 to enable (default: off)
//!
//! Cache location:
//! - ~/.cache/rocmforge/launch_autotune_v5.json

use super::error::{GpuError, GpuResult};
use super::safety::gpu_safe_mode_enabled;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

const AUTOTUNE_ENV: &str = "ROCMFORGE_ENABLE_LAUNCH_AUTOTUNE";
const CACHE_VERSION: &str = "v5";
const CACHE_FILENAME: &str = "launch_autotune_v5.json";
const MAX_CANDIDATES: usize = 3;
const WARMUP_RUNS: usize = 1;
const TIMED_RUNS: usize = 2;

/// Shape class key for autotune cache.
///
/// Uses discrete buckets to keep the cache bounded and avoid
/// over-fitting to exact dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ShapeKey {
    /// Operation type identifier
    pub op: OpType,
    /// Bucketed input dimension (rounded to nearest 64)
    pub in_dim_bucket: u32,
    /// Bucketed output dimension (rounded to nearest 64)
    pub out_dim_bucket: u32,
    /// Optional secondary dimension (e.g., ff_size for gate_up)
    pub aux_dim_bucket: u32,
}

/// Operation types supported for autotuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum OpType {
    QkvFused = 1,
    GateUpSwigluQ8 = 2,
    LmHeadQ8 = 3,
    Q4_0Q8Residual = 4,
    Q4_1Residual = 5,
}

impl ShapeKey {
    /// Create a shape key from raw dimensions.
    ///
    /// Buckets dimensions to nearest 64 to keep cache bounded.
    pub fn new(op: OpType, in_dim: usize, out_dim: usize, aux_dim: usize) -> Self {
        Self {
            op,
            in_dim_bucket: Self::bucket_dim(in_dim),
            out_dim_bucket: Self::bucket_dim(out_dim),
            aux_dim_bucket: Self::bucket_dim(aux_dim),
        }
    }

    fn bucket_dim(dim: usize) -> u32 {
        ((dim + 63) / 64 * 64) as u32
    }
}

/// Variant identifier for a specific launch configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum VariantId {
    /// Default/baseline variant
    Baseline = 0,
    /// Optimized variant 1 (e.g., different block size)
    Variant1 = 1,
    /// Optimized variant 2
    Variant2 = 2,
}

impl Default for VariantId {
    fn default() -> Self {
        VariantId::Baseline
    }
}

/// Persistent cache for autotune decisions.
#[derive(Debug, Default)]
struct AutotuneCache {
    version: String,
    entries: HashMap<ShapeKey, VariantId>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PersistedCacheEntry {
    key: ShapeKey,
    variant: VariantId,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PersistedAutotuneCache {
    version: String,
    entries: Vec<PersistedCacheEntry>,
}

impl From<&AutotuneCache> for PersistedAutotuneCache {
    fn from(cache: &AutotuneCache) -> Self {
        let mut entries: Vec<PersistedCacheEntry> = cache
            .entries
            .iter()
            .map(|(key, variant)| PersistedCacheEntry {
                key: *key,
                variant: *variant,
            })
            .collect();
        entries.sort_by_key(|entry| {
            (
                entry.key.op as u8,
                entry.key.in_dim_bucket,
                entry.key.out_dim_bucket,
                entry.key.aux_dim_bucket,
            )
        });
        Self {
            version: cache.version.clone(),
            entries,
        }
    }
}

impl From<PersistedAutotuneCache> for AutotuneCache {
    fn from(cache: PersistedAutotuneCache) -> Self {
        let mut entries = HashMap::with_capacity(cache.entries.len());
        for entry in cache.entries {
            entries.insert(entry.key, entry.variant);
        }
        Self {
            version: cache.version,
            entries,
        }
    }
}

/// Launch autotuner with in-memory and on-disk cache.
pub struct LaunchAutotuner {
    cache: RwLock<AutotuneCache>,
    cache_path: PathBuf,
    enabled: AtomicBool,
    /// Tracks keys currently being tuned to avoid concurrent tuning
    tuning_in_progress: Mutex<Vec<ShapeKey>>,
    /// Runtime disable flag per process
    runtime_disabled: AtomicBool,
}

impl LaunchAutotuner {
    /// Create or load the global autotuner.
    fn new() -> Self {
        let cache_path = Self::cache_path();
        let cache = Self::load_cache(&cache_path);
        let enabled = Self::check_env_enabled();

        Self {
            cache: RwLock::new(cache),
            cache_path,
            enabled: AtomicBool::new(enabled),
            tuning_in_progress: Mutex::new(Vec::new()),
            runtime_disabled: AtomicBool::new(false),
        }
    }

    fn cache_path() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("rocmforge")
            .join(CACHE_FILENAME)
    }

    fn check_env_enabled() -> bool {
        if gpu_safe_mode_enabled() {
            return false;
        }
        match std::env::var(AUTOTUNE_ENV).ok() {
            Some(v) => matches!(v.as_str(), "1" | "true" | "yes" | "on"),
            None => false,
        }
    }

    fn load_cache(path: &PathBuf) -> AutotuneCache {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(cache) = serde_json::from_str::<PersistedAutotuneCache>(&contents) {
                if cache.version == CACHE_VERSION {
                    return cache.into();
                }
            }
        }
        AutotuneCache {
            version: CACHE_VERSION.to_string(),
            entries: HashMap::new(),
        }
    }

    fn save_cache(&self) {
        let cache = match self.cache.read() {
            Ok(c) => c,
            Err(_) => return,
        };

        // Ensure parent directory exists
        if let Some(parent) = self.cache_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // Write atomically via temp file
        let temp_path = self.cache_path.with_extension("tmp");
        let persisted = PersistedAutotuneCache::from(&*cache);
        if let Ok(json) = serde_json::to_string_pretty(&persisted) {
            if std::fs::write(&temp_path, json).is_ok() {
                let _ = std::fs::rename(&temp_path, &self.cache_path);
            }
        }
    }

    /// Check if autotune is enabled.
    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) && !self.runtime_disabled.load(Ordering::Relaxed)
    }

    /// Disable autotune for this process due to failure.
    fn disable_runtime(&self, reason: &str) {
        if !self.runtime_disabled.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[rocmforge][launch_autotune] autotune auto-disabled for this process: {}",
                reason
            );
        }
    }

    /// Get the cached variant for a shape key, if available.
    fn get_cached_variant(&self, key: ShapeKey) -> Option<VariantId> {
        let cache = self.cache.read().ok()?;
        cache.entries.get(&key).copied()
    }

    /// Store a variant decision in cache.
    fn store_variant(&self, key: ShapeKey, variant: VariantId) {
        if let Ok(mut cache) = self.cache.write() {
            cache.entries.insert(key, variant);
            // Limit cache size to prevent unbounded growth
            if cache.entries.len() > 1000 {
                // Simple LRU: clear half the entries if too large
                let keys_to_remove: Vec<_> = cache
                    .entries
                    .keys()
                    .take(cache.entries.len() / 2)
                    .copied()
                    .collect();
                for k in keys_to_remove {
                    cache.entries.remove(&k);
                }
            }
        }
        self.save_cache();
    }

    /// Check if tuning is already in progress for this key.
    fn is_tuning(&self, key: ShapeKey) -> bool {
        let tuning = match self.tuning_in_progress.lock() {
            Ok(t) => t,
            Err(_) => return true, // Fail safe: assume tuning if lock fails
        };
        tuning.contains(&key)
    }

    /// Mark tuning as in progress.
    fn start_tuning(&self, key: ShapeKey) {
        if let Ok(mut tuning) = self.tuning_in_progress.lock() {
            tuning.push(key);
        }
    }

    /// Mark tuning as complete.
    fn finish_tuning(&self, key: ShapeKey) {
        if let Ok(mut tuning) = self.tuning_in_progress.lock() {
            tuning.retain(|&k| k != key);
        }
    }
}

/// Global autotuner singleton.
static AUTOTUNER: std::sync::OnceLock<LaunchAutotuner> = std::sync::OnceLock::new();

fn get_autotuner() -> &'static LaunchAutotuner {
    AUTOTUNER.get_or_init(LaunchAutotuner::new)
}

/// Timing helper for micro-benchmarking.
struct GpuTimer {
    start: Instant,
}

impl GpuTimer {
    fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
}

/// Result of a variant benchmark.
struct VariantResult {
    variant: VariantId,
    avg_time_us: u64,
}

/// Autotune candidate function type.
pub type AutotuneCandidateFn = Arc<dyn Fn(VariantId) -> GpuResult<()> + Send + Sync>;

/// Get the selected variant for a shape key.
///
/// Returns cached decision if available, otherwise runs micro-benchmark
/// to select between available variants.
pub fn select_variant(
    key: ShapeKey,
    candidates: &[VariantId],
    benchmark_fn: impl Fn(VariantId) -> GpuResult<()>,
) -> VariantId {
    let autotuner = get_autotuner();

    if !autotuner.is_enabled() || candidates.is_empty() {
        return VariantId::Baseline;
    }

    // Return cached if available
    if let Some(variant) = autotuner.get_cached_variant(key) {
        return variant;
    }

    // Avoid concurrent tuning for same key
    if autotuner.is_tuning(key) {
        return VariantId::Baseline;
    }

    // Limit candidates
    let candidates: Vec<_> = candidates.iter().take(MAX_CANDIDATES).copied().collect();

    autotuner.start_tuning(key);

    let result = run_micro_benchmark(&candidates, &benchmark_fn);

    autotuner.finish_tuning(key);

    match result {
        Ok(selected) => {
            autotuner.store_variant(key, selected);
            selected
        }
        Err(e) => {
            autotuner.disable_runtime(&format!("benchmark failed: {}", e));
            VariantId::Baseline
        }
    }
}

/// Read a cached variant for a shape key without triggering benchmarking.
///
/// Returns `None` when autotune is disabled, runtime-disabled, or if no cached
/// entry exists yet.
pub fn lookup_cached_variant(key: ShapeKey) -> Option<VariantId> {
    let autotuner = get_autotuner();
    if !autotuner.is_enabled() {
        return None;
    }
    autotuner.get_cached_variant(key)
}

/// Run conservative micro-benchmark (warmup + timed repeats).
fn run_micro_benchmark(
    candidates: &[VariantId],
    benchmark_fn: impl Fn(VariantId) -> GpuResult<()>,
) -> GpuResult<VariantId> {
    let mut results = Vec::with_capacity(candidates.len());

    for &variant in candidates {
        // Warmup
        for _ in 0..WARMUP_RUNS {
            benchmark_fn(variant)?;
        }

        // Synchronize before timing (caller is responsible for stream sync)
        std::thread::yield_now();

        // Timed runs
        let mut times = Vec::with_capacity(TIMED_RUNS);
        for _ in 0..TIMED_RUNS {
            let timer = GpuTimer::new();
            benchmark_fn(variant)?;
            times.push(timer.elapsed_us());
        }

        // Average time
        let avg_time = times.iter().sum::<u64>() / times.len() as u64;
        results.push(VariantResult {
            variant,
            avg_time_us: avg_time,
        });
    }

    // Select fastest
    results
        .into_iter()
        .min_by_key(|r| r.avg_time_us)
        .map(|r| r.variant)
        .ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "no benchmark results".to_string(),
        })
}

/// Convenience: get variant for QKV fused launch.
pub fn select_qkv_variant(
    h: usize,
    q_size: usize,
    kv_size: usize,
    benchmark_fn: impl Fn(VariantId) -> GpuResult<()>,
) -> VariantId {
    let key = ShapeKey::new(OpType::QkvFused, h, q_size, kv_size);
    select_variant(
        key,
        &[VariantId::Baseline, VariantId::Variant1],
        benchmark_fn,
    )
}

/// Convenience: read cached variant for QKV fused launch.
pub fn lookup_qkv_variant(h: usize, q_size: usize, kv_size: usize) -> Option<VariantId> {
    let key = ShapeKey::new(OpType::QkvFused, h, q_size, kv_size);
    lookup_cached_variant(key)
}

/// Convenience: get variant for gate_up_swiglu q8 inline launch.
pub fn select_gate_up_swiglu_q8_variant(
    h: usize,
    ff_size: usize,
    benchmark_fn: impl Fn(VariantId) -> GpuResult<()>,
) -> VariantId {
    let key = ShapeKey::new(OpType::GateUpSwigluQ8, h, ff_size, 0);
    select_variant(
        key,
        &[
            VariantId::Baseline,
            VariantId::Variant1,
            VariantId::Variant2,
        ],
        benchmark_fn,
    )
}

/// Convenience: read cached variant for gate_up_swiglu q8 inline launch.
pub fn lookup_gate_up_swiglu_q8_variant(h: usize, ff_size: usize) -> Option<VariantId> {
    let key = ShapeKey::new(OpType::GateUpSwigluQ8, h, ff_size, 0);
    lookup_cached_variant(key)
}

/// Convenience: get variant for LM-head q8 launch.
pub fn select_lm_head_q8_variant(
    n_rows: usize,
    ncols_dst: usize,
    benchmark_fn: impl Fn(VariantId) -> GpuResult<()>,
) -> VariantId {
    let key = ShapeKey::new(OpType::LmHeadQ8, n_rows, ncols_dst, 0);
    select_variant(
        key,
        &[VariantId::Baseline, VariantId::Variant1],
        benchmark_fn,
    )
}

/// Convenience: read cached variant for LM-head q8 launch.
pub fn lookup_lm_head_q8_variant(n_rows: usize, ncols_dst: usize) -> Option<VariantId> {
    let key = ShapeKey::new(OpType::LmHeadQ8, n_rows, ncols_dst, 0);
    lookup_cached_variant(key)
}

/// Convenience: get variant for q4_0 q8 inline residual launch.
pub fn select_q4_0_q8_residual_variant(
    n_rows: usize,
    ncols_dst: usize,
    benchmark_fn: impl Fn(VariantId) -> GpuResult<()>,
) -> VariantId {
    let key = ShapeKey::new(OpType::Q4_0Q8Residual, n_rows, ncols_dst, 0);
    select_variant(
        key,
        &[
            VariantId::Baseline,
            VariantId::Variant1,
            VariantId::Variant2,
        ],
        benchmark_fn,
    )
}

/// Convenience: read cached variant for q4_0 q8 inline residual launch.
pub fn lookup_q4_0_q8_residual_variant(n_rows: usize, ncols_dst: usize) -> Option<VariantId> {
    let key = ShapeKey::new(OpType::Q4_0Q8Residual, n_rows, ncols_dst, 0);
    lookup_cached_variant(key)
}

/// Convenience: get variant for q4_1 residual launch (FFN-down decode path).
pub fn select_q4_1_residual_variant(
    n_rows: usize,
    ncols_dst: usize,
    benchmark_fn: impl Fn(VariantId) -> GpuResult<()>,
) -> VariantId {
    let key = ShapeKey::new(OpType::Q4_1Residual, n_rows, ncols_dst, 0);
    select_variant(
        key,
        &[VariantId::Baseline, VariantId::Variant1],
        benchmark_fn,
    )
}

/// Convenience: read cached variant for q4_1 residual launch.
pub fn lookup_q4_1_residual_variant(n_rows: usize, ncols_dst: usize) -> Option<VariantId> {
    let key = ShapeKey::new(OpType::Q4_1Residual, n_rows, ncols_dst, 0);
    lookup_cached_variant(key)
}

/// Check if launch autotune is enabled.
pub fn launch_autotune_enabled() -> bool {
    get_autotuner().is_enabled()
}

/// Refresh the autotuner state (for testing).
pub fn refresh_launch_autotune_state() {
    if let Some(autotuner) = AUTOTUNER.get() {
        let enabled = LaunchAutotuner::check_env_enabled();
        autotuner.enabled.store(enabled, Ordering::Relaxed);
        autotuner.runtime_disabled.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_key_bucketing() {
        let key1 = ShapeKey::new(OpType::QkvFused, 100, 200, 50);
        let key2 = ShapeKey::new(OpType::QkvFused, 110, 210, 60);
        // Both should bucket to same values
        assert_eq!(key1.in_dim_bucket, key2.in_dim_bucket);
        assert_eq!(key1.out_dim_bucket, key2.out_dim_bucket);
        assert_eq!(key1.aux_dim_bucket, key2.aux_dim_bucket);
    }

    #[test]
    fn shape_key_different_ops_not_equal() {
        let key1 = ShapeKey::new(OpType::QkvFused, 128, 256, 0);
        let key2 = ShapeKey::new(OpType::LmHeadQ8, 128, 256, 0);
        assert_ne!(key1.op, key2.op);
    }

    #[test]
    fn select_variant_returns_baseline_when_disabled() {
        unsafe { std::env::remove_var(AUTOTUNE_ENV) };
        refresh_launch_autotune_state();

        let key = ShapeKey::new(OpType::QkvFused, 128, 256, 0);
        let variant = select_variant(key, &[VariantId::Baseline], |_v| Ok(()));
        assert_eq!(variant, VariantId::Baseline);
    }

    #[test]
    fn cache_serialization_roundtrip() {
        let mut cache = AutotuneCache {
            version: CACHE_VERSION.to_string(),
            entries: HashMap::new(),
        };

        let key1 = ShapeKey::new(OpType::QkvFused, 128, 256, 0);
        let key2 = ShapeKey::new(OpType::LmHeadQ8, 512, 1024, 0);
        cache.entries.insert(key1, VariantId::Variant1);
        cache.entries.insert(key2, VariantId::Variant2);

        let persisted = PersistedAutotuneCache::from(&cache);
        let json = serde_json::to_string_pretty(&persisted).unwrap();
        let loaded: AutotuneCache = serde_json::from_str::<PersistedAutotuneCache>(&json)
            .unwrap()
            .into();

        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries.get(&key1), Some(&VariantId::Variant1));
        assert_eq!(loaded.entries.get(&key2), Some(&VariantId::Variant2));
    }
}
