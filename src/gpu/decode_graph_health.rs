//! Decode-graph health telemetry.
//!
//! HIP decode graphs are a performance-critical fast path, but capture failures,
//! replay failures, or unexpected fallbacks to the non-graph path are easy to
//! miss in production.  This module exposes atomic counters for those events
//! and a snapshot that callers can read at any time.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

pub const OBSERVE_DECODE_GRAPH_HEALTH_ENV: &str = "ROCMFORGE_OBSERVE_DECODE_GRAPH_HEALTH";

const ENV_UNKNOWN: u8 = 0;
const ENV_DISABLED: u8 = 1;
const ENV_ENABLED: u8 = 2;

static OBSERVE_DECODE_GRAPH_HEALTH_FLAG: AtomicU8 = AtomicU8::new(ENV_UNKNOWN);

fn parse_env_flag(value: Option<String>, default: bool) -> bool {
    match value.map(|value| value.trim().to_ascii_lowercase()) {
        Some(value) => matches!(value.as_str(), "1" | "true" | "yes" | "on"),
        None => default,
    }
}

pub(crate) fn decode_graph_health_observing_enabled() -> bool {
    match OBSERVE_DECODE_GRAPH_HEALTH_FLAG.load(Ordering::Relaxed) {
        ENV_DISABLED => false,
        ENV_ENABLED => true,
        _ => {
            let enabled =
                parse_env_flag(std::env::var(OBSERVE_DECODE_GRAPH_HEALTH_ENV).ok(), false);
            OBSERVE_DECODE_GRAPH_HEALTH_FLAG.store(
                if enabled { ENV_ENABLED } else { ENV_DISABLED },
                Ordering::Relaxed,
            );
            enabled
        }
    }
}

pub(crate) fn refresh_decode_graph_health_env_flag() {
    OBSERVE_DECODE_GRAPH_HEALTH_FLAG.store(ENV_UNKNOWN, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuDecodeGraphHealthSnapshot {
    pub captures_attempted: u64,
    pub captures_succeeded: u64,
    pub captures_failed: u64,
    pub replays_attempted: u64,
    pub replays_succeeded: u64,
    pub replays_failed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub updates_attempted: u64,
    pub updates_succeeded: u64,
    pub fallbacks_to_non_graph: u64,
    pub runtime_disables: u64,
}

struct HealthStore {
    snapshot: GpuDecodeGraphHealthSnapshot,
}

fn health_store() -> &'static Mutex<HealthStore> {
    static STORE: OnceLock<Mutex<HealthStore>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(HealthStore {
            snapshot: GpuDecodeGraphHealthSnapshot::default(),
        })
    })
}

pub fn decode_graph_health_snapshot() -> GpuDecodeGraphHealthSnapshot {
    health_store().lock().snapshot
}

pub fn reset_decode_graph_health() {
    let mut guard = health_store().lock();
    guard.snapshot = GpuDecodeGraphHealthSnapshot::default();
}

fn log_event(message: &str) {
    if decode_graph_health_observing_enabled() {
        eprintln!("[rocmforge][decode-graph-health] {}", message);
    }
}

fn increment(field: fn(&mut GpuDecodeGraphHealthSnapshot) -> &mut u64) {
    let mut guard = health_store().lock();
    let counter = field(&mut guard.snapshot);
    *counter = counter.wrapping_add(1);
}

pub(crate) fn record_decode_graph_capture_attempted() {
    increment(|s| &mut s.captures_attempted);
}

pub(crate) fn record_decode_graph_capture_succeeded() {
    increment(|s| &mut s.captures_succeeded);
    log_event("graph captured");
}

pub(crate) fn record_decode_graph_capture_failed(reason: &str) {
    increment(|s| &mut s.captures_failed);
    log_event(&format!("graph capture failed: {}", reason));
}

pub(crate) fn record_decode_graph_replay_attempted() {
    increment(|s| &mut s.replays_attempted);
}

pub(crate) fn record_decode_graph_replay_succeeded() {
    increment(|s| &mut s.replays_succeeded);
}

pub(crate) fn record_decode_graph_replay_failed(reason: &str) {
    increment(|s| &mut s.replays_failed);
    log_event(&format!("graph replay failed: {}", reason));
}

pub(crate) fn record_decode_graph_cache_hit() {
    increment(|s| &mut s.cache_hits);
}

pub(crate) fn record_decode_graph_cache_miss() {
    increment(|s| &mut s.cache_misses);
    log_event("graph cache miss");
}

pub(crate) fn record_decode_graph_update_attempted() {
    increment(|s| &mut s.updates_attempted);
}

pub(crate) fn record_decode_graph_update_succeeded() {
    increment(|s| &mut s.updates_succeeded);
}

pub(crate) fn record_decode_graph_fallback_to_non_graph(reason: &str) {
    increment(|s| &mut s.fallbacks_to_non_graph);
    log_event(&format!("fallback to non-graph path: {}", reason));
}

pub(crate) fn record_decode_graph_runtime_disabled(reason: &str) {
    increment(|s| &mut s.runtime_disables);
    log_event(&format!("decode graph runtime disabled: {}", reason));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_counters_increment_and_reset() {
        reset_decode_graph_health();

        record_decode_graph_capture_attempted();
        record_decode_graph_capture_succeeded();
        record_decode_graph_replay_attempted();
        record_decode_graph_replay_failed("test failure");
        record_decode_graph_cache_hit();
        record_decode_graph_cache_miss();
        record_decode_graph_fallback_to_non_graph("test fallback");
        record_decode_graph_runtime_disabled("test disable");

        let snap = decode_graph_health_snapshot();
        assert_eq!(snap.captures_attempted, 1);
        assert_eq!(snap.captures_succeeded, 1);
        assert_eq!(snap.captures_failed, 0);
        assert_eq!(snap.replays_attempted, 1);
        assert_eq!(snap.replays_succeeded, 0);
        assert_eq!(snap.replays_failed, 1);
        assert_eq!(snap.cache_hits, 1);
        assert_eq!(snap.cache_misses, 1);
        assert_eq!(snap.fallbacks_to_non_graph, 1);
        assert_eq!(snap.runtime_disables, 1);

        reset_decode_graph_health();
        assert_eq!(
            decode_graph_health_snapshot(),
            GpuDecodeGraphHealthSnapshot::default()
        );
    }

    #[test]
    fn env_flag_refresh_works() {
        use parking_lot::Mutex;
        static ENV_MUTEX: Mutex<()> = Mutex::new(());
        let _guard = ENV_MUTEX.lock();

        unsafe {
            std::env::set_var(OBSERVE_DECODE_GRAPH_HEALTH_ENV, "1");
        }
        refresh_decode_graph_health_env_flag();
        assert!(decode_graph_health_observing_enabled());

        unsafe {
            std::env::set_var(OBSERVE_DECODE_GRAPH_HEALTH_ENV, "0");
        }
        refresh_decode_graph_health_env_flag();
        assert!(!decode_graph_health_observing_enabled());

        unsafe {
            std::env::remove_var(OBSERVE_DECODE_GRAPH_HEALTH_ENV);
        }
        refresh_decode_graph_health_env_flag();
    }
}
