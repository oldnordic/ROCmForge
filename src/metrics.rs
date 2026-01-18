//! Prometheus-compatible metrics collection for ROCmForge
//!
//! Provides thread-safe metrics for monitoring inference operations.
//! Metrics are exported in Prometheus text format via the /metrics endpoint.
//!
//! Uses prometheus-client crate for thread-safe metric collection.

use prometheus_client::{
    encoding::text::encode,
    metrics::counter::Counter,
    metrics::gauge::Gauge,
    metrics::histogram::{exponential_buckets, Histogram},
    registry::Registry,
};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Primary metrics collection structure
///
/// Thread-safe singleton holding all ROCmForge metrics.
/// All metric updates use lock-free atomic operations.
#[derive(Debug)]
pub struct Metrics {
    /// Registry containing all metrics
    pub registry: Registry,

    /// Total requests started
    pub requests_started: Counter<u64>,

    /// Total requests completed
    pub requests_completed: Counter<u64>,

    /// Total requests failed
    pub requests_failed: Counter<u64>,

    /// Total requests cancelled
    pub requests_cancelled: Counter<u64>,

    /// Total number of tokens generated
    pub tokens_generated_total: Counter<u64>,

    /// Prefill phase duration histogram
    pub prefill_duration_seconds: Histogram,

    /// Decode phase duration histogram
    pub decode_duration_seconds: Histogram,

    /// Total inference duration histogram
    pub total_duration_seconds: Histogram,

    /// Current queue length
    pub queue_length: Gauge<i64>,

    /// Current number of active requests
    pub active_requests: Gauge<i64>,

    /// Time to first token (TTFT) in seconds
    pub ttft_seconds: Histogram,

    /// Tokens per second throughput
    pub tokens_per_second: Gauge<f64, AtomicU64>,
}

impl Metrics {
    /// Create a new metrics collection
    pub fn new() -> Self {
        let mut registry = Registry::default();

        // Request counters
        let requests_started = Counter::default();
        registry.register(
            "rocmforge_requests_started_total",
            "Total number of inference requests started",
            requests_started.clone(),
        );

        let requests_completed = Counter::default();
        registry.register(
            "rocmforge_requests_completed_total",
            "Total number of inference requests completed",
            requests_completed.clone(),
        );

        let requests_failed = Counter::default();
        registry.register(
            "rocmforge_requests_failed_total",
            "Total number of inference requests failed",
            requests_failed.clone(),
        );

        let requests_cancelled = Counter::default();
        registry.register(
            "rocmforge_requests_cancelled_total",
            "Total number of inference requests cancelled",
            requests_cancelled.clone(),
        );

        // Token generation counter
        let tokens_generated_total = Counter::default();
        registry.register(
            "rocmforge_tokens_generated_total",
            "Total number of tokens generated",
            tokens_generated_total.clone(),
        );

        // Phase duration histograms
        // Buckets: 0.001s, 0.01s, 0.1s, 1s, 10s, 100s
        let prefill_duration_seconds = Histogram::new(exponential_buckets(0.001, 10.0, 6));
        registry.register(
            "rocmforge_prefill_duration_seconds",
            "Prefill phase duration in seconds",
            prefill_duration_seconds.clone(),
        );

        let decode_duration_seconds = Histogram::new(exponential_buckets(0.001, 10.0, 6));
        registry.register(
            "rocmforge_decode_duration_seconds",
            "Decode phase duration in seconds",
            decode_duration_seconds.clone(),
        );

        let total_duration_seconds = Histogram::new(exponential_buckets(0.001, 10.0, 6));
        registry.register(
            "rocmforge_total_duration_seconds",
            "Total inference duration in seconds",
            total_duration_seconds.clone(),
        );

        // Queue length gauge
        let queue_length = Gauge::default();
        registry.register(
            "rocmforge_queue_length",
            "Current number of requests in queue",
            queue_length.clone(),
        );

        // Active requests gauge
        let active_requests = Gauge::default();
        registry.register(
            "rocmforge_active_requests",
            "Current number of active requests",
            active_requests.clone(),
        );

        // TTFT histogram (Time To First Token)
        let ttft_seconds = Histogram::new(exponential_buckets(0.001, 10.0, 6));
        registry.register(
            "rocmforge_ttft_seconds",
            "Time to first token in seconds",
            ttft_seconds.clone(),
        );

        // Tokens per second gauge
        let tokens_per_second = Gauge::<f64, AtomicU64>::default();
        registry.register(
            "rocmforge_tokens_per_second",
            "Tokens generated per second",
            tokens_per_second.clone(),
        );

        Metrics {
            registry,
            requests_started,
            requests_completed,
            requests_failed,
            requests_cancelled,
            tokens_generated_total,
            prefill_duration_seconds,
            decode_duration_seconds,
            total_duration_seconds,
            queue_length,
            active_requests,
            ttft_seconds,
            tokens_per_second,
        }
    }

    /// Record a request starting
    pub fn record_request_start(&self) {
        self.requests_started.inc();
        self.active_requests.inc();
    }

    /// Record a request completing successfully
    pub fn record_request_complete(&self, token_count: u64) {
        self.requests_completed.inc();
        self.tokens_generated_total.inc_by(token_count);
        self.active_requests.dec();
    }

    /// Record a request failure
    pub fn record_request_failed(&self) {
        self.requests_failed.inc();
        self.active_requests.dec();
    }

    /// Record a request cancellation
    pub fn record_request_cancelled(&self) {
        self.requests_cancelled.inc();
        self.active_requests.dec();
    }

    /// Record prefill phase duration
    pub fn record_prefill_duration(&self, duration_sec: f64) {
        self.prefill_duration_seconds.observe(duration_sec);
    }

    /// Record decode phase duration
    pub fn record_decode_duration(&self, duration_sec: f64) {
        self.decode_duration_seconds.observe(duration_sec);
    }

    /// Record total inference duration
    pub fn record_total_duration(&self, duration_sec: f64) {
        self.total_duration_seconds.observe(duration_sec);
    }

    /// Update queue length
    pub fn set_queue_length(&self, length: u64) {
        self.queue_length.set(length as i64);
    }

    /// Update active requests count
    pub fn set_active_requests(&self, count: u64) {
        self.active_requests.set(count as i64);
    }

    /// Record time to first token
    pub fn record_ttft(&self, duration_sec: f64) {
        self.ttft_seconds.observe(duration_sec);
    }

    /// Update tokens per second throughput
    pub fn set_tokens_per_second(&self, tps: f64) {
        self.tokens_per_second.set(tps);
    }

    /// Export metrics in Prometheus text format
    pub fn export(&self) -> String {
        let mut buffer = String::new();
        encode(&mut buffer, &self.registry).expect("encoding should succeed");
        buffer
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global metrics accessor
///
/// Provides thread-safe access to the global metrics instance.
/// Use get() to access metrics from anywhere in the codebase.
#[derive(Debug, Clone)]
pub struct MetricRegistry {
    inner: Arc<RwLock<Option<Arc<Metrics>>>>,
}

impl MetricRegistry {
    /// Create a new empty metric registry
    pub fn new() -> Self {
        MetricRegistry {
            inner: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize the metrics instance
    pub async fn init(&self, metrics: Arc<Metrics>) {
        let mut guard = self.inner.write().await;
        *guard = Some(metrics);
    }

    /// Get a reference to the metrics, if initialized
    pub async fn get(&self) -> Option<Arc<Metrics>> {
        let guard = self.inner.read().await;
        guard.as_ref().map(Arc::clone)
    }

    /// Export metrics in Prometheus text format
    pub async fn export(&self) -> String {
        if let Some(metrics) = self.get().await {
            metrics.export()
        } else {
            "# ROCmForge Metrics\n# Metrics not initialized\n".to_string()
        }
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for timing inference phases
///
/// Records the duration when dropped.
pub struct PhaseTimer {
    metrics: Option<Arc<Metrics>>,
    phase: Phase,
    start: std::time::Instant,
}

/// Inference phase for timing
#[derive(Debug, Clone, Copy)]
pub enum Phase {
    Prefill,
    Decode,
    Total,
}

impl PhaseTimer {
    /// Create a new phase timer
    pub fn new(metrics: Option<Arc<Metrics>>, phase: Phase) -> Self {
        PhaseTimer {
            metrics,
            phase,
            start: std::time::Instant::now(),
        }
    }

    /// Complete the timer early
    pub fn finish(self) {
        if let Some(metrics) = &self.metrics {
            let duration = self.start.elapsed().as_secs_f64();
            match self.phase {
                Phase::Prefill => metrics.record_prefill_duration(duration),
                Phase::Decode => metrics.record_decode_duration(duration),
                Phase::Total => metrics.record_total_duration(duration),
            }
        }
        // Prevent double-recording on drop
        std::mem::forget(self);
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        if let Some(metrics) = &self.metrics {
            let duration = self.start.elapsed().as_secs_f64();
            match self.phase {
                Phase::Prefill => metrics.record_prefill_duration(duration),
                Phase::Decode => metrics.record_decode_duration(duration),
                Phase::Total => metrics.record_total_duration(duration),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new();
        // Metrics created successfully
        let export = metrics.export();
        assert!(!export.is_empty());
    }

    #[test]
    fn test_request_lifecycle() {
        let metrics = Metrics::new();

        metrics.record_request_start();
        // Active requests incremented
        let export = metrics.export();
        eprintln!("Export after request_start:\n{}", export);
        assert!(export.contains("rocmforge_requests_started_total"));
        assert!(export.contains("rocmforge_active_requests"));

        metrics.record_request_complete(10);
        // Active requests decremented
        let export = metrics.export();
        assert!(export.contains("rocmforge_active_requests"));
        assert!(export.contains("rocmforge_tokens_generated_total"));
    }

    #[test]
    fn test_queue_length() {
        let metrics = Metrics::new();

        metrics.set_queue_length(5);
        let export = metrics.export();
        assert!(export.contains("rocmforge_queue_length 5"));

        metrics.set_queue_length(0);
        let export = metrics.export();
        assert!(export.contains("rocmforge_queue_length 0"));
    }

    #[test]
    fn test_phase_durations() {
        let metrics = Metrics::new();

        metrics.record_prefill_duration(0.1);
        metrics.record_decode_duration(0.05);
        metrics.record_total_duration(1.0);

        let export = metrics.export();
        assert!(export.contains("rocmforge_prefill_duration_seconds"));
        assert!(export.contains("rocmforge_decode_duration_seconds"));
        assert!(export.contains("rocmforge_total_duration_seconds"));
    }

    #[test]
    fn test_metrics_export_format() {
        let metrics = Metrics::new();

        metrics.set_queue_length(3);
        metrics.set_active_requests(2);
        metrics.record_request_start();
        metrics.record_request_complete(5);

        let export = metrics.export();

        // Verify key metrics are present
        assert!(export.contains("rocmforge_requests_started_total"));
        assert!(export.contains("rocmforge_tokens_generated_total"));
        assert!(export.contains("rocmforge_queue_length 3"));
        assert!(export.contains("rocmforge_active_requests 2"));
    }

    #[tokio::test]
    async fn test_metric_registry() {
        let registry = MetricRegistry::new();

        // Metrics not initialized
        assert!(registry.get().await.is_none());

        // Initialize and check
        let metrics = Arc::new(Metrics::new());
        registry.init(metrics.clone()).await;

        let retrieved = registry.get().await.unwrap();
        assert!(Arc::ptr_eq(&metrics, &retrieved));
    }

    #[tokio::test]
    async fn test_metric_registry_export() {
        let registry = MetricRegistry::new();

        // Export before initialization
        let export = registry.export().await;
        assert!(export.contains("Metrics not initialized"));

        // Initialize and export again
        let metrics = Arc::new(Metrics::new());
        metrics.set_queue_length(7);
        registry.init(metrics.clone()).await;

        let export = registry.export().await;
        assert!(export.contains("rocmforge_queue_length 7"));
    }

    #[test]
    fn test_phase_timer() {
        let metrics = Arc::new(Metrics::new());

        {
            let _timer = PhaseTimer::new(Some(metrics.clone()), Phase::Prefill);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let export = metrics.export();
        assert!(export.contains("rocmforge_prefill_duration_seconds"));
    }

    #[test]
    fn test_phase_timer_finish() {
        let metrics = Arc::new(Metrics::new());

        let timer = PhaseTimer::new(Some(metrics.clone()), Phase::Decode);
        std::thread::sleep(std::time::Duration::from_millis(10));
        timer.finish();

        let export = metrics.export();
        assert!(export.contains("rocmforge_decode_duration_seconds"));
    }

    #[test]
    fn test_ttft_recording() {
        let metrics = Metrics::new();

        metrics.record_ttft(0.05);
        metrics.record_ttft(0.15);
        metrics.record_ttft(0.25);

        let export = metrics.export();
        assert!(export.contains("rocmforge_ttft_seconds"));
    }

    #[test]
    fn test_tokens_per_second() {
        let metrics = Metrics::new();

        metrics.set_tokens_per_second(42.5);
        let export = metrics.export();
        assert!(export.contains("rocmforge_tokens_per_second 42.5"));

        metrics.set_tokens_per_second(100.0);
        let export = metrics.export();
        assert!(export.contains("rocmforge_tokens_per_second 100"));
    }

    #[test]
    fn test_all_request_counters() {
        let metrics = Metrics::new();

        metrics.record_request_start();
        metrics.record_request_complete(5);
        metrics.record_request_failed();
        metrics.record_request_cancelled();

        let export = metrics.export();

        // Check that all counters are present in the export
        assert!(export.contains("rocmforge_requests_started_total"));
        assert!(export.contains("rocmforge_requests_completed_total"));
        assert!(export.contains("rocmforge_requests_failed_total"));
        assert!(export.contains("rocmforge_requests_cancelled_total"));
    }

    #[test]
    fn test_metrics_prometheus_format() {
        let metrics = Metrics::new();

        metrics.set_queue_length(2);
        metrics.record_request_start();
        metrics.record_prefill_duration(0.123);
        metrics.record_ttft(0.089);

        let export = metrics.export();

        // Check Prometheus format basics
        assert!(export.contains("# HELP"));
        assert!(export.contains("# TYPE"));
        assert!(export.contains("rocmforge_"));
    }
}
