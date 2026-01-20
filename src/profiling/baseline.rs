//! Performance baseline storage and regression detection
//!
//! This module provides utilities for storing, loading, and comparing performance
//! baselines. Baselines are used to track performance over time and detect
//! regressions in critical metrics.
//!
//! # Example
//!
//! ```rust,ignore
//! use rocmforge::profiling::baseline::{PerformanceBaseline, BaselineMetrics};
//!
//! // Create a new baseline
//! let baseline = PerformanceBaseline::new("my_benchmark", BaselineMetrics {
//!     avg_ms: 10.5,
//!     min_ms: 9.8,
//!     max_ms: 11.2,
//!     p50_ms: 10.3,
//!     p95_ms: 10.9,
//!     p99_ms: 11.1,
//!     iterations: 100,
//! });
//!
//! // Save to file
//! baseline.save("baselines/my_benchmark.json").unwrap();
//!
//! // Load and compare
//! let loaded = PerformanceBaseline::load("baselines/my_benchmark.json").unwrap();
//! let current = BaselineMetrics { avg_ms: 11.6, /* ... */ };
//! let comparison = loaded.compare_metrics("current", &current, 0.10); // 10% threshold
//! ```

pub use super::baseline_types::*;
pub use super::baseline_storage::{save_baseline, load_baseline, save_collection, load_collection, chrono_from_timestamp};

use std::collections::HashMap;
use std::path::Path;

impl PerformanceBaseline {
    /// Create a new performance baseline
    pub fn new(name: impl Into<String>, metrics: BaselineMetrics) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        PerformanceBaseline {
            name: name.into(),
            timestamp: now,
            hardware: HardwareInfo::default(),
            metrics,
            metadata: HashMap::new(),
        }
    }

    /// Set the hardware info for this baseline
    pub fn with_hardware(mut self, hardware: HardwareInfo) -> Self {
        self.hardware = hardware;
        self
    }

    /// Add metadata to this baseline
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Save the baseline to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), BaselineError> {
        save_baseline(self, path)
    }

    /// Load a baseline from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BaselineError> {
        load_baseline(path)
    }

    /// Compare current metrics against this baseline
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the current run
    /// * `current` - Current metrics to compare
    /// * `threshold_pct` - Acceptable regression threshold (e.g., 0.10 for 10%)
    ///
    /// # Returns
    ///
    /// A `ComparisonResult` indicating if performance is acceptable
    pub fn compare_metrics(
        &self,
        name: &str,
        current: &BaselineMetrics,
        threshold_pct: f64,
    ) -> ComparisonResult {
        // Check hardware compatibility first
        if !self.hardware.is_compatible(&HardwareInfo::default()) {
            return ComparisonResult::HardwareMismatch {
                reason: format!(
                    "Baseline hardware ({:?}) != current hardware ({:?})",
                    self.hardware, HardwareInfo::default()
                ),
            };
        }

        // Compare average time (primary metric)
        let avg_diff_pct = (current.avg_ms - self.metrics.avg_ms) / self.metrics.avg_ms;

        if avg_diff_pct > threshold_pct {
            // Regression detected
            return ComparisonResult::Regression {
                metric: format!("{}.avg_ms", name),
                regression_pct: avg_diff_pct * 100.0,
                baseline_value: self.metrics.avg_ms,
                current_value: current.avg_ms,
            };
        }

        if avg_diff_pct < -threshold_pct {
            // Performance improved
            return ComparisonResult::Improved {
                metric: format!("{}.avg_ms", name),
                improvement_pct: -avg_diff_pct * 100.0,
            };
        }

        // Check other critical metrics
        for (metric_name, baseline_val, current_val) in [
            ("p95_ms", self.metrics.p95_ms, current.p95_ms),
            ("p99_ms", self.metrics.p99_ms, current.p99_ms),
        ] {
            let diff_pct = (current_val - baseline_val) / baseline_val;
            if diff_pct > threshold_pct {
                return ComparisonResult::Regression {
                    metric: format!("{}.{}", name, metric_name),
                    regression_pct: diff_pct * 100.0,
                    baseline_value: baseline_val,
                    current_value: current_val,
                };
            }
        }

        ComparisonResult::Ok
    }

    /// Get a human-readable summary of this baseline
    pub fn summary(&self) -> String {
        format!(
            "Baseline '{}':\n  Created: {}\n  Hardware: {:?}\n  Avg: {:.3} ms ({:.2} ops/sec)\n  Min: {:.3} ms\n  Max: {:.3} ms\n  P50: {:.3} ms\n  P95: {:.3} ms\n  P99: {:.3} ms\n  Iterations: {}",
            self.name,
            chrono_from_timestamp(self.timestamp),
            self.hardware,
            self.metrics.avg_ms,
            self.metrics.ops_per_sec(),
            self.metrics.min_ms,
            self.metrics.max_ms,
            self.metrics.p50_ms,
            self.metrics.p95_ms,
            self.metrics.p99_ms,
            self.metrics.iterations
        )
    }
}

impl BaselineCollection {
    /// Create a new empty collection
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        BaselineCollection {
            baselines: HashMap::new(),
            metadata: HashMap::new(),
            hardware: HardwareInfo::default(),
            timestamp: now,
        }
    }

    /// Create a new collection with specific hardware info
    pub fn with_hardware(hardware: HardwareInfo) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        BaselineCollection {
            baselines: HashMap::new(),
            metadata: HashMap::new(),
            hardware,
            timestamp: now,
        }
    }

    /// Add a baseline to the collection
    pub fn add(&mut self, baseline: PerformanceBaseline) {
        self.baselines.insert(baseline.name.clone(), baseline);
    }

    /// Add or update a baseline in the collection
    pub fn upsert(&mut self, baseline: PerformanceBaseline) {
        self.baselines.insert(baseline.name.clone(), baseline);
    }

    /// Get a baseline by name
    pub fn get(&self, name: &str) -> Option<&PerformanceBaseline> {
        self.baselines.get(name)
    }

    /// Remove a baseline from the collection
    pub fn remove(&mut self, name: &str) -> Option<PerformanceBaseline> {
        self.baselines.remove(name)
    }

    /// Get the number of baselines in the collection
    pub fn len(&self) -> usize {
        self.baselines.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.baselines.is_empty()
    }

    /// Get all baseline names
    pub fn names(&self) -> Vec<String> {
        self.baselines.keys().cloned().collect()
    }

    /// Merge another collection into this one
    ///
    /// Existing baselines are replaced with ones from `other`
    pub fn merge(&mut self, other: BaselineCollection) {
        for (name, baseline) in other.baselines {
            self.baselines.insert(name, baseline);
        }
        // Keep the newer timestamp
        if other.timestamp > self.timestamp {
            self.timestamp = other.timestamp;
        }
    }

    /// Filter baselines by a predicate
    pub fn filter<F>(&self, predicate: F) -> BaselineCollection
    where
        F: Fn(&PerformanceBaseline) -> bool,
    {
        let mut filtered = BaselineCollection {
            baselines: HashMap::new(),
            metadata: self.metadata.clone(),
            hardware: self.hardware.clone(),
            timestamp: self.timestamp,
        };

        for (name, baseline) in &self.baselines {
            if predicate(baseline) {
                filtered.baselines.insert(name.clone(), baseline.clone());
            }
        }

        filtered
    }

    /// Get baselines by metadata key-value pair
    pub fn find_by_metadata(&self, key: &str, value: &str) -> Vec<&PerformanceBaseline> {
        self.baselines
            .values()
            .filter(|b| b.metadata.get(key).map(|v| v == value).unwrap_or(false))
            .collect()
    }

    /// Save the collection to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), BaselineError> {
        save_collection(self, path)
    }

    /// Load a collection from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BaselineError> {
        load_collection(path)
    }

    /// Compare current metrics against all baselines in the collection
    ///
    /// Returns a map of benchmark name to comparison result
    pub fn compare_all(
        &self,
        current: &HashMap<String, BaselineMetrics>,
        threshold_pct: f64,
    ) -> HashMap<String, ComparisonResult> {
        let mut results = HashMap::new();

        for (name, baseline) in &self.baselines {
            if let Some(current_metrics) = current.get(name) {
                let result = baseline.compare_metrics(name, current_metrics, threshold_pct);
                results.insert(name.clone(), result);
            }
        }

        results
    }

    /// Compare current metrics against all baselines and generate a report
    ///
    /// Returns a `RegressionReport` with detailed results
    pub fn compare_and_report(
        &self,
        current: &HashMap<String, BaselineMetrics>,
        threshold_pct: f64,
    ) -> RegressionReport {
        let results = self.compare_all(current, threshold_pct);
        RegressionReport::from_results(results)
    }

    /// Compare a single benchmark against its baseline
    pub fn compare_one(
        &self,
        name: &str,
        current: &BaselineMetrics,
        threshold_pct: f64,
    ) -> Option<ComparisonResult> {
        self.baselines.get(name)
            .map(|baseline| baseline.compare_metrics(name, current, threshold_pct))
    }

    /// Check hardware compatibility for this collection
    pub fn check_hardware_compatibility(&self) -> Result<(), ComparisonResult> {
        let current_hardware = HardwareInfo::default();
        if !self.hardware.is_compatible(&current_hardware) {
            return Err(ComparisonResult::HardwareMismatch {
                reason: format!(
                    "Collection hardware ({:?}) != current hardware ({:?})",
                    self.hardware, current_hardware
                ),
            });
        }
        Ok(())
    }
}

/// Helper functions for benchmarks to create and save baselines
pub struct BenchmarkBaseline {
    /// Collection being built
    collection: BaselineCollection,
    /// Default hardware info
    hardware: HardwareInfo,
    /// Threshold for comparisons
    threshold: f64,
}

impl BenchmarkBaseline {
    /// Create a new benchmark baseline helper
    pub fn new() -> Self {
        BenchmarkBaseline {
            collection: BaselineCollection::new(),
            hardware: HardwareInfo::default(),
            threshold: 0.10, // 10% default threshold
        }
    }

    /// Create a new benchmark baseline helper with specific hardware info
    pub fn with_hardware(hardware: HardwareInfo) -> Self {
        let mut collection = BaselineCollection::with_hardware(hardware.clone());
        collection.metadata.insert("created_by".to_string(), "ROCmForge benchmark".to_string());
        BenchmarkBaseline {
            collection,
            hardware,
            threshold: 0.10,
        }
    }

    /// Set the regression threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Add a benchmark result to the collection
    ///
    /// # Arguments
    ///
    /// * `name` - Benchmark name
    /// * `durations` - Array of duration measurements in milliseconds
    pub fn add_benchmark(&mut self, name: impl Into<String>, durations_ms: &[f64]) {
        let metrics = BaselineMetrics::from_ms_durations(durations_ms);
        let baseline = PerformanceBaseline::new(name, metrics)
            .with_hardware(self.hardware.clone());
        self.collection.add(baseline);
    }

    /// Add a benchmark result with metadata
    pub fn add_benchmark_with_metadata(
        &mut self,
        name: impl Into<String>,
        durations_ms: &[f64],
        metadata: HashMap<String, String>,
    ) {
        let metrics = BaselineMetrics::from_ms_durations(durations_ms);
        let mut baseline = PerformanceBaseline::new(name, metrics)
            .with_hardware(self.hardware.clone());
        baseline.metadata = metadata;
        self.collection.add(baseline);
    }

    /// Add metrics directly to the collection
    pub fn add_metrics(&mut self, name: impl Into<String>, metrics: BaselineMetrics) {
        let baseline = PerformanceBaseline::new(name, metrics)
            .with_hardware(self.hardware.clone());
        self.collection.add(baseline);
    }

    /// Save the collection to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), BaselineError> {
        self.collection.save(path)
    }

    /// Get the collection
    pub fn collection(&self) -> &BaselineCollection {
        &self.collection
    }

    /// Mutably borrow the collection
    pub fn collection_mut(&mut self) -> &mut BaselineCollection {
        &mut self.collection
    }

    /// Compare current run against the baselines
    pub fn compare(&self, current: &HashMap<String, BaselineMetrics>) -> RegressionReport {
        self.collection.compare_and_report(current, self.threshold)
    }
}

impl Default for BenchmarkBaseline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_metrics(avg_ms: f64) -> BaselineMetrics {
        BaselineMetrics {
            avg_ms,
            min_ms: avg_ms * 0.9,
            max_ms: avg_ms * 1.1,
            p50_ms: avg_ms * 0.95,
            p95_ms: avg_ms * 1.05,
            p99_ms: avg_ms * 1.08,
            iterations: 100,
        }
    }

    #[test]
    fn test_performance_baseline_creation() {
        let metrics = create_test_metrics(10.0);
        let baseline = PerformanceBaseline::new("test_benchmark", metrics.clone());

        assert_eq!(baseline.name, "test_benchmark");
        assert_eq!(baseline.metrics, metrics);
        assert!(baseline.timestamp > 0);
    }

    #[test]
    fn test_performance_baseline_with_hardware() {
        let metrics = create_test_metrics(10.0);
        let hardware = HardwareInfo::with_gpu("RX 7900 XTX", "RDNA3", "6.0.0");
        let baseline = PerformanceBaseline::new("test", metrics)
            .with_hardware(hardware);

        assert_eq!(baseline.hardware.gpu_name, Some("RX 7900 XTX".to_string()));
        assert_eq!(baseline.hardware.gpu_architecture, Some("RDNA3".to_string()));
        assert_eq!(baseline.hardware.rocm_version, Some("6.0.0".to_string()));
    }

    #[test]
    fn test_performance_baseline_with_metadata() {
        let metrics = create_test_metrics(10.0);
        let baseline = PerformanceBaseline::new("test", metrics)
            .with_metadata("model_size", "7B")
            .with_metadata("quantization", "Q4_K_M");

        assert_eq!(baseline.metadata.get("model_size"), Some(&"7B".to_string()));
        assert_eq!(baseline.metadata.get("quantization"), Some(&"Q4_K_M".to_string()));
    }

    #[test]
    fn test_comparison_result_passing() {
        // Within threshold - should pass
        let baseline = PerformanceBaseline::new("test", create_test_metrics(10.0));
        let current = create_test_metrics(10.5); // 5% slower, under 10% threshold
        let result = baseline.compare_metrics("test", &current, 0.10);

        assert!(result.passed());
        assert!(!result.is_regression());
    }

    #[test]
    fn test_comparison_result_regression() {
        // Exceeds threshold - should detect regression
        let baseline = PerformanceBaseline::new("test", create_test_metrics(10.0));
        let current = create_test_metrics(11.5); // 15% slower, over 10% threshold
        let result = baseline.compare_metrics("test", &current, 0.10);

        assert!(!result.passed());
        assert!(result.is_regression());

        match result {
            ComparisonResult::Regression { metric, regression_pct, .. } => {
                assert!(metric.contains("avg_ms"));
                assert!((regression_pct - 15.0).abs() < 1.0);
            }
            _ => panic!("Expected Regression result"),
        }
    }

    #[test]
    fn test_comparison_result_improvement() {
        // Performance improved
        let baseline = PerformanceBaseline::new("test", create_test_metrics(10.0));
        let current = create_test_metrics(8.5); // 15% faster
        let result = baseline.compare_metrics("test", &current, 0.10);

        assert!(result.passed());
        assert!(result.is_improved());

        match result {
            ComparisonResult::Improved { improvement_pct, .. } => {
                assert!((improvement_pct - 15.0).abs() < 1.0);
            }
            _ => panic!("Expected Improved result"),
        }
    }

    // Tests for BaselineCollection
    #[test]
    fn test_baseline_collection_new() {
        let collection = BaselineCollection::new();
        assert!(collection.is_empty());
        assert_eq!(collection.len(), 0);
        assert!(collection.hardware.gpu_architecture.is_none());
    }

    #[test]
    fn test_baseline_collection_add() {
        let mut collection = BaselineCollection::new();
        collection.add(PerformanceBaseline::new("bench1", create_test_metrics(10.0)));
        collection.add(PerformanceBaseline::new("bench2", create_test_metrics(20.0)));

        assert_eq!(collection.len(), 2);
        assert!(collection.get("bench1").is_some());
        assert!(collection.get("bench2").is_some());
    }

    #[test]
    fn test_baseline_collection_names() {
        let mut collection = BaselineCollection::new();
        collection.add(PerformanceBaseline::new("bench1", create_test_metrics(10.0)));
        collection.add(PerformanceBaseline::new("bench2", create_test_metrics(20.0)));

        let mut names = collection.names();
        names.sort();
        assert_eq!(names, vec!["bench1", "bench2"]);
    }

    #[test]
    fn test_baseline_collection_remove() {
        let mut collection = BaselineCollection::new();
        collection.add(PerformanceBaseline::new("bench1", create_test_metrics(10.0)));
        collection.add(PerformanceBaseline::new("bench2", create_test_metrics(20.0)));

        assert_eq!(collection.len(), 2);
        collection.remove("bench1");
        assert_eq!(collection.len(), 1);
        assert!(collection.get("bench1").is_none());
        assert!(collection.get("bench2").is_some());
    }

    #[test]
    fn test_baseline_collection_merge() {
        let mut collection1 = BaselineCollection::new();
        collection1.add(PerformanceBaseline::new("bench1", create_test_metrics(10.0)));

        let mut collection2 = BaselineCollection::new();
        collection2.add(PerformanceBaseline::new("bench2", create_test_metrics(20.0)));

        collection1.merge(collection2);
        assert_eq!(collection1.len(), 2);
        assert!(collection1.get("bench1").is_some());
        assert!(collection1.get("bench2").is_some());
    }

    #[test]
    fn test_baseline_collection_find_by_metadata() {
        let mut collection = BaselineCollection::new();
        let baseline1 = PerformanceBaseline::new("bench1", create_test_metrics(10.0))
            .with_metadata("format", "Q4_0");
        let baseline2 = PerformanceBaseline::new("bench2", create_test_metrics(20.0))
            .with_metadata("format", "Q8_0");
        collection.add(baseline1);
        collection.add(baseline2);

        let q4_results = collection.find_by_metadata("format", "Q4_0");
        assert_eq!(q4_results.len(), 1);
        assert_eq!(q4_results[0].name, "bench1");

        let q8_results = collection.find_by_metadata("format", "Q8_0");
        assert_eq!(q8_results.len(), 1);
        assert_eq!(q8_results[0].name, "bench2");
    }

    // Tests for BenchmarkBaseline
    #[test]
    fn test_benchmark_baseline_new() {
        let helper = BenchmarkBaseline::new();
        assert!(helper.collection().is_empty());
    }

    #[test]
    fn test_benchmark_baseline_add_benchmark() {
        let mut helper = BenchmarkBaseline::new();
        let durations = vec![10.0, 11.0, 9.0, 10.5, 9.5];
        helper.add_benchmark("test_bench", &durations);

        assert_eq!(helper.collection().len(), 1);
        let baseline = helper.collection().get("test_bench").unwrap();
        assert!((baseline.metrics.avg_ms - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_baseline_with_hardware() {
        let hardware = HardwareInfo::with_gpu("RX 7900 XTX", "RDNA3", "6.0.0");
        let helper = BenchmarkBaseline::with_hardware(hardware.clone());

        assert_eq!(helper.collection().hardware.gpu_name, Some("RX 7900 XTX".to_string()));
    }
}
