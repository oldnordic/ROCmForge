//! Performance baseline storage and regression detection
//!
//! This module provides utilities for storing, loading, and comparing performance
//! baselines. Baselines are used to track performance over time and detect
//! regressions in critical metrics.
//!
//! # Example
//!
//! ```rust
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

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// Hardware metadata for a baseline
///
/// This information helps determine if baselines are comparable
/// across different systems.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareInfo {
    /// GPU device name (if applicable)
    pub gpu_name: Option<String>,
    /// GPU architecture (e.g., "RDNA3", "RDNA2")
    pub gpu_architecture: Option<String>,
    /// ROCm version (if applicable)
    pub rocm_version: Option<String>,
    /// CPU architecture
    pub cpu_arch: String,
    /// OS information
    pub os: String,
    /// Rust compiler version
    pub rustc_version: String,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        // Get rustc version from build-time environment variable, or runtime, or use a placeholder
        let rustc_version = option_env!("RUSTC_VERSION")
            .map(ToString::to_string)
            .unwrap_or_else(|| {
                std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
            });

        HardwareInfo {
            gpu_name: None,
            gpu_architecture: None,
            rocm_version: None,
            cpu_arch: std::env::consts::ARCH.to_string(),
            os: std::env::consts::OS.to_string(),
            rustc_version,
        }
    }
}

impl HardwareInfo {
    /// Create hardware info with GPU details
    pub fn with_gpu(
        gpu_name: impl Into<String>,
        gpu_architecture: impl Into<String>,
        rocm_version: impl Into<String>,
    ) -> Self {
        let mut info = Self::default();
        info.gpu_name = Some(gpu_name.into());
        info.gpu_architecture = Some(gpu_architecture.into());
        info.rocm_version = Some(rocm_version.into());
        info
    }

    /// Check if this hardware info is compatible with another for comparison
    pub fn is_compatible(&self, other: &HardwareInfo) -> bool {
        // CPU architecture must match
        if self.cpu_arch != other.cpu_arch {
            return false;
        }

        // If both have GPU architecture, they must match
        match (&self.gpu_architecture, &other.gpu_architecture) {
            (Some(arch1), Some(arch2)) if arch1 != arch2 => return false,
            _ => {}
        }

        true
    }
}

/// Performance metrics for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineMetrics {
    /// Average execution time in milliseconds
    pub avg_ms: f64,
    /// Minimum execution time in milliseconds
    pub min_ms: f64,
    /// Maximum execution time in milliseconds
    pub max_ms: f64,
    /// 50th percentile (median) in milliseconds
    pub p50_ms: f64,
    /// 95th percentile in milliseconds
    pub p95_ms: f64,
    /// 99th percentile in milliseconds
    pub p99_ms: f64,
    /// Number of iterations
    pub iterations: usize,
}

impl BaselineMetrics {
    /// Create new metrics from duration samples
    pub fn from_durations(durations: &[std::time::Duration]) -> Self {
        let mut sorted_durations = durations.to_vec();
        sorted_durations.sort();

        let min = sorted_durations.first().unwrap();
        let max = sorted_durations.last().unwrap();
        let total: std::time::Duration = sorted_durations.iter().sum();
        let avg = total / sorted_durations.len() as u32;

        let p50 = &sorted_durations[sorted_durations.len() / 2];
        let p95 = &sorted_durations[(sorted_durations.len() * 95) / 100];
        let p99 = &sorted_durations[(sorted_durations.len() * 99) / 100];

        BaselineMetrics {
            avg_ms: avg.as_secs_f64() * 1000.0,
            min_ms: min.as_secs_f64() * 1000.0,
            max_ms: max.as_secs_f64() * 1000.0,
            p50_ms: p50.as_secs_f64() * 1000.0,
            p95_ms: p95.as_secs_f64() * 1000.0,
            p99_ms: p99.as_secs_f64() * 1000.0,
            iterations: durations.len(),
        }
    }

    /// Calculate throughput in operations per second
    pub fn ops_per_sec(&self) -> f64 {
        if self.avg_ms > 0.0 {
            1000.0 / self.avg_ms
        } else {
            0.0
        }
    }
}

/// Result of comparing current metrics against a baseline
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonResult {
    /// Performance is within acceptable range
    Ok,
    /// Performance has improved (faster than baseline)
    Improved {
        /// Metric name that improved
        metric: String,
        /// Percentage improvement (positive value)
        improvement_pct: f64,
    },
    /// Performance has regressed (slower than baseline)
    Regression {
        /// Metric name that regressed
        metric: String,
        /// Percentage regression (positive value)
        regression_pct: f64,
        /// Baseline value
        baseline_value: f64,
        /// Current value
        current_value: f64,
    },
    /// Hardware mismatch - baselines not comparable
    HardwareMismatch {
        /// Reason for mismatch
        reason: String,
    },
}

impl ComparisonResult {
    /// Check if the comparison indicates a regression
    pub fn is_regression(&self) -> bool {
        matches!(self, ComparisonResult::Regression { .. })
    }

    /// Check if the comparison indicates an improvement
    pub fn is_improved(&self) -> bool {
        matches!(self, ComparisonResult::Improved { .. })
    }

    /// Check if the comparison passed (no regression)
    pub fn passed(&self) -> bool {
        !self.is_regression() && !matches!(self, ComparisonResult::HardwareMismatch { .. })
    }
}

/// A performance baseline for a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Unique identifier for this baseline
    pub name: String,
    /// Timestamp when baseline was created
    pub timestamp: i64,
    /// Hardware information
    pub hardware: HardwareInfo,
    /// Metrics for the benchmark
    pub metrics: BaselineMetrics,
    /// Additional metadata as key-value pairs
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

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
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| BaselineError::SerializationError(e.to_string()))?;

        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| BaselineError::IoError(e.to_string()))?;
        }

        let mut file = std::fs::File::create(path)
            .map_err(|e| BaselineError::IoError(e.to_string()))?;
        file.write_all(json.as_bytes())
            .map_err(|e| BaselineError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load a baseline from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BaselineError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| BaselineError::IoError(format!("Failed to open baseline file: {}", e)))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| BaselineError::IoError(e.to_string()))?;

        serde_json::from_str(&contents)
            .map_err(|e| BaselineError::SerializationError(format!("Invalid baseline JSON: {}", e)))
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

/// Collection of baselines for multiple benchmarks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BaselineCollection {
    /// Map of benchmark name to baseline
    pub baselines: HashMap<String, PerformanceBaseline>,
    /// Collection metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl BaselineCollection {
    /// Create a new empty collection
    pub fn new() -> Self {
        BaselineCollection {
            baselines: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a baseline to the collection
    pub fn add(&mut self, baseline: PerformanceBaseline) {
        self.baselines.insert(baseline.name.clone(), baseline);
    }

    /// Get a baseline by name
    pub fn get(&self, name: &str) -> Option<&PerformanceBaseline> {
        self.baselines.get(name)
    }

    /// Save the collection to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), BaselineError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| BaselineError::SerializationError(e.to_string()))?;

        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| BaselineError::IoError(e.to_string()))?;
        }

        let mut file = std::fs::File::create(path)
            .map_err(|e| BaselineError::IoError(e.to_string()))?;
        file.write_all(json.as_bytes())
            .map_err(|e| BaselineError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load a collection from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BaselineError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| BaselineError::IoError(format!("Failed to open collection file: {}", e)))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| BaselineError::IoError(e.to_string()))?;

        serde_json::from_str(&contents)
            .map_err(|e| BaselineError::SerializationError(format!("Invalid collection JSON: {}", e)))
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
}

/// Errors that can occur when working with baselines
#[derive(Debug, Clone, PartialEq)]
pub enum BaselineError {
    /// File I/O error
    IoError(String),
    /// JSON serialization/deserialization error
    SerializationError(String),
    /// Validation error
    ValidationError(String),
}

impl std::fmt::Display for BaselineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BaselineError::IoError(msg) => write!(f, "I/O error: {}", msg),
            BaselineError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            BaselineError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for BaselineError {}

/// Helper function to convert timestamp to readable date
fn chrono_from_timestamp(secs: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    if let Some(d) = UNIX_EPOCH.checked_add(Duration::from_secs(secs as u64)) {
        // Simple format - could use chrono crate for better formatting
        let datetime = format!("{:?}", d);
        // Extract the date part from debug output
        datetime
    } else {
        "Invalid timestamp".to_string()
    }
}

/// Regression threshold configuration
#[derive(Debug, Clone, Copy)]
pub struct RegressionThreshold {
    /// Percentage threshold for average time (default 10%)
    pub avg_threshold_pct: f64,
    /// Percentage threshold for p95 time (default 15%)
    pub p95_threshold_pct: f64,
    /// Percentage threshold for p99 time (default 20%)
    pub p99_threshold_pct: f64,
}

impl Default for RegressionThreshold {
    fn default() -> Self {
        RegressionThreshold {
            avg_threshold_pct: 0.10,  // 10%
            p95_threshold_pct: 0.15, // 15%
            p99_threshold_pct: 0.20, // 20%
        }
    }
}

impl RegressionThreshold {
    /// Create a new threshold with custom values
    pub fn new(avg_pct: f64, p95_pct: f64, p99_pct: f64) -> Self {
        RegressionThreshold {
            avg_threshold_pct: avg_pct,
            p95_threshold_pct: p95_pct,
            p99_threshold_pct: p99_pct,
        }
    }

    /// Check if a value exceeds the threshold for a given metric
    pub fn exceeds_threshold(&self, metric: &str, diff_pct: f64) -> bool {
        let threshold = match metric {
            "avg_ms" => self.avg_threshold_pct,
            "p95_ms" => self.p95_threshold_pct,
            "p99_ms" => self.p99_threshold_pct,
            _ => self.avg_threshold_pct,
        };
        diff_pct > threshold
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
    fn test_baseline_metrics_from_durations() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(13),
            Duration::from_millis(10),
        ];

        let metrics = BaselineMetrics::from_durations(&durations);

        assert_eq!(metrics.iterations, 5);
        assert_eq!(metrics.min_ms, 10.0);
        assert_eq!(metrics.max_ms, 13.0);
        assert!(metrics.avg_ms > 10.0 && metrics.avg_ms < 13.0);
    }

    #[test]
    fn test_baseline_metrics_ops_per_sec() {
        let metrics = create_test_metrics(10.0);
        assert!((metrics.ops_per_sec() - 100.0).abs() < 0.1);

        let metrics_slow = create_test_metrics(100.0);
        assert!((metrics_slow.ops_per_sec() - 10.0).abs() < 0.1);
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

    #[test]
    fn test_hardware_info_compatibility() {
        let info1 = HardwareInfo::default();
        let info2 = HardwareInfo::default();
        assert!(info1.is_compatible(&info2));

        let gpu_info = HardwareInfo::with_gpu("GPU", "RDNA3", "6.0");
        assert!(info1.is_compatible(&gpu_info)); // Different GPU arch is compatible
        assert!(gpu_info.is_compatible(&info1));
    }
}
