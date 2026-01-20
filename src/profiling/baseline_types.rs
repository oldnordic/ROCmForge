//! Types for performance baseline storage and comparison
//!
//! This module contains the core data structures used for performance baseline
//! management, including metrics, hardware information, comparison results, and
//! regression reports.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

    /// Create metrics from an array of durations in milliseconds
    ///
    /// This is a convenience function for benchmarks that measure time directly
    /// in milliseconds (e.g., using `Instant::elapsed().as_secs_f64() * 1000.0`).
    pub fn from_ms_durations(durations_ms: &[f64]) -> Self {
        if durations_ms.is_empty() {
            return BaselineMetrics {
                avg_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                iterations: 0,
            };
        }

        let mut sorted = durations_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted.first().unwrap();
        let max = sorted.last().unwrap();
        let avg = sorted.iter().sum::<f64>() / sorted.len() as f64;

        let p50 = &sorted[sorted.len() / 2];
        let p95 = &sorted[(sorted.len() * 95) / 100];
        let p99 = &sorted[(sorted.len() * 99) / 100];

        BaselineMetrics {
            avg_ms: avg,
            min_ms: *min,
            max_ms: *max,
            p50_ms: *p50,
            p95_ms: *p95,
            p99_ms: *p99,
            iterations: durations_ms.len(),
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

    /// Get a human-readable description of the result
    pub fn description(&self) -> String {
        match self {
            ComparisonResult::Ok => "Performance within acceptable range".to_string(),
            ComparisonResult::Improved { metric, improvement_pct } => {
                format!("Improved: {} is {:.1}% faster than baseline", metric, improvement_pct)
            }
            ComparisonResult::Regression { metric, regression_pct, baseline_value, current_value } => {
                format!("Regression: {} is {:.1}% slower (baseline: {:.3} ms, current: {:.3} ms)",
                    metric, regression_pct, baseline_value, current_value)
            }
            ComparisonResult::HardwareMismatch { reason } => {
                format!("Hardware mismatch: {}", reason)
            }
        }
    }
}

/// Summary report for baseline comparisons
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Number of benchmarks that passed
    pub passed: usize,
    /// Number of benchmarks that regressed
    pub regressed: usize,
    /// Number of benchmarks that improved
    pub improved: usize,
    /// Individual benchmark results
    pub results: Vec<(String, ComparisonResult)>,
    /// Overall pass/fail status
    pub overall_passed: bool,
}

impl RegressionReport {
    /// Create a new regression report from comparison results
    pub fn from_results(results: HashMap<String, ComparisonResult>) -> Self {
        let mut passed = 0;
        let mut regressed = 0;
        let mut improved = 0;
        let mut result_vec = Vec::new();

        for (name, result) in results {
            passed += if result.passed() { 1 } else { 0 };
            regressed += if result.is_regression() { 1 } else { 0 };
            improved += if result.is_improved() { 1 } else { 0 };
            result_vec.push((name, result));
        }

        // Sort by name for consistent output
        result_vec.sort_by(|a, b| a.0.cmp(&b.0));

        let overall_passed = regressed == 0;

        RegressionReport {
            passed,
            regressed,
            improved,
            results: result_vec,
            overall_passed,
        }
    }

    /// Print the report to stdout
    pub fn print(&self) {
        println!("\n=== Baseline Comparison Report ===");
        println!("Total benchmarks: {}", self.results.len());
        println!("Passed: {}", self.passed);
        println!("Improved: {}", self.improved);
        println!("Regressed: {}", self.regressed);
        println!("Overall: {}", if self.overall_passed { "PASS" } else { "FAIL" });

        if self.regressed > 0 {
            println!("\n--- Regressions Detected ---");
            for (name, result) in &self.results {
                if result.is_regression() {
                    println!("  {}: {}", name, result.description());
                }
            }
        }

        if self.improved > 0 {
            println!("\n--- Improvements ---");
            for (name, result) in &self.results {
                if result.is_improved() {
                    println!("  {}: {}", name, result.description());
                }
            }
        }

        println!("====================================\n");
    }

    /// Get the report as a string
    pub fn to_string(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Baseline Comparison Report ===\n");
        output.push_str(&format!("Total benchmarks: {}\n", self.results.len()));
        output.push_str(&format!("Passed: {}\n", self.passed));
        output.push_str(&format!("Improved: {}\n", self.improved));
        output.push_str(&format!("Regressed: {}\n", self.regressed));
        output.push_str(&format!("Overall: {}\n", if self.overall_passed { "PASS" } else { "FAIL" }));

        if self.regressed > 0 {
            output.push_str("\n--- Regressions Detected ---\n");
            for (name, result) in &self.results {
                if result.is_regression() {
                    output.push_str(&format!("  {}: {}\n", name, result.description()));
                }
            }
        }

        if self.improved > 0 {
            output.push_str("\n--- Improvements ---\n");
            for (name, result) in &self.results {
                if result.is_improved() {
                    output.push_str(&format!("  {}: {}\n", name, result.description()));
                }
            }
        }

        output.push_str("====================================\n");
        output
    }

    /// Check if the report indicates any failures
    pub fn has_failures(&self) -> bool {
        !self.overall_passed
    }

    /// Get all failed benchmark names
    pub fn failed_benchmarks(&self) -> Vec<&str> {
        self.results
            .iter()
            .filter(|(_, r)| !r.passed())
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// Get the total number of benchmarks in the report
    pub fn total_benchmarks(&self) -> usize {
        self.results.len()
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

/// Collection of baselines for multiple benchmarks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BaselineCollection {
    /// Map of benchmark name to baseline
    pub baselines: HashMap<String, PerformanceBaseline>,
    /// Collection metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Hardware info for the entire collection
    #[serde(default)]
    pub hardware: HardwareInfo,
    /// Collection creation timestamp
    #[serde(default)]
    pub timestamp: i64,
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
