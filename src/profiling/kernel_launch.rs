//! Kernel launch overhead profiler and optimizer.
//!
//! This module provides infrastructure for measuring and reducing kernel launch overhead.
//! It includes:
//! - Launch overhead measurement (time between kernel launches)
//! - Statistics tracking for optimization decisions
//!
//! # Example
//!
//! ```rust
//! use rocmforge::profiling::kernel_launch::{LaunchOverheadTracker, LaunchOverheadStats};
//! use rocmforge::error::ForgeResult;
//!
//! // Track kernel launch overhead
//! let mut tracker = LaunchOverheadTracker::new();
//! tracker.enable()?;
//!
//! // Measure launch overhead
//! let result = tracker.measure_launch("matmul", || {
//!     my_kernel.launch(&args)
//! });
//!
//! // Get statistics
//! let stats = tracker.get_stats("matmul").unwrap();
//! println!("Average overhead: {:.2} us", stats.avg_overhead_us);
//! # Ok::<(), rocmforge::error::RocmForgeError>(())
//! ```

use std::time::Instant;
use std::sync::Mutex;
use std::collections::HashMap;

use crate::error::{ForgeResult, RocmForgeError};

/// Statistics about kernel launch overhead
#[derive(Debug, Clone)]
pub struct LaunchOverheadStats {
    /// Name of the kernel or operation
    pub kernel_name: String,
    /// Number of launches measured
    pub launch_count: u64,
    /// Total overhead time across all launches (microseconds)
    pub total_overhead_us: u64,
    /// Average overhead per launch (microseconds)
    pub avg_overhead_us: f64,
    /// Minimum overhead observed (microseconds)
    pub min_overhead_us: u64,
    /// Maximum overhead observed (microseconds)
    pub max_overhead_us: u64,
    /// Standard deviation of overhead (microseconds)
    pub std_dev_us: f64,
}

impl LaunchOverheadStats {
    /// Calculate overhead percentage relative to total execution time
    pub fn overhead_percentage(&self, total_time_us: u64) -> f64 {
        if total_time_us == 0 {
            0.0
        } else {
            (self.total_overhead_us as f64 / total_time_us as f64) * 100.0
        }
    }

    /// Check if this kernel has high overhead (>100us average)
    pub fn has_high_overhead(&self) -> bool {
        self.avg_overhead_us > 100.0
    }

    /// Check if overhead variance is high (std_dev > avg * 0.5)
    pub fn has_high_variance(&self) -> bool {
        self.std_dev_us > self.avg_overhead_us * 0.5
    }
}

/// Tracks kernel launch overhead across multiple launches
#[derive(Debug)]
pub struct LaunchOverheadTracker {
    /// Per-kernel overhead statistics
    stats: HashMap<String, OverheadAccumulator>,
    /// Whether overhead tracking is enabled
    enabled: Mutex<bool>,
}

#[derive(Debug)]
struct OverheadAccumulator {
    count: u64,
    total_us: u64,
    min_us: u64,
    max_us: u64,
    values: Vec<u64>, // For std dev calculation
}

impl OverheadAccumulator {
    fn new() -> Self {
        Self {
            count: 0,
            total_us: 0,
            min_us: u64::MAX,
            max_us: 0,
            values: Vec::with_capacity(1000),
        }
    }

    fn add(&mut self, overhead_us: u64) {
        self.count += 1;
        self.total_us += overhead_us;
        self.min_us = self.min_us.min(overhead_us);
        self.max_us = self.max_us.max(overhead_us);

        // Keep last 1000 samples for std dev
        if self.values.len() >= 1000 {
            self.values.remove(0);
        }
        self.values.push(overhead_us);
    }

    fn avg(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_us as f64 / self.count as f64
        }
    }

    fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            0.0
        } else {
            let avg = self.avg();
            let variance = self.values.iter()
                .map(|&v| {
                    let diff = v as f64 - avg;
                    diff * diff
                })
                .sum::<f64>() / self.values.len() as f64;
            variance.sqrt()
        }
    }
}

impl LaunchOverheadTracker {
    /// Create a new launch overhead tracker
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            enabled: Mutex::new(false),
        }
    }

    /// Enable overhead tracking
    pub fn enable(&self) -> ForgeResult<()> {
        let mut enabled = self.enabled.lock().map_err(|e| {
            RocmForgeError::LockPoisoned(format!("Failed to acquire lock in enable(): {}", e))
        })?;
        *enabled = true;
        Ok(())
    }

    /// Disable overhead tracking
    pub fn disable(&self) -> ForgeResult<()> {
        let mut enabled = self.enabled.lock().map_err(|e| {
            RocmForgeError::LockPoisoned(format!("Failed to acquire lock in disable(): {}", e))
        })?;
        *enabled = false;
        Ok(())
    }

    /// Check if tracking is enabled
    ///
    /// Returns false on lock failure (graceful degradation)
    pub fn is_enabled(&self) -> bool {
        self.enabled
            .lock()
            .map(|guard| *guard)
            .unwrap_or(false)
    }

    /// Measure launch overhead for a kernel launch
    ///
    /// This measures the CPU-side time spent preparing and launching the kernel,
    /// excluding the actual GPU execution time.
    ///
    /// # Arguments
    ///
    /// * `kernel_name` - Name of the kernel being launched
    /// * `launch_fn` - Function that performs the kernel launch
    ///
    /// # Returns
    ///
    /// The result of the launch function
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = tracker.measure_launch("my_kernel", || {
    ///     backend.launch_kernel(&kernel, &args)
    /// });
    /// ```
    pub fn measure_launch<F, R>(&mut self, kernel_name: &str, launch_fn: F) -> R
    where
        F: FnOnce() -> R,
    {
        let enabled = self.is_enabled();
        let start = if enabled {
            Some(Instant::now())
        } else {
            None
        };

        // Execute the launch
        let result = launch_fn();

        // Record overhead if enabled
        if let Some(start_time) = start {
            let overhead = start_time.elapsed().as_micros() as u64;
            self.record_overhead(kernel_name, overhead);
        }

        result
    }

    /// Record overhead for a kernel launch
    fn record_overhead(&mut self, kernel_name: &str, overhead_us: u64) {
        let accumulator = self.stats
            .entry(kernel_name.to_string())
            .or_insert_with(OverheadAccumulator::new);
        accumulator.add(overhead_us);
    }

    /// Get statistics for a specific kernel
    pub fn get_stats(&self, kernel_name: &str) -> Option<LaunchOverheadStats> {
        let acc = self.stats.get(kernel_name)?;
        Some(LaunchOverheadStats {
            kernel_name: kernel_name.to_string(),
            launch_count: acc.count,
            total_overhead_us: acc.total_us,
            avg_overhead_us: acc.avg(),
            min_overhead_us: acc.min_us,
            max_overhead_us: acc.max_us,
            std_dev_us: acc.std_dev(),
        })
    }

    /// Get statistics for all tracked kernels
    ///
    /// Returns stats sorted by average overhead (descending)
    pub fn get_all_stats(&self) -> Vec<LaunchOverheadStats> {
        let mut stats: Vec<_> = self.stats.iter()
            .map(|(name, acc)| LaunchOverheadStats {
                kernel_name: name.clone(),
                launch_count: acc.count,
                total_overhead_us: acc.total_us,
                avg_overhead_us: acc.avg(),
                min_overhead_us: acc.min_us,
                max_overhead_us: acc.max_us,
                std_dev_us: acc.std_dev(),
            })
            .collect();

        // Sort by average overhead (descending)
        stats.sort_by(|a, b| b.avg_overhead_us.partial_cmp(&a.avg_overhead_us).unwrap_or(std::cmp::Ordering::Equal));
        stats
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.stats.clear();
    }

    /// Print a summary of tracked overhead
    pub fn print_summary(&self) {
        let stats = self.get_all_stats();
        if stats.is_empty() {
            println!("No kernel launch overhead statistics collected.");
            return;
        }

        println!("=== Kernel Launch Overhead Summary ===");
        println!("{:<30} {:>10} {:>12} {:>12} {:>12} {:>12}",
            "Kernel", "Launches", "Avg (us)", "Min (us)", "Max (us)", "StdDev (us)");
        println!("{}", "-".repeat(90));

        for stat in stats {
            let flag = if stat.has_high_overhead() { " [HIGH]" } else { "" };
            println!("{:<30} {:>10} {:>12.2} {:>12} {:>12} {:>12.2}{}",
                stat.kernel_name,
                stat.launch_count,
                stat.avg_overhead_us,
                stat.min_overhead_us,
                stat.max_overhead_us,
                stat.std_dev_us,
                flag
            );
        }
    }

    /// Get kernels with high overhead (>100us average)
    pub fn get_high_overhead_kernels(&self) -> Vec<String> {
        self.get_all_stats()
            .into_iter()
            .filter(|s| s.has_high_overhead())
            .map(|s| s.kernel_name)
            .collect()
    }

    /// Get total overhead across all tracked launches
    pub fn get_total_overhead_us(&self) -> u64 {
        self.stats.values()
            .map(|acc| acc.total_us)
            .sum()
    }
}

impl Default for LaunchOverheadTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for kernel batching
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of kernels to batch together
    pub max_batch_size: usize,
    /// Maximum time to wait before flushing the batch (microseconds)
    pub max_batch_delay_us: u64,
    /// Minimum kernel execution time to consider for batching (microseconds)
    /// Kernels faster than this are good candidates for batching
    pub min_kernel_time_us: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10,
            max_batch_delay_us: 100, // 100 microseconds
            min_kernel_time_us: 50,  // 50 microseconds
        }
    }
}

impl BatchConfig {
    /// Create a new batch configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set maximum batch delay
    pub fn with_max_batch_delay_us(mut self, delay_us: u64) -> Self {
        self.max_batch_delay_us = delay_us;
        self
    }

    /// Set minimum kernel time for batching consideration
    pub fn with_min_kernel_time_us(mut self, time_us: u64) -> Self {
        self.min_kernel_time_us = time_us;
        self
    }
}

/// Recommendations for reducing kernel launch overhead
#[derive(Debug, Clone)]
pub struct OverheadOptimizationRecommendation {
    /// The kernel name
    pub kernel_name: String,
    /// Current average overhead in microseconds
    pub current_overhead_us: f64,
    /// Recommendation type
    pub recommendation: RecommendationType,
}

/// Type of optimization recommendation
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationType {
    /// Use deferred synchronization (launch multiple kernels before syncing)
    DeferSynchronization,
    /// Batch small operations together
    BatchOperations,
    /// Use HIP Graph for kernel sequences
    UseHipGraph,
    /// No significant overhead - no action needed
    NoAction,
}

impl LaunchOverheadTracker {
    /// Generate optimization recommendations based on collected data
    pub fn get_recommendations(&self) -> Vec<OverheadOptimizationRecommendation> {
        let stats = self.get_all_stats();
        let mut recommendations = Vec::new();

        for stat in stats {
            let recommendation = if stat.avg_overhead_us > 200.0 {
                // Very high overhead - consider batching and HIP Graph
                OverheadOptimizationRecommendation {
                    kernel_name: stat.kernel_name.clone(),
                    current_overhead_us: stat.avg_overhead_us,
                    recommendation: RecommendationType::UseHipGraph,
                }
            } else if stat.avg_overhead_us > 100.0 {
                // High overhead - consider batching
                OverheadOptimizationRecommendation {
                    kernel_name: stat.kernel_name.clone(),
                    current_overhead_us: stat.avg_overhead_us,
                    recommendation: RecommendationType::BatchOperations,
                }
            } else if stat.avg_overhead_us > 50.0 {
                // Moderate overhead - defer synchronization
                OverheadOptimizationRecommendation {
                    kernel_name: stat.kernel_name.clone(),
                    current_overhead_us: stat.avg_overhead_us,
                    recommendation: RecommendationType::DeferSynchronization,
                }
            } else {
                // Low overhead - no action needed
                OverheadOptimizationRecommendation {
                    kernel_name: stat.kernel_name.clone(),
                    current_overhead_us: stat.avg_overhead_us,
                    recommendation: RecommendationType::NoAction,
                }
            };

            recommendations.push(recommendation);
        }

        recommendations
    }

    /// Print optimization recommendations
    pub fn print_recommendations(&self) {
        let recommendations = self.get_recommendations();

        if recommendations.is_empty() {
            println!("No kernel launch data available for recommendations.");
            return;
        }

        println!("=== Kernel Launch Overhead Optimization Recommendations ===");
        println!("{:<30} {:>15} {:>40}", "Kernel", "Overhead (us)", "Recommendation");
        println!("{}", "-".repeat(90));

        for rec in recommendations {
            let rec_str = match rec.recommendation {
                RecommendationType::DeferSynchronization => "Defer synchronization",
                RecommendationType::BatchOperations => "Batch small operations",
                RecommendationType::UseHipGraph => "Use HIP Graph for sequence",
                RecommendationType::NoAction => "No action needed",
            };
            println!("{:<30} {:>15.2} {:>40}",
                rec.kernel_name,
                rec.current_overhead_us,
                rec_str
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_overhead_tracker_creation() {
        let tracker = LaunchOverheadTracker::new();
        assert!(!tracker.is_enabled());

        tracker.enable().unwrap();
        assert!(tracker.is_enabled());

        tracker.disable().unwrap();
        assert!(!tracker.is_enabled());
    }

    #[test]
    fn test_launch_overhead_measurement() {
        let mut tracker = LaunchOverheadTracker::new();
        tracker.enable().unwrap();

        // Simulate a kernel launch with overhead
        tracker.measure_launch("test_kernel", || {
            std::thread::sleep(Duration::from_micros(10));
        });

        let stats = tracker.get_stats("test_kernel");
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.kernel_name, "test_kernel");
        assert_eq!(stats.launch_count, 1);
        assert!(stats.avg_overhead_us >= 10.0);
    }

    #[test]
    fn test_multiple_launches_tracking() {
        let mut tracker = LaunchOverheadTracker::new();
        tracker.enable().unwrap();

        for _ in 0..10 {
            tracker.measure_launch("repeated_kernel", || {
                std::thread::sleep(Duration::from_micros(5));
            });
        }

        let stats = tracker.get_stats("repeated_kernel");
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.launch_count, 10);
        assert!(stats.min_overhead_us <= stats.max_overhead_us);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = LaunchOverheadTracker::new();
        tracker.enable().unwrap();

        tracker.measure_launch("temp_kernel", || ());
        assert!(tracker.get_stats("temp_kernel").is_some());

        tracker.reset();
        assert!(tracker.get_stats("temp_kernel").is_none());
    }

    #[test]
    fn test_get_all_stats() {
        let mut tracker = LaunchOverheadTracker::new();
        tracker.enable().unwrap();

        tracker.measure_launch("kernel_a", || ());
        tracker.measure_launch("kernel_b", || ());

        let all_stats = tracker.get_all_stats();
        assert_eq!(all_stats.len(), 2);
    }

    #[test]
    fn test_overhead_percentage() {
        let stats = LaunchOverheadStats {
            kernel_name: "test".to_string(),
            launch_count: 10,
            total_overhead_us: 100,
            avg_overhead_us: 10.0,
            min_overhead_us: 8,
            max_overhead_us: 15,
            std_dev_us: 2.0,
        };

        // 100us overhead out of 1000us total = 10%
        assert!((stats.overhead_percentage(1000) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_high_overhead_detection() {
        let stats = LaunchOverheadStats {
            kernel_name: "test".to_string(),
            launch_count: 10,
            total_overhead_us: 2000,
            avg_overhead_us: 200.0,
            min_overhead_us: 150,
            max_overhead_us: 250,
            std_dev_us: 20.0,
        };

        assert!(stats.has_high_overhead());
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 10);
        assert_eq!(config.max_batch_delay_us, 100);
        assert_eq!(config.min_kernel_time_us, 50);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new()
            .with_max_batch_size(20)
            .with_max_batch_delay_us(200)
            .with_min_kernel_time_us(100);

        assert_eq!(config.max_batch_size, 20);
        assert_eq!(config.max_batch_delay_us, 200);
        assert_eq!(config.min_kernel_time_us, 100);
    }

    #[test]
    fn test_recommendations() {
        let mut tracker = LaunchOverheadTracker::new();
        tracker.enable().unwrap();

        // Add some simulated data
        for _ in 0..10 {
            tracker.measure_launch("fast_kernel", || {
                std::thread::sleep(Duration::from_micros(10));
            });
        }

        let recommendations = tracker.get_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_get_high_overhead_kernels() {
        let mut tracker = LaunchOverheadTracker::new();
        tracker.enable().unwrap();

        tracker.measure_launch("fast_kernel", || {
            std::thread::sleep(Duration::from_micros(10));
        });

        let high_overhead = tracker.get_high_overhead_kernels();
        assert!(high_overhead.is_empty());
    }
}
