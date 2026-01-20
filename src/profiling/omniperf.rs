//! ROCm omniperf profiling tool integration
//!
//! This module provides integration with omniperf, the comprehensive profiling
//! and analysis tool for ROCm applications.
//!
//! # Overview
//!
//! [`OmniperfProfileBuilder`] provides a fluent interface for building omniperf
//! profiling commands. The module also includes memory bandwidth analysis utilities
//! that work with omniperf output.
//!
//! # Example
//!
//! ```rust,ignore
//! use rocmforge::profiling::omniperf::OmniperfProfileBuilder;
//!
//! let builder = OmniperfProfileBuilder::new("/tmp/profile")
//!     .target_arch("gfx1100")
//!     .command("my_app")
//!     .arg("--input")
//!     .arg("data.txt")
//!     .mode("detailed");
//!
//! let cmd = builder.build()?;
//! println!("Run: {:?}", cmd);
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

// Re-export common types from the parent module
pub use crate::profiling::types::{ProfilingError, ProfilingResult, ProfilingTool};

// Import for memory bandwidth analysis
use crate::profiling::rocprof::ProfilingResults;

/// Helper for building omniperf profile commands
#[derive(Debug, Clone)]
pub struct OmniperfProfileBuilder {
    /// Output directory
    output_dir: PathBuf,
    /// Target GPU architecture
    target_arch: Option<String>,
    /// Application command
    command: Option<String>,
    /// Command arguments
    args: Vec<String>,
    /// Profile mode (basic, detailed, etc.)
    mode: String,
}

impl OmniperfProfileBuilder {
    /// Create a new omniperf profile builder
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        OmniperfProfileBuilder {
            output_dir: output_dir.as_ref().to_path_buf(),
            target_arch: None,
            command: None,
            args: Vec::new(),
            mode: "basic".to_string(),
        }
    }

    /// Set the target GPU architecture
    pub fn target_arch(mut self, arch: impl Into<String>) -> Self {
        self.target_arch = Some(arch.into());
        self
    }

    /// Set the application to profile
    pub fn command(mut self, cmd: impl Into<String>) -> Self {
        self.command = Some(cmd.into());
        self
    }

    /// Add arguments to the application command
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Set the profiling mode
    pub fn mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = mode.into();
        self
    }

    /// Build the omniperf command
    pub fn build(&self) -> ProfilingResult<Command> {
        let tool = ProfilingTool::Omniperf;
        if !tool.is_available() {
            return Err(ProfilingError::ToolNotFound("omniperf".to_string()));
        }

        let cmd_ref = self.command.as_ref()
            .ok_or_else(|| ProfilingError::InvalidConfig("No command specified".to_string()))?;

        let mut cmd = Command::new("omniperf");
        cmd.arg("profile")
            .arg("-n")
            .arg("rocmforge")
            .arg("-d")
            .arg(&self.output_dir)
            .arg("--mode")
            .arg(&self.mode);

        if let Some(arch) = &self.target_arch {
            cmd.arg("--target").arg(arch);
        }

        cmd.arg("--").arg(cmd_ref).args(&self.args);

        Ok(cmd)
    }
}

/// Memory bandwidth analysis from profiling results
#[derive(Debug, Clone)]
pub struct MemoryBandwidthAnalysis {
    /// Total bytes read from memory
    pub bytes_read: u64,
    /// Total bytes written to memory
    pub bytes_written: u64,
    /// Total bytes transferred
    pub bytes_total: u64,
    /// Execution time in seconds
    pub duration_secs: f64,
    /// Memory bandwidth in GB/s
    pub bandwidth_gbps: f64,
    /// L2 cache hit rate (0.0 to 1.0)
    pub l2_hit_rate: Option<f64>,
    /// Memory stall percentage
    pub stall_pct: Option<f64>,
    /// Theoretical peak bandwidth (for comparison)
    pub peak_bandwidth_gbps: f64,
    /// Bandwidth utilization (0.0 to 1.0)
    pub utilization: f64,
}

impl MemoryBandwidthAnalysis {
    /// Calculate memory bandwidth from operation metadata
    ///
    /// # Arguments
    ///
    /// * `bytes_read` - Total bytes read from memory
    /// * `bytes_written` - Total bytes written to memory
    /// * `duration_secs` - Execution time in seconds
    /// * `peak_bandwidth_gbps` - Theoretical peak bandwidth in GB/s (default: 560 for RX 7900 XT)
    ///
    /// # Example
    ///
    /// ```rust
    /// use rocmforge::profiling::omniperf::MemoryBandwidthAnalysis;
    ///
    /// // Analyze a matmul operation
    /// let analysis = MemoryBandwidthAnalysis::from_operation(
    ///     1024 * 1024 * 1024, // 1 GB read
    ///     512 * 1024 * 1024,  // 512 MB written
    ///     0.01,               // 10ms duration
    ///     560.0,              // 560 GB/s peak
    /// );
    ///
    /// println!("Bandwidth: {:.2} GB/s", analysis.bandwidth_gbps);
    /// println!("Utilization: {:.1}%", analysis.utilization * 100.0);
    /// ```
    pub fn from_operation(
        bytes_read: u64,
        bytes_written: u64,
        duration_secs: f64,
        peak_bandwidth_gbps: f64,
    ) -> Self {
        let bytes_total = bytes_read + bytes_written;
        let bandwidth_gbps = if duration_secs > 0.0 {
            (bytes_total as f64 / 1e9) / duration_secs
        } else {
            0.0
        };
        let utilization = bandwidth_gbps / peak_bandwidth_gbps;

        MemoryBandwidthAnalysis {
            bytes_read,
            bytes_written,
            bytes_total,
            duration_secs,
            bandwidth_gbps,
            l2_hit_rate: None,
            stall_pct: None,
            peak_bandwidth_gbps,
            utilization,
        }
    }

    /// Calculate bandwidth from profiling results
    pub fn from_profiling_results(
        results: &ProfilingResults,
        duration_secs: f64,
        peak_bandwidth_gbps: f64,
    ) -> Self {
        // Estimate bytes transferred from counter data
        // This is approximate - actual measurements require kernel instrumentation

        // Get cache accesses to estimate memory traffic
        let cache_accesses = results.get_counter("TCP_TOTAL_CACHE_ACCESSES").unwrap_or(0.0);
        let cache_misses = results.get_counter("TCP_TOTAL_CACHE_MISSES").unwrap_or(0.0);

        // Estimate: 64 bytes per cache line
        let bytes_read = (cache_misses * 64.0) as u64;
        let bytes_written = bytes_read / 2; // Assume 2:1 read:write ratio

        let l2_hit_rate = if cache_accesses > 0.0 {
            Some((cache_accesses - cache_misses) / cache_accesses)
        } else {
            None
        };

        let bytes_total = bytes_read + bytes_written;
        let bandwidth_gbps = if duration_secs > 0.0 {
            (bytes_total as f64 / 1e9) / duration_secs
        } else {
            0.0
        };

        let utilization = bandwidth_gbps / peak_bandwidth_gbps;

        MemoryBandwidthAnalysis {
            bytes_read,
            bytes_written,
            bytes_total,
            duration_secs,
            bandwidth_gbps,
            l2_hit_rate,
            stall_pct: None,
            peak_bandwidth_gbps,
            utilization,
        }
    }

    /// Format bytes as human readable
    pub fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * 1024;
        const GB: u64 = 1024 * 1024 * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Get a summary report
    pub fn summary(&self) -> String {
        let mut report = String::from("Memory Bandwidth Analysis:\n");
        report.push_str(&format!("  Bytes Read:        {}\n", Self::format_bytes(self.bytes_read)));
        report.push_str(&format!("  Bytes Written:     {}\n", Self::format_bytes(self.bytes_written)));
        report.push_str(&format!("  Total Transfer:    {}\n", Self::format_bytes(self.bytes_total)));
        report.push_str(&format!("  Duration:          {:.3} ms\n", self.duration_secs * 1000.0));
        report.push_str(&format!("  Bandwidth:         {:.2} GB/s\n", self.bandwidth_gbps));
        report.push_str(&format!("  Utilization:       {:.1}%", self.utilization * 100.0));

        if let Some(hit_rate) = self.l2_hit_rate {
            report.push_str(&format!("  L2 Hit Rate:       {:.1}%\n", hit_rate * 100.0));
        }

        if let Some(stall) = self.stall_pct {
            report.push_str(&format!("  Memory Stall:      {:.1}%\n", stall * 100.0));
        }

        report
    }

    /// Check if bandwidth utilization is good (>60%)
    pub fn is_good_utilization(&self) -> bool {
        self.utilization > 0.6
    }

    /// Check if bandwidth utilization is excellent (>80%)
    pub fn is_excellent_utilization(&self) -> bool {
        self.utilization > 0.8
    }

    /// Get bottleneck description
    pub fn bottleneck_description(&self) -> &'static str {
        if self.utilization < 0.3 {
            "SEVERE: Memory bandwidth severely underutilized. Check memory access patterns for coalescing."
        } else if self.utilization < 0.5 {
            "MODERATE: Memory bandwidth underutilized. Consider cache blocking or shared memory."
        } else if self.utilization < 0.7 {
            "FAIR: Memory bandwidth utilization is acceptable. Room for optimization remains."
        } else {
            "GOOD: Memory bandwidth utilization is high. Kernel is likely compute-bound."
        }
    }
}

/// Memory access pattern analysis for kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccessPattern {
    /// Sequential access (best case)
    Sequential,
    /// Strided access with known stride
    Strided { stride: usize },
    /// Random access (worst case)
    Random,
    /// Coalesced access across threads
    Coalesced,
    /// Uncoalesced access across threads
    Uncoalesced,
}

impl MemoryAccessPattern {
    /// Get expected bandwidth efficiency (0.0 to 1.0) for this pattern
    pub fn expected_efficiency(&self) -> f64 {
        match self {
            MemoryAccessPattern::Sequential => 0.95,
            MemoryAccessPattern::Coalesced => 0.90,
            MemoryAccessPattern::Strided { stride } if *stride <= 8 => 0.75,
            MemoryAccessPattern::Strided { stride } if *stride <= 32 => 0.50,
            MemoryAccessPattern::Strided { .. } => 0.30,
            MemoryAccessPattern::Uncoalesced => 0.40,
            MemoryAccessPattern::Random => 0.20,
        }
    }

    /// Get description of the access pattern
    pub fn description(&self) -> String {
        match self {
            MemoryAccessPattern::Sequential => "Sequential access - optimal cache utilization".to_string(),
            MemoryAccessPattern::Coalesced => "Coalesced access - threads access contiguous memory".to_string(),
            MemoryAccessPattern::Strided { stride } => {
                format!("Strided access with stride {} - reduced cache line utilization", stride)
            }
            MemoryAccessPattern::Uncoalesced => "Uncoalesced access - each thread accesses different cache line".to_string(),
            MemoryAccessPattern::Random => "Random access - poor cache utilization".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omniperf_profile_builder() {
        let builder = OmniperfProfileBuilder::new("/tmp/output")
            .target_arch("gfx1100")
            .command("my_app")
            .arg("--input")
            .arg("data.txt")
            .mode("detailed");

        assert_eq!(builder.output_dir, PathBuf::from("/tmp/output"));
        assert_eq!(builder.target_arch, Some("gfx1100".to_string()));
        assert_eq!(builder.command, Some("my_app".to_string()));
        assert_eq!(builder.args, vec!["--input", "data.txt"]);
        assert_eq!(builder.mode, "detailed");
    }

    #[test]
    fn test_omniperf_profile_builder_no_command() {
        let builder = OmniperfProfileBuilder::new("/tmp/output")
            .target_arch("gfx1100");

        let result = builder.build();
        // Should fail either because tool not found (CI) or invalid config (no command)
        assert!(result.is_err());
        match result {
            Err(ProfilingError::ToolNotFound(_)) => {
                // Expected in CI environments without omniperf installed
            }
            Err(ProfilingError::InvalidConfig(_)) => {
                // Expected when omniperf is available but no command set
            }
            _ => panic!("Expected ToolNotFound or InvalidConfig error"),
        }
    }

    #[test]
    fn test_memory_bandwidth_analysis_from_operation() {
        let analysis = MemoryBandwidthAnalysis::from_operation(
            1024 * 1024 * 1024, // 1 GiB read
            512 * 1024 * 1024,  // 512 MiB written
            0.01,               // 10ms duration
            560.0,              // 560 GB/s peak
        );

        assert_eq!(analysis.bytes_read, 1024 * 1024 * 1024);
        assert_eq!(analysis.bytes_written, 512 * 1024 * 1024);
        assert_eq!(analysis.bytes_total, 1536 * 1024 * 1024);
        assert_eq!(analysis.duration_secs, 0.01);
        // 1.5 GiB / 0.01s = 161.06 GB/s (using GiB, dividing by 1e9)
        assert!((analysis.bandwidth_gbps - 161.06).abs() < 1.0);
        // 161.06 / 560 ≈ 0.288
        assert!((analysis.utilization - 0.288).abs() < 0.01);
    }

    #[test]
    fn test_memory_bandwidth_format_bytes() {
        assert_eq!(MemoryBandwidthAnalysis::format_bytes(512), "512 B");
        assert_eq!(MemoryBandwidthAnalysis::format_bytes(2048), "2.00 KB");
        assert_eq!(MemoryBandwidthAnalysis::format_bytes(2 * 1024 * 1024), "2.00 MB");
        assert_eq!(MemoryBandwidthAnalysis::format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_memory_bandwidth_utilization_checks() {
        let excellent = MemoryBandwidthAnalysis::from_operation(
            500 * 1024 * 1024 * 1024,
            0,
            1.0,
            560.0,
        );
        // 500 GB/s / 560 GB/s ≈ 89%
        assert!(excellent.is_excellent_utilization());
        assert!(excellent.is_good_utilization());

        let good = MemoryBandwidthAnalysis::from_operation(
            400 * 1024 * 1024 * 1024,
            0,
            1.0,
            560.0,
        );
        // 400 GB/s / 560 GB/s ≈ 71%
        assert!(!good.is_excellent_utilization());
        assert!(good.is_good_utilization());

        let poor = MemoryBandwidthAnalysis::from_operation(
            100 * 1024 * 1024 * 1024,
            0,
            1.0,
            560.0,
        );
        // 100 GB/s / 560 GB/s ≈ 18%
        assert!(!poor.is_good_utilization());
        assert!(!poor.is_excellent_utilization());
    }

    #[test]
    fn test_memory_bandwidth_bottleneck_description() {
        let poor = MemoryBandwidthAnalysis::from_operation(100, 0, 1.0, 560.0);
        assert!(poor.bottleneck_description().contains("SEVERE"));

        let good = MemoryBandwidthAnalysis::from_operation(400 * 1024 * 1024 * 1024, 0, 1.0, 560.0);
        assert!(good.bottleneck_description().contains("GOOD"));
    }

    #[test]
    fn test_memory_access_pattern_efficiency() {
        assert_eq!(MemoryAccessPattern::Sequential.expected_efficiency(), 0.95);
        assert_eq!(MemoryAccessPattern::Coalesced.expected_efficiency(), 0.90);
        assert_eq!(MemoryAccessPattern::Strided { stride: 4 }.expected_efficiency(), 0.75);
        assert_eq!(MemoryAccessPattern::Strided { stride: 16 }.expected_efficiency(), 0.50);
        assert_eq!(MemoryAccessPattern::Strided { stride: 64 }.expected_efficiency(), 0.30);
        assert_eq!(MemoryAccessPattern::Uncoalesced.expected_efficiency(), 0.40);
        assert_eq!(MemoryAccessPattern::Random.expected_efficiency(), 0.20);
    }

    #[test]
    fn test_memory_access_pattern_description() {
        assert!(MemoryAccessPattern::Sequential.description().contains("Sequential"));
        assert!(MemoryAccessPattern::Coalesced.description().contains("Coalesced"));
        assert!(MemoryAccessPattern::Strided { stride: 8 }.description().contains("stride 8"));
        assert!(MemoryAccessPattern::Random.description().contains("Random"));
    }

    #[test]
    fn test_memory_bandwidth_summary() {
        let analysis = MemoryBandwidthAnalysis::from_operation(
            1024 * 1024 * 1024,
            512 * 1024 * 1024,
            0.01,
            560.0,
        );

        let summary = analysis.summary();
        assert!(summary.contains("Memory Bandwidth Analysis"));
        assert!(summary.contains("1.00 GB") || summary.contains("1.00 GiB"));
        assert!(summary.contains("512.00 MB") || summary.contains("512.00 MiB"));
        assert!(summary.contains("Bandwidth"));
    }
}
