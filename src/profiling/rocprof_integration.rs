//! ROCm Profiling Tools Integration
//!
//! This module provides integration with ROCm profiling tools including:
//! - **rocprof**: HSA trace collection and kernel profiling
//! - **rocperf**: Performance counter collection
//! - **omniperf**: Comprehensive profiling and analysis
//!
//! # Overview
//!
//! The integration provides helper functions and structs for working with
//! ROCm profiling tools, enabling performance measurement and bottleneck
//! detection for GPU kernels.
//!
//! # Tool Availability
//!
//! These tools are **external binaries** that must be installed separately:
//! - `rocprof` - Available in ROCm installations
//! - `omniperf` - Install via `pip install omniperf`
//!
//! This module provides command builders and output parsers for working
//! with these tools from Rust code.
//!
//! # Example
//!
//! ```rust,ignore
//! use rocmforge::profiling::rocprof_integration::{RocprofSession, ProfilingConfig};
//!
//! // Create a profiling session
//! let config = ProfilingConfig::default()
//!     .with_counters(vec!["SQ_WAVES", "SQ_INSTS"]);
//!
//! let session = RocprofSession::new("/tmp/profile_output", config)?;
//!
//! // Build command to run your application under rocprof
//! let cmd = session.build_command("my_app", &["--arg1", "arg2"]);
//! println!("Run: {:?}", cmd);
//!
//! // After running, parse results
//! let results = session.parse_results()?;
//! println!("GPM Waves: {:?}", results.get_counter("SQ_WAVES"));
//! ```
//!
//! # Module Organization
//!
//! This module re-exports types from the decomposed profiling modules:
//! - [`types`] - Common types (ProfilingError, ProfilingTool, CounterCategory)
//! - [`rocprof`] - rocprof-specific integration
//! - [`omniperf`] - omniperf-specific integration

// Re-export all types from the decomposed modules for backward compatibility

// Common types
pub use crate::profiling::types::{
    ProfilingError, ProfilingResult, ProfilingTool, CounterCategory,
};

// rocprof types
pub use crate::profiling::rocprof::{
    ProfilingConfig, RocprofSession, ProfilingResults, ProfilingMetrics,
    KernelExecution,
};

// rocprof helpers
pub use crate::profiling::rocprof::helpers::{
    profile_kernel, profile_memory, profile_memory_detailed,
    profile_matmul_memory, profile_compute_unit,
    available_tools, print_available_tools,
};

// omniperf types
pub use crate::profiling::omniperf::{
    OmniperfProfileBuilder, MemoryBandwidthAnalysis, MemoryAccessPattern,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiling_config_default() {
        let config = ProfilingConfig::default();
        assert_eq!(config.output_dir, std::path::PathBuf::from("/tmp/rocmforge_profile"));
        assert!(config.counters.is_empty());
        assert_eq!(config.categories.len(), 3);
        assert!(config.enable_hsa_trace);
        assert!(!config.enable_i_trace);
    }

    #[test]
    fn test_profiling_config_builder() {
        let config = ProfilingConfig::new("/tmp/test")
            .with_counter("TEST_COUNTER")
            .with_category(CounterCategory::Instructions)
            .with_hsa_trace(false)
            .with_i_trace(true);

        assert_eq!(config.output_dir, std::path::PathBuf::from("/tmp/test"));
        assert!(config.counters.contains(&"TEST_COUNTER".to_string()));
        assert!(config.categories.contains(&CounterCategory::Instructions));
        assert!(!config.enable_hsa_trace);
        assert!(config.enable_i_trace);
    }

    #[test]
    fn test_profiling_config_validate_success() {
        let config = ProfilingConfig::new("/tmp/test");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_profiling_tool_command_names() {
        assert_eq!(ProfilingTool::Rocprof.command_name(), "rocprof");
        assert_eq!(ProfilingTool::Omniperf.command_name(), "omniperf");
        assert_eq!(ProfilingTool::Rocperf.command_name(), "rocperf");
    }

    #[test]
    fn test_counter_category_common_counters() {
        let inst_counters = CounterCategory::Instructions.common_counters();
        assert!(inst_counters.contains(&"SQ_INSTS"));

        let wave_counters = CounterCategory::Waves.common_counters();
        assert!(wave_counters.contains(&"SQ_WAVES"));
    }

    #[test]
    fn test_omniperf_profile_builder() {
        let builder = OmniperfProfileBuilder::new("/tmp/output")
            .target_arch("gfx1100")
            .command("my_app")
            .arg("--input")
            .arg("data.txt")
            .mode("detailed");

        assert_eq!(builder.output_dir, std::path::PathBuf::from("/tmp/output"));
        assert_eq!(builder.target_arch, Some("gfx1100".to_string()));
        assert_eq!(builder.command, Some("my_app".to_string()));
        assert_eq!(builder.args, vec!["--input", "data.txt"]);
        assert_eq!(builder.mode, "detailed");
    }

    #[test]
    fn test_memory_bandwidth_from_operation() {
        use std::path::Path;
        let analysis = MemoryBandwidthAnalysis::from_operation(
            1024 * 1024 * 1024, // 1 GB read
            512 * 1024 * 1024,  // 512 MB written
            0.01,               // 10ms duration
            560.0,              // 560 GB/s peak
        );

        assert_eq!(analysis.bytes_read, 1024 * 1024 * 1024);
        assert_eq!(analysis.bytes_written, 512 * 1024 * 1024);
        assert!(analysis.bandwidth_gbps > 0.0);
    }

    #[test]
    fn test_memory_access_pattern_efficiency() {
        assert_eq!(MemoryAccessPattern::Sequential.expected_efficiency(), 0.95);
        assert_eq!(MemoryAccessPattern::Coalesced.expected_efficiency(), 0.90);
        assert_eq!(MemoryAccessPattern::Random.expected_efficiency(), 0.20);
    }
}
