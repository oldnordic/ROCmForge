//! Common types for ROCm profiling tools integration
//!
//! This module provides shared types used across different profiling tool integrations.

use std::process::Command;

/// Error type for ROCm profiling operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProfilingError {
    #[error("Profiling tool not found: {0}")]
    ToolNotFound(String),

    #[error("Invalid profiling configuration: {0}")]
    InvalidConfig(String),

    #[error("Failed to parse profiling output: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Profiling session error: {0}")]
    SessionError(String),
}

/// Result type for profiling operations
pub type ProfilingResult<T> = Result<T, ProfilingError>;

/// Available ROCm profiling tools
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingTool {
    /// rocprof - HSA tracer and profiler
    Rocprof,
    /// omniperf - Comprehensive profiler with GUI analysis
    Omniperf,
    /// rocperf - Performance counter collector
    Rocperf,
}

impl ProfilingTool {
    /// Get the command name for this tool
    pub fn command_name(&self) -> &str {
        match self {
            ProfilingTool::Rocprof => "rocprof",
            ProfilingTool::Omniperf => "omniperf",
            ProfilingTool::Rocperf => "rocperf",
        }
    }

    /// Check if this tool is available in PATH
    pub fn is_available(&self) -> bool {
        Self::check_command(self.command_name())
    }

    fn check_command(name: &str) -> bool {
        Command::new("which")
            .arg(name)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

/// Performance counter categories for GPU profiling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterCategory {
    /// Instruction counters
    Instructions,
    /// Wavefront/warp activity
    Waves,
    /// Memory bandwidth and transactions
    Memory,
    /// Cache hit/miss rates
    Cache,
    /// Compute unit utilization
    ComputeUnit,
    /// LDS (Local Data Share) usage
    Lds,
    /// Pipeline stalls
    Stalls,
}

impl CounterCategory {
    /// Get common counter names for this category
    pub fn common_counters(&self) -> &[&str] {
        match self {
            CounterCategory::Instructions => &[
                "SQ_INSTS",
                "SQ_INSTS_VALU",
                "SQ_INSTS_SALU",
                "SQ_INSTS_VMEM",
                "SQ_INSTS_FLAT",
            ],
            CounterCategory::Waves => &[
                "SQ_WAVES",
                "GRBM_GUI_ACTIVE",
            ],
            CounterCategory::Memory => &[
                "PM_MARKER",
                "SQ_LDS_IDX_ACTIVE",
                "SQ_LDS_BANK_ACTIVE",
            ],
            CounterCategory::Cache => &[
                "TCP_TOTAL_CACHE_ACCESSES",
                "TCP_TOTAL_CACHE_MISSES",
                "TCP_TOTAL_HIT_RATE",
            ],
            CounterCategory::ComputeUnit => &[
                "GRBM_GUI_ACTIVE",
                "GRBM_COUNT",
            ],
            CounterCategory::Lds => &[
                "SQ_LDS_IDX_ACTIVE",
                "SQ_LDS_BANK_ACTIVE",
            ],
            CounterCategory::Stalls => &[
                "SQ_INSTS",
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
