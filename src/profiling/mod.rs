//! Profiling infrastructure for ROCmForge
//!
//! This module provides profiling and timing utilities for measuring GPU kernel
//! execution time and CPU operations, as well as integration with ROCm profiling tools.
//!
//! # Modules
//!
//! - [`kernel_timer`] - Timer for measuring GPU kernel and CPU operation execution time
//! - [`kernel_launch`] - Kernel launch overhead profiling and batching optimization
//! - [`ttft`] - Time to First Token (TTFT) profiling for inference latency analysis
//! - [`baseline`] - Performance baseline storage and regression detection
//! - [`types`] - Common types for ROCm profiling tools
//! - [`rocprof`] - rocprof profiling tool integration
//! - [`omniperf`] - omniperf profiling tool integration
//!
//! # Example
//!
//! ```rust
//! use rocmforge::profiling::KernelTimer;
//!
//! // Time a GPU kernel
//! let mut timer = KernelTimer::for_kernel("matmul");
//! timer.start(&stream)?;
//!
//! // ... execute kernel ...
//!
//! timer.stop(&stream)?;
//!
//! println!("Elapsed: {:.2} ms", timer.elapsed_unwrap());
//! ```
//!
//! # ROCm Profiling Tools Integration
//!
//! The profiling modules provide helpers for working with external ROCm profiling tools:
//!
//! ```rust,ignore
//! use rocmforge::profiling::{RocprofSession, ProfilingConfig};
//!
//! // Create a profiling session
//! let config = ProfilingConfig::default()
//!     .with_counters(vec!["SQ_WAVES", "SQ_INSTS"]);
//!
//! let session = RocprofSession::with_config(config)?;
//!
//! // Build command to run application under rocprof
//! let cmd = session.build_command("./my_app", &["--arg1"]);
//! ```
//!
//! # TTFT Profiling
//!
//! The [`ttft`] module provides detailed breakdown of Time to First Token (TTFT) latency:
//!
//! ```rust
//! use rocmforge::profiling::ttft::TtftProfiler;
//!
//! let mut profiler = TtftProfiler::new();
//! profiler.start_ttft();
//!
//! // Measure model loading
//! profiler.start_model_loading();
//! // ... load model ...
//! profiler.stop_model_loading();
//!
//! // Measure prompt processing
//! profiler.start_prompt_processing();
//! // ... process prompt ...
//! profiler.stop_prompt_processing();
//!
//! let breakdown = profiler.finish_ttft();
//! println!("{}", breakdown);
//! ```

pub mod baseline;
pub mod baseline_types;
pub mod baseline_storage;
pub mod kernel_timer;
pub mod kernel_launch;
pub mod ttft;

// ROCm profiling tools modules
pub mod types;
pub mod rocprof;
pub mod omniperf;

// Legacy re-export for backward compatibility
pub mod rocprof_integration {
    //! Re-export of profiling types for backward compatibility.
    //!
    //! This module re-exports all public types from the decomposed profiling modules.
    //! New code should import directly from `rocmforge::profiling::{rocprof, omniperf, types}`.

    // Re-export common types
    pub use crate::profiling::types::{
        ProfilingError, ProfilingResult, ProfilingTool, CounterCategory,
    };

    // Re-export rocprof types
    pub use crate::profiling::rocprof::{
        ProfilingConfig, RocprofSession, ProfilingResults, ProfilingMetrics,
        KernelExecution,
    };

    // Re-export rocprof helpers
    pub use crate::profiling::rocprof::helpers;

    // Re-export omniperf types
    pub use crate::profiling::omniperf::{
        OmniperfProfileBuilder, MemoryBandwidthAnalysis, MemoryAccessPattern,
    };
}

// Public exports from baseline (re-exports from baseline_types)
pub use baseline_types::{
    PerformanceBaseline, BaselineMetrics, BaselineCollection,
    ComparisonResult, HardwareInfo, BaselineError, RegressionThreshold,
    RegressionReport,
};
pub use baseline::BenchmarkBaseline;

// Public exports from kernel_timer
pub use kernel_timer::{KernelTimer, ScopedTimer};

// Public exports from kernel_launch
pub use kernel_launch::{
    LaunchOverheadTracker, LaunchOverheadStats, BatchConfig,
    OverheadOptimizationRecommendation, RecommendationType,
};

// Public exports from types (common profiling types)
pub use types::{
    ProfilingError, ProfilingResult, ProfilingTool, CounterCategory,
};

// Public exports from rocprof
pub use rocprof::{
    ProfilingConfig, RocprofSession, ProfilingResults, ProfilingMetrics,
    KernelExecution,
};

// Public exports from rocprof helpers
pub use rocprof::helpers::{
    profile_kernel, profile_memory, profile_memory_detailed,
    profile_matmul_memory, profile_compute_unit,
    available_tools, print_available_tools,
};

// Public exports from omniperf
pub use omniperf::{
    OmniperfProfileBuilder, MemoryBandwidthAnalysis, MemoryAccessPattern,
};

// Public exports from ttft
pub use ttft::{
    TtftBreakdown, TtftProfiler, KernelTiming,
    create_ttft_breakdown,
};
