//! Profiling infrastructure for ROCmForge
//!
//! This module provides profiling and timing utilities for measuring GPU kernel
//! execution time and CPU operations, as well as integration with ROCm profiling tools.
//!
//! # Modules
//!
//! - [`kernel_timer`] - Timer for measuring GPU kernel and CPU operation execution time
//! - [`baseline`] - Performance baseline storage and regression detection
//! - [`rocprof_integration`] - Integration with ROCm profiling tools (rocprof, omniperf, rocperf)
//!
//! # Example
//!
//! ```rust
//! use rocmforge::profiling::KernelTimer;
//!
//! // Time a GPU kernel
//! let mut timer = KernelTimer::for_kernel("matmul");
//! #[cfg(feature = "rocm")]
//! timer.start(&stream)?;
//! #[cfg(not(feature = "rocm"))]
//! timer.start_cpu();
//!
//! // ... execute kernel ...
//!
//! #[cfg(feature = "rocm")]
//! timer.stop(&stream)?;
//! #[cfg(not(feature = "rocm"))]
//! timer.stop_cpu();
//!
//! println!("Elapsed: {:.2} ms", timer.elapsed_unwrap());
//! ```
//!
//! # ROCm Profiling Tools Integration
//!
//! The [`rocprof_integration`] module provides helpers for working with external
//! ROCm profiling tools:
//!
//! ```rust,ignore
//! use rocmforge::profiling::rocprof_integration::{RocprofSession, ProfilingConfig};
//!
//! // Create a profiling session
//! let session = RocprofSession::new("/tmp/profile")?;
//!
//! // Build command to run application under rocprof
//! let cmd = session.build_command("./my_app", &["--arg1"]);
//! ```
//!
//! See [`rocprof_integration`] for more details on profiling tool integration.

pub mod baseline;
pub mod kernel_timer;
pub mod rocprof_integration;

// Public exports from baseline
pub use baseline::{
    PerformanceBaseline, BaselineMetrics, BaselineCollection,
    ComparisonResult, HardwareInfo, BaselineError, RegressionThreshold,
};

// Public exports from kernel_timer
pub use kernel_timer::{KernelTimer, ScopedTimer};

// Public exports from rocprof_integration
pub use rocprof_integration::{
    ProfilingTool, CounterCategory, ProfilingConfig, RocprofSession,
    ProfilingResults, ProfilingMetrics, ProfilingError, KernelExecution,
    OmniperfProfileBuilder,
};
