//! Profiling infrastructure for ROCmForge
//!
//! This module provides profiling and timing utilities for measuring GPU kernel
//! execution time and CPU operations.
//!
//! # Modules
//!
//! - [`kernel_timer`] - Timer for measuring GPU kernel and CPU operation execution time
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

pub mod kernel_timer;

// Public exports
pub use kernel_timer::{KernelTimer, ScopedTimer};
