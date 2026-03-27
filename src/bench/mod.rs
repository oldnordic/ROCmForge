//! Benchmark utilities and report generation.
//!
//! This module provides shared utilities for benchmarking:
//! - Model discovery for real GGUF files
//! - Report generation (markdown + CSV)
//! - Timing utilities

pub mod discovery;
pub mod reporter;

pub use discovery::discover_models;
pub use reporter::{generate_report, export_csv};
