//! Helper functions for tempfile/tempdir usage in tests
//!
//! This module provides wrappers around tempfile crate with consistent
//! error handling and context for test files.

use anyhow::Context;
use std::path::PathBuf;

/// Create a named temp file with a helpful error message.
///
/// Wrapper around tempfile::NamedTempFile with consistent error context.
pub fn create_temp_file() -> anyhow::Result<tempfile::NamedTempFile> {
    tempfile::NamedTempFile::new()
        .context("Failed to create temporary file for test")
}

/// Create a temp directory with a helpful error message.
///
/// Wrapper around tempfile::tempdir with consistent error context.
pub fn create_temp_dir() -> anyhow::Result<tempfile::TempDir> {
    tempfile::tempdir()
        .context("Failed to create temporary directory for test")
}

/// Create a temp file with a specific suffix.
///
/// Useful for files that need specific extensions (e.g., ".gguf").
pub fn create_temp_file_with_suffix(suffix: &str) -> anyhow::Result<tempfile::NamedTempFile> {
    tempfile::NamedTempFile::with_suffix(suffix)
        .context("Failed to create temporary file with suffix")
}

/// Get a temp path that doesn't exist yet (for path testing).
///
/// This generates a unique path based on the process ID. Useful for
/// testing file operations without actually creating the file.
pub fn temp_path() -> PathBuf {
    std::env::temp_dir().join(format!("rocmforge_test_{}", std::process::id()))
}

// Re-export commonly used tempfile types for convenience
// Note: These are exported for external use even if not used directly in this module
#[allow(dead_code)]
pub use tempfile::{NamedTempFile, TempDir};
