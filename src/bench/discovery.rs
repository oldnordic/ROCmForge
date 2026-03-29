//! Model discovery utilities for benchmarks.
//!
//! Finds GGUF model files in known locations with support for
//! environment variables and command-line overrides.

use std::env;
use std::path::{Path, PathBuf};

/// Default model search locations in priority order.
const DEFAULT_MODEL_PATHS: &[&str] = &["/home/feanor/Projects/Memoria/models", "./models"];

/// Discover GGUF model files from search paths.
///
/// # Arguments
/// * `explicit_dir` - Optional explicit directory from command line
///
/// # Returns
/// Vector of paths to `.gguf` files found
pub fn discover_models(explicit_dir: Option<&str>) -> Vec<PathBuf> {
    let search_paths = get_search_paths(explicit_dir);
    let mut models = Vec::new();

    for path in search_paths {
        if !path.exists() {
            continue;
        }

        // Read directory and collect .gguf files
        if let Ok(entries) = std::fs::read_dir(&path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_path = entry.path();
                if file_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                    models.push(file_path);
                }
            }
        }
    }

    models
}

/// Get search paths in priority order.
fn get_search_paths(explicit_dir: Option<&str>) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Explicit --model-dir argument (highest priority)
    if let Some(dir) = explicit_dir {
        paths.push(PathBuf::from(dir));
    }

    // 2. ROCFORGE_MODEL_DIR environment variable
    if let Ok(dir) = env::var("ROCMFORGE_MODEL_DIR") {
        paths.push(PathBuf::from(dir));
    }

    // 3. Default locations
    for path in DEFAULT_MODEL_PATHS {
        paths.push(PathBuf::from(path));
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_search_paths_includes_defaults() {
        let paths = get_search_paths(None);
        assert!(paths.len() >= 2);
        assert!(paths.iter().any(|p| p.ends_with("Memoria/models")));
        assert!(paths.iter().any(|p| p.ends_with("models")));
    }

    #[test]
    fn explicit_dir_overrides_defaults() {
        let paths = get_search_paths(Some("/custom/path"));
        assert_eq!(paths[0], PathBuf::from("/custom/path"));
    }
}
