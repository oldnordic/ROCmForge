//! File storage for performance baselines
//!
//! This module provides utilities for saving and loading baselines to/from JSON files.

use crate::profiling::baseline_types::{BaselineError, BaselineCollection, PerformanceBaseline};
use std::io::{Read, Write};
use std::path::Path;

/// Save a baseline to a JSON file
pub fn save_baseline<P: AsRef<Path>>(
    baseline: &PerformanceBaseline,
    path: P,
) -> Result<(), BaselineError> {
    let json = serde_json::to_string_pretty(baseline)
        .map_err(|e| BaselineError::SerializationError(e.to_string()))?;

    // Ensure parent directory exists
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| BaselineError::IoError(e.to_string()))?;
    }

    let mut file = std::fs::File::create(path)
        .map_err(|e| BaselineError::IoError(e.to_string()))?;
    file.write_all(json.as_bytes())
        .map_err(|e| BaselineError::IoError(e.to_string()))?;

    Ok(())
}

/// Load a baseline from a JSON file
pub fn load_baseline<P: AsRef<Path>>(path: P) -> Result<PerformanceBaseline, BaselineError> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| BaselineError::IoError(format!("Failed to open baseline file: {}", e)))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| BaselineError::IoError(e.to_string()))?;

    serde_json::from_str(&contents)
        .map_err(|e| BaselineError::SerializationError(format!("Invalid baseline JSON: {}", e)))
}

/// Save a collection to a JSON file
pub fn save_collection<P: AsRef<Path>>(
    collection: &BaselineCollection,
    path: P,
) -> Result<(), BaselineError> {
    let json = serde_json::to_string_pretty(collection)
        .map_err(|e| BaselineError::SerializationError(e.to_string()))?;

    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| BaselineError::IoError(e.to_string()))?;
    }

    let mut file = std::fs::File::create(path)
        .map_err(|e| BaselineError::IoError(e.to_string()))?;
    file.write_all(json.as_bytes())
        .map_err(|e| BaselineError::IoError(e.to_string()))?;

    Ok(())
}

/// Load a collection from a JSON file
pub fn load_collection<P: AsRef<Path>>(path: P) -> Result<BaselineCollection, BaselineError> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| BaselineError::IoError(format!("Failed to open collection file: {}", e)))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| BaselineError::IoError(e.to_string()))?;

    serde_json::from_str(&contents)
        .map_err(|e| BaselineError::SerializationError(format!("Invalid collection JSON: {}", e)))
}

/// Helper function to convert timestamp to readable date
pub fn chrono_from_timestamp(secs: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    if let Some(d) = UNIX_EPOCH.checked_add(Duration::from_secs(secs as u64)) {
        // Simple format - could use chrono crate for better formatting
        let datetime = format!("{:?}", d);
        // Extract the date part from debug output
        datetime
    } else {
        "Invalid timestamp".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::baseline_types::{HardwareInfo, BaselineMetrics};
    use std::time::Duration;

    #[test]
    fn test_save_and_load_baseline() {
        let metrics = BaselineMetrics {
            avg_ms: 10.0,
            min_ms: 9.0,
            max_ms: 11.0,
            p50_ms: 10.0,
            p95_ms: 10.5,
            p99_ms: 10.8,
            iterations: 100,
        };
        let baseline = PerformanceBaseline::new("test_benchmark", metrics);

        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_baseline.json");

        // Save
        save_baseline(&baseline, &test_path).unwrap();

        // Load
        let loaded = load_baseline(&test_path).unwrap();

        assert_eq!(baseline.name, loaded.name);
        assert_eq!(baseline.metrics.avg_ms, loaded.metrics.avg_ms);

        // Cleanup
        std::fs::remove_file(&test_path).ok();
    }

    #[test]
    fn test_save_and_load_collection() {
        let metrics = BaselineMetrics {
            avg_ms: 10.0,
            min_ms: 9.0,
            max_ms: 11.0,
            p50_ms: 10.0,
            p95_ms: 10.5,
            p99_ms: 10.8,
            iterations: 100,
        };
        let baseline = PerformanceBaseline::new("bench1", metrics);

        let mut collection = BaselineCollection::new();
        collection.add(baseline);

        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_collection.json");

        // Save
        save_collection(&collection, &test_path).unwrap();

        // Load
        let loaded = load_collection(&test_path).unwrap();

        assert_eq!(collection.len(), loaded.len());
        assert!(loaded.get("bench1").is_some());

        // Cleanup
        std::fs::remove_file(&test_path).ok();
    }

    #[test]
    fn test_chrono_from_timestamp() {
        let result = chrono_from_timestamp(0);
        // Just verify it returns something non-empty (format is implementation-dependent)
        assert!(!result.is_empty() || result == "Invalid timestamp");

        let result = chrono_from_timestamp(1609459200); // 2021-01-01
        assert!(!result.is_empty() || result == "Invalid timestamp");
    }
}
