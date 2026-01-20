//! Engine statistics and health monitoring
//!
//! This module provides types for monitoring the inference engine:
//! - [`EngineStats`] - Runtime statistics
//! - [`HealthStatus`] - Health information for monitoring endpoints

use serde::Serialize;

/// Runtime statistics for the inference engine
#[derive(Debug, Clone)]
pub struct EngineStats {
    /// Whether the engine is running
    pub is_running: bool,

    /// Scheduler statistics
    pub scheduler_stats: crate::scheduler::QueueStats,

    /// Cache statistics
    pub cache_stats: crate::kv_cache::CacheStats,

    /// Whether a model is loaded
    pub model_loaded: bool,
}

/// Health status information for monitoring endpoints
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    /// Overall health status: healthy, degraded, or unhealthy
    pub status: String,

    /// Whether the inference engine is running
    pub engine_running: bool,

    /// Whether a model is currently loaded
    pub model_loaded: bool,

    /// Whether GPU is available
    pub gpu_available: bool,

    /// Free GPU memory in bytes (if available)
    pub gpu_memory_free: Option<usize>,

    /// Total GPU memory in bytes (if available)
    pub gpu_memory_total: Option<usize>,

    /// GPU error message (if GPU query failed)
    pub gpu_error: Option<String>,

    /// Number of currently processing requests
    pub active_requests: usize,

    /// Number of queued requests
    pub queued_requests: usize,

    /// Number of KV cache pages in use
    pub cache_pages_used: usize,

    /// Total number of KV cache pages
    pub cache_pages_total: usize,

    /// Number of active sequences in cache
    pub active_sequences: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_serializable() {
        // HealthStatus should be serializable for HTTP endpoints
        let status = HealthStatus {
            status: "healthy".to_string(),
            engine_running: true,
            model_loaded: true,
            gpu_available: true,
            gpu_memory_free: Some(1_000_000_000),
            gpu_memory_total: Some(8_000_000_000),
            gpu_error: None,
            active_requests: 2,
            queued_requests: 1,
            cache_pages_used: 50,
            cache_pages_total: 1000,
            active_sequences: 2,
        };

        // Verify it can be serialized to JSON
        let json = serde_json::to_string(&status);
        assert!(json.is_ok(), "HealthStatus should be JSON-serializable");

        let json_str = json.unwrap();
        assert!(json_str.contains("\"status\":\"healthy\""));
        assert!(json_str.contains("\"engine_running\":true"));
        assert!(json_str.contains("\"active_requests\":2"));
    }

    #[test]
    fn test_health_status_with_error() {
        let status = HealthStatus {
            status: "unhealthy".to_string(),
            engine_running: false,
            model_loaded: false,
            gpu_available: false,
            gpu_memory_free: None,
            gpu_memory_total: None,
            gpu_error: Some("GPU initialization failed".to_string()),
            active_requests: 0,
            queued_requests: 0,
            cache_pages_used: 0,
            cache_pages_total: 0,
            active_sequences: 0,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("GPU initialization failed"));
    }
}
