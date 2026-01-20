//! Main inference engine for ROCmForge
//!
//! This module provides the core inference engine that coordinates backend,
//! cache, scheduler, and sampler for LLM inference on AMD GPUs.
//!
//! ## Module Structure
//!
//! - [`types`] - Core error types and retry configuration
//! - [`config`] - Engine configuration and builders
//! - [`inference`] - Main InferenceEngine implementation
//! - [`stats`] - Engine statistics and health monitoring
//!
//! ## Public API
//!
//! The main types are re-exported here for convenience:
//! - [`InferenceEngine`] - Main inference engine
//! - [`EngineConfig`] - Engine configuration
//! - [`EngineError`] - Error types
//! - [`EngineResult`] - Result type alias
//! - [`RetryConfig`] - Retry configuration
//! - [`EngineStats`] - Engine statistics
//! - [`HealthStatus`] - Health monitoring

pub mod types;
pub mod config;
pub mod inference;
pub mod stats;

// Re-exports for backward compatibility
pub use types::{EngineError, EngineResult, RetryConfig};
pub use config::EngineConfig;
pub use inference::InferenceEngine;
pub use stats::{EngineStats, HealthStatus};
