//! Core error types and retry configuration for the inference engine
//!
//! This module defines:
//! - [`EngineError`] - Error types for engine operations
//! - [`EngineResult<T>`] - Result type alias for engine operations
//! - [`RetryConfig`] - Configuration for retrying temporary GPU errors

use std::time::Duration;
use thiserror::Error;

/// Error types for the inference engine
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Backend initialization failed: {0}")]
    BackendFailed(String),

    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    #[error("Cache initialization failed: {0}")]
    CacheFailed(String),

    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type alias for engine operations
pub type EngineResult<T> = Result<T, EngineError>;

/// Retry configuration for temporary GPU errors
///
/// Phase 10-20: Production Hardening - Retry Logic
///
/// This config defines how the engine handles temporary GPU errors
/// with exponential backoff retry. Only recoverable errors are retried.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts for recoverable errors
    pub max_retries: usize,

    /// Initial delay before first retry (milliseconds)
    pub initial_delay_ms: u64,

    /// Multiplier for exponential backoff (e.g., 2.0 = double each time)
    pub backoff_multiplier: f64,

    /// Maximum delay between retries (milliseconds)
    pub max_delay_ms: u64,

    /// Whether to add jitter to retry delays (prevents thundering herd)
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10,
            backoff_multiplier: 2.0,
            max_delay_ms: 1000,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a retry config with no retries (for testing)
    pub fn no_retry() -> Self {
        RetryConfig {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Set maximum retry attempts
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set initial delay in milliseconds
    pub fn with_initial_delay_ms(mut self, delay_ms: u64) -> Self {
        self.initial_delay_ms = delay_ms;
        self
    }

    /// Set backoff multiplier (exponential)
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set maximum delay in milliseconds
    pub fn with_max_delay_ms(mut self, max_delay_ms: u64) -> Self {
        self.max_delay_ms = max_delay_ms;
        self
    }

    /// Enable or disable jitter
    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Calculate delay for the given retry attempt
    ///
    /// # Arguments
    /// * `attempt` - Retry attempt number (0-based)
    ///
    /// # Returns
    /// Duration to wait before this retry attempt
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        let base_delay = self.initial_delay_ms as f64
            * self.backoff_multiplier.powi(attempt as i32);

        let delay_ms = base_delay.min(self.max_delay_ms as f64) as u64;

        if self.jitter {
            // Add up to 25% random jitter
            let jitter_range = delay_ms / 4;
            let jitter_amt = if jitter_range > 0 {
                use std::time::SystemTime;
                let nanos = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos() as u64;
                nanos % jitter_range
            } else {
                0
            };
            Duration::from_millis(delay_ms + jitter_amt)
        } else {
            Duration::from_millis(delay_ms)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3, "Default max_retries should be 3");
        assert_eq!(config.initial_delay_ms, 10, "Default initial delay should be 10ms");
        assert_eq!(config.backoff_multiplier, 2.0, "Default multiplier should be 2.0");
        assert_eq!(config.max_delay_ms, 1000, "Default max delay should be 1000ms");
        assert!(config.jitter, "Jitter should be enabled by default");
    }

    #[test]
    fn test_retry_config_builder() {
        let config = RetryConfig::new()
            .with_max_retries(5)
            .with_initial_delay_ms(100)
            .with_backoff_multiplier(3.0)
            .with_max_delay_ms(5000)
            .with_jitter(false);

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay_ms, 100);
        assert_eq!(config.backoff_multiplier, 3.0);
        assert_eq!(config.max_delay_ms, 5000);
        assert!(!config.jitter);
    }

    #[test]
    fn test_retry_config_no_retry() {
        let config = RetryConfig::no_retry();
        assert_eq!(config.max_retries, 0, "no_retry should have 0 retries");
    }

    #[test]
    fn test_retry_config_delay_calculation() {
        let config = RetryConfig::new()
            .with_initial_delay_ms(10)
            .with_backoff_multiplier(2.0)
            .with_max_delay_ms(100)
            .with_jitter(false);

        let delay_0 = config.delay_for_attempt(0);
        assert_eq!(delay_0.as_millis(), 10, "First retry delay should be 10ms");

        let delay_1 = config.delay_for_attempt(1);
        assert_eq!(delay_1.as_millis(), 20, "Second retry delay should be 20ms");

        let delay_2 = config.delay_for_attempt(2);
        assert_eq!(delay_2.as_millis(), 40, "Third retry delay should be 40ms");

        let delay_10 = config.delay_for_attempt(10);
        // 10 * 2^10 = 10240, capped at 100
        assert_eq!(delay_10.as_millis(), 100, "Delay should be capped at max_delay_ms");
    }

    #[test]
    fn test_retry_config_jitter_in_range() {
        let config = RetryConfig::new()
            .with_initial_delay_ms(100)
            .with_backoff_multiplier(1.0)
            .with_max_delay_ms(200)
            .with_jitter(true);

        // Test that jitter adds some variability but stays within bounds
        let delay_0 = config.delay_for_attempt(0);
        // Base delay is 100ms, jitter adds up to 25% (125ms max)
        assert!(delay_0.as_millis() >= 100, "Jittered delay should be >= base delay");
        assert!(delay_0.as_millis() <= 125, "Jittered delay should be <= base + 25%");
    }
}
