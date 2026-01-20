//! Configuration for the inference engine
//!
//! This module defines [`EngineConfig`] which controls the behavior
//! of the inference engine including batch size, cache configuration,
//! and model architecture parameters.

use std::time::Duration;

use super::types::RetryConfig;

/// Configuration for the inference engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum batch size for concurrent inference
    pub max_batch_size: usize,

    /// Maximum sequence length for generated text
    pub max_sequence_length: usize,

    /// Cache page size in tokens
    pub cache_page_size: usize,

    /// Maximum number of cache pages
    pub max_cache_pages: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Dimension of each attention head
    pub head_dim: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Timeout for batching requests
    pub batch_timeout: Duration,

    /// Retry configuration for temporary GPU errors
    pub retry_config: RetryConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            max_batch_size: 32,
            max_sequence_length: 4096,
            cache_page_size: 16,
            max_cache_pages: 1000,
            num_heads: 32,
            head_dim: 128,
            num_layers: 24,
            batch_timeout: Duration::from_millis(50),
            retry_config: RetryConfig::default(),
        }
    }
}

impl EngineConfig {
    /// Create a new engine config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }

    /// Set maximum sequence length
    pub fn with_max_sequence_length(mut self, max_sequence_length: usize) -> Self {
        self.max_sequence_length = max_sequence_length;
        self
    }

    /// Set cache page size
    pub fn with_cache_page_size(mut self, cache_page_size: usize) -> Self {
        self.cache_page_size = cache_page_size;
        self
    }

    /// Set maximum cache pages
    pub fn with_max_cache_pages(mut self, max_cache_pages: usize) -> Self {
        self.max_cache_pages = max_cache_pages;
        self
    }

    /// Set number of attention heads
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set head dimension
    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set batch timeout
    pub fn with_batch_timeout(mut self, batch_timeout: Duration) -> Self {
        self.batch_timeout = batch_timeout;
        self
    }

    /// Set retry configuration
    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_sequence_length, 4096);
        assert_eq!(config.cache_page_size, 16);
        assert_eq!(config.max_cache_pages, 1000);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_layers, 24);
    }

    #[test]
    fn test_engine_config_builder() {
        let config = EngineConfig::new()
            .with_max_batch_size(16)
            .with_max_sequence_length(2048)
            .with_cache_page_size(8)
            .with_max_cache_pages(500)
            .with_num_heads(16)
            .with_head_dim(64)
            .with_num_layers(12)
            .with_batch_timeout(Duration::from_millis(100));

        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.max_sequence_length, 2048);
        assert_eq!(config.cache_page_size, 8);
        assert_eq!(config.max_cache_pages, 500);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.batch_timeout.as_millis(), 100);
    }

    #[test]
    fn test_engine_config_includes_retry_config() {
        let config = EngineConfig::default();
        // Verify that EngineConfig includes retry_config with default values
        assert_eq!(config.retry_config.max_retries, 3);
        assert_eq!(config.retry_config.initial_delay_ms, 10);
    }
}
