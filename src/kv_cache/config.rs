//! KV cache configuration types
//!
//! This module contains configuration types for the paged KV cache,
//! including presets for common model sizes.

use super::types::KvCacheResult;
use super::KvCacheError;

/// Configuration for paged KV cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub page_size: usize,
    pub max_pages: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    /// Enable cache compaction for long-running inference
    pub enable_compaction: bool,
    /// Minimum free ratio before triggering compaction (0.0-1.0)
    pub compaction_threshold: f64,
}

/// Preset configurations for common model sizes
///
/// These presets are optimized for typical workloads and balance
/// memory usage vs. context length capacity.
#[derive(Debug, Clone, Copy)]
pub enum CachePreset {
    /// Small models (1B-3B parameters)
    /// Optimized for edge devices and limited memory
    Small,
    /// Medium models (7B-13B parameters)
    /// Optimized for typical consumer GPUs
    Medium,
    /// Large models (30B-70B parameters)
    /// Optimized for data center GPUs
    Large,
    /// Custom configuration with explicit parameters
    Custom { page_size: usize, max_pages: usize },
}

impl CachePreset {
    /// Get the optimal page size for this preset
    ///
    /// Page size is a trade-off:
    /// - Smaller pages: Less wasted memory for short sequences
    /// - Larger pages: Less page table overhead for long sequences
    pub fn page_size(self) -> usize {
        match self {
            CachePreset::Small => 16,   // Short contexts expected
            CachePreset::Medium => 32,  // Balanced
            CachePreset::Large => 64,   // Long contexts expected
            CachePreset::Custom { page_size, .. } => page_size,
        }
    }

    /// Get the recommended max pages for this preset
    ///
    /// This is calculated based on typical GPU memory sizes:
    /// - Small: ~4GB VRAM (edge devices)
    /// - Medium: ~12GB VRAM (consumer GPUs)
    /// - Large: ~40GB VRAM (data center)
    pub fn max_pages(self, num_heads: usize, head_dim: usize, _num_layers: usize) -> usize {
        // Estimate memory per page in bytes
        let bytes_per_token = num_heads * head_dim * 2 * std::mem::size_of::<f32>(); // K + V
        let bytes_per_page = self.page_size() * bytes_per_token;

        // Target VRAM sizes for each preset
        let target_vram_bytes = match self {
            CachePreset::Small => 4 * 1024 * 1024 * 1024,   // 4GB
            CachePreset::Medium => 12 * 1024 * 1024 * 1024,  // 12GB
            CachePreset::Large => 40 * 1024 * 1024 * 1024,   // 40GB
            CachePreset::Custom { .. } => 8 * 1024 * 1024 * 1024, // 8GB default for custom
        };

        // Reserve 50% of VRAM for KV cache (rest for model weights, activations)
        let kv_cache_budget = target_vram_bytes / 2;

        // Calculate max pages
        let max_pages = kv_cache_budget / bytes_per_page;

        // Clamp to reasonable range
        max_pages.clamp(64, 8192)
    }
}

impl CacheConfig {
    pub fn new(
        page_size: usize,
        max_pages: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> KvCacheResult<Self> {
        if page_size == 0 || max_pages == 0 || num_heads == 0 || head_dim == 0 || num_layers == 0 {
            return Err(KvCacheError::InvalidConfiguration);
        }

        Ok(CacheConfig {
            page_size,
            max_pages,
            num_heads,
            head_dim,
            num_layers,
            enable_compaction: false,
            compaction_threshold: 0.3, // Default: compact when 30% free
        })
    }

    /// Create a CacheConfig from a preset
    ///
    /// This is the recommended way to create a cache configuration
    /// as it automatically calculates optimal parameters.
    ///
    /// # Arguments
    /// * `preset` - The cache preset (Small, Medium, Large, Custom)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `num_layers` - Number of transformer layers
    ///
    /// # Example
    /// ```ignore
    /// let config = CacheConfig::from_preset(
    ///     CachePreset::Medium,
    ///     32,  // num_heads
    ///     128, // head_dim
    ///     32,  // num_layers
    /// )?;
    /// ```
    pub fn from_preset(
        preset: CachePreset,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> KvCacheResult<Self> {
        if num_heads == 0 || head_dim == 0 || num_layers == 0 {
            return Err(KvCacheError::InvalidConfiguration);
        }

        let page_size = preset.page_size();
        let max_pages = preset.max_pages(num_heads, head_dim, num_layers);

        Ok(CacheConfig {
            page_size,
            max_pages,
            num_heads,
            head_dim,
            num_layers,
            enable_compaction: true,
            compaction_threshold: 0.3,
        })
    }

    /// Create a CacheConfig optimized for specific context length
    ///
    /// This method calculates the optimal page size and max pages
    /// based on the target context length.
    ///
    /// # Arguments
    /// * `target_context_len` - Target maximum context length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `num_layers` - Number of transformer layers
    /// * `vram_budget_bytes` - Optional VRAM budget (defaults to 50% of typical GPU)
    pub fn for_context_length(
        target_context_len: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
        vram_budget_bytes: Option<usize>,
    ) -> KvCacheResult<Self> {
        if num_heads == 0 || head_dim == 0 || num_layers == 0 || target_context_len == 0 {
            return Err(KvCacheError::InvalidConfiguration);
        }

        // Choose page size based on context length
        // Short contexts (< 1K): smaller pages for less waste
        // Medium contexts (1K-4K): balanced page size
        // Long contexts (> 4K): larger pages for less overhead
        let page_size = if target_context_len < 1024 {
            16
        } else if target_context_len < 4096 {
            32
        } else {
            64
        };

        // Calculate required pages
        let pages_needed = (target_context_len + page_size - 1) / page_size;

        // Default to 8GB VRAM budget if not specified
        let vram_budget = vram_budget_bytes.unwrap_or(8 * 1024 * 1024 * 1024);
        let kv_cache_budget = vram_budget / 2;

        // Calculate max pages based on VRAM budget
        let bytes_per_token = num_heads * head_dim * 2 * std::mem::size_of::<f32>();
        let bytes_per_page = page_size * bytes_per_token;
        let max_pages_from_vam = kv_cache_budget / bytes_per_page;

        // Use the larger of: context requirement or VRAM limit
        let max_pages = pages_needed.max(max_pages_from_vam).min(8192);

        Ok(CacheConfig {
            page_size,
            max_pages,
            num_heads,
            head_dim,
            num_layers,
            enable_compaction: target_context_len > 2048, // Enable compaction for long contexts
            compaction_threshold: 0.3,
        })
    }

    /// Enable cache compaction for long-running inference
    ///
    /// Cache compaction reorganizes memory to reduce fragmentation
    /// during long inference runs.
    pub fn with_compaction(mut self, enabled: bool) -> Self {
        self.enable_compaction = enabled;
        self
    }

    /// Set the compaction threshold
    ///
    /// Compaction is triggered when the free ratio exceeds this value.
    pub fn with_compaction_threshold(mut self, threshold: f64) -> Self {
        self.compaction_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Calculate expected memory usage in bytes
    pub fn estimated_memory_bytes(&self) -> usize {
        let bytes_per_token = self.num_heads * self.head_dim * 2 * std::mem::size_of::<f32>();
        self.max_pages * self.page_size * bytes_per_token
    }

    /// Calculate expected memory usage in human-readable format
    pub fn estimated_memory_human(&self) -> String {
        let bytes = self.estimated_memory_bytes();
        const KB: usize = 1024;
        const MB: usize = 1024 * 1024;
        const GB: usize = 1024 * 1024 * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        }
    }

    /// Calculate maximum context length supported by this config
    pub fn max_context_length(&self) -> usize {
        self.max_pages * self.page_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_creation() {
        let config = CacheConfig::new(1024, 100, 32, 128, 24);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.page_size, 1024);
        assert_eq!(config.max_pages, 100);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_layers, 24);
    }

    #[test]
    fn test_invalid_cache_config() {
        let config = CacheConfig::new(0, 100, 32, 128, 24);
        assert!(config.is_err());
        assert!(matches!(config, Err(KvCacheError::InvalidConfiguration)));
    }

    #[test]
    fn test_cache_preset_page_size() {
        assert_eq!(CachePreset::Small.page_size(), 16);
        assert_eq!(CachePreset::Medium.page_size(), 32);
        assert_eq!(CachePreset::Large.page_size(), 64);
        assert_eq!(CachePreset::Custom { page_size: 8, max_pages: 100 }.page_size(), 8);
    }

    #[test]
    fn test_cache_config_from_preset() {
        let config = CacheConfig::from_preset(CachePreset::Medium, 32, 128, 24);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.page_size, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert!(config.enable_compaction);
    }

    #[test]
    fn test_cache_config_with_compaction() {
        let config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
        assert!(!config.enable_compaction);

        let config = config.with_compaction(true);
        assert!(config.enable_compaction);
    }

    #[test]
    fn test_cache_config_max_context_length() {
        let config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
        assert_eq!(config.max_context_length(), 1600);
    }

    #[test]
    fn test_cache_config_estimated_memory() {
        let config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
        // 100 pages * 16 tokens/page * 32 heads * 128 dim * 2 (K+V) * 4 bytes/f32
        let expected = 100 * 16 * 32 * 128 * 2 * 4;
        assert_eq!(config.estimated_memory_bytes(), expected);
    }

    #[test]
    fn test_cache_config_for_context_length() {
        let config = CacheConfig::for_context_length(500, 32, 128, 24, None);
        assert!(config.is_ok());

        let config = config.unwrap();
        // Short context -> 16 token pages
        assert_eq!(config.page_size, 16);
    }
}
