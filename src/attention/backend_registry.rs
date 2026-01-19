//! Attention backend registry with pluggable implementations
//! Supports CPU and GPU (ROCm) backends with runtime selection

use thiserror::Error;

/// Error type for attention backend operations
#[derive(Error, Debug)]
pub enum AttentionBackendError {
    #[error("Backend not found: {0}")]
    NotFound(String),
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Operation failed: {0}")]
    OperationFailed(String),
    #[error("Configuration not supported: {0}")]
    NotSupported(String),
}

pub type AttentionBackendResult<T> = Result<T, AttentionBackendError>;

/// Trait for attention backend implementations
///
/// This trait allows pluggable attention backends (CPU, GPU, future optimized variants).
/// Each backend declares its capabilities and can be selected at runtime.
///
/// Note: This is the pluggable backend interface. For the simple enum used
/// for backend selection in the Attention struct, see `BackendType` in backend.rs.
pub trait BackendImplementation: Send + Sync {
    /// Get the name of this backend (e.g., "cpu", "gpu", "flash_attention")
    fn name(&self) -> &str;

    /// Check if this backend supports the given configuration
    fn supports(&self, config: &AttentionConfig) -> bool;

    /// Get the required KV cache layout (if any)
    /// Returns None if the backend doesn't have specific layout requirements
    fn required_kv_layout(&self) -> Option<KvCacheLayout>;

    /// Execute attention computation
    ///
    /// # Arguments
    /// * `config` - Attention configuration (dimensions, heads, etc.)
    /// * `q` - Query tensor [batch_size, seq_len, head_dim]
    /// * `k` - Key tensor [batch_size, seq_len, head_dim]
    /// * `v` - Value tensor [batch_size, seq_len, head_dim]
    /// * `mask` - Optional attention mask [batch_size, seq_len, seq_len]
    ///
    /// # Returns
    /// Output tensor [batch_size, seq_len, head_dim]
    fn forward(
        &self,
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
    ) -> AttentionBackendResult<Vec<f32>>;
}

/// Configuration for attention operations
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Model dimension
    pub dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension (must equal dim / num_heads)
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Whether to use causal masking
    pub is_causal: bool,
    /// Dropout probability
    pub dropout: Option<f32>,
}

impl AttentionConfig {
    pub fn new(dim: usize, num_heads: usize, head_dim: usize) -> Self {
        AttentionConfig {
            dim,
            num_heads,
            head_dim,
            max_sequence_length: 4096,
            is_causal: false,
            dropout: None,
        }
    }

    pub fn with_causal(mut self, is_causal: bool) -> Self {
        self.is_causal = is_causal;
        self
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = Some(dropout);
        self
    }

    pub fn with_max_sequence_length(mut self, max_sequence_length: usize) -> Self {
        self.max_sequence_length = max_sequence_length;
        self
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> Result<(), String> {
        if self.dim % self.num_heads != 0 {
            return Err(format!(
                "dim ({}) must be divisible by num_heads ({})",
                self.dim, self.num_heads
            ));
        }
        if self.head_dim != self.dim / self.num_heads {
            return Err(format!(
                "head_dim ({}) must equal dim ({}) / num_heads ({}), got {}",
                self.head_dim,
                self.dim,
                self.num_heads,
                self.dim / self.num_heads
            ));
        }
        Ok(())
    }
}

/// KV cache layout options
///
/// Different attention backends may require different KV cache layouts
/// for optimal performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheLayout {
    /// Standard contiguous layout [batch, seq_len, heads, head_dim]
    Contiguous,
    /// Block-sparse layout (for PagedAttention)
    BlockSparse,
    /// FlashAttention-optimized layout
    FlashAttention,
}

/// Attention backend registry with pluggable implementations
///
/// The registry manages multiple attention backends and selects the best one
/// based on configuration and system capabilities.
pub struct AttentionBackendRegistry {
    backends: Vec<Box<dyn BackendImplementation>>,
    default_backend: Option<String>,
}

impl AttentionBackendRegistry {
    pub fn new() -> Self {
        let mut backends: Vec<Box<dyn BackendImplementation>> =
            vec![Box::new(cpu_backend::CpuAttentionBackend::new())];

        #[cfg(feature = "rocm")]
        {
            backends.push(Box::new(gpu_backend::GpuAttentionBackend::new()));
            // FlashAttention backend - registered but not auto-selected by default
            // Users can explicitly set it as default via set_default()
            backends.push(Box::new(flash_attention_backend::FlashAttentionBackend::new()));
        }

        AttentionBackendRegistry {
            backends,
            default_backend: None,
        }
    }

    /// Register a new backend
    pub fn register(&mut self, backend: Box<dyn BackendImplementation>) {
        self.backends.push(backend);
    }

    /// Select the best backend for the given configuration
    ///
    /// Selection logic:
    /// 1. If default_backend is set, use it (if it supports the config)
    /// 2. Otherwise, auto-select the first backend that supports the config
    pub fn select_backend(
        &self,
        config: &AttentionConfig,
    ) -> AttentionBackendResult<&dyn BackendImplementation> {
        // First try default if set
        if let Some(ref default_name) = self.default_backend {
            if let Some(backend) = self.backends.iter().find(|b| b.name() == default_name) {
                if backend.supports(config) {
                    return Ok(backend.as_ref());
                }
                return Err(AttentionBackendError::NotSupported(format!(
                    "Default backend '{}' does not support this configuration",
                    default_name
                )));
            }
            return Err(AttentionBackendError::NotFound(format!(
                "Default backend '{}' not found",
                default_name
            )));
        }

        // Auto-select based on configuration
        for backend in &self.backends {
            if backend.supports(config) {
                return Ok(backend.as_ref());
            }
        }

        Err(AttentionBackendError::NotFound(
            "No suitable backend found for configuration".to_string(),
        ))
    }

    /// Set the default backend by name
    pub fn set_default(&mut self, name: String) -> AttentionBackendResult<()> {
        let exists = self.backends.iter().any(|b| b.name() == name);
        if exists {
            self.default_backend = Some(name);
            Ok(())
        } else {
            Err(AttentionBackendError::NotFound(name))
        }
    }

    /// Get all registered backend names
    pub fn list_backends(&self) -> Vec<String> {
        self.backends.iter().map(|b| b.name().to_string()).collect()
    }

    /// Get backend by name
    pub fn get_backend(&self, name: &str) -> AttentionBackendResult<&dyn BackendImplementation> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
            .ok_or_else(|| AttentionBackendError::NotFound(name.to_string()))
    }
}

impl Default for AttentionBackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CPU Backend Implementation
// ============================================================================

/// CPU backend implementation using existing cpu::CpuBackend
pub mod cpu_backend {
    use super::*;

    pub struct CpuAttentionBackend;

    impl CpuAttentionBackend {
        pub fn new() -> Self {
            CpuAttentionBackend
        }
    }

    impl BackendImplementation for CpuAttentionBackend {
        fn name(&self) -> &str {
            "cpu"
        }

        fn supports(&self, _config: &AttentionConfig) -> bool {
            // CPU always supports everything (fallback)
            true
        }

        fn required_kv_layout(&self) -> Option<KvCacheLayout> {
            Some(KvCacheLayout::Contiguous)
        }

        fn forward(
            &self,
            config: &AttentionConfig,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            mask: Option<&[f32]>,
        ) -> AttentionBackendResult<Vec<f32>> {
            // Validate config
            config
                .validate()
                .map_err(|e| AttentionBackendError::NotSupported(e))?;

            // Call existing CPU implementation
            super::super::cpu::CpuBackend::forward(config.dim, q, k, v, mask, config.dropout)
                .map_err(|e| AttentionBackendError::OperationFailed(e.to_string()))
        }
    }
}

// ============================================================================
// GPU Backend Implementation
// ============================================================================

#[cfg(feature = "rocm")]
/// GPU backend implementation using existing gpu::GpuBackend
pub mod gpu_backend {
    use super::*;

    pub struct GpuAttentionBackend {
        #[allow(dead_code)] // Will be used when FlashAttention is implemented
        use_flash_attention: bool,
    }

    impl GpuAttentionBackend {
        pub fn new() -> Self {
            GpuAttentionBackend {
                use_flash_attention: false, // TODO: detect from system config
            }
        }

        pub fn with_flash_attention(mut self, use_flash: bool) -> Self {
            self.use_flash_attention = use_flash;
            self
        }
    }

    impl BackendImplementation for GpuAttentionBackend {
        fn name(&self) -> &str {
            "gpu"
        }

        fn supports(&self, config: &AttentionConfig) -> bool {
            // GPU supports configurations where dim is divisible by num_heads
            config.validate().is_ok()
        }

        fn required_kv_layout(&self) -> Option<KvCacheLayout> {
            if self.use_flash_attention {
                Some(KvCacheLayout::FlashAttention)
            } else {
                Some(KvCacheLayout::BlockSparse)
            }
        }

        fn forward(
            &self,
            config: &AttentionConfig,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            mask: Option<&[f32]>,
        ) -> AttentionBackendResult<Vec<f32>> {
            // Validate config
            config
                .validate()
                .map_err(|e| AttentionBackendError::NotSupported(e))?;

            // Call existing GPU implementation
            super::super::gpu::GpuBackend::forward(config.dim, q, k, v, mask, config.dropout)
                .map_err(|e| AttentionBackendError::OperationFailed(e.to_string()))
        }
    }
}

// ============================================================================
// FlashAttention Backend Implementation
// ============================================================================

#[cfg(feature = "rocm")]
/// FlashAttention backend implementation using fused kernels
pub mod flash_attention_backend {
    use super::*;

    pub struct FlashAttentionBackend {
        #[allow(dead_code)] // Will be configured in future
        max_seq_len: usize,
    }

    impl FlashAttentionBackend {
        pub fn new() -> Self {
            FlashAttentionBackend {
                max_seq_len: 2048, // Default maximum sequence length
            }
        }

        pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
            self.max_seq_len = max_seq_len;
            self
        }
    }

    impl BackendImplementation for FlashAttentionBackend {
        fn name(&self) -> &str {
            "flash_attention"
        }

        fn supports(&self, config: &AttentionConfig) -> bool {
            // Flash attention requires:
            // - ROCm feature enabled (checked by #[cfg])
            // - Head dimension <= 128 (register limit)
            // - Sequence length <= max_sequence_length
            config.head_dim <= 128
                && config.max_sequence_length <= self.max_seq_len
        }

        fn required_kv_layout(&self) -> Option<KvCacheLayout> {
            Some(KvCacheLayout::FlashAttention)
        }

        fn forward(
            &self,
            config: &AttentionConfig,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            mask: Option<&[f32]>,
        ) -> AttentionBackendResult<Vec<f32>> {
            // Validate config
            config
                .validate()
                .map_err(|e| AttentionBackendError::NotSupported(e))?;

            // For now, delegate to GPU implementation
            // In phase 06-03, this will call the actual flash kernels
            super::super::gpu::GpuBackend::forward(config.dim, q, k, v, mask, config.dropout)
                .map_err(|e| AttentionBackendError::OperationFailed(e.to_string()))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = AttentionBackendRegistry::new();
        let backends = registry.list_backends();

        #[cfg(feature = "rocm")]
        assert_eq!(backends.len(), 3); // cpu + gpu + flash_attention
        #[cfg(not(feature = "rocm"))]
        assert_eq!(backends.len(), 1); // cpu only

        assert!(backends.contains(&"cpu".to_string()));
    }

    #[test]
    fn test_cpu_backend_selection() {
        let registry = AttentionBackendRegistry::new();
        let config = AttentionConfig::new(512, 8, 64);

        let backend = registry.select_backend(&config).unwrap();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_gpu_backend_selection() {
        let _registry = AttentionBackendRegistry::new();
        let _config = AttentionConfig::new(512, 8, 64);

        #[cfg(feature = "rocm")]
        {
            // With rocm feature, GPU should be selected
            // Note: Backend selection depends on available hardware
            // This test just verifies the config can be created
        }
    }

    #[test]
    fn test_set_default_backend() {
        let mut registry = AttentionBackendRegistry::new();

        // Set CPU as default
        registry.set_default("cpu".to_string()).unwrap();

        let config = AttentionConfig::new(512, 8, 64);
        let backend = registry.select_backend(&config).unwrap();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_set_nonexistent_backend() {
        let mut registry = AttentionBackendRegistry::new();
        let result = registry.set_default("nonexistent".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_get_backend_by_name() {
        let registry = AttentionBackendRegistry::new();

        let cpu_backend = registry.get_backend("cpu").unwrap();
        assert_eq!(cpu_backend.name(), "cpu");

        let result = registry.get_backend("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_get_flash_attention_backend() {
        let registry = AttentionBackendRegistry::new();

        let flash_backend = registry.get_backend("flash_attention").unwrap();
        assert_eq!(flash_backend.name(), "flash_attention");
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_set_flash_attention_as_default() {
        let mut registry = AttentionBackendRegistry::new();

        // Set flash_attention as default
        registry.set_default("flash_attention".to_string()).unwrap();

        // Flash attention should be selected for compatible configs
        let config = AttentionConfig::new(512, 8, 64)
            .with_max_sequence_length(1024);

        let backend = registry.select_backend(&config).unwrap();
        assert_eq!(backend.name(), "flash_attention");
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_flash_attention_supports_valid_config() {
        let registry = AttentionBackendRegistry::new();
        let flash_backend = registry.get_backend("flash_attention").unwrap();

        // Valid config: head_dim=64 <= 128, max_sequence_length=1024 <= 2048
        let config = AttentionConfig::new(512, 8, 64)
            .with_max_sequence_length(1024);

        assert!(flash_backend.supports(&config));
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_flash_attention_does_not_support_large_head_dim() {
        let registry = AttentionBackendRegistry::new();
        let flash_backend = registry.get_backend("flash_attention").unwrap();

        // Invalid config: head_dim=129 > 128
        let config = AttentionConfig::new(516, 4, 129)
            .with_max_sequence_length(1024);

        assert!(!flash_backend.supports(&config));
    }

    #[test]
    fn test_attention_config_validation() {
        // Valid config
        let config = AttentionConfig::new(512, 8, 64);
        assert!(config.validate().is_ok());

        // Invalid: dim not divisible by num_heads
        let config = AttentionConfig::new(513, 8, 64);
        assert!(config.validate().is_err());

        // Invalid: head_dim doesn't match
        let config = AttentionConfig::new(512, 8, 63);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_attention_config_builder() {
        let config = AttentionConfig::new(512, 8, 64)
            .with_causal(true)
            .with_dropout(0.1)
            .with_max_sequence_length(8192);

        assert_eq!(config.dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.is_causal, true);
        assert_eq!(config.dropout, Some(0.1));
        assert_eq!(config.max_sequence_length, 8192);
    }

    #[test]
    fn test_cpu_backend_forward() {
        let registry = AttentionBackendRegistry::new();
        let backend = registry.get_backend("cpu").unwrap();

        // CPU backend expects seq_len = dim, input shape = [batch_size, seq_len, dim]
        let config = AttentionConfig::new(16, 4, 4);

        // Test data: batch=1, seq=16, dim=16
        // Total size = batch_size * seq_len * dim = 1 * 16 * 16 = 256
        let q = vec![0.1f32; 256];
        let k = vec![0.2f32; 256];
        let v = vec![0.3f32; 256];

        let result = backend.forward(&config, &q, &k, &v, None);
        assert!(
            result.is_ok(),
            "CPU backend forward failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert_eq!(output.len(), q.len());
    }
}
