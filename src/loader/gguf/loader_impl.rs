//! GGUF Loader Implementation
//!
//! This module contains the core implementation of GgufLoader:
//! - File loading and parsing
//! - Model config conversion
//! - Internal helper methods

use crate::loader::gguf::{header, metadata, tensor_info, types::GgufLoader, GgufTensor};
use crate::loader::{lazy_tensor::LazyTensor, TensorShape};
use crate::model::config::ModelConfig;
use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

/// Create new GGUF loader from file path
///
/// # Phase 1 Lazy Loading
///
/// This method now initializes the loader with lazy loading support:
/// - Opens the GGUF file and memory-maps it for zero-copy access
/// - Parses metadata (KV pairs, tensor info)
/// - Creates `LazyTensor` handles for all tensors (metadata only)
/// - Does NOT load tensor data into RAM
///
/// # Performance
///
/// - Before Phase 1: ~60s (loaded all tensor data)
/// - After Phase 1: ~5s (metadata only)
pub fn new_loader(path: &str) -> Result<GgufLoader> {
    use crate::loader::mmap::MmapGguf;
    use crate::loader::metadata::GgufMetadata;

    // Create memory-mapped file for zero-copy access
    let mmap = MmapGguf::open(Path::new(path))
        .map_err(|e| anyhow!("Failed to memory-map GGUF file '{}': {}", path, e))?;

    let mut loader = GgufLoader {
        path: path.to_string(),
        metadata: GgufMetadata::default(),
        tensors: std::collections::HashMap::new(),
        mmap: Some(Arc::new(mmap)),
        lazy_tensors: std::collections::HashMap::new(),
        gpu_cache: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
    };

    // Parse metadata and tensor info (but NOT tensor data)
    load_from_disk_impl(&mut loader, true)?;
    Ok(loader)
}

/// Internal implementation for loading from disk
///
/// # Arguments
///
/// * `loader` - Mutable reference to the loader
/// * `load_tensors` - If true, parse tensor info; if false, metadata only
pub fn load_from_disk_impl(loader: &mut GgufLoader, load_tensors: bool) -> Result<()> {
    let mut file = File::open(&loader.path)?;

    // Read and verify magic number
    header::validate_gguf_magic(&mut file)?;

    // Read version
    let version = header::read_gguf_version(&mut file)?;

    // Read tensor count
    let tensor_count = header::read_tensor_count(&mut file)?;

    // Read KV count
    let kv_count = header::read_kv_count(&mut file)?;

    // Parse KV pairs (metadata)
    metadata::parse_kv_pairs(&mut file, kv_count, &mut loader.metadata)?;

    // Calculate default head_dim if not set by GGUF
    // Per llama.cpp pattern: calculate default before optional override
    loader.metadata.calculate_default_head_dim();

    if load_tensors {
        // Parse tensor info (creates LazyTensor handles)
        let (tensors, lazy_tensors) = tensor_info::parse_tensor_info(&mut file, tensor_count)?;

        loader.tensors = tensors;
        loader.lazy_tensors = lazy_tensors;

        // Phase 1 Lazy Loading: Skip read_tensor_data()
        // Tensor data is loaded on-demand from memory-mapped file via load_tensor_to_gpu()
        // This reduces RAM usage and initial load time significantly
        tracing::debug!("Phase 1: Skipping tensor data load - will use mmap on-demand");
    }

    tracing::debug!(
        "Loaded GGUF v{} with {} tensors, {} KV pairs",
        version,
        tensor_count,
        kv_count
    );

    Ok(())
}

/// Convert loader metadata to ModelConfig
pub fn to_model_config(loader: &GgufLoader) -> Result<ModelConfig> {
    use crate::model::config::ModelType;

    // Determine vocab_size: metadata, inference, or default
    let vocab_size = if loader.metadata.vocab_size > 0 {
        // Metadata has explicit vocab_size
        loader.metadata.vocab_size
    } else {
        // Try to infer from tensor shapes
        match loader.infer_vocab_size_from_tensors() {
            Some(inferred) => inferred,
            None => {
                // Last resort: architecture-specific defaults
                let default = match loader.metadata.architecture.as_str() {
                    "qwen2" => 151936,
                    "llama" => 32000,
                    "glm" => 151552,
                    _ => 32000,
                };
                tracing::info!(
                    "GGUF: Using default vocab_size={} for '{}'",
                    default,
                    loader.metadata.architecture
                );
                default
            }
        }
    };

    // Determine intermediate_size: metadata, inference, or default
    let intermediate_size = if loader.metadata.intermediate_size > 0 {
        // Metadata has explicit intermediate_size
        loader.metadata.intermediate_size
    } else {
        // Try to infer from tensor shapes (MLP gate weights)
        match loader.infer_intermediate_size_from_tensors() {
            Some(inferred) => inferred,
            None => {
                // Last resort: use 4x hidden_size (common FFN expansion ratio)
                let default = loader.metadata.hidden_size * 4;
                tracing::info!(
                    "GGUF: Using default intermediate_size={} (4x hidden_size) for '{}'",
                    default,
                    loader.metadata.architecture
                );
                default
            }
        }
    };

    Ok(ModelConfig {
        num_hidden_layers: loader.metadata.num_layers,
        num_attention_heads: loader.metadata.num_heads,
        num_kv_heads: loader.metadata.num_kv_heads,
        hidden_size: loader.metadata.hidden_size,
        intermediate_size,
        max_position_embeddings: loader.metadata.max_position_embeddings,
        vocab_size,
        rms_norm_eps: loader.metadata.rms_norm_eps,
        use_rotary_embeddings: loader.metadata.use_rotary_embeddings,
        model_type: if loader.metadata.architecture == "glm" {
            ModelType::Llama // Use Llama as placeholder for now
        } else {
            ModelType::Llama
        },
        head_dim: if loader.metadata.head_dim > 0 {
            loader.metadata.head_dim
        } else if loader.metadata.num_heads > 0 {
            loader.metadata.hidden_size / loader.metadata.num_heads
        } else {
            128 // Safe fallback
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_model_config_basic() {
        use crate::loader::metadata::GgufMetadata;

        let mut loader = GgufLoader {
            path: "test.gguf".to_string(),
            metadata: GgufMetadata {
                architecture: "qwen2".to_string(),
                num_layers: 24,
                num_heads: 14,
                num_kv_heads: Some(2),
                hidden_size: 896,
                intermediate_size: 4864,
                max_position_embeddings: 32768,
                vocab_size: 151936,
                head_dim: 64,
                rms_norm_eps: 1e-6,
                use_rotary_embeddings: true,
                ..Default::default()
            },
            tensors: std::collections::HashMap::new(),
            mmap: None,
            lazy_tensors: std::collections::HashMap::new(),
            gpu_cache: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        };

        let config = to_model_config(&loader).unwrap();
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_kv_heads, Some(2));
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.intermediate_size, 4864);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_to_model_config_fallback() {
        use crate::loader::metadata::GgufMetadata;

        let mut loader = GgufLoader {
            path: "test.gguf".to_string(),
            metadata: GgufMetadata {
                architecture: "llama".to_string(),
                num_layers: 32,
                num_heads: 32,
                num_kv_heads: None,
                hidden_size: 4096,
                intermediate_size: 0, // Will use 4x hidden_size fallback
                max_position_embeddings: 2048,
                vocab_size: 0, // Will use default 32000
                head_dim: 0,   // Will be calculated
                rms_norm_eps: 1e-6,
                use_rotary_embeddings: true,
                ..Default::default()
            },
            tensors: std::collections::HashMap::new(),
            mmap: None,
            lazy_tensors: std::collections::HashMap::new(),
            gpu_cache: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        };

        let config = to_model_config(&loader).unwrap();
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.vocab_size, 32000); // Default for llama
        assert_eq!(config.intermediate_size, 4096 * 4); // 4x fallback
        assert_eq!(config.head_dim, 4096 / 32); // Calculated
    }
}
