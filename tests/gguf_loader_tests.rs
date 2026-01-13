//! GGUF Loader Tests
//!
//! Test suite for GGUF file parsing and tensor loading functionality.

use serial_test::serial;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use rocmforge::backend::hip_backend::{DeviceTensor, HipBackend};
use rocmforge::loader::gguf::{GgufLoader, GgufMetadata, GgufTensor};
use rocmforge::loader::TensorShape;
use rocmforge::model::config::ModelConfig;

/// Create a minimal synthetic GGUF file for testing
fn create_minimal_gguf_file(path: &Path) -> anyhow::Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // GGUF magic number: "GGUF"
    file.write_all(b"GGUF")?;

    // Version: 3
    file.write_all(&3u32.to_le_bytes())?;

    // Tensor count: 4
    file.write_all(&4u64.to_le_bytes())?;

    // KV count: 6
    file.write_all(&6u64.to_le_bytes())?;

    // Write KV pairs (metadata)
    let metadata = vec![
        ("general.architecture", "glm"),
        ("general.file_type", "1"), // Q8_0
        ("glm.n_layers", "2"),
        ("glm.n_heads", "8"),
        ("glm.n_embd", "512"),
        ("glm.intermediate_size", "2048"),
    ];

    for (key, value) in metadata {
        // Key length and key
        let key_bytes = key.as_bytes();
        file.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
        file.write_all(key_bytes)?;

        // Value type: string (8)
        file.write_all(&8u8.to_le_bytes())?;

        // Value length and value
        let value_bytes = value.as_bytes();
        file.write_all(&(value_bytes.len() as u64).to_le_bytes())?;
        file.write_all(value_bytes)?;
    }

    // Write tensor info
    let tensors = vec![
        ("token_embd.weight", vec![512, 1000]), // vocab_size=1000
        ("layers.0.attention.qkv.weight", vec![1536, 512]), // 3 * hidden_size
        ("layers.0.mlp.fc1.weight", vec![2048, 512]),
        ("layers.0.mlp.fc2.weight", vec![512, 2048]),
    ];

    for (name, shape) in &tensors {
        // Name length and name
        let name_bytes = name.as_bytes();
        file.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
        file.write_all(name_bytes)?;

        // Number of dimensions
        file.write_all(&(shape.len() as u32).to_le_bytes())?;

        // Dimensions
        for &dim in shape {
            file.write_all(&(dim as u64).to_le_bytes())?;
        }

        // Tensor type: Q8_0 (1)
        file.write_all(&1u32.to_le_bytes())?;

        // Tensor offset (placeholder, will be updated)
        file.write_all(&0u64.to_le_bytes())?;
    }

    // Write tensor data (simplified - just zeros)
    for (name, shape) in &tensors {
        let total_elements: usize = shape.iter().product();
        let data_size = if name.contains("embd") {
            // FP32 for embeddings
            total_elements * 4
        } else {
            // Q8_0 quantized: block_size=32, each block has 1 scale + 32 quants
            let blocks = (total_elements + 31) / 32;
            blocks * (4 + 32) // scale (f32) + quants (u8)
        };

        // Write zeros as placeholder data
        file.write_all(&vec![0u8; data_size])?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_gguf_file_parsing() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let gguf_path = temp_dir.path().join("test_model.gguf");

        // Create minimal GGUF file
        create_minimal_gguf_file(&gguf_path)?;

        // Parse the file
        let loader = GgufLoader::new(&gguf_path.to_string_lossy())?;
        let metadata = loader.metadata();

        // Verify metadata
        assert_eq!(metadata.architecture, "glm");
        assert_eq!(metadata.num_layers, 2);
        assert_eq!(metadata.num_heads, 8);
        assert_eq!(metadata.hidden_size, 512);
        assert_eq!(metadata.intermediate_size, 2048);

        Ok(())
    }

    #[test]
    fn test_tensor_loading() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let gguf_path = temp_dir.path().join("test_model.gguf");

        // Create minimal GGUF file
        create_minimal_gguf_file(&gguf_path)?;

        // Load tensors
        let loader = GgufLoader::new(&gguf_path.to_string_lossy())?;
        let tensors = loader.load_tensors()?;

        // Verify tensor count
        assert_eq!(tensors.len(), 4);

        // Verify specific tensors
        let token_embd = tensors.get("token_embd.weight").unwrap();
        assert_eq!(token_embd.shape.dims(), &[512, 1000]);

        let qkv_weight = tensors.get("layers.0.attention.qkv.weight").unwrap();
        assert_eq!(qkv_weight.shape.dims(), &[1536, 512]);

        let fc1_weight = tensors.get("layers.0.mlp.fc1.weight").unwrap();
        assert_eq!(fc1_weight.shape.dims(), &[2048, 512]);

        let fc2_weight = tensors.get("layers.0.mlp.fc2.weight").unwrap();
        assert_eq!(fc2_weight.shape.dims(), &[512, 2048]);

        Ok(())
    }

    #[test]
    #[serial]
    fn test_gpu_tensor_upload() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let gguf_path = temp_dir.path().join("test_model.gguf");

        // Create minimal GGUF file
        create_minimal_gguf_file(&gguf_path)?;

        // Load and upload to GPU
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let loader = GgufLoader::new(&gguf_path.to_string_lossy())?;
        let tensors = loader.load_to_gpu(&backend)?;

        // Verify GPU tensors
        assert_eq!(tensors.len(), 4);

        for (name, tensor) in &tensors {
            assert!(
                !tensor.as_ptr().is_null(),
                "Tensor {} should have valid GPU pointer",
                name
            );
            assert!(
                tensor.size() > 0,
                "Tensor {} should have non-zero size",
                name
            );
        }

        // Check for memory leaks
        fixture.assert_no_leak(5);

        Ok(())
    }

    #[test]
    fn test_model_config_integration() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let gguf_path = temp_dir.path().join("test_model.gguf");

        // Create minimal GGUF file
        create_minimal_gguf_file(&gguf_path)?;

        // Load and create model config
        let loader = GgufLoader::new(&gguf_path.to_string_lossy())?;
        let config = loader.to_model_config()?;

        // Verify config
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.intermediate_size, 2048);
        assert_eq!(config.vocab_size, 1000);

        Ok(())
    }

    #[test]
    fn test_quantization_handling() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let gguf_path = temp_dir.path().join("test_model.gguf");

        // Create minimal GGUF file
        create_minimal_gguf_file(&gguf_path)?;

        // Load and verify quantization types
        let loader = GgufLoader::new(&gguf_path.to_string_lossy())?;
        let tensors = loader.load_tensors()?;

        // Embeddings should be FP32
        let token_embd = tensors.get("token_embd.weight").unwrap();
        assert_eq!(token_embd.quant_type, "FP32");

        // Other weights should be Q8_0
        for (name, tensor) in &tensors {
            if name != "token_embd.weight" {
                assert_eq!(tensor.quant_type, "Q8_0");
            }
        }

        Ok(())
    }

    #[test]
    fn test_shape_validation() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let gguf_path = temp_dir.path().join("test_model.gguf");

        // Create minimal GGUF file
        create_minimal_gguf_file(&gguf_path)?;

        // Load and validate shapes
        let loader = GgufLoader::new(&gguf_path.to_string_lossy())?;
        let tensors = loader.load_tensors()?;

        // Validate QKV weight shape: [3 * hidden_size, hidden_size]
        let qkv_weight = tensors.get("layers.0.attention.qkv.weight").unwrap();
        assert_eq!(qkv_weight.shape.dims()[0], 3 * 512); // 3 * hidden_size
        assert_eq!(qkv_weight.shape.dims()[1], 512); // hidden_size

        // Validate MLP shapes
        let fc1_weight = tensors.get("layers.0.mlp.fc1.weight").unwrap();
        assert_eq!(fc1_weight.shape.dims(), &[2048, 512]); // [intermediate_size, hidden_size]

        let fc2_weight = tensors.get("layers.0.mlp.fc2.weight").unwrap();
        assert_eq!(fc2_weight.shape.dims(), &[512, 2048]); // [hidden_size, intermediate_size]

        Ok(())
    }
}
