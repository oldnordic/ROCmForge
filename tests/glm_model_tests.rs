//! GLM Model Integration Tests
//!
//! Test suite for GLM model loading and inference functionality.

use serial_test::serial;
use std::fs;
use std::io::Write;
use std::path::Path;

use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use rocmforge::backend::hip_backend::{DeviceTensor, HipBackend, ModelRuntime};
use rocmforge::loader::gguf::GgufLoader;
use rocmforge::model::config::ModelConfig;
use rocmforge::model::execution_plan::ExecutionPlan;

/// Create a synthetic GLM GGUF model for testing
fn create_synthetic_glm_model(path: &Path) -> anyhow::Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // GGUF magic number: "GGUF"
    file.write_all(b"GGUF")?;

    // Version: 3
    file.write_all(&3u32.to_le_bytes())?;

    // Tensor count: comprehensive GLM model
    let tensor_count = 1 + // token_embd
        2 * (1 + // layer 0
        1 + // layer 1
        1) + // output
        1; // final_norm
    file.write_all(&(tensor_count as u64).to_le_bytes())?;

    // KV count: GLM metadata
    let kv_count = 10;
    file.write_all(&(kv_count as u64).to_le_bytes())?;

    // Write KV pairs (metadata)
    let metadata = vec![
        ("general.architecture", "glm"),
        ("general.file_type", "1"), // Q8_0
        ("glm.n_layers", "2"),
        ("glm.n_heads", "8"),
        ("glm.n_embd", "512"),
        ("glm.intermediate_size", "2048"),
        ("glm.head_dim", "64"),
        ("glm.max_position_embeddings", "2048"),
        ("glm.vocab_size", "1000"),
        ("glm.rms_norm_eps", "1e-6"),
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

    // Define all tensors for a complete GLM model
    let mut tensors = Vec::new();

    // Token embeddings
    tensors.push(("token_embd.weight", vec![512, 1000])); // [hidden_size, vocab_size]

    // Layer 0
    tensors.push(("layers.0.attention.qkv.weight", vec![1536, 512])); // [3 * hidden_size, hidden_size]
    tensors.push(("layers.0.attention.o_proj.weight", vec![512, 1536])); // [hidden_size, 3 * hidden_size]
    tensors.push(("layers.0.mlp.gate_proj.weight", vec![2048, 512])); // [intermediate_size, hidden_size]
    tensors.push(("layers.0.mlp.up_proj.weight", vec![2048, 512])); // [intermediate_size, hidden_size]
    tensors.push(("layers.0.mlp.down_proj.weight", vec![512, 2048])); // [hidden_size, intermediate_size]
    tensors.push(("layers.0.attention_norm.weight", vec![512])); // [hidden_size]
    tensors.push(("layers.0.ffn_norm.weight", vec![512])); // [hidden_size]

    // Layer 1
    tensors.push(("layers.1.attention.qkv.weight", vec![1536, 512]));
    tensors.push(("layers.1.attention.o_proj.weight", vec![512, 1536]));
    tensors.push(("layers.1.mlp.gate_proj.weight", vec![2048, 512]));
    tensors.push(("layers.1.mlp.up_proj.weight", vec![2048, 512]));
    tensors.push(("layers.1.mlp.down_proj.weight", vec![512, 2048]));
    tensors.push(("layers.1.attention_norm.weight", vec![512]));
    tensors.push(("layers.1.ffn_norm.weight", vec![512]));

    // Final norm and output
    tensors.push(("norm.weight", vec![512])); // [hidden_size]
    tensors.push(("lm_head.weight", vec![1000, 512])); // [vocab_size, hidden_size]

    // Write tensor info
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

        // Tensor type: Q8_0 (1) for weights, FP32 (0) for embeddings and norms
        let tensor_type =
            if name.contains("embd") || name.contains("norm") || name.contains("lm_head") {
                0u32 // FP32
            } else {
                1u32 // Q8_0
            };
        file.write_all(&tensor_type.to_le_bytes())?;

        // Tensor offset (placeholder, will be updated)
        file.write_all(&0u64.to_le_bytes())?;
    }

    // Write tensor data (simplified - just zeros)
    for (name, shape) in &tensors {
        let total_elements: usize = shape.iter().product();
        let data_size =
            if name.contains("embd") || name.contains("norm") || name.contains("lm_head") {
                // FP32
                total_elements * 4
            } else {
                // Q8_0 quantized
                let blocks = (total_elements + 31) / 32;
                blocks * (4 + 32)
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
    fn test_glm_model_loading() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("synthetic_glm.gguf");

        // Create synthetic GLM model
        create_synthetic_glm_model(&model_path)?;

        // Load model
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let loader = GgufLoader::new(&model_path.to_string_lossy())?;
        let config = loader.to_model_config()?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;

        // Verify model structure
        assert_eq!(execution_plan.num_layers(), 2);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.hidden_size, 512);

        // Verify layer plans have all required tensors
        for layer_idx in 0..2 {
            let layer_plan = execution_plan.layers().get(layer_idx).context("TODO: add error context")?;

            // Check QKV projection
            assert_eq!(layer_plan.qkv_weight.shape().context("TODO: add error context")?, &[1536, 512]);

            // Check MLP projections
            assert_eq!(layer_plan.mlp_gate_proj.shape().context("TODO: add error context")?, &[2048, 512]);
            assert_eq!(layer_plan.mlp_down_proj.shape().context("TODO: add error context")?, &[512, 2048]);

            // Check layer norms
            assert_eq!(layer_plan.norm1_weight.shape().context("TODO: add error context")?, &[512]);
            assert_eq!(layer_plan.norm2_weight.shape().context("TODO: add error context")?, &[512]);
        }

        Ok(())
    }

    #[test]
    fn test_single_token_decode() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("synthetic_glm.gguf");

        // Create synthetic GLM model
        create_synthetic_glm_model(&model_path)?;

        // Load model
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let loader = GgufLoader::new(&model_path.to_string_lossy())?;
        let config = loader.to_model_config()?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;

        // Create model runtime
        let mut runtime = ModelRuntime::from_execution_plan_with_backend(execution_plan)?;

        // Create input tensor (single token)
        let input_shape = rocmforge::loader::TensorShape::from_dims(&[1, 512]); // [seq_len=1, hidden_size]
        let mut input_data = vec![0.1f32; 512]; // Simple test input
        let input_tensor = DeviceTensor::from_host_vec(backend, input_data, input_shape)?;

        // Run decode step
        let output = runtime.decode_step(&input_tensor)?;

        // Verify output logits shape
        assert_eq!(output.shape().dims(), &[config.vocab_size]);
        assert!(!output.as_ptr().is_null());

        // Verify output is finite (no NaN or Inf)
        let output_data = output.to_host_vec()?;
        for &val in &output_data {
            assert!(val.is_finite(), "Output contains non-finite values");
        }

        Ok(())
    }

    // REMOVED: Duplicate test_model_runtime_creation
    // This test is already in model_runtime_tests.rs:14

    #[test]
    fn test_glm_layer_structure() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("synthetic_glm.gguf");

        // Create synthetic GLM model
        create_synthetic_glm_model(&model_path)?;

        // Load model
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let loader = GgufLoader::new(&model_path.to_string_lossy())?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;

        // Verify GLM layer structure: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
        for layer_idx in 0..2 {
            let layer_plan = execution_plan.layers().get(layer_idx).context("TODO: add error context")?;

            // Should have attention norms (attention_norm, ffn_norm in GLM)
            assert!(layer_plan.norm1_weight.shape().context("TODO: add error context")?.iter().product::<usize>() > 0);
            assert!(layer_plan.norm2_weight.shape().context("TODO: add error context")?.iter().product::<usize>() > 0);

            // Should have QKV and output projections
            assert!(layer_plan.qkv_weight.shape().context("TODO: add error context")?.iter().product::<usize>() > 0);
            assert!(layer_plan.o_proj.shape().context("TODO: add error context")?.iter().product::<usize>() > 0);

            // Should have MLP projections (gate, up, down in GLM)
            assert!(layer_plan.mlp_gate_proj.shape().context("TODO: add error context")?.iter().product::<usize>() > 0); // gate_proj
            assert!(layer_plan.mlp_down_proj.shape().context("TODO: add error context")?.iter().product::<usize>() > 0); // down_proj
        }

        Ok(())
    }

    #[test]
    fn test_position_ids_handling() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("synthetic_glm.gguf");

        // Create synthetic GLM model
        create_synthetic_glm_model(&model_path)?;

        // Load model
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let loader = GgufLoader::new(&model_path.to_string_lossy())?;
        let config = loader.to_model_config()?;

        // Verify GLM position embedding configuration
        assert_eq!(config.max_position_embeddings, 2048);
        assert_eq!(config.head_dim, 64);

        // GLM uses rotary embeddings with specific position handling
        // This test verifies the configuration is correctly loaded
        assert!(config.use_rotary_embeddings);

        Ok(())
    }

    #[test]
    fn test_multi_query_attention_support() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("synthetic_glm.gguf");

        // Create synthetic GLM model
        create_synthetic_glm_model(&model_path)?;

        // Load model
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let loader = GgufLoader::new(&model_path.to_string_lossy())?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;

        // Verify QKV projection structure supports multi-query attention
        for layer_idx in 0..2 {
            let layer_plan = execution_plan.layers().get(layer_idx).context("TODO: add error context")?;
            let qkv_shape = layer_plan.qkv_weight.shape().context("TODO: add error context")?;

            // QKV should be [3 * hidden_size, hidden_size] for standard attention
            // or [hidden_size + 2 * head_dim, hidden_size] for multi-query
            assert_eq!(qkv_shape[1], 512); // hidden_size
            assert!(qkv_shape[0] == 1536); // 3 * hidden_size (standard)
        }

        Ok(())
    }

    #[test]
    fn test_output_shape_correctness() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("synthetic_glm.gguf");

        // Create synthetic GLM model
        create_synthetic_glm_model(&model_path)?;

        // Load model
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let loader = GgufLoader::new(&model_path.to_string_lossy())?;
        let config = loader.to_model_config()?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;

        // Create model runtime
        let mut runtime = ModelRuntime::from_execution_plan_with_backend(execution_plan)?;

        // Test different sequence lengths
        for seq_len in [1, 2, 4, 8] {
            let input_shape = rocmforge::loader::TensorShape::from_dims(&[seq_len, 512]);
            let input_data = vec![0.1f32; seq_len * 512];
            let input_tensor = DeviceTensor::from_host_vec(backend, input_data, input_shape)?;

            let output = runtime.decode_step(&input_tensor)?;

            // Output should map hidden states to vocab logits
            if seq_len == 1 {
                assert_eq!(output.shape().dims(), &[config.vocab_size]);
                assert_eq!(output.len(), config.vocab_size);
            } else {
                assert_eq!(output.shape().dims(), &[seq_len, config.vocab_size]);
                assert_eq!(output.len(), seq_len * config.vocab_size);
            }
        }

        Ok(())
    }
}
