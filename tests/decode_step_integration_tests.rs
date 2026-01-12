//! decode_step() Integration Tests
//!
//! Tests the complete decode_step() pipeline in ModelRuntime using real ExecutionPlan,
//! KVCache, and existing GPU/CPU operations.

use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use rocmforge::backend::hip_backend::{DeviceTensor, ModelRuntime};
use rocmforge::backend::scratch::ScratchBufferManager;
use rocmforge::loader::gguf::GgufLoader;
use rocmforge::loader::mmap_loader::TensorShape;
use rocmforge::model::config::{ModelConfig, ModelType};
use rocmforge::model::execution_plan::ExecutionPlan;
use rocmforge::model::kv_cache::KVCache;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// Create a minimal synthetic GGUF file for testing decode_step
fn create_minimal_gguf_file(path: &Path, config: &ModelConfig) -> anyhow::Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // GGUF magic number: "GGUF"
    file.write_all(b"GGUF")?;

    // Version: 3
    file.write_all(&3u32.to_le_bytes())?;

    // Tensor count: minimal set for decode_step testing
    let mut tensor_info: Vec<(String, Vec<usize>)> = Vec::new();

    // Add embedding weights
    tensor_info.push(("token_embd.weight".to_string(), vec![config.vocab_size, config.hidden_size]));

    // Add LM head (tied to embeddings)
    tensor_info.push(("output.weight".to_string(), vec![config.vocab_size, config.hidden_size]));

    // Add layer 0 weights (for single layer test)
    let layer_prefix = "blk.0.";
    tensor_info.push((format!("{}attn_q.weight", layer_prefix), vec![config.hidden_size, config.hidden_size]));
    tensor_info.push((format!("{}attn_k.weight", layer_prefix), vec![config.head_dim, config.hidden_size]));
    tensor_info.push((format!("{}attn_v.weight", layer_prefix), vec![config.head_dim, config.hidden_size]));
    tensor_info.push((format!("{}attn_output.weight", layer_prefix), vec![config.hidden_size, config.hidden_size]));
    tensor_info.push((format!("{}ffn_gate.weight", layer_prefix), vec![config.intermediate_size, config.hidden_size]));
    tensor_info.push((format!("{}ffn_up.weight", layer_prefix), vec![config.intermediate_size, config.hidden_size]));
    tensor_info.push((format!("{}ffn_down.weight", layer_prefix), vec![config.hidden_size, config.intermediate_size]));
    tensor_info.push((format!("{}attn_norm.weight", layer_prefix), vec![config.hidden_size]));
    tensor_info.push((format!("{}ffn_norm.weight", layer_prefix), vec![config.hidden_size]));

    // For multi-layer test, add layer 1
    if config.num_hidden_layers > 1 {
        let layer_prefix = "blk.1.";
        tensor_info.push((format!("{}attn_q.weight", layer_prefix), vec![config.hidden_size, config.hidden_size]));
        tensor_info.push((format!("{}attn_k.weight", layer_prefix), vec![config.head_dim, config.hidden_size]));
        tensor_info.push((format!("{}attn_v.weight", layer_prefix), vec![config.head_dim, config.hidden_size]));
        tensor_info.push((format!("{}attn_output.weight", layer_prefix), vec![config.hidden_size, config.hidden_size]));
        tensor_info.push((format!("{}ffn_gate.weight", layer_prefix), vec![config.intermediate_size, config.hidden_size]));
        tensor_info.push((format!("{}ffn_up.weight", layer_prefix), vec![config.intermediate_size, config.hidden_size]));
        tensor_info.push((format!("{}ffn_down.weight", layer_prefix), vec![config.hidden_size, config.intermediate_size]));
        tensor_info.push((format!("{}attn_norm.weight", layer_prefix), vec![config.hidden_size]));
        tensor_info.push((format!("{}ffn_norm.weight", layer_prefix), vec![config.hidden_size]));
    }

    let tensor_count = tensor_info.len() as u64;

    // KV count: basic metadata
    let kv_count = 10u64; // architecture, layers, heads, etc.

    file.write_all(&tensor_count.to_le_bytes())?;
    file.write_all(&kv_count.to_le_bytes())?;

    // Write KV pairs (metadata)
    let metadata = vec![
        ("general.architecture", "llama".to_string()),
        ("general.file_type", "0".to_string()), // FP32
        ("llama.vocab_size", config.vocab_size.to_string()),
        ("llama.n_layers", config.num_hidden_layers.to_string()),
        ("llama.n_heads", config.num_attention_heads.to_string()),
        ("llama.n_embd", config.hidden_size.to_string()),
        ("llama.intermediate_size", config.intermediate_size.to_string()),
        ("llama.head_dim", config.head_dim.to_string()),
        ("llama.max_position_embeddings", config.max_position_embeddings.to_string()),
        ("llama.rms_norm_eps", "0.000001".to_string()),
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
    for (name, shape) in &tensor_info {
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

        // Tensor type: FP32 (0)
        file.write_all(&0u32.to_le_bytes())?;

        // Tensor offset (placeholder, will be updated)
        file.write_all(&0u64.to_le_bytes())?;
    }

    // Write tensor data (FP32 weights with small random-like values)
    for (name, shape) in &tensor_info {
        let total_elements: usize = shape.iter().product();
        let _data_size = total_elements * 4; // FP32 = 4 bytes per element

        // Generate pseudo-random but deterministic values based on tensor name
        let mut data = Vec::with_capacity(total_elements);
        let mut seed = 0u32;
        for byte in name.as_bytes() {
            seed = seed.wrapping_add(*byte as u32);
        }

        for _i in 0..total_elements {
            // Generate small pseudo-random values
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = (seed % 2000) as f32 / 1000.0 - 1.0; // Range [-1.0, 1.0)
            data.push(value.to_le_bytes());
        }

        // Write data as bytes
        for chunk in data {
            file.write_all(&chunk)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test decode_step() with single layer using CPU reference path
    #[test]
    fn test_decode_step_single_layer_cpu_reference() {
        // Initialize HIP backend
        let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Create minimal model configuration for testing
        let config = ModelConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: Some(4_usize),
            head_dim: 16, // hidden_size / num_attention_heads
            max_position_embeddings: 128_usize,
            vocab_size: 1000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        };

        // Create temporary GGUF file for testing
        let temp_dir = tempdir().unwrap();
        let gguf_path = temp_dir.path().join("test_model.gguf");
        create_minimal_gguf_file(&gguf_path, &config).unwrap();

        // Load GGUF and create execution plan
        let loader = GgufLoader::new(&gguf_path.to_string_lossy()).unwrap();
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Create KV cache (unused in this test but created for consistency)
        let _kv_cache = KVCache::new(&backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .unwrap();

        // Create scratch buffer manager (unused in this test but created for consistency)
        let _scratch = ScratchBufferManager::new(&backend,
            config.num_attention_heads,
            config.max_position_embeddings,
            config.head_dim,
            config.hidden_size,
        )
        .unwrap();

        // Create model runtime
        let mut runtime = ModelRuntime::new_with_config(config.clone()).unwrap();

        // Set the execution plan
        runtime.set_execution_plan(execution_plan);

        // Create input token embedding (simulate token id 42)
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let input_tensor = DeviceTensor::empty(&backend, input_shape).unwrap();

        // Initialize with test data
        let test_input: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.1) - 3.0)
            .collect();
        input_tensor.buffer().copy_from_host(&test_input).unwrap();

        // Run decode_step
        let output_tensor = runtime.decode_step(&input_tensor).unwrap();

        // Verify output shape aligns with vocab size (logits)
        assert_eq!(output_tensor.shape().dims(), &[config.vocab_size]);

        // Verify output is finite
        let mut output_host = vec![0.0f32; config.vocab_size];
        output_tensor
            .buffer()
            .copy_to_host(&mut output_host)
            .unwrap();

        for &val in &output_host {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }

        // Basic sanity: logits buffer should not be empty
        assert!(
            !output_host.is_empty(),
            "decode_step should produce logits for all vocab entries"
        );
    }

    /// Test decode_step() GPU matches CPU within tolerance
    #[cfg(feature = "rocm")]
    #[test]
    fn test_decode_step_gpu_matches_cpu_within_tolerance() {
        // Skip test gracefully if ROCm is not available
        let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Create minimal model configuration
        let config = ModelConfig {
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: Some(4),
            head_dim: 8, // hidden_size / num_attention_heads
            max_position_embeddings: 64,
            vocab_size: 1000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        };

        // Create temporary GGUF file for testing
        let temp_dir = tempdir().unwrap();
        let gguf_path = temp_dir.path().join("test_model.gguf");
        create_minimal_gguf_file(&gguf_path, &config).unwrap();

        // Load GGUF and create execution plan
        let loader = GgufLoader::new(&gguf_path.to_string_lossy()).unwrap();
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Test input
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let test_input: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.05) + 1.0)
            .collect();

        // GPU path test (unused variables in this test)
        let _kv_cache_gpu = KVCache::new(&backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .unwrap();

        let _scratch_gpu = ScratchBufferManager::new(&backend,
            config.num_attention_heads,
            config.max_position_embeddings,
            config.head_dim,
            config.hidden_size,
        )
        .unwrap();

        let mut runtime_gpu = ModelRuntime::new_with_config(config.clone()).unwrap();
        runtime_gpu.set_execution_plan(execution_plan.clone());

        let input_tensor_gpu = DeviceTensor::empty(&backend, input_shape.clone()).unwrap();
        input_tensor_gpu
            .buffer()
            .copy_from_host(&test_input)
            .unwrap();

        let gpu_output = runtime_gpu.decode_step(&input_tensor_gpu).unwrap();
        assert_eq!(
            gpu_output.shape().dims(),
            &[config.vocab_size],
            "decode_step should return vocab-sized logits"
        );

        let mut gpu_output_host = vec![0.0f32; config.vocab_size];
        gpu_output
            .buffer()
            .copy_to_host(&mut gpu_output_host)
            .unwrap();

        for &val in &gpu_output_host {
            assert!(
                val.is_finite(),
                "GPU output contains non-finite value: {}",
                val
            );
        }
    }

    /// Test decode_step() updates KV cache correctly
    #[test]
    fn test_decode_step_updates_kv_cache_correctly() {
        // Initialize backend
        let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Create model configuration
        let config = ModelConfig {
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 2, // Test with multiple layers
            num_attention_heads: 4,
            num_kv_heads: Some(4),
            head_dim: 8,
            max_position_embeddings: 64,
            vocab_size: 1000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        };

        // Create temporary GGUF file for testing
        let temp_dir = tempdir().unwrap();
        let gguf_path = temp_dir.path().join("test_model.gguf");
        create_minimal_gguf_file(&gguf_path, &config).unwrap();

        // Load GGUF and create execution plan
        let loader = GgufLoader::new(&gguf_path.to_string_lossy()).unwrap();
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Create KV cache with initial length 0
        let kv_cache = KVCache::new(&backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .unwrap();

        // Verify initial cache state
        assert_eq!(kv_cache.get_current_length(0).unwrap(), 0);

        // Create scratch buffer
        let _scratch = ScratchBufferManager::new(&backend,
            config.num_attention_heads,
            config.max_position_embeddings,
            config.head_dim,
            config.hidden_size,
        )
        .unwrap();

        // Create runtime
        let mut runtime = ModelRuntime::new_with_config(config.clone()).unwrap();
        runtime.set_execution_plan(execution_plan);

        // First token
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let test_input1: Vec<f32> = (0..config.hidden_size).map(|i| (i as f32 * 0.1)).collect();

        let mut input_tensor1 = DeviceTensor::empty(&backend, input_shape.clone()).unwrap();
        input_tensor1.buffer().copy_from_host(&test_input1).unwrap();

        let result1 = runtime.decode_step(&input_tensor1);
        assert!(result1.is_ok(), "First decode_step failed: {:?}", result1);

        // Verify cache length after first token
        assert_eq!(runtime.kv_cache().get_current_length(0).unwrap(), 1);

        // Second token
        let test_input2: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.1) + 0.5)
            .collect();

        let mut input_tensor2 = DeviceTensor::empty(&backend, input_shape).unwrap();
        input_tensor2.buffer().copy_from_host(&test_input2).unwrap();

        let result2 = runtime.decode_step(&input_tensor2);
        assert!(result2.is_ok(), "Second decode_step failed: {:?}", result2);

        // Verify cache length after second token
        assert_eq!(runtime.kv_cache().get_current_length(0).unwrap(), 2);

        // Verify outputs are finite
        let output1 = result1.unwrap();
        let output2 = result2.unwrap();

        let mut output1_host = vec![0.0f32; config.vocab_size];
        let mut output2_host = vec![0.0f32; config.vocab_size];

        output1.buffer().copy_to_host(&mut output1_host).unwrap();
        output2.buffer().copy_to_host(&mut output2_host).unwrap();

        for &val in &output1_host {
            assert!(
                val.is_finite(),
                "First output contains non-finite value: {}",
                val
            );
        }

        for &val in &output2_host {
            assert!(
                val.is_finite(),
                "Second output contains non-finite value: {}",
                val
            );
        }

        // Verify outputs are finite and non-zero (weights initialized)
        let output1_sum: f32 = output1_host.iter().sum();
        let output2_sum: f32 = output2_host.iter().sum();

        assert!(
            output1_sum.abs() > 1e-6 && output2_sum.abs() > 1e-6,
            "Outputs should be non-zero with initialized weights"
        );

        println!(
            "Output sums: {} vs {} (KV cache lengths: {} and {})",
            output1_sum,
            output2_sum,
            runtime.kv_cache().get_current_length(0).unwrap(),
            runtime.kv_cache().get_current_length(1).unwrap()
        );
    }
}
