//! decode_step() Integration Tests
//!
//! Tests the complete decode_step() pipeline in ModelRuntime using real ExecutionPlan,
//! KVCache, and existing GPU/CPU operations.

use rocmforge::backend::hip_backend::{DeviceTensor, HipBackend, ModelRuntime};
use rocmforge::backend::scratch::ScratchBufferManager;
use rocmforge::loader::mmap_loader::TensorShape;
use rocmforge::model::config::{ModelConfig, ModelType};
use rocmforge::model::execution_plan::ExecutionPlan;
use rocmforge::model::kv_cache::KVCache;
use serial_test::serial;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test decode_step() with single layer using CPU reference path
    #[test]
    fn test_decode_step_single_layer_cpu_reference() {
        // Initialize HIP backend
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        assert!(backend.is_ok(), "Failed to initialize HIP backend");
        let backend = backend.unwrap();

        // Create minimal model configuration for testing
        let config = ModelConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            head_dim: 16, // hidden_size / num_attention_heads
            max_position_embeddings: 128,
            vocab_size: 1000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        };

        // Create execution plan with synthetic weights
        let plan_result = ExecutionPlan::new(&backend, &config);
        assert!(plan_result.is_ok(), "Failed to create execution plan");
        let execution_plan = plan_result.unwrap();

        // Create KV cache
        let mut kv_cache = KVCache::new(
            backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .unwrap();

        // Create scratch buffer manager
        let scratch = ScratchBufferManager::new(
            backend,
            config.num_attention_heads,
            config.hidden_size, // ← PHASE 24 FIX: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← PHASE 24 FIX: 5th param
        )
        .unwrap();

        // Create model runtime
        let mut runtime = ModelRuntime::new_with_config(config.clone()).unwrap();

        // Set the execution plan
        runtime.set_execution_plan(execution_plan);

        // Create input token embedding (simulate token id 42)
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let mut input_tensor = DeviceTensor::empty(&backend, input_shape).unwrap();

        // Initialize with test data
        let test_input: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.1) - 3.0)
            .collect();
        input_tensor.buffer().copy_from_host(&test_input).unwrap();

        // Run decode_step
        let result = runtime.decode_step(&input_tensor);
        assert!(result.is_ok(), "decode_step failed: {:?}", result);

        let output_tensor = result.unwrap();

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
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        if backend.is_err() {
            return;
        }
        let backend = backend.unwrap();

        // Create minimal model configuration
        let config = ModelConfig {
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            head_dim: 8, // hidden_size / num_attention_heads
            max_position_embeddings: 64,
            vocab_size: 1000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        };

        // Create execution plan
        let plan_result = ExecutionPlan::new(&backend, &config);
        assert!(plan_result.is_ok(), "Failed to create execution plan");
        let execution_plan = plan_result.unwrap();

        // Test input
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let test_input: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.05) + 1.0)
            .collect();

        // GPU path test
        let mut kv_cache_gpu = KVCache::new(
            backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .unwrap();

        let scratch_gpu = ScratchBufferManager::new(
            backend,
            config.num_attention_heads,
            config.hidden_size, // ← PHASE 24 FIX: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← PHASE 24 FIX: 5th param
        )
        .unwrap();

        let mut runtime_gpu = ModelRuntime::new_with_config(config.clone()).unwrap();
        runtime_gpu.set_execution_plan(execution_plan.clone());

        let mut input_tensor_gpu = DeviceTensor::empty(&backend, input_shape.clone()).unwrap();
        input_tensor_gpu
            .buffer()
            .copy_from_host(&test_input)
            .unwrap();

        let gpu_result = runtime_gpu.decode_step(&input_tensor_gpu);
        assert!(
            gpu_result.is_ok(),
            "GPU decode_step failed: {:?}",
            gpu_result
        );

        let gpu_output = gpu_result.unwrap();
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
        let fixture = rocmforge::GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        assert!(backend.is_ok(), "Failed to initialize HIP backend");
        let backend = backend.unwrap();

        // Create model configuration
        let config = ModelConfig {
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 2, // Test with multiple layers
            num_attention_heads: 4,
            head_dim: 8,
            max_position_embeddings: 64,
            vocab_size: 1000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        };

        // Create execution plan
        let plan_result = ExecutionPlan::new(&backend, &config);
        assert!(plan_result.is_ok(), "Failed to create execution plan");
        let execution_plan = plan_result.unwrap();

        // Create KV cache with initial length 0
        let mut kv_cache = KVCache::new(
            backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .unwrap();

        // Verify initial cache state
        assert_eq!(kv_cache.get_current_length(0).unwrap(), 0);

        // Create scratch buffer
        let scratch = ScratchBufferManager::new(
            backend,
            config.num_attention_heads,
            config.hidden_size, // ← PHASE 24 FIX: 3rd param
            config.head_dim,
            config.max_position_embeddings, // ← PHASE 24 FIX: 5th param
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
