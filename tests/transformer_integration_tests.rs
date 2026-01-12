//! End-to-end transformer layer tests
//! Tests the complete transformer pipeline including LayerNorm, attention, and MLP

use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use rocmforge::backend::hip_backend::DeviceTensor;
use rocmforge::backend::scratch::ScratchBufferManager;
use rocmforge::loader::mmap_loader::TensorShape;
use rocmforge::model::config::{ModelConfig, ModelType};
use rocmforge::model::execution_plan::ExecutionPlan;
use rocmforge::model::kv_cache::KVCache;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_integration() {
        // Initialize HIP backend
        let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

        // Create input tensor [batch=2, seq_len=3, hidden_size=128]
        let batch_size = 2;
        let seq_len = 3;
        let hidden_size = 128;
        let input_shape = TensorShape::from_dims(&[batch_size, seq_len, hidden_size]);
        let mut input_tensor = DeviceTensor::empty(&backend, input_shape).unwrap();

        // Create weight and bias tensors
        let norm_shape = TensorShape::from_dims(&[hidden_size]);
        let mut weight_tensor = DeviceTensor::empty(&backend, norm_shape.clone()).unwrap();
        let mut bias_tensor = DeviceTensor::empty(&backend, norm_shape).unwrap();

        // Create output tensor
        let output_shape = TensorShape::from_dims(&[batch_size, seq_len, hidden_size]);
        let mut output_tensor = DeviceTensor::empty(&backend, output_shape).unwrap();

        // Initialize with test data
        let total_elements = batch_size * seq_len * hidden_size;
        let test_input: Vec<f32> = (0..total_elements)
            .map(|i| ((i as f32) * 0.1 - 5.0))
            .collect();
        input_tensor.buffer().copy_from_host(&test_input).unwrap();

        let test_weight: Vec<f32> = (0..hidden_size).map(|i| 0.5 + (i as f32 * 0.01)).collect();
        weight_tensor.buffer().copy_from_host(&test_weight).unwrap();

        let test_bias: Vec<f32> = (0..hidden_size).map(|i| (i as f32 * 0.02) - 1.0).collect();
        bias_tensor.buffer().copy_from_host(&test_bias).unwrap();

        // Test LayerNorm
        let result = backend.layernorm(
            &input_tensor,
            &weight_tensor,
            Some(&bias_tensor),
            &mut output_tensor,
            1e-6,
        );

        assert!(result.is_ok(), "LayerNorm failed: {:?}", result);

        // Verify output
        let mut output_host = vec![0.0f32; total_elements];
        output_tensor
            .buffer()
            .copy_to_host(&mut output_host)
            .unwrap();

        // Check that outputs are finite
        for &val in &output_host {
            assert!(
                val.is_finite(),
                "LayerNorm output contains non-finite value: {}",
                val
            );
        }

        // Verify LayerNorm properties: each row should have mean≈0 and std≈1 after normalization
        for row_idx in 0..(batch_size * seq_len) {
            let start_idx = row_idx * hidden_size;
            let end_idx = start_idx + hidden_size;
            let row = &output_host[start_idx..end_idx];

            // Calculate mean and std of normalized output
            let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
            let variance: f32 =
                row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;
            let std = variance.sqrt();

            // Allow some tolerance due to floating point precision
            assert!(
                (mean).abs() < 1e-4,
                "Row {} mean should be ~0, got {}",
                row_idx,
                mean
            );
            assert!(
                (std - 1.0).abs() < 1e-4,
                "Row {} std should be ~1, got {}",
                row_idx,
                std
            );
        }
    }

    #[test]
    fn test_mlp_swiglu_integration() {
        // Initialize HIP backend
        let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

        // Test dimensions
        let seq_len = 2;
        let hidden_size = 256;
        let intermediate_size = 1024;

        // Create input tensor [seq_len, hidden_size]
        let input_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let mut input_tensor = DeviceTensor::empty(&backend, input_shape).unwrap();

        // Create weight tensors
        let gate_shape = TensorShape::from_dims(&[hidden_size, intermediate_size]);
        let up_shape = TensorShape::from_dims(&[hidden_size, intermediate_size]);
        let down_shape = TensorShape::from_dims(&[intermediate_size, hidden_size]);

        let mut gate_weight = DeviceTensor::empty(&backend, gate_shape).unwrap();
        let mut up_weight = DeviceTensor::empty(&backend, up_shape).unwrap();
        let mut down_weight = DeviceTensor::empty(&backend, down_shape).unwrap();

        // Create output tensor
        let output_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let mut output_tensor = DeviceTensor::empty(&backend, output_shape).unwrap();

        // Initialize with test data
        let test_input: Vec<f32> = (0..(seq_len * hidden_size))
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        input_tensor.buffer().copy_from_host(&test_input).unwrap();

        let test_gate: Vec<f32> = (0..(hidden_size * intermediate_size))
            .map(|i| (i as f32 * 0.02) - 1.0)
            .collect();
        gate_weight.buffer().copy_from_host(&test_gate).unwrap();

        let test_up: Vec<f32> = (0..(hidden_size * intermediate_size))
            .map(|i| (i as f32 * 0.015) + 0.5)
            .collect();
        up_weight.buffer().copy_from_host(&test_up).unwrap();

        let test_down: Vec<f32> = (0..(intermediate_size * hidden_size))
            .map(|i| (i as f32 * 0.025) - 0.3)
            .collect();
        down_weight.buffer().copy_from_host(&test_down).unwrap();

        // Test MLP SwiGLU
        let result = backend.mlp_swiglu(
            &input_tensor,
            &gate_weight,
            &up_weight,
            &down_weight,
            &mut output_tensor,
        );

        assert!(result.is_ok(), "MLP SwiGLU failed: {:?}", result);

        // Verify output
        let mut output_host = vec![0.0f32; seq_len * hidden_size];
        output_tensor
            .buffer()
            .copy_to_host(&mut output_host)
            .unwrap();

        // Check that outputs are finite
        for &val in &output_host {
            assert!(
                val.is_finite(),
                "MLP output contains non-finite value: {}",
                val
            );
        }

        // Verify output shape
        assert_eq!(output_tensor.shape().dims(), &[seq_len, hidden_size]);

        // Check that computation occurred (output differs from input)
        let mut computation_occurred = false;
        for i in 0..test_input.len().min(output_host.len()) {
            if (test_input[i] - output_host[i]).abs() > 1e-6 {
                computation_occurred = true;
                break;
            }
        }
        assert!(computation_occurred, "MLP computation should have occurred");
    }

    #[test]
    fn test_transformer_component_shapes() {
        // Initialize HIP backend
        let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Create model configuration
        let config = ModelConfig {
            hidden_size: 256,
            intermediate_size: 1024,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: Some(4),
            head_dim: 64, // hidden_size / num_attention_heads
            max_position_embeddings: 512,
            vocab_size: 1000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        };

        // Create execution plan
        let plan_result = ExecutionPlan::new(&backend, &config);
        assert!(plan_result.is_ok(), "Failed to create execution plan");
        let execution_plan = plan_result.unwrap();

        // Verify we have the expected number of layers
        assert_eq!(execution_plan.layers().len(), config.num_hidden_layers);

        // Test first layer
        let layer_plan = &execution_plan.layers()[0];

        // Create input tensor [batch=1, seq_len=4, hidden_size=256]
        let seq_len = 4;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let input_shape = TensorShape::from_dims(&[1, seq_len, hidden_size]);
        let mut input_tensor = DeviceTensor::empty(&backend, input_shape).unwrap();

        // Initialize input with test data
        let test_input: Vec<f32> = (0..(1 * seq_len * hidden_size))
            .map(|i| (i as f32 * 0.01).cos())
            .collect();
        input_tensor.buffer().copy_from_host(&test_input).unwrap();

        // Create output tensors
        let output_shape = TensorShape::from_dims(&[1, seq_len, hidden_size]);
        let mut attention_output = DeviceTensor::empty(&backend, output_shape.clone()).unwrap();
        let mut mlp_output = DeviceTensor::empty(&backend, output_shape.clone()).unwrap();

        // Create scratch buffer manager
        let scratch = ScratchBufferManager::new(&backend,
            config.num_attention_heads,
            hidden_size,
            config.head_dim,
            config.max_position_embeddings,
        );

        // Create KV cache
        let mut kv_cache = KVCache::new(&backend,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
            config.max_position_embeddings,
        )
        .unwrap();

        // Test that we can create the necessary tensors and they have correct shapes
        assert_eq!(input_tensor.shape().dims(), &[1, seq_len, hidden_size]);
        assert_eq!(attention_output.shape().dims(), &[1, seq_len, hidden_size]);
        assert_eq!(mlp_output.shape().dims(), &[1, seq_len, hidden_size]);

        // Verify layer plan tensors have expected shapes
        assert_eq!(
            layer_plan.qkv_weight.shape().unwrap(),
            &[3 * hidden_size, hidden_size]
        );
        assert_eq!(
            layer_plan.o_proj.shape().unwrap(),
            &[hidden_size, hidden_size]
        );
        assert_eq!(
            layer_plan.mlp_gate_proj.shape().unwrap(),
            &[intermediate_size, hidden_size]
        );
        assert_eq!(
            layer_plan.mlp_down_proj.shape().unwrap(),
            &[hidden_size, intermediate_size]
        );
        assert_eq!(layer_plan.norm1_weight.shape().unwrap(), &[hidden_size]);
        assert_eq!(layer_plan.norm2_weight.shape().unwrap(), &[hidden_size]);

        // TODO: Test tensor operations once lazy loading is properly implemented
        // For now, just verify that the tensors exist and have the correct shapes
        // (Actual computation testing requires loading LazyTensors to DeviceTensors)
    }
}
