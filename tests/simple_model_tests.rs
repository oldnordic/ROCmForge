//! Integration tests for simple model functionality

use rocmforge::model::{
    Linear, ModelBackend, SimpleAttention, SimpleModel, SimpleTransformerBlock,
};
use serial_test::serial;

#[test]
#[serial]
fn test_simple_model_forward_runs() {
    let model = SimpleModel::new(100, 8, 1, 4, ModelBackend::Cpu, 42);
    let input_tokens = vec![1, 2, 3, 4];

    // Should not panic
    let result = model.forward(&input_tokens);

    assert!(result.is_ok(), "Model forward pass should succeed");

    let output = result.unwrap();

    // Check output shape
    assert_eq!(
        output.len(),
        4 * 8,
        "Output should have seq_len * dim elements"
    );

    // Check all outputs are finite
    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );

    // Check output is not all zeros (model should produce some variation)
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() > 1e-6, "Output should not be all zeros");
}

#[test]
#[serial]
fn test_linear_layer_forward() {
    let linear = Linear::new(4, 2, 42);
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];

    let result = linear.forward(&input);

    assert!(result.is_ok(), "Linear forward should succeed");

    let output = result.unwrap();
    assert_eq!(output.len(), 2, "Output should have out_features elements");
    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );
}

#[test]
#[serial]
fn test_simple_attention_forward() {
    let attention = SimpleAttention::new(4, ModelBackend::Cpu, 42);
    let input = vec![1.0f32; 8]; // seq_len=2, dim=4

    let result = attention.forward(&input);

    assert!(result.is_ok(), "Attention forward should succeed");

    let output = result.unwrap();
    assert_eq!(output.len(), 8, "Output should preserve input length");
    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );
}

#[test]
#[serial]
fn test_transformer_block_forward() {
    let block = SimpleTransformerBlock::new(4, ModelBackend::Cpu, 42);
    let input = vec![1.0f32; 8]; // seq_len=2, dim=4

    let result = block.forward(&input);

    assert!(result.is_ok(), "Transformer block forward should succeed");

    let output = result.unwrap();
    assert_eq!(output.len(), 8, "Output should preserve input length");
    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );
}

#[test]
#[serial]
fn test_model_with_different_sequence_lengths() {
    let model = SimpleModel::new(50, 6, 1, 8, ModelBackend::Cpu, 123);

    // Test with different sequence lengths
    for seq_len in 1..=4 {
        let input_tokens: Vec<u32> = (1..=seq_len).map(|x| x as u32).collect();

        let result = model.forward(&input_tokens);
        assert!(result.is_ok(), "Model should handle seq_len {}", seq_len);

        let output = result.unwrap();
        assert_eq!(
            output.len(),
            seq_len * 6,
            "Output length should match seq_len * dim"
        );
        assert!(
            output.iter().all(|x| x.is_finite()),
            "All outputs should be finite"
        );
    }
}

#[test]
#[serial]
fn test_model_error_handling() {
    let model = SimpleModel::new(10, 4, 1, 4, ModelBackend::Cpu, 42);

    // Test with token ID out of range
    let input_tokens = vec![15]; // vocab size is 10, so 15 is out of range
    let result = model.forward(&input_tokens);
    assert!(result.is_err(), "Should fail with out-of-range token ID");

    // Test with sequence too long
    let input_tokens = vec![1, 2, 3, 4, 5]; // max_seq_len is 4
    let result = model.forward(&input_tokens);
    assert!(result.is_err(), "Should fail with sequence too long");
}

#[test]
#[serial]
fn test_model_deterministic_with_same_seed() {
    let model1 = SimpleModel::new(20, 4, 1, 3, ModelBackend::Cpu, 999);
    let model2 = SimpleModel::new(20, 4, 1, 3, ModelBackend::Cpu, 999);

    let input_tokens = vec![1, 2, 3];

    let output1 = model1.forward(&input_tokens).unwrap();
    let output2 = model2.forward(&input_tokens).unwrap();

    // Outputs should be identical with same seed
    for (i, (&v1, &v2)) in output1.iter().zip(output2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-6,
            "Output at index {} differs: {} vs {}",
            i,
            v1,
            v2
        );
    }
}

#[test]
#[serial]
fn test_simple_model_forward_gpu_close_to_cpu() {
    // Create identical models with CPU and GPU backends
    let cpu_model = SimpleModel::new(50, 8, 1, 4, ModelBackend::Cpu, 42);
    let gpu_model = SimpleModel::new(50, 8, 1, 4, ModelBackend::Gpu, 42);

    let input_tokens = vec![1, 5, 9, 3];

    // Run forward pass on both backends
    let cpu_output = cpu_model.forward(&input_tokens).unwrap();
    let gpu_output = gpu_model.forward(&input_tokens).unwrap();

    assert_eq!(
        cpu_output.len(),
        gpu_output.len(),
        "Outputs should have same length"
    );

    // Compare outputs with tolerance (GPU and CPU might have small numerical differences)
    for (i, (&cpu_val, &gpu_val)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        assert!(
            diff < 1e-4,
            "Output at index {} differs too much: CPU={}, GPU={}, diff={}",
            i,
            cpu_val,
            gpu_val,
            diff
        );
    }
}

#[test]
#[serial]
fn test_model_with_multiple_layers() {
    let model = SimpleModel::new(30, 6, 3, 4, ModelBackend::Cpu, 777);
    let input_tokens = vec![1, 2, 3, 4];

    let result = model.forward(&input_tokens);
    assert!(result.is_ok(), "Model with multiple layers should work");

    let output = result.unwrap();
    assert_eq!(output.len(), 4 * 6, "Output should have correct length");
    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );

    // Multiple layers should produce different output than single layer
    let single_layer_model = SimpleModel::new(30, 6, 1, 4, ModelBackend::Cpu, 777);
    let single_output = single_layer_model.forward(&input_tokens).unwrap();

    let sum_diff: f32 = output
        .iter()
        .zip(single_output.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        sum_diff > 1e-3,
        "Multiple layers should produce different output than single layer"
    );
}
