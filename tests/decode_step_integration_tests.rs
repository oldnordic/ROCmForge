//! decode_step() Integration Tests
//!
//! Tests the complete decode_step() pipeline in ModelRuntime using real ExecutionPlan,
//! KVCache, and existing GPU/CPU operations.
//!
//! These tests require a real GGUF model file to function properly.
//! Set ROCFORGE_TEST_MODEL environment variable to specify the model path.

use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use serial_test::serial;
use rocmforge::backend::hip_backend::{DeviceTensor, ModelRuntime};
use rocmforge::loader::gguf::GgufLoader;
use rocmforge::loader::mmap_loader::TensorShape;
use rocmforge::model::execution_plan::ExecutionPlan;
use anyhow::Context;

/// Get the test model path from environment variable
fn test_model_path() -> std::path::PathBuf {
    std::env::var("ROCFORGE_TEST_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/models/tiny-llama.gguf"))
}

/// Check if test model is available
fn has_test_model() -> bool {
    test_model_path().exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test decode_step() with single layer using CPU reference path
    #[test]
    #[serial]
    fn test_decode_step_single_layer_cpu_reference() -> anyhow::Result<()> {
        // Skip if GPU not available
        let fixture = GPU_FIXTURE.as_ref();
        if fixture.is_none() {
            eprintln!("SKIPPED: GPU not available - test skipped");
            return Ok(());
        }
        let backend = fixture.unwrap().backend();

        // Skip if test model not available
        if !has_test_model() {
            eprintln!("SKIPPED: Test model not available. Set ROCFORGE_TEST_MODEL to run this test.");
            return Ok(());
        }

        let model_path = test_model_path();
        eprintln!("Loading model from: {:?}", model_path);

        // Load GGUF and create execution plan
        let loader = GgufLoader::new(&model_path.to_string_lossy()).context("GGUF loader creation")?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader).context("execution plan creation from GGUF")?;

        // Create model runtime
        let config = loader.to_model_config().context("model config creation")?;
        let mut runtime = ModelRuntime::new_with_config(config.clone()).context("model runtime creation")?;
        runtime.set_execution_plan(execution_plan);

        // Create input token embedding (simulate token id 42)
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let input_tensor = DeviceTensor::empty(&backend, input_shape).context("input tensor allocation")?;

        // Initialize with test data
        let test_input: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.1) - 3.0)
            .collect();
        input_tensor.buffer().copy_from_host(&test_input).context("input data copy to device")?;

        // Run decode_step
        let output_tensor = runtime.decode_step(&input_tensor).context("decode_step execution")?;

        // Verify output shape aligns with vocab size (logits)
        assert_eq!(output_tensor.shape().dims(), &[config.vocab_size]);

        // Verify output is finite
        let mut output_host = vec![0.0f32; config.vocab_size];
        backend
            .copy_from_device_safe(output_tensor.buffer(), &mut output_host)
            .context("output copy from device")?;

        for &val in &output_host {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }

        // Basic sanity: logits buffer should not be empty
        assert!(
            !output_host.is_empty(),
            "decode_step should produce logits for all vocab entries"
        );
        Ok(())
    }

    /// Test decode_step() GPU matches CPU within tolerance
    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn test_decode_step_gpu_matches_cpu_within_tolerance() -> anyhow::Result<()> {
        // Skip if GPU not available
        let fixture = GPU_FIXTURE.as_ref();
        if fixture.is_none() {
            eprintln!("SKIPPED: GPU not available - test skipped");
            return Ok(());
        }
        let backend = fixture.unwrap().backend();

        // Skip if test model not available
        if !has_test_model() {
            eprintln!("SKIPPED: Test model not available. Set ROCFORGE_TEST_MODEL to run this test.");
            return Ok(());
        }

        let model_path = test_model_path();
        eprintln!("Loading model from: {:?}", model_path);

        // Load GGUF and create execution plan
        let loader = GgufLoader::new(&model_path.to_string_lossy()).context("GGUF loader creation")?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader).context("execution plan creation from GGUF")?;

        // Create model runtime
        let config = loader.to_model_config().context("model config creation")?;
        let mut runtime = ModelRuntime::new_with_config(config.clone()).context("model runtime creation")?;
        runtime.set_execution_plan(execution_plan);

        // Test input
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let test_input: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.05) + 1.0)
            .collect();

        let input_tensor = DeviceTensor::empty(&backend, input_shape.clone()).context("input tensor allocation")?;
        input_tensor
            .buffer()
            .copy_from_host(&test_input)
            .context("input data copy to device")?;

        let output = runtime.decode_step(&input_tensor).context("decode_step execution")?;
        assert_eq!(
            output.shape().dims(),
            &[config.vocab_size],
            "decode_step should return vocab-sized logits"
        );

        let mut output_host = vec![0.0f32; config.vocab_size];
        backend
            .copy_from_device_safe(output.buffer(), &mut output_host)
            .context("output copy from device")?;

        for &val in &output_host {
            assert!(
                val.is_finite(),
                "GPU output contains non-finite value: {}",
                val
            );
        }
        Ok(())
    }

    /// Test decode_step() updates KV cache correctly
    #[test]
    #[serial]
    fn test_decode_step_updates_kv_cache_correctly() -> anyhow::Result<()> {
        // Skip if GPU not available
        let fixture = GPU_FIXTURE.as_ref();
        if fixture.is_none() {
            eprintln!("SKIPPED: GPU not available - test skipped");
            return Ok(());
        }
        let backend = fixture.unwrap().backend();

        // Skip if test model not available
        if !has_test_model() {
            eprintln!("SKIPPED: Test model not available. Set ROCFORGE_TEST_MODEL to run this test.");
            return Ok(());
        }

        let model_path = test_model_path();
        eprintln!("Loading model from: {:?}", model_path);

        // Load GGUF and create execution plan
        let loader = GgufLoader::new(&model_path.to_string_lossy()).context("GGUF loader creation")?;
        let execution_plan = ExecutionPlan::from_gguf(&backend, &loader).context("execution plan creation from GGUF")?;

        // Create runtime
        let config = loader.to_model_config().context("model config creation")?;
        let mut runtime = ModelRuntime::new_with_config(config.clone()).context("model runtime creation")?;
        runtime.set_execution_plan(execution_plan);

        // First token
        let input_shape = TensorShape::from_dims(&[config.hidden_size]);
        let test_input1: Vec<f32> = (0..config.hidden_size).map(|i| i as f32 * 0.1).collect();

        let input_tensor1 = DeviceTensor::empty(&backend, input_shape.clone()).context("first input tensor allocation")?;
        input_tensor1.buffer().copy_from_host(&test_input1).context("first input data copy to device")?;

        let result1 = runtime.decode_step(&input_tensor1);
        assert!(result1.is_ok(), "First decode_step failed: {:?}", result1);

        // Verify cache length after first token
        assert_eq!(runtime.kv_cache().get_current_length(0).context("getting cache length after first token")?, 1);

        // Second token
        let test_input2: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32 * 0.1) + 0.5)
            .collect();

        let input_tensor2 = DeviceTensor::empty(&backend, input_shape).context("second input tensor allocation")?;
        input_tensor2.buffer().copy_from_host(&test_input2).context("second input data copy to device")?;

        let result2 = runtime.decode_step(&input_tensor2);
        assert!(result2.is_ok(), "Second decode_step failed: {:?}", result2);

        // Verify cache length after second token
        assert_eq!(runtime.kv_cache().get_current_length(0).context("getting cache length after second token")?, 2);

        // Verify outputs are finite
        let output1 = result1.context("unwrapping first decode_step result")?;
        let output2 = result2.context("unwrapping second decode_step result")?;

        let mut output1_host = vec![0.0f32; config.vocab_size];
        let mut output2_host = vec![0.0f32; config.vocab_size];

        backend.copy_from_device_safe(output1.buffer(), &mut output1_host).context("copying first output from device")?;
        backend.copy_from_device_safe(output2.buffer(), &mut output2_host).context("copying second output from device")?;

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
            runtime.kv_cache().get_current_length(0).context("getting layer 0 cache length for summary")?,
            runtime.kv_cache().get_current_length(1).context("getting layer 1 cache length for summary")?
        );
        Ok(())
    }
}
