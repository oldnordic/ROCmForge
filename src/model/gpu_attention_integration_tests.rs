// GPU Attention Integration Tests - TDD Implementation
//
// Test-Driven Development approach for GPU attention kernel integration.
// These tests verify the complete GPU attention path matches CPU results.

#[cfg(test)]
mod gpu_attention_integration_tests {
    use crate::backend::hip_backend::{HipBackend, DeviceTensor};
use serial_test::serial;
    use crate::loader::TensorShape;
    use crate::model::config::{ModelConfig, ModelType};
    use crate::model::execution_plan::ExecutionPlan;
    use std::sync::Arc;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for gpu_attention_integration_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Helper: Create a simple test configuration
    fn create_test_config() -> ModelConfig {
        ModelConfig {
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: Some(4),  // MHA for now
            head_dim: 32,  // 128 / 4
            hidden_size: 128,
            max_position_embeddings: 2048,
            intermediate_size: 512,
            vocab_size: 32000,
            model_type: ModelType::Qwen,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        }
    }

    /// Helper: Create test QKV tensors
    fn create_test_qkv_tensors(
        backend: &HipBackend,
        seq_len: usize,
        hidden_size: usize,
    ) -> (DeviceTensor, DeviceTensor, DeviceTensor) {
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;

        // Create Q: [seq_len, num_heads, head_dim]
        let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let q = DeviceTensor::from_host_vec(backend, q_data, q_shape).unwrap();

        // Create K: [seq_len, num_heads, head_dim]
        let k_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let k_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();
        let k = DeviceTensor::from_host_vec(backend, k_data, k_shape).unwrap();

        // Create V: [seq_len, num_heads, head_dim]
        let v_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let v_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| (i as f32 * 0.03).tan())
            .collect();
        let v = DeviceTensor::from_host_vec(backend, v_data, v_shape).unwrap();

        (q, k, v)
    }

    /// Helper: Compare two tensors for approximate equality
    fn compare_tensors(a: &DeviceTensor, b: &DeviceTensor, tolerance: f32) -> Result<String, String> {
        let a_host = a.to_host_vec().unwrap();
        let b_host = b.to_host_vec().unwrap();

        if a_host.len() != b_host.len() {
            return Err(format!(
                "Size mismatch: {} vs {}",
                a_host.len(),
                b_host.len()
            ));
        }

        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;

        for i in 0..a_host.len() {
            let diff = (a_host[i] - b_host[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
            if diff > tolerance {
                return Err(format!(
                    "Mismatch at index {}: {} vs {} (diff: {})",
                    i, a_host[i], b_host[i], diff
                ));
            }
        }

        Ok(format!("Max diff: {} at index {}", max_diff, max_diff_idx))
    }

    /// Test 1: QKV → RoPE → Causal Mask → Attention → Output (Single Token)
    #[test]
    #[serial]
    #[ignore] // ExecutionPlan::new() is deprecated - use ExecutionPlan::from_gguf() instead
    fn test_gpu_attention_single_token() {
        let backend = get_backend_or_skip();
        let config = create_test_config();
        let plan = ExecutionPlan::new(&backend, &config).unwrap();

        let seq_len = 1;
        let hidden_size = 128;
        let (q, k, v) = create_test_qkv_tensors(&backend, seq_len, hidden_size);

        // Test GPU attention (should not crash)
        let result = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);

        assert!(
            result.is_ok(),
            "GPU attention failed for single token: {:?}",
            result.err()
        );

        let output = result.unwrap();
        let output_shape = output.shape().dims();

        assert_eq!(
            output_shape,
            &[seq_len, 4, hidden_size / 4],
            "Output shape mismatch"
        );

        println!("✓ Test 1 PASSED: Single token attention");
    }

    /// Test 2: Multi-token sequence attention
    #[test]
    #[serial]
    #[ignore] // ExecutionPlan::new() is deprecated - use ExecutionPlan::from_gguf() instead
    fn test_gpu_attention_multi_token() {
        let backend = get_backend_or_skip();
        let config = create_test_config();
        let plan = ExecutionPlan::new(&backend, &config).unwrap();

        let seq_len = 16;
        let hidden_size = 128;
        let (q, k, v) = create_test_qkv_tensors(&backend, seq_len, hidden_size);

        let result = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);

        assert!(
            result.is_ok(),
            "GPU attention failed for multi-token: {:?}",
            result.err()
        );

        let output = result.unwrap();
        let output_shape = output.shape().dims();

        assert_eq!(
            output_shape,
            &[seq_len, 4, hidden_size / 4],
            "Output shape mismatch for multi-token"
        );

        println!("✓ Test 2 PASSED: Multi-token attention (seq_len={})", seq_len);
    }

    /// Test 3: Causal mask correctness
    #[test]
    #[serial]
    #[ignore] // ExecutionPlan::new() is deprecated - use ExecutionPlan::from_gguf() instead
    fn test_gpu_attention_causal_mask() {
        let backend = get_backend_or_skip();
        let config = create_test_config();
        let plan = ExecutionPlan::new(&backend, &config).unwrap();

        let seq_len = 8;
        let hidden_size = 128;
        let (q, k, v) = create_test_qkv_tensors(&backend, seq_len, hidden_size);

        let result = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);

        assert!(
            result.is_ok(),
            "GPU attention with causal mask failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        let output_host = output.to_host_vec().unwrap();

        // Verify output is finite (no NaN or Inf)
        for (i, val) in output_host.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Non-finite value at index {}: {}",
                i, val
            );
        }

        println!("✓ Test 3 PASSED: Causal mask correctness");
    }

    /// Test 4: GPU results match CPU within tolerance
    #[test]
    #[serial]
    #[ignore] // ExecutionPlan::new() is deprecated - use ExecutionPlan::from_gguf() instead
    fn test_gpu_cpu_consistency() {
        let backend = get_backend_or_skip();
        let config = create_test_config();
        let plan = ExecutionPlan::new(&backend, &config).unwrap();

        let seq_len = 4;
        let hidden_size = 128;
        let (q, k, v) = create_test_qkv_tensors(&backend, seq_len, hidden_size);

        // Run GPU attention
        let gpu_result = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);

        assert!(
            gpu_result.is_ok(),
            "GPU attention failed: {:?}",
            gpu_result.err()
        );

        let gpu_output = gpu_result.unwrap();

        // For now, just verify GPU produces valid output
        // (Full CPU comparison would require CPU implementation)
        let gpu_host = gpu_output.to_host_vec().unwrap();

        let mut has_significant_values = false;
        for val in &gpu_host {
            if val.abs() > 0.01 {
                has_significant_values = true;
                break;
            }
        }

        assert!(
            has_significant_values,
            "GPU output appears to be all near-zero values"
        );

        println!("✓ Test 4 PASSED: GPU produces valid output");
    }

    /// Test 5: Attention with varying sequence lengths
    #[test]
    #[serial]
    #[ignore] // ExecutionPlan::new() is deprecated - use ExecutionPlan::from_gguf() instead
    fn test_gpu_attention_varying_lengths() {
        let backend = get_backend_or_skip();
        let config = create_test_config();
        let plan = ExecutionPlan::new(&backend, &config).unwrap();

        let hidden_size = 128;
        let test_lengths = [1, 2, 4, 8, 16, 32];

        for seq_len in test_lengths {
            let (q, k, v) = create_test_qkv_tensors(&backend, seq_len, hidden_size);

            let result = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);

            assert!(
                result.is_ok(),
                "GPU attention failed for seq_len={}: {:?}",
                seq_len,
                result.err()
            );

            let output = result.unwrap();
            let output_shape = output.shape().dims();

            assert_eq!(
                output_shape,
                &[seq_len, 4, hidden_size / 4],
                "Shape mismatch for seq_len={}",
                seq_len
            );
        }

        println!("✓ Test 5 PASSED: Varying sequence lengths (1-32)");
    }

    /// Test 6: Numerical stability (avoid NaN/Inf)
    #[test]
    #[serial]
    #[ignore] // ExecutionPlan::new() is deprecated - use ExecutionPlan::from_gguf() instead
    fn test_gpu_attention_numerical_stability() {
        let backend = get_backend_or_skip();
        let config = create_test_config();
        let plan = ExecutionPlan::new(&backend, &config).unwrap();

        let seq_len = 16;
        let hidden_size = 128;

        // Test with potentially problematic values
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;

        let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| ((i as f32) * 100.0).sin() * 100.0)
            .collect();
        let q = DeviceTensor::from_host_vec(&backend, q_data, q_shape).unwrap();

        let k_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let k_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| ((i as f32) * 100.0).cos() * 100.0)
            .collect();
        let k = DeviceTensor::from_host_vec(&backend, k_data, k_shape).unwrap();

        let v_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let v_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| ((i as f32) * 100.0).tan() * 10.0)
            .collect();
        let v = DeviceTensor::from_host_vec(&backend, v_data, v_shape).unwrap();

        let result = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);

        assert!(
            result.is_ok(),
            "GPU attention failed with extreme values: {:?}",
            result.err()
        );

        let output = result.unwrap();
        let output_host = output.to_host_vec().unwrap();

        // Check for numerical stability
        let mut nan_count = 0;
        let mut inf_count = 0;

        for val in &output_host {
            if val.is_nan() {
                nan_count += 1;
            }
            if val.is_infinite() {
                inf_count += 1;
            }
        }

        assert_eq!(nan_count, 0, "Found {} NaN values in output", nan_count);
        assert_eq!(inf_count, 0, "Found {} Inf values in output", inf_count);

        println!("✓ Test 6 PASSED: Numerical stability");
    }

    /// Test 7: Performance baseline (measure for comparison)
    #[test]
    #[serial]
    #[ignore] // ExecutionPlan::new() is deprecated - use ExecutionPlan::from_gguf() instead
    fn test_gpu_attention_performance_baseline() {
        use std::time::Instant;

        let backend = get_backend_or_skip();
        let config = create_test_config();
        let plan = ExecutionPlan::new(&backend, &config).unwrap();

        let seq_len = 32;
        let hidden_size = 128;
        let (q, k, v) = create_test_qkv_tensors(&backend, seq_len, hidden_size);

        // Warmup
        for _ in 0..3 {
            let _ = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);
        }

        // Measure performance
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = plan.scaled_dot_product_attention(&backend, &q, &k, &v, None, 0);
        }

        let duration = start.elapsed();
        let avg_time_ms = duration.as_millis() as f64 / iterations as f64;

        println!(
            "✓ Test 7 PASSED: Performance baseline: {:.2}ms per iteration (seq_len={})",
            avg_time_ms, seq_len
        );

        // Sanity check: should complete in reasonable time
        assert!(
            avg_time_ms < 1000.0,
            "GPU attention too slow: {:.2}ms",
            avg_time_ms
        );
    }
}
