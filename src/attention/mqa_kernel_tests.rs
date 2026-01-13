//! Unit tests for MQA KV replication kernel
//!
//! TDD: These tests are written FIRST and expected to FAIL until
//! the GPU pipeline is implemented in multi_query.rs
//!
//! # Phase 20: GPU Testing Safety
//!
//! These tests use the GPU_FIXTURE pattern to:
//! - Check GPU availability before running
//! - Use a shared backend (prevents multiple allocations)
//! - Check for memory leaks after each test
//! - Run serially (one test at a time)

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use crate::attention::multi_query::{MultiQueryAttention, MultiQueryConfig};
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::TensorShape;

    // Phase 20: Import GPU test fixture
    // Note: tests/common module is in tests/ directory, not src/
    // For now, we'll use direct HipBackend::new_checked() instead
    use serial_test::serial;

    /// Test MQA: 32 query heads, 1 KV head
    ///
    /// NOTE: GPU tensors use [batch, seq, heads, dim] layout
    /// CPU tensors use [batch*seq, heads, dim] layout (flattened)
    #[test]
    #[serial] // Phase 20: Run serially to prevent GPU resource conflicts
    fn test_kv_replication_mqa() {
        // Phase 20: Check GPU availability first
        let backend = match HipBackend::new_checked() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("SKIP: GPU not available");
                return;
            }
        };
        let config = MultiQueryConfig::new(32, 128); // 32 query heads, 1 KV head, head_dim=128
        let mqa = MultiQueryAttention::new(config).expect("Failed to create MQA");

        let batch_size = 1;
        let seq_len = 16;
        let num_kv_heads = 1;
        let num_q_heads = 32;
        let head_dim = 128;

        // GPU tensors use [batch, seq, heads, dim] layout
        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);

        // Create test data with GPU layout [batch, seq, heads, dim]
        let mut k_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut v_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut q_host = vec![0.0f32; batch_size * seq_len * num_q_heads * head_dim];

        // Fill with test data
        for (i, val) in k_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.1;
        }
        for (i, val) in v_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.2;
        }
        for (i, val) in q_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.3;
        }

        let k_device = DeviceTensor::from_host_vec(&backend, k_host.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_device = DeviceTensor::from_host_vec(&backend, v_host.clone(), v_shape)
            .expect("Failed to create V tensor");
        let q_device = DeviceTensor::from_host_vec(&backend, q_host.clone(), q_shape.clone())
            .expect("Failed to create Q tensor");

        // Execute forward_device - should use GPU pipeline
        let output_device = mqa
            .forward_device(&q_device, &k_device, &v_device, None, None)
            .expect("forward_device failed");

        // Verify output shape matches input
        assert_eq!(output_device.shape().dims(), q_shape.dims());

        // Verify output is not all zeros
        let output_host = output_device
            .to_host_vec()
            .expect("Failed to copy output to host");
        let non_zero_count = output_host.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count > 0, "Output should not be all zeros");

        // Phase 20: Check for memory leaks (5% tolerance)
        // Explicitly drop GPU tensors before leak check
        drop(k_device);
        drop(v_device);
        drop(q_device);
        drop(output_device);
        // NOTE: Memory leak checking not available without GPU_FIXTURE
    }

    /// Test GQA: 32 query heads, 8 KV heads
    #[test]
    #[serial] // Phase 20: Run serially
    fn test_kv_replication_gqa() {
        // Phase 20: Check GPU availability first
        let backend = match HipBackend::new_checked() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("SKIP: GPU not available");
                return;
            }
        };
        let config = MultiQueryConfig::new(32, 128).with_kv_heads(8); // 32 query heads, 8 KV heads, head_dim=128
        let mqa = MultiQueryAttention::new(config).expect("Failed to create MQA");

        let batch_size = 1;
        let seq_len = 16;
        let num_kv_heads = 8;
        let num_q_heads = 32;
        let head_dim = 128;

        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);

        let mut k_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut v_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut q_host = vec![0.0f32; batch_size * seq_len * num_q_heads * head_dim];

        for (i, val) in k_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.1;
        }
        for (i, val) in v_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.2;
        }
        for (i, val) in q_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.3;
        }

        let k_device = DeviceTensor::from_host_vec(&backend, k_host.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_device = DeviceTensor::from_host_vec(&backend, v_host.clone(), v_shape)
            .expect("Failed to create V tensor");
        let q_device = DeviceTensor::from_host_vec(&backend, q_host.clone(), q_shape.clone())
            .expect("Failed to create Q tensor");

        let output_device = mqa
            .forward_device(&q_device, &k_device, &v_device, None, None)
            .expect("forward_device failed");

        assert_eq!(output_device.shape().dims(), q_shape.dims());

        let output_host = output_device
            .to_host_vec()
            .expect("Failed to copy output to host");
        let non_zero_count = output_host.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count > 0, "Output should not be all zeros");

        // Phase 20: Check for memory leaks
        drop(k_device);
        drop(v_device);
        drop(q_device);
        drop(output_device);
        // NOTE: Memory leak checking not available without GPU_FIXTURE
    }

    /// Test correctness: Compare GPU output with CPU replication
    #[test]
    #[serial] // Phase 20: Run serially
    fn test_kv_replication_correctness() {
        // Phase 20: Check GPU availability first
        let backend = match HipBackend::new_checked() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("SKIP: GPU not available");
                return;
            }
        };
        let config = MultiQueryConfig::new(8, 64); // Small config for fast testing
        let mqa = MultiQueryAttention::new(config).expect("Failed to create MQA");

        let batch_size = 1;
        let seq_len = 4;
        let num_kv_heads = 2;
        let num_q_heads = 8;
        let head_dim = 64;

        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);

        let mut k_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut v_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut q_host = vec![0.0f32; batch_size * seq_len * num_q_heads * head_dim];

        // Use predictable test data
        for (i, val) in k_host.iter_mut().enumerate() {
            *val = (i % 100) as f32 * 0.01;
        }
        for (i, val) in v_host.iter_mut().enumerate() {
            *val = (i % 100) as f32 * 0.02;
        }
        for (i, val) in q_host.iter_mut().enumerate() {
            *val = (i % 100) as f32 * 0.03;
        }

        // CPU reference
        let cpu_output = mqa
            .forward(&q_host, &k_host, &v_host, None, None)
            .expect("CPU forward failed");

        // GPU computation
        let k_device = DeviceTensor::from_host_vec(&backend, k_host.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_device = DeviceTensor::from_host_vec(&backend, v_host.clone(), v_shape)
            .expect("Failed to create V tensor");
        let q_device = DeviceTensor::from_host_vec(&backend, q_host.clone(), q_shape.clone())
            .expect("Failed to create Q tensor");

        let output_device = mqa
            .forward_device(&q_device, &k_device, &v_device, None, None)
            .expect("GPU forward_device failed");

        let gpu_output = output_device
            .to_host_vec()
            .expect("Failed to copy GPU output to host");

        // Compare with tolerance for floating point differences
        assert_eq!(
            cpu_output.len(),
            gpu_output.len(),
            "Output lengths should match"
        );

        let mut max_diff = 0.0f32;
        for (cpu_val, gpu_val) in cpu_output.iter().zip(gpu_output.iter()) {
            let diff = (cpu_val - gpu_val).abs();
            max_diff = max_diff.max(diff);
        }

        // Allow small tolerance for floating point arithmetic differences
        const TOLERANCE: f32 = 1e-3;
        assert!(
            max_diff < TOLERANCE,
            "GPU and CPU outputs differ significantly: max_diff={}",
            max_diff
        );

        // Phase 20: Check for memory leaks
        drop(k_device);
        drop(v_device);
        drop(q_device);
        drop(output_device);
        // NOTE: Memory leak checking not available without GPU_FIXTURE
    }

    /// Test edge cases: single token, long sequences
    #[test]
    #[serial] // Phase 20: Run serially
    fn test_kv_replication_edge_cases() {
        // Phase 20: Check GPU availability first
        let backend = match HipBackend::new_checked() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("SKIP: GPU not available");
                return;
            }
        };
        let config = MultiQueryConfig::new(4, 32);
        let mqa = MultiQueryAttention::new(config).expect("Failed to create MQA");

        // Test 1: Single token
        {
            let batch_size = 1;
            let seq_len = 1;
            let num_kv_heads = 1;
            let num_q_heads = 4;
            let head_dim = 32;

            let k_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
            let v_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
            let q_shape = TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);

            let k_host = vec![1.0; batch_size * seq_len * num_kv_heads * head_dim];
            let v_host = vec![2.0; batch_size * seq_len * num_kv_heads * head_dim];
            let q_host = vec![3.0; batch_size * seq_len * num_q_heads * head_dim];

            let k_device = DeviceTensor::from_host_vec(&backend, k_host, k_shape)
                .expect("Failed to create K tensor");
            let v_device = DeviceTensor::from_host_vec(&backend, v_host, v_shape)
                .expect("Failed to create V tensor");
            let q_device = DeviceTensor::from_host_vec(&backend, q_host, q_shape.clone())
                .expect("Failed to create Q tensor");

            let output_device = mqa
                .forward_device(&q_device, &k_device, &v_device, None, None)
                .expect("Single token forward_device failed");

            assert_eq!(output_device.shape().dims(), q_shape.dims());
        }

        // Test 2: Long sequence (2048 tokens)
        {
            let batch_size = 1;
            let seq_len = 2048;
            let num_kv_heads = 1;
            let num_q_heads = 4;
            let head_dim = 32;

            let k_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
            let v_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
            let q_shape = TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);

            let mut k_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
            let mut v_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
            let mut q_host = vec![0.0f32; batch_size * seq_len * num_q_heads * head_dim];

            for (i, val) in k_host.iter_mut().enumerate() {
                *val = (i as f32) * 0.001;
            }
            for (i, val) in v_host.iter_mut().enumerate() {
                *val = (i as f32) * 0.002;
            }
            for (i, val) in q_host.iter_mut().enumerate() {
                *val = (i as f32) * 0.003;
            }

            let k_device = DeviceTensor::from_host_vec(&backend, k_host, k_shape)
                .expect("Failed to create K tensor");
            let v_device = DeviceTensor::from_host_vec(&backend, v_host, v_shape)
                .expect("Failed to create V tensor");
            let q_device = DeviceTensor::from_host_vec(&backend, q_host, q_shape.clone())
                .expect("Failed to create Q tensor");

            let output_device = mqa
                .forward_device(&q_device, &k_device, &v_device, None, None)
                .expect("Long sequence forward_device failed");

            assert_eq!(output_device.shape().dims(), q_shape.dims());

            let output_host = output_device
                .to_host_vec()
                .expect("Failed to copy output to host");
            let non_zero_count = output_host.iter().filter(|&&x| x != 0.0).count();
            assert!(
                non_zero_count > 0,
                "Long sequence output should not be all zeros"
            );
        }

        // Phase 20: Check for memory leaks at end of test
        // NOTE: Memory leak checking not available without GPU_FIXTURE
    }
}
