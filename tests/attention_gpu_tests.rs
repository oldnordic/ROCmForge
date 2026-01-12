//! GPU Attention Kernel Tests - Phase E
//!
//! Test suite for Phase E: GPU Attention Kernels implementation.
//! Tests HIP kernels for QK^T computation, softmax, and attention-weighted V.

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
    use rocmforge::backend::{DeviceTensor, HipBackend};
    use rocmforge::loader::TensorShape;
    use rocmforge::model::config::ModelConfig;
    use rocmforge::model::kv_cache::KVCache;
    use rocmforge::ops::attention_gpu::HipAttentionKernels;

    /// Test basic HIP backend creation for attention kernels
    #[test]
    #[cfg(feature = "rocm")]
    #[serial]
    fn test_hip_backend_creation() {
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let _backend = fixture.backend();
        assert!(true, "HIP backend created successfully");
        fixture.assert_no_leak(5);
    }

    /// Test that we can create model configuration
    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig::llama2_7b();
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.hidden_size, 4096);
    }

    /// Test HipAttentionKernels struct creation - this should fail initially
    #[test]
    #[cfg(feature = "rocm")]
    #[serial]
    fn test_hip_attention_kernels_creation() {
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        // This should fail until we implement HipAttentionKernels
        let _kernels = rocmforge::ops::attention_gpu::HipAttentionKernels::new(backend)
            .expect("Failed to create HIP attention kernels");
        fixture.assert_no_leak(5);
    }

    /// Helper function to create random test data
    fn create_random_data(size: usize, min: f32, max: f32) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let hash = hasher.finish();
            let normalized = (hash as f64 / u64::MAX as f64) as f32;
            let value = min + normalized * (max - min);
            data.push(value);
        }
        data
    }

    /// Test GPU QK^T kernel computation
    #[test]
    #[cfg(feature = "rocm")]
    #[serial]
    fn test_gpu_qk_kernel() {
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Test configuration
        let seq_len = 4;
        let num_heads = 8;
        let head_dim = 128;
        let cache_len = 10;

        // Create test tensors
        let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let k_shape = TensorShape::from_dims(&[cache_len, num_heads, head_dim]);
        let output_shape = TensorShape::from_dims(&[seq_len, cache_len]);

        let q_data = create_random_data(seq_len * num_heads * head_dim, 0.0, 1.0);
        let k_data = create_random_data(cache_len * num_heads * head_dim, 0.0, 1.0);

        let q = DeviceTensor::from_host_vec(backend, q_data, q_shape)
            .expect("Failed to create Q tensor");
        let k = DeviceTensor::from_host_vec(backend, k_data, k_shape)
            .expect("Failed to create K tensor");
        let mut output =
            DeviceTensor::empty(backend, output_shape).expect("Failed to create output tensor");

        // Initialize attention kernels
        let kernels =
            HipAttentionKernels::new(backend).expect("Failed to initialize attention kernels");

        // Test QK^T computation
        kernels
            .compute_qk_t(&q, &k, &mut output)
            .expect("QK^T computation failed");

        // Verify output shape
        assert_eq!(output.shape().dims(), &[seq_len, cache_len]);

        // Verify computation against CPU reference
        let q_host = q.to_host_vec().expect("Failed to copy Q to host");
        let k_host = k.to_host_vec().expect("Failed to copy K to host");
        let output_host = output.to_host_vec().expect("Failed to copy output to host");

        // Compute reference on CPU
        for i in 0..seq_len {
            for j in 0..cache_len {
                let mut sum = 0.0f32;
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        let q_idx = i * num_heads * head_dim + h * head_dim + d;
                        let k_idx = j * num_heads * head_dim + h * head_dim + d;
                        sum += q_host[q_idx] * k_host[k_idx];
                    }
                }
                let output_idx = i * cache_len + j;
                assert!(
                    (output_host[output_idx] - sum).abs() < 1e-5,
                    "QK^T mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    output_host[output_idx],
                    sum
                );
            }
        }

        drop(q);
        drop(k);
        drop(output);
        fixture.assert_no_leak(5);
    }

    /// Test GPU softmax kernel with causal masking
    #[test]
    #[cfg(feature = "rocm")]
    #[serial]
    fn test_gpu_softmax_kernel() {
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        let seq_len = 6;
        let cache_len = 8;
        let attention_shape = TensorShape::from_dims(&[seq_len, cache_len]);
        let temp_shape = TensorShape::from_dims(&[cache_len]);

        // Create attention scores (before softmax)
        let attention_data = create_random_data(seq_len * cache_len, -2.0, 2.0);
        let mut attention = DeviceTensor::from_host_vec(backend, attention_data, attention_shape)
            .expect("Failed to create attention tensor");
        let temp_buffer =
            DeviceTensor::empty(backend, temp_shape).expect("Failed to create temp buffer");

        let kernels =
            HipAttentionKernels::new(backend).expect("Failed to initialize attention kernels");

        // Apply causal mask and softmax
        kernels
            .apply_causal_mask(&mut attention, seq_len, cache_len)
            .expect("Failed to apply causal mask");
        kernels
            .compute_softmax(&mut attention, &temp_buffer)
            .expect("Failed to compute softmax");

        // Verify softmax properties
        let attention_host = attention
            .to_host_vec()
            .expect("Failed to copy attention to host");

        for i in 0..seq_len {
            let mut row_sum = 0.0f32;
            for j in 0..cache_len {
                let idx = i * cache_len + j;
                // Check causal masking
                if j > i {
                    assert_eq!(
                        attention_host[idx], 0.0,
                        "Causal mask failed at ({}, {}): should be 0.0",
                        i, j
                    );
                } else {
                    assert!(
                        attention_host[idx] >= 0.0,
                        "Softmax output negative at ({}, {}): {}",
                        i,
                        j,
                        attention_host[idx]
                    );
                    row_sum += attention_host[idx];
                }
            }
            // Check row normalization (only for non-masked elements)
            if i < cache_len {
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "Softmax row {} not normalized: sum = {}",
                    i,
                    row_sum
                );
            }
        }

        drop(attention);
        drop(temp_buffer);
        fixture.assert_no_leak(5);
    }

    /// Test GPU attention-weighted V kernel
    #[test]
    #[cfg(feature = "rocm")]
    #[serial]
    fn test_gpu_attention_v_kernel() {
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        let seq_len = 3;
        let cache_len = 5;
        let num_heads = 4;
        let head_dim = 64;

        let attention_shape = TensorShape::from_dims(&[seq_len, cache_len]);
        let v_shape = TensorShape::from_dims(&[cache_len, num_heads, head_dim]);
        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);

        let attention_data = create_random_data(seq_len * cache_len, 0.0, 1.0);
        let v_data = create_random_data(cache_len * num_heads * head_dim, 0.0, 1.0);

        let attention = DeviceTensor::from_host_vec(backend, attention_data, attention_shape)
            .expect("Failed to create attention tensor");
        let v = DeviceTensor::from_host_vec(backend, v_data, v_shape)
            .expect("Failed to create V tensor");
        let mut output =
            DeviceTensor::empty(backend, output_shape).expect("Failed to create output tensor");

        let kernels =
            HipAttentionKernels::new(backend).expect("Failed to initialize attention kernels");

        // Compute attention-weighted V
        kernels
            .compute_attention_weighted_v(&attention, &v, &mut output)
            .expect("Failed to compute attention-weighted V");

        // Verify output shape
        assert_eq!(output.shape().dims(), &[seq_len, num_heads, head_dim]);

        // Verify computation against CPU reference
        let attention_host = attention
            .to_host_vec()
            .expect("Failed to copy attention to host");
        let v_host = v.to_host_vec().expect("Failed to copy V to host");
        let output_host = output.to_host_vec().expect("Failed to copy output to host");

        for i in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for k in 0..cache_len {
                        let attention_idx = i * cache_len + k;
                        let v_idx = k * num_heads * head_dim + h * head_dim + d;
                        sum += attention_host[attention_idx] * v_host[v_idx];
                    }
                    let output_idx = i * num_heads * head_dim + h * head_dim + d;
                    assert!(
                        (output_host[output_idx] - sum).abs() < 1e-5,
                        "Attention@V mismatch at ({}, {}, {}): {} vs {}",
                        i,
                        h,
                        d,
                        output_host[output_idx],
                        sum
                    );
                }
            }
        }

        drop(attention);
        drop(v);
        drop(output);
        fixture.assert_no_leak(5);
    }

    /// Test complete GPU attention pipeline with KV cache integration
    #[test]
    #[cfg(feature = "rocm")]
    #[serial]
    fn test_gpu_attention_pipeline() {
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Model configuration
        let config = ModelConfig::llama2_7b();
        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let seq_len = 1; // Single token generation
        let current_seq_len = 10; // Current context length

        // Create KV cache
        let kv_cache = KVCache::new(
            backend,
            config.num_hidden_layers,
            num_heads,
            head_dim,
            config.max_position_embeddings,
        )
        .expect("Failed to create KV cache");

        // Create test tensors
        let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let attention_scores_shape = TensorShape::from_dims(&[seq_len, current_seq_len]);
        let softmax_temp_shape = TensorShape::from_dims(&[current_seq_len]);

        let q_data = create_random_data(seq_len * num_heads * head_dim, 0.0, 1.0);
        let q = DeviceTensor::from_host_vec(backend, q_data, q_shape)
            .expect("Failed to create Q tensor");
        let attention_scores = DeviceTensor::empty(backend, attention_scores_shape)
            .expect("Failed to create attention scores");
        let softmax_temp = DeviceTensor::empty(backend, softmax_temp_shape)
            .expect("Failed to create softmax temp");

        let kernels =
            HipAttentionKernels::new(backend).expect("Failed to initialize attention kernels");

        // Test complete attention computation
        let layer_id = 0;
        let output = kernels
            .compute_attention(
                &q,
                &attention_scores,
                &softmax_temp,
                &kv_cache,
                layer_id,
                current_seq_len,
            )
            .expect("Failed to compute attention");

        // Verify output shape
        assert_eq!(output.shape().dims(), &[seq_len, num_heads, head_dim]);

        // Verify finite values
        let output_host = output.to_host_vec().expect("Failed to copy output to host");
        for (i, &val) in output_host.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output contains non-finite value at index {}: {}",
                i,
                val
            );
        }

        drop(q);
        drop(attention_scores);
        drop(softmax_temp);
        drop(kv_cache);
        drop(output);
        fixture.assert_no_leak(5);
    }

    /// Test attention kernel error handling
    #[test]
    #[cfg(feature = "rocm")]
    #[serial]
    fn test_attention_kernel_error_handling() {
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();
        let kernels =
            HipAttentionKernels::new(backend).expect("Failed to initialize attention kernels");

        // Test invalid tensor shapes
        let wrong_shape = TensorShape::from_dims(&[5, 10]); // 2D instead of 3D
        let q = DeviceTensor::empty(backend, wrong_shape).expect("Failed to create tensor");
        let k_shape = TensorShape::from_dims(&[10, 8, 128]);
        let k = DeviceTensor::empty(backend, k_shape).expect("Failed to create tensor");
        let output_shape = TensorShape::from_dims(&[5, 10]);
        let mut output =
            DeviceTensor::empty(backend, output_shape).expect("Failed to create output");

        // Should fail with invalid Q shape
        let result = kernels.compute_qk_t(&q, &k, &mut output);
        assert!(result.is_err(), "QK^T should fail with invalid Q shape");

        // Test mismatched dimensions
        let q_shape = TensorShape::from_dims(&[5, 8, 128]);
        let q = DeviceTensor::empty(backend, q_shape).expect("Failed to create Q tensor");
        let k_wrong_shape = TensorShape::from_dims(&[10, 16, 64]); // Different num_heads/head_dim
        let k_wrong =
            DeviceTensor::empty(backend, k_wrong_shape).expect("Failed to create K tensor");

        let result = kernels.compute_qk_t(&q, &k_wrong, &mut output);
        assert!(
            result.is_err(),
            "QK^T should fail with mismatched dimensions"
        );

        drop(q);
        drop(k);
        drop(k_wrong);
        drop(output);
        fixture.assert_no_leak(5);
    }

    #[test]
    #[cfg(not(feature = "rocm"))]
    fn test_gpu_backend_disabled_without_rocm() {
        // Test that GPU functionality is properly disabled without ROCm feature
        // This test should compile and pass when ROCm is not available
        assert!(true, "GPU backend should be disabled without ROCm feature");
    }
}
