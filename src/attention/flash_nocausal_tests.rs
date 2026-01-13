//! Fused non-causal FlashAttention tests - Phase 3a.5
//!
//! Tests verify fused attention: QK^T → scale → softmax → softmax × V
//! Layout: [batch, heads, seq, dim] explicit for all tensors
//!
//! This is the final piece of Phase 3a - combining all atomic operations
//! into a single kernel for maximum efficiency.

#[cfg(feature = "rocm")]
#[cfg(test)]
mod flash_nocausal_tests {
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;

    const TEST_TOLERANCE: f32 = 1e-4;
    const TEST_TOLERANCE_LARGE: f32 = 2e-3; // Fused kernel has more FP operations

    /// Helper: Create Q tensor [batch, heads, seq, dim]
    fn create_q_tensor(batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let total = batch * heads * seq * dim;
        (0..total).map(|i| (i as f32) * 0.1).collect()
    }

    /// Helper: Create K tensor [batch, heads, seq, dim]
    fn create_k_tensor(batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let total = batch * heads * seq * dim;
        (0..total).map(|i| (i as f32) * 0.1 + 1.0).collect()
    }

    /// Helper: Create V tensor [batch, heads, seq, dim]
    fn create_v_tensor(batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let total = batch * heads * seq * dim;
        (0..total).map(|i| (i as f32) * 0.1 + 2.0).collect()
    }

    /// CPU reference for non-causal FlashAttention
    ///
    /// Computes: QK^T → scale → softmax → softmax × V
    fn flash_attention_nocausal_cpu_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        heads: usize,
        seq_len: usize,
        dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * heads * seq_len * dim];

        for b in 0..batch {
            for h in 0..heads {
                let q_offset = b * heads * seq_len * dim + h * seq_len * dim;
                let k_offset = b * heads * seq_len * dim + h * seq_len * dim;
                let v_offset = b * heads * seq_len * dim + h * seq_len * dim;
                let out_offset = b * heads * seq_len * dim + h * seq_len * dim;

                // For each query position
                for query_pos in 0..seq_len {
                    // Step 1: Compute QK^T scores
                    let mut scores = vec![0.0f32; seq_len];
                    for key_pos in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..dim {
                            let q_val = q[q_offset + query_pos * dim + d];
                            let k_val = k[k_offset + key_pos * dim + d];
                            dot += q_val * k_val;
                        }
                        scores[key_pos] = dot * scale;
                    }

                    // Step 2: Softmax (numerically stable)
                    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut weights = vec![0.0f32; seq_len];
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    for (i, s) in scores.iter().enumerate() {
                        weights[i] = s / exp_sum;
                    }

                    // Step 3: Weighted × V
                    for d in 0..dim {
                        let mut sum = 0.0f32;
                        for key_pos in 0..seq_len {
                            let v_val = v[v_offset + key_pos * dim + d];
                            sum += weights[key_pos] * v_val;
                        }
                        output[out_offset + query_pos * dim + d] = sum;
                    }
                }
            }
        }

        output
    }

    /// Test 1: Fused non-causal matches CPU - small dimensions
    #[test]
    fn test_flash_nocausal_matches_cpu_small() {
        let batch = 1;
        let heads = 2;
        let seq_len = 4;
        let dim = 8;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        // CPU reference
        let cpu_result =
            flash_attention_nocausal_cpu_reference(&q, &k, &v, batch, heads, seq_len, dim, scale);

        // GPU run
        let backend = HipBackend::new().expect("Failed to create HIP backend");

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q, q_shape.clone())
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k, k_shape.clone())
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v, v_shape.clone())
            .expect("Failed to create V tensor");
        let mut out_gpu = DeviceTensor::empty(&backend, out_shape.clone())
            .expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::flash_attention_nocausal_gpu_kernel(
                q_gpu.as_ptr() as *const f32,
                k_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                scale,
                batch as u32,
                seq_len as u32,
                heads as u32,
                dim as u32,
            )
        };

        assert_eq!(result, Ok(()), "GPU kernel failed");

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        assert_eq!(cpu_result.len(), gpu_result.len());

        let mut max_diff = 0.0f32;
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < TEST_TOLERANCE,
                "Flash non-causal mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
        println!("Flash non-causal small max diff: {}", max_diff);
    }

    /// Test 2: Fused non-causal matches CPU - 16×16 (larger)
    #[test]
    fn test_flash_nocausal_matches_cpu_16x16() {
        let batch = 1;
        let heads = 2;
        let seq_len = 16;
        let dim = 16;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        let cpu_result =
            flash_attention_nocausal_cpu_reference(&q, &k, &v, batch, heads, seq_len, dim, scale);

        let backend = HipBackend::new().expect("Failed to create HIP backend");

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q, q_shape.clone())
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k, k_shape.clone())
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v, v_shape.clone())
            .expect("Failed to create V tensor");
        let mut out_gpu = DeviceTensor::empty(&backend, out_shape.clone())
            .expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::flash_attention_nocausal_gpu_kernel(
                q_gpu.as_ptr() as *const f32,
                k_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                scale,
                batch as u32,
                seq_len as u32,
                heads as u32,
                dim as u32,
            )
        };

        assert_eq!(result, Ok(()));

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        let mut max_diff = 0.0f32;
        for (cpu_val, gpu_val) in cpu_result.iter().zip(gpu_result.iter()) {
            max_diff = max_diff.max((cpu_val - gpu_val).abs());
        }
        println!("Flash non-causal 16x16 max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }

    /// Test 3: Fused non-causal with 32×32 (correctness at scale)
    #[test]
    fn test_flash_nocausal_matches_cpu_32x32() {
        let batch = 2;
        let heads = 4;
        let seq_len = 32;
        let dim = 32;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        let cpu_result =
            flash_attention_nocausal_cpu_reference(&q, &k, &v, batch, heads, seq_len, dim, scale);

        let backend = HipBackend::new().expect("Failed to create HIP backend");

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q, q_shape.clone())
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k, k_shape.clone())
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v, v_shape.clone())
            .expect("Failed to create V tensor");
        let mut out_gpu = DeviceTensor::empty(&backend, out_shape.clone())
            .expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::flash_attention_nocausal_gpu_kernel(
                q_gpu.as_ptr() as *const f32,
                k_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                scale,
                batch as u32,
                seq_len as u32,
                heads as u32,
                dim as u32,
            )
        };

        assert_eq!(result, Ok(()));

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        let mut max_diff = 0.0f32;
        for (cpu_val, gpu_val) in cpu_result.iter().zip(gpu_result.iter()) {
            max_diff = max_diff.max((cpu_val - gpu_val).abs());
        }
        println!("Flash non-causal 32x32 max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }

    /// Test 4: Verify row-wise softmax properties (rows sum to ~1)
    #[test]
    fn test_flash_nocausal_softmax_properties() {
        let batch = 1;
        let heads = 1;
        let seq_len = 8;
        let dim = 16;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        // Create V with all ones - output will be sum of softmax weights
        let v_ones = vec![1.0f32; batch * heads * seq_len * dim];

        let backend = HipBackend::new().expect("Failed to create HIP backend");

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q, q_shape.clone())
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k, k_shape.clone())
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v_ones, v_shape.clone())
            .expect("Failed to create V tensor");
        let mut out_gpu = DeviceTensor::empty(&backend, out_shape.clone())
            .expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::flash_attention_nocausal_gpu_kernel(
                q_gpu.as_ptr() as *const f32,
                k_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                scale,
                batch as u32,
                seq_len as u32,
                heads as u32,
                dim as u32,
            )
        };

        assert_eq!(result, Ok(()));

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // With V=all ones, each output dimension should equal the softmax weight sum (~1.0)
        for d in 0..dim {
            for query_pos in 0..seq_len {
                let idx = query_pos * dim + d;
                let val = gpu_result[idx];
                assert!(
                    (val - 1.0).abs() < 1e-3,
                    "Softmax weights don't sum to 1.0: query_pos={}, dim={}, value={}",
                    query_pos,
                    d,
                    val
                );
            }
        }

        println!("Flash non-causal softmax properties verified");
    }

    /// Test 5: Compare fused kernel vs separate kernels (consistency check)
    #[test]
    fn test_flash_nocausal_vs_separate_kernels() {
        let batch = 1;
        let heads = 2;
        let seq_len = 8;
        let dim = 16;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        let backend = HipBackend::new().expect("Failed to create HIP backend");

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let scores_shape = TensorShape::from_dims(&[batch, heads, seq_len, seq_len]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);

        // Method 1: Separate kernels
        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape.clone()).unwrap();
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape.clone()).unwrap();
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape.clone()).unwrap();
        let mut scores_gpu = DeviceTensor::empty(&backend, scores_shape.clone()).unwrap();
        let mut weights_gpu = DeviceTensor::empty(&backend, scores_shape.clone()).unwrap();
        let mut out_separate = DeviceTensor::empty(&backend, out_shape.clone()).unwrap();

        // QK^T with scale
        unsafe {
            crate::attention::kernels::qkt_matmul_gpu_kernel_scaled(
                q_gpu.as_ptr() as *const f32,
                k_gpu.as_ptr() as *const f32,
                scores_gpu.buffer().as_mut_ptr() as *mut f32,
                batch as u32,
                seq_len as u32,
                seq_len as u32,
                heads as u32,
                dim as u32,
                scale,
            )
            .unwrap();
        }

        // Softmax (in-place on scores_gpu)
        let total_rows = (batch * heads * seq_len) as u32;
        let row_len = seq_len as u32;
        unsafe {
            crate::attention::kernels::softmax_gpu_kernel(
                scores_gpu.buffer().as_mut_ptr() as *mut f32,
                total_rows,
                row_len,
            );
        }

        // Copy scores to weights (same buffer after softmax)
        weights_gpu = DeviceTensor::from_host_vec(
            &backend,
            scores_gpu.to_host_vec().unwrap(),
            scores_shape.clone(),
        )
        .unwrap();

        // Weighted × V
        unsafe {
            crate::attention::kernels::weighted_matmul_gpu_kernel(
                weights_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_separate.buffer().as_mut_ptr() as *mut f32,
                batch as u32,
                seq_len as u32,
                seq_len as u32,
                heads as u32,
                dim as u32,
            )
            .unwrap();
        }

        backend.synchronize().unwrap();
        let result_separate = out_separate.to_host_vec().unwrap();

        // Method 2: Fused kernel
        let q_gpu2 = DeviceTensor::from_host_vec(&backend, q, q_shape).unwrap();
        let k_gpu2 = DeviceTensor::from_host_vec(&backend, k, k_shape).unwrap();
        let v_gpu2 = DeviceTensor::from_host_vec(&backend, v, v_shape).unwrap();
        let mut out_fused = DeviceTensor::empty(&backend, out_shape).unwrap();

        unsafe {
            crate::attention::kernels::flash_attention_nocausal_gpu_kernel(
                q_gpu2.as_ptr() as *const f32,
                k_gpu2.as_ptr() as *const f32,
                v_gpu2.as_ptr() as *const f32,
                out_fused.buffer().as_mut_ptr() as *mut f32,
                scale,
                batch as u32,
                seq_len as u32,
                heads as u32,
                dim as u32,
            )
            .unwrap();
        }

        backend.synchronize().unwrap();
        let result_fused = out_fused.to_host_vec().unwrap();

        // Compare: fused vs separate should be very close
        let mut max_diff = 0.0f32;
        for (i, (sep, fused)) in result_separate.iter().zip(result_fused.iter()).enumerate() {
            let diff = (sep - fused).abs();
            max_diff = max_diff.max(diff);
        }
        println!("Flash fused vs separate max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }
}
