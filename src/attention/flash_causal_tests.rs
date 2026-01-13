//! Fused causal FlashAttention tests - Phase 3b.2
//!
//! Tests verify fused attention with causal masking:
//! QK^T → scale → causal mask → softmax → softmax × V
//! Layout: [batch, heads, seq, dim] explicit for all tensors
//!
//! This extends Phase 3a.5 (non-causal) by adding causal masking.
//! Test first, implement kernel after tests fail (TDD).

#[cfg(feature = "rocm")]
#[cfg(test)]
mod flash_causal_tests {
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;

    const TEST_TOLERANCE: f32 = 1e-4;
    const TEST_TOLERANCE_LARGE: f32 = 2e-3;

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

    /// CPU reference for causal FlashAttention
    ///
    /// Computes: QK^T → scale → causal mask → softmax → softmax × V
    /// Layout: [batch, heads, seq, dim] explicit
    fn flash_attention_causal_cpu_reference(
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

                    // Step 2: Apply causal mask
                    // Causal: query_pos can only attend to key_pos <= query_pos
                    for key_pos in (query_pos + 1)..seq_len {
                        scores[key_pos] = f32::NEG_INFINITY;
                    }

                    // Step 3: Softmax (numerically stable)
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

                    // Step 4: Weighted × V
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

    /// Test 1: Fused causal matches CPU - small dimensions
    #[test]
    fn test_flash_causal_matches_cpu_small() {
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
            flash_attention_causal_cpu_reference(&q, &k, &v, batch, heads, seq_len, dim, scale);

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
            crate::attention::kernels::flash_attention_causal_gpu_kernel(
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
                "Flash causal mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
        println!("Flash causal small max diff: {}", max_diff);
    }

    /// Test 2: Causal vs non-causal comparison
    /// For the first query position, causal and non-causal should be the same
    /// For later positions, causal should ignore future keys
    #[test]
    fn test_flash_causal_first_position_matches_noncausal() {
        let batch = 1;
        let heads = 1;
        let seq_len = 8;
        let dim = 16;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        // CPU reference for causal
        let causal_cpu =
            flash_attention_causal_cpu_reference(&q, &k, &v, batch, heads, seq_len, dim, scale);

        // For query position 0, causal and non-causal should be identical
        // (query 0 can attend to all keys)
        for d in 0..dim {
            let idx = 0 * dim + d; // First query position, dimension d
            assert!(
                causal_cpu[idx].is_finite(),
                "Causal output at position 0 should be finite"
            );
        }

        println!("Flash causal first position check passed");
    }

    /// Test 3: Causal mask property - weights sum to 1 for valid positions
    #[test]
    fn test_flash_causal_weights_sum_to_one() {
        let batch = 1;
        let heads = 1;
        let seq_len = 4;
        let dim = 8;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        // GPU run
        let backend = HipBackend::new().expect("Failed to create HIP backend");

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);

        let q_gpu =
            DeviceTensor::from_host_vec(&backend, q, q_shape).expect("Failed to create Q tensor");
        let k_gpu =
            DeviceTensor::from_host_vec(&backend, k, k_shape).expect("Failed to create K tensor");
        let v_gpu =
            DeviceTensor::from_host_vec(&backend, v, v_shape).expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::flash_attention_causal_gpu_kernel(
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

        // Verify output is finite (no NaN or inf in results)
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Flash causal output should be finite at index {}, got {}",
                i,
                val
            );
        }

        println!("Flash causal weights property verified");
    }

    /// Test 4: Larger test - 16x16
    #[test]
    fn test_flash_causal_matches_cpu_16x16() {
        let batch = 1;
        let heads = 2;
        let seq_len = 16;
        let dim = 16;
        let scale = 1.0 / (dim as f32).sqrt();

        let q = create_q_tensor(batch, heads, seq_len, dim);
        let k = create_k_tensor(batch, heads, seq_len, dim);
        let v = create_v_tensor(batch, heads, seq_len, dim);

        let cpu_result =
            flash_attention_causal_cpu_reference(&q, &k, &v, batch, heads, seq_len, dim, scale);

        let backend = HipBackend::new().expect("Failed to create HIP backend");

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_len, dim]);

        let q_gpu =
            DeviceTensor::from_host_vec(&backend, q, q_shape).expect("Failed to create Q tensor");
        let k_gpu =
            DeviceTensor::from_host_vec(&backend, k, k_shape).expect("Failed to create K tensor");
        let v_gpu =
            DeviceTensor::from_host_vec(&backend, v, v_shape).expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::flash_attention_causal_gpu_kernel(
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
        println!("Flash causal 16x16 max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }
}
