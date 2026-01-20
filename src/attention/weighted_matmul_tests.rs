//! Weighted × V matmul tests - Phase 3a.4
//!
//! Tests verify softmax weights × V GPU implementation matches CPU reference.
//! Layout: [batch, heads, seq_q, seq_k] × [batch, heads, seq_k, dim] → [batch, heads, seq_q, dim]
//!
//! This computes: output[query_pos, dim] = sum over seq_k of (weights[query_pos, key_pos] * V[key_pos, dim])

#[cfg(feature = "rocm")]
#[cfg(test)]
mod weighted_matmul_tests {
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;
    use serial_test::serial;

    const TEST_TOLERANCE: f32 = 1e-4;
    const TEST_TOLERANCE_LARGE: f32 = 1e-3; // For larger inputs due to FP reduction order

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    ///
    /// This follows llama.cpp's approach of checking GPU availability before
    /// running tests. If GPU is not available, the test is skipped gracefully
    /// rather than crashing or failing.
    fn get_backend_or_skip() -> std::sync::Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for weighted_matmul_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP"); // Special panic to indicate skip
            }
        }
    }

    /// Helper: Create softmax weights [batch, heads, seq_q, seq_k]
    fn create_weights_tensor(batch: usize, heads: usize, seq_q: usize, seq_k: usize) -> Vec<f32> {
        let total = batch * heads * seq_q * seq_k;
        // Create softmax-like weights (row-normalized positive values)
        let mut data = Vec::with_capacity(total);
        for b in 0..batch {
            for h in 0..heads {
                for sq in 0..seq_q {
                    // Create values that sum to 1.0 per row
                    let row: Vec<f32> = (0..seq_k).map(|i| (i + 1) as f32).collect();
                    let sum: f32 = row.iter().sum();
                    for sk in 0..seq_k {
                        data.push(row[sk] / sum);
                    }
                }
            }
        }
        data
    }

    /// Helper: Create V tensor [batch, heads, seq_k, dim]
    fn create_v_tensor(batch: usize, heads: usize, seq_k: usize, dim: usize) -> Vec<f32> {
        let total = batch * heads * seq_k * dim;
        (0..total).map(|i| (i as f32) * 0.1 + 2.0).collect()
    }

    /// CPU reference for weighted × V matmul with explicit layout
    ///
    /// Input:  weights [batch, heads, seq_q, seq_k]
    ///         V       [batch, heads, seq_k, dim]
    /// Output: result  [batch, heads, seq_q, dim]
    ///
    /// For each (batch, head): output[seq_q, dim] = weights[seq_q, seq_k] @ V[seq_k, dim]
    fn weighted_matmul_cpu_reference(
        weights: &[f32],
        v: &[f32],
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * heads * seq_q * dim];

        for b in 0..batch {
            for h in 0..heads {
                let weights_offset = b * heads * seq_q * seq_k + h * seq_q * seq_k;
                let v_offset = b * heads * seq_k * dim + h * seq_k * dim;
                let out_offset = b * heads * seq_q * dim + h * seq_q * dim;

                // output[query_pos, d] = sum over key_pos of (weights[query_pos, key_pos] * V[key_pos, d])
                for sq in 0..seq_q {
                    for d in 0..dim {
                        let mut sum = 0.0f32;
                        for sk in 0..seq_k {
                            let w = weights[weights_offset + sq * seq_k + sk];
                            let val = v[v_offset + sk * dim + d];
                            sum += w * val;
                        }
                        output[out_offset + sq * dim + d] = sum;
                    }
                }
            }
        }

        output
    }

    /// Test 1: Weighted matmul matches CPU - small dimensions
    #[test]
    #[serial]
    fn test_weighted_matmul_matches_cpu_small() {
        let batch = 1;
        let heads = 2;
        let seq_q = 4;
        let seq_k = 4;
        let dim = 8;

        let weights = create_weights_tensor(batch, heads, seq_q, seq_k);
        let v = create_v_tensor(batch, heads, seq_k, dim);

        // CPU reference
        let cpu_result =
            weighted_matmul_cpu_reference(&weights, &v, batch, heads, seq_q, seq_k, dim);

        // GPU run (llama.cpp pattern: skip if GPU not available)
        let backend = get_backend_or_skip();

        let weights_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_k, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_q, dim]);

        let weights_gpu = DeviceTensor::from_host_vec(&backend, weights.clone(), weights_shape)
            .expect("Failed to create weights tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape)
            .expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::weighted_matmul_gpu_kernel(
                weights_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                batch as u32,
                seq_q as u32,
                seq_k as u32,
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
                "Weighted matmul mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
        println!("Weighted matmul small max diff: {}", max_diff);
    }

    /// Test 2: Weighted matmul matches CPU - 32×32×4×32 (larger test)
    #[test]
    #[serial]
    fn test_weighted_matmul_matches_cpu_32x32() {
        let batch = 2;
        let heads = 4;
        let seq_q = 32;
        let seq_k = 32;
        let dim = 32;

        let weights = create_weights_tensor(batch, heads, seq_q, seq_k);
        let v = create_v_tensor(batch, heads, seq_k, dim);

        let cpu_result =
            weighted_matmul_cpu_reference(&weights, &v, batch, heads, seq_q, seq_k, dim);

        let backend = get_backend_or_skip();

        let weights_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_k, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_q, dim]);

        let weights_gpu = DeviceTensor::from_host_vec(&backend, weights, weights_shape)
            .expect("Failed to create weights tensor");
        let v_gpu =
            DeviceTensor::from_host_vec(&backend, v, v_shape).expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::weighted_matmul_gpu_kernel(
                weights_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                batch as u32,
                seq_q as u32,
                seq_k as u32,
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
        println!("Weighted matmul 32x32 max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }

    /// Test 3: Non-square sequences (seq_q != seq_k)
    #[test]
    #[serial]
    fn test_weighted_matmul_non_square_sequences() {
        let batch = 1;
        let heads = 2;
        let seq_q = 8;
        let seq_k = 16;
        let dim = 8;

        let weights = create_weights_tensor(batch, heads, seq_q, seq_k);
        let v = create_v_tensor(batch, heads, seq_k, dim);

        let cpu_result =
            weighted_matmul_cpu_reference(&weights, &v, batch, heads, seq_q, seq_k, dim);

        let backend = get_backend_or_skip();

        let weights_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);
        let v_shape = TensorShape::from_dims(&[batch, heads, seq_k, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_q, dim]);

        let weights_gpu = DeviceTensor::from_host_vec(&backend, weights, weights_shape)
            .expect("Failed to create weights tensor");
        let v_gpu =
            DeviceTensor::from_host_vec(&backend, v, v_shape).expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::attention::kernels::weighted_matmul_gpu_kernel(
                weights_gpu.as_ptr() as *const f32,
                v_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                batch as u32,
                seq_q as u32,
                seq_k as u32,
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
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            max_diff = max_diff.max(diff);
            if diff >= TEST_TOLERANCE_LARGE {
                panic!(
                    "Mismatch at {}: CPU={}, GPU={}, diff={}",
                    i, cpu_val, gpu_val, diff
                );
            }
        }
        println!("Weighted matmul non-square max diff: {}", max_diff);

        // Output shape should be [batch, heads, seq_q, dim] = [1, 2, 8, 8] = 128
        assert_eq!(gpu_result.len(), 128);
    }

    /// Test 4: Verify explicit layout indexing is correct
    #[test]
    #[serial]
    fn test_weighted_matmul_explicit_layout_indexing() {
        let batch = 1;
        let heads = 2;
        let seq_q = 4;
        let seq_k = 4;
        let dim = 8;

        let weights = create_weights_tensor(batch, heads, seq_q, seq_k);
        let v = create_v_tensor(batch, heads, seq_k, dim);

        // Our explicit layout reference
        let explicit_result =
            weighted_matmul_cpu_reference(&weights, &v, batch, heads, seq_q, seq_k, dim);

        // Spot check: verify a few manual calculations
        // For head=0, query_pos=0, d=0:
        // output = sum over key_pos of (weights[0,0,0,key_pos] * V[0,0,key_pos,0])
        let h = 0;
        let sq = 0;
        let d = 0;

        let weights_offset = h * seq_q * seq_k; // batch=0
        let v_offset = h * seq_k * dim;

        let mut manual_sum = 0.0f32;
        for sk in 0..seq_k {
            let w = weights[weights_offset + sq * seq_k + sk];
            let val = v[v_offset + sk * dim + d];
            manual_sum += w * val;
        }

        let out_offset = h * seq_q * dim;
        let explicit_val = explicit_result[out_offset + sq * dim + d];

        assert!(
            (manual_sum - explicit_val).abs() < 1e-6,
            "Layout verification failed: manual={}, explicit={}",
            manual_sum,
            explicit_val
        );

        println!("Explicit layout indexing verified correctly");
    }
}
