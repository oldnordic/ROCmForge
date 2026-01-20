//! QK^T matrix multiply tests - Phase 3a.1.1 (TDD)
//!
//! Tests verify QK^T matmul GPU implementation matches CPU reference.
//! Layout: [batch, heads, seq_q, dim] x [batch, heads, seq_k, dim] → [batch, heads, seq_q, seq_k]
//!
//! IMPORTANT: Per Phase 3a "explicit layout" requirement, we use [batch, heads, seq, dim]
//! where each [seq, dim] slice per head is CONTIGUOUS in memory.
//! This makes matmul_cpu's [seq, dim] expectation work without reshaping.
//!
//! This is the FIRST operation in the divide-and-conquer approach.
//! Following TDD: these tests WILL FAIL until kernel is implemented.

#[cfg(feature = "rocm")]
#[cfg(test)]
mod qkt_matmul_tests {
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;
    use serial_test::serial;

    const TEST_TOLERANCE: f32 = 1e-3;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> std::sync::Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for qkt_matmul_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Helper: Create explicit layout Q tensor [batch, heads, seq, dim]
    ///
    /// Layout is row-major with dimensions ordered [batch, heads, seq, dim]
    /// This makes each [seq, dim] slice per head contiguous for easy matmul.
    fn create_q_tensor_explicit(batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let total = batch * heads * seq * dim;
        (0..total).map(|i| (i as f32) * 0.1).collect()
    }

    /// Helper: Create explicit layout K tensor [batch, heads, seq, dim]
    fn create_k_tensor_explicit(batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let total = batch * heads * seq * dim;
        (0..total).map(|i| (i as f32) * 0.1 + 1.0).collect()
    }

    /// CPU reference for QK^T with explicit layout
    ///
    /// Input:  Q [batch, heads, seq_q, dim]
    ///         K [batch, heads, seq_k, dim]
    /// Output: Scores [batch, heads, seq_q, seq_k]
    ///
    /// This computes QK^T per head, where for each (batch, head):
    ///   scores[seq_q, seq_k] = Q[seq_q, dim] @ K[seq_k, dim]^T
    ///   = sum over dim of (Q[seq_q, d] * K[seq_k, d])
    ///
    /// Layout: [batch, heads, seq, dim] row-major
    /// Index: batch * heads * seq * dim + head * seq * dim + seq * dim + d
    fn qkt_matmul_cpu_reference(
        q: &[f32],
        k: &[f32],
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * heads * seq_q * seq_k];

        for b in 0..batch {
            for h in 0..heads {
                // Each head's data is contiguous
                let q_head_offset = b * heads * seq_q * dim + h * seq_q * dim;
                let k_head_offset = b * heads * seq_k * dim + h * seq_k * dim;
                let out_head_offset = b * heads * seq_q * seq_k + h * seq_q * seq_k;

                // Extract this head's [seq, dim] slices
                let q_head = &q[q_head_offset..q_head_offset + seq_q * dim];
                let k_head = &k[k_head_offset..k_head_offset + seq_k * dim];

                // Compute QK^T directly: output[query_pos, key_pos] = sum_d(Q[query_pos, d] * K[key_pos, d])
                for sq in 0..seq_q {
                    for sk in 0..seq_k {
                        let mut sum = 0.0f32;
                        for d in 0..dim {
                            let q_val = q_head[sq * dim + d];
                            let k_val = k_head[sk * dim + d];
                            sum += q_val * k_val;
                        }
                        let out_idx = out_head_offset + sq * seq_k + sk;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        output
    }

    /// Test 1: QK^T matches CPU - small dimensions (4×4×2×8)
    ///
    /// This is the TDD entry point - it WILL FAIL until we implement:
    /// - kernels/qkt_matmul.hip
    /// - build.rs integration
    /// - kernels.rs wrapper
    #[test]
    #[serial]
    fn test_qkt_matmul_matches_cpu_small() {
        let batch = 1;
        let heads = 2;
        let seq_q = 4;
        let seq_k = 4;
        let dim = 8;

        let q = create_q_tensor_explicit(batch, heads, seq_q, dim);
        let k = create_k_tensor_explicit(batch, heads, seq_k, dim);

        // CPU reference with explicit layout
        let cpu_result = qkt_matmul_cpu_reference(&q, &k, batch, heads, seq_q, seq_k, dim);

        // GPU computation
        let backend = get_backend_or_skip();

        let q_shape = TensorShape::from_dims(&[batch, heads, seq_q, dim]);
        let k_shape = TensorShape::from_dims(&[batch, heads, seq_k, dim]);
        let out_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape)
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape)
            .expect("Failed to create K tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        // Call qkt_matmul_gpu_kernel
        // Parameters: (q, k, output, batch_size, seq_q, seq_k, num_heads, head_dim)
        let result = unsafe {
            crate::attention::kernels::qkt_matmul_gpu_kernel(
                q_gpu.as_ptr() as *const f32,
                k_gpu.as_ptr() as *const f32,
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
                "QK^T mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
        println!("QK^T Max difference: {}", max_diff);
    }

    /// Test 2: QK^T matches CPU - 32×32×4×32 (larger test)
    #[test]
    #[serial]
    fn test_qkt_matmul_matches_cpu_32x32() {
        let batch = 2;
        let heads = 4;
        let seq_q = 32;
        let seq_k = 32;
        let dim = 32;

        let q = create_q_tensor_explicit(batch, heads, seq_q, dim);
        let k = create_k_tensor_explicit(batch, heads, seq_k, dim);

        let cpu_result = qkt_matmul_cpu_reference(&q, &k, batch, heads, seq_q, seq_k, dim);

        let expected_len = batch * heads * seq_q * seq_k;
        assert_eq!(cpu_result.len(), expected_len);
        assert!(cpu_result[0] != 0.0);

        // GPU test to be added once kernel exists
    }

    /// Test 3: Explicit layout index math verification
    ///
    /// This test verifies that our explicit layout indexing is correct
    /// by computing QK^T per head and checking the result is consistent.
    #[test]
    #[serial]
    fn test_qkt_matmul_explicit_layout_indexing() {
        let batch = 1;
        let heads = 2;
        let seq_q = 4;
        let seq_k = 4;
        let dim = 8;

        let q = create_q_tensor_explicit(batch, heads, seq_q, dim);
        let k = create_k_tensor_explicit(batch, heads, seq_k, dim);

        // Our explicit layout reference
        let explicit_result = qkt_matmul_cpu_reference(&q, &k, batch, heads, seq_q, seq_k, dim);

        // Verify by computing per-head QK^T directly and checking consistency
        for h in 0..heads {
            let q_head_offset = h * seq_q * dim;
            let k_head_offset = h * seq_k * dim;

            let q_head = &q[q_head_offset..q_head_offset + seq_q * dim];
            let k_head = &k[k_head_offset..k_head_offset + seq_k * dim];

            // Compute QK^T for this head: output[sq, sk] = sum_d(Q[sq, d] * K[sk, d])
            for sq in 0..seq_q {
                for sk in 0..seq_k {
                    let mut sum = 0.0f32;
                    for d in 0..dim {
                        sum += q_head[sq * dim + d] * k_head[sk * dim + d];
                    }

                    let explicit_idx = h * seq_q * seq_k + sq * seq_k + sk;
                    let diff = (explicit_result[explicit_idx] - sum).abs();
                    assert!(
                        diff < 1e-5,
                        "Layout mismatch at head={}, sq={}, sk={}: explicit={}, direct={}, diff={}",
                        h,
                        sq,
                        sk,
                        explicit_result[explicit_idx],
                        sum,
                        diff
                    );
                }
            }
        }

        println!("Explicit layout indexing verified correctly");
    }

    /// Test 4: Non-square sequence lengths (seq_q != seq_k)
    #[test]
    #[serial]
    fn test_qkt_matmul_non_square_sequences() {
        let batch = 1;
        let heads = 2;
        let seq_q = 8;
        let seq_k = 16;
        let dim = 8;

        let q = create_q_tensor_explicit(batch, heads, seq_q, dim);
        let k = create_k_tensor_explicit(batch, heads, seq_k, dim);

        let cpu_result = qkt_matmul_cpu_reference(&q, &k, batch, heads, seq_q, seq_k, dim);

        let expected_len = batch * heads * seq_q * seq_k;
        assert_eq!(cpu_result.len(), expected_len);

        // Output shape should be [batch, heads, seq_q, seq_k] = [1, 2, 8, 16] = 256
        assert_eq!(cpu_result.len(), 256);
    }
}
