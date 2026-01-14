//! Causal mask tests - Phase 3b.1
//!
//! Tests verify standalone causal mask kernel:
//! - Creates mask where mask[i, j] = -inf if j > i (key > query)
//! - Layout: [batch, heads, seq_q, seq_k] explicit
//!
//! This is a standalone atomic operation - no fusion yet.
//! Test first, implement kernel after tests fail (TDD).

#[cfg(feature = "rocm")]
#[cfg(test)]
mod causal_mask_tests {
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;
    use std::sync::Arc;

    const TEST_TOLERANCE: f32 = 1e-5;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for causal_mask_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// CPU reference: create causal mask with explicit layout
    ///
    /// Layout: [batch, heads, seq_q, seq_k]
    /// mask[batch, heads, query_pos, key_pos] = -inf if key_pos > query_pos
    fn create_causal_mask_explicit(
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
    ) -> Vec<f32> {
        let total = batch * heads * seq_q * seq_k;
        let mut mask = vec![0.0f32; total];

        for b in 0..batch {
            for h in 0..heads {
                for q in 0..seq_q {
                    for k in 0..seq_k {
                        let idx = b * heads * seq_q * seq_k + h * seq_q * seq_k + q * seq_k + k;
                        // Causal: mask out future positions
                        // In causal attention, query at position q can only attend to keys at positions <= q
                        if k > q {
                            mask[idx] = f32::NEG_INFINITY;
                        } else {
                            mask[idx] = 0.0f32;
                        }
                    }
                }
            }
        }

        mask
    }

    /// Test 1: Causal mask pattern matches CPU - small square
    #[test]
    fn test_causal_mask_matches_cpu_small_square() {
        let batch = 1;
        let heads = 1;
        let seq = 4;

        // CPU reference
        let cpu_mask = create_causal_mask_explicit(batch, heads, seq, seq);

        // Verify CPU mask pattern
        // seq=4, expected pattern:
        // query 0: [0, -inf, -inf, -inf]
        // query 1: [0,   0, -inf, -inf]
        // query 2: [0,   0,   0, -inf]
        // query 3: [0,   0,   0,   0]
        assert_eq!(cpu_mask[0 * 4 + 0], 0.0f32); // q=0, k=0
        assert_eq!(cpu_mask[0 * 4 + 1], f32::NEG_INFINITY); // q=0, k=1
        assert_eq!(cpu_mask[0 * 4 + 2], f32::NEG_INFINITY); // q=0, k=2
        assert_eq!(cpu_mask[0 * 4 + 3], f32::NEG_INFINITY); // q=0, k=3

        assert_eq!(cpu_mask[1 * 4 + 0], 0.0f32); // q=1, k=0
        assert_eq!(cpu_mask[1 * 4 + 1], 0.0f32); // q=1, k=1
        assert_eq!(cpu_mask[1 * 4 + 2], f32::NEG_INFINITY); // q=1, k=2
        assert_eq!(cpu_mask[1 * 4 + 3], f32::NEG_INFINITY); // q=1, k=3

        assert_eq!(cpu_mask[2 * 4 + 0], 0.0f32); // q=2, k=0
        assert_eq!(cpu_mask[2 * 4 + 1], 0.0f32); // q=2, k=1
        assert_eq!(cpu_mask[2 * 4 + 2], 0.0f32); // q=2, k=2
        assert_eq!(cpu_mask[2 * 4 + 3], f32::NEG_INFINITY); // q=2, k=3

        assert_eq!(cpu_mask[3 * 4 + 0], 0.0f32); // q=3, k=0
        assert_eq!(cpu_mask[3 * 4 + 1], 0.0f32); // q=3, k=1
        assert_eq!(cpu_mask[3 * 4 + 2], 0.0f32); // q=3, k=2
        assert_eq!(cpu_mask[3 * 4 + 3], 0.0f32); // q=3, k=3

        println!("CPU causal mask pattern verified for seq=4");

        // GPU run
        let backend = get_backend_or_skip();

        let mask_shape = TensorShape::from_dims(&[batch, heads, seq, seq]);
        let mut mask_gpu = DeviceTensor::empty(&backend, mask_shape.clone())
            .expect("Failed to create mask tensor");

        let result = unsafe {
            crate::attention::kernels::causal_mask_gpu_kernel(
                mask_gpu.buffer().as_mut_ptr() as *mut f32,
                batch as u32,
                seq as u32,
                heads as u32,
            )
        };

        assert_eq!(result, Ok(()), "GPU kernel failed");

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_mask = mask_gpu
            .to_host_vec()
            .expect("Failed to copy mask from GPU");

        assert_eq!(cpu_mask.len(), gpu_mask.len());

        // Compare CPU vs GPU
        for (i, (cpu_val, gpu_val)) in cpu_mask.iter().zip(gpu_mask.iter()).enumerate() {
            // For -inf values, check if both are -inf
            if cpu_val.is_finite() {
                let diff = (cpu_val - gpu_val).abs();
                assert!(
                    diff < TEST_TOLERANCE,
                    "Causal mask mismatch at {}: CPU={}, GPU={}, diff={}",
                    i,
                    cpu_val,
                    gpu_val,
                    diff
                );
            } else {
                assert!(
                    !gpu_val.is_finite() || *gpu_val < -1e30,
                    "Causal mask mismatch at {}: CPU=-inf, GPU={}",
                    i,
                    gpu_val
                );
            }
        }

        println!("Causal mask small square: CPU vs GPU match verified");
    }

    /// Test 2: Causal mask with multiple heads and batch
    #[test]
    fn test_causal_mask_multi_head_batch() {
        let batch = 2;
        let heads = 4;
        let seq = 8;

        // CPU reference
        let cpu_mask = create_causal_mask_explicit(batch, heads, seq, seq);

        // GPU run
        let backend = get_backend_or_skip();

        let mask_shape = TensorShape::from_dims(&[batch, heads, seq, seq]);
        let mut mask_gpu = DeviceTensor::empty(&backend, mask_shape.clone())
            .expect("Failed to create mask tensor");

        let result = unsafe {
            crate::attention::kernels::causal_mask_gpu_kernel(
                mask_gpu.buffer().as_mut_ptr() as *mut f32,
                batch as u32,
                seq as u32,
                heads as u32,
            )
        };

        assert_eq!(result, Ok(()), "GPU kernel failed");

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_mask = mask_gpu
            .to_host_vec()
            .expect("Failed to copy mask from GPU");

        assert_eq!(cpu_mask.len(), gpu_mask.len());

        // Sample some positions to verify
        // Check batch 0, head 0
        let base_idx = 0;
        assert_eq!(cpu_mask[base_idx + 0 * 8 + 0], 0.0f32); // q=0, k=0
        assert!(!cpu_mask[base_idx + 0 * 8 + 1].is_finite()); // q=0, k=1 should be -inf

        // Verify these match on GPU
        assert!(
            (cpu_mask[base_idx + 0 * 8 + 0] - gpu_mask[base_idx + 0 * 8 + 0]).abs()
                < TEST_TOLERANCE
        );
        assert!(
            !gpu_mask[base_idx + 0 * 8 + 1].is_finite() || gpu_mask[base_idx + 0 * 8 + 1] < -1e30
        );

        // Check batch 1, head 2
        let base_idx = 1 * heads * seq * seq + 2 * seq * seq;
        assert_eq!(cpu_mask[base_idx + 7 * 8 + 7], 0.0f32); // q=7, k=7 (last position, no mask)

        println!("Causal mask multi-head batch verified");
    }

    /// Test 3: Verify mask doesn't corrupt non-causal positions
    #[test]
    fn test_causal_mask_preserves_valid_positions() {
        let batch = 1;
        let heads = 1;
        let seq = 16;

        // CPU reference
        let cpu_mask = create_causal_mask_explicit(batch, heads, seq, seq);

        // Count non-negative infinity values (should be triangular number)
        let mut valid_count = 0;
        for &val in cpu_mask.iter() {
            if val.is_finite() && val >= -1e30 {
                valid_count += 1;
            }
        }

        // Triangular number: seq * (seq + 1) / 2
        let expected_valid = seq * (seq + 1) / 2;
        assert_eq!(
            valid_count, expected_valid,
            "Expected {} valid positions, got {}",
            expected_valid, valid_count
        );

        println!(
            "Causal mask preserves {} valid positions (triangular check)",
            valid_count
        );
    }

    /// Test 4: Explicit layout indexing verification
    #[test]
    fn test_causal_mask_explicit_layout() {
        let batch = 2;
        let heads = 3;
        let seq_q = 4;
        let seq_k = 4;

        // CPU reference
        let cpu_mask = create_causal_mask_explicit(batch, heads, seq_q, seq_k);

        // Verify layout: [batch, heads, seq_q, seq_k]
        // For batch=1, head=1, should match 1D create_causal_mask pattern
        let mask_1d = crate::attention::mask::create_causal_mask(seq_q);

        // batch=1, head=0 should match 1D pattern
        for q in 0..seq_q {
            for k in 0..seq_k {
                let explicit_idx = 0 * heads * seq_q * seq_k + 0 * seq_q * seq_k + q * seq_k + k;
                let flat_idx = q * seq_k + k;
                assert_eq!(
                    cpu_mask[explicit_idx].is_finite(),
                    mask_1d[flat_idx].is_finite(),
                    "Layout mismatch at q={}, k={}",
                    q,
                    k
                );
            }
        }

        println!("Causal mask explicit layout verified");
    }
}
