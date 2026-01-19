//! GPU RoPE kernel tests - CPU vs GPU comparison
//!
//! Tests verify that GPU RoPE implementation matches CPU reference.

#[cfg(feature = "rocm")]
#[cfg(test)]
mod phase2_rope_tests {
    use crate::attention::rope::{Rope, RopeConfig};
    use crate::backend::DeviceTensor;

    const TEST_TOLERANCE: f32 = 1e-5;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> std::sync::Arc<crate::backend::HipBackend> {
        match crate::backend::HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for rope_gpu_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Test RoPE GPU matches CPU - small dimensions
    #[test]
    fn test_rope_gpu_matches_cpu_small() {
        // Small config for testing
        let config = RopeConfig::new(4, 8); // head_dim=4, max_seq_len=8
        let rope = Rope::new(config);

        // Test tensor: [batch=1, seq_len=2, num_heads=1, head_dim=4]
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // position 0
            5.0, 6.0, 7.0, 8.0, // position 1
        ];
        let position_ids = vec![0, 1];
        let num_heads = 1;

        // CPU reference
        let mut cpu_result = input.clone();
        rope.apply_q(&mut cpu_result, &position_ids, num_heads)
            .expect("CPU RoPE failed");

        // GPU run
        let backend = get_backend_or_skip();

        let shape = crate::loader::TensorShape::from_dims(&[2, 1, 4]); // [seq_len, num_heads, head_dim]
        let mut gpu_tensor = DeviceTensor::from_host_vec(&backend, input.clone(), shape)
            .expect("Failed to create GPU tensor");

        rope.apply_q_device(&mut gpu_tensor, &position_ids, num_heads)
            .expect("GPU RoPE failed");

        let gpu_result = gpu_tensor.to_host_vec().expect("Failed to copy from GPU");

        // Compare
        assert_eq!(cpu_result.len(), gpu_result.len());
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "RoPE mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
    }

    /// Test RoPE GPU matches CPU - multiple heads
    #[test]
    fn test_rope_gpu_matches_cpu_multi_head() {
        // head_dim=8, 2 heads
        let config = RopeConfig::new(8, 16);
        let rope = Rope::new(config);

        // Test tensor: [batch=1, seq_len=2, num_heads=2, head_dim=8]
        let input: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let position_ids = vec![0, 1];
        let num_heads = 2;

        // CPU reference
        let mut cpu_result = input.clone();
        rope.apply_q(&mut cpu_result, &position_ids, num_heads)
            .expect("CPU RoPE failed");

        // GPU run
        let backend = get_backend_or_skip();

        let shape = crate::loader::TensorShape::from_dims(&[2, 2, 8]); // [seq_len, num_heads, head_dim]
        let mut gpu_tensor = DeviceTensor::from_host_vec(&backend, input.clone(), shape)
            .expect("Failed to create GPU tensor");

        rope.apply_q_device(&mut gpu_tensor, &position_ids, num_heads)
            .expect("GPU RoPE failed");

        let gpu_result = gpu_tensor.to_host_vec().expect("Failed to copy from GPU");

        // Compare
        assert_eq!(cpu_result.len(), gpu_result.len());
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "RoPE mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
    }

    /// Test RoPE with different positions
    #[test]
    fn test_rope_gpu_different_positions() {
        let config = RopeConfig::new(4, 16);
        let rope = Rope::new(config);

        // Test at position 5 (not position 0)
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let position_ids = vec![5];
        let num_heads = 1;

        // CPU reference
        let mut cpu_result = input.clone();
        rope.apply_k(&mut cpu_result, &position_ids, num_heads)
            .expect("CPU RoPE failed");

        // GPU run
        let backend = get_backend_or_skip();

        let shape = crate::loader::TensorShape::from_dims(&[1, 1, 4]);
        let mut gpu_tensor = DeviceTensor::from_host_vec(&backend, input.clone(), shape)
            .expect("Failed to create GPU tensor");

        rope.apply_k_device(&mut gpu_tensor, &position_ids, num_heads)
            .expect("GPU RoPE failed");

        let gpu_result = gpu_tensor.to_host_vec().expect("Failed to copy from GPU");

        // Compare
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "RoPE position 5 mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
    }

    /// Test that values actually change (not identity)
    #[test]
    fn test_rope_gpu_actually_rotates() {
        let config = RopeConfig::new(8, 16);
        let rope = Rope::new(config);

        let input: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let position_ids = vec![1]; // Non-zero position should cause rotation
        let num_heads = 1;

        // GPU run
        let backend = get_backend_or_skip();

        let shape = crate::loader::TensorShape::from_dims(&[1, 1, 8]);
        let mut gpu_tensor = DeviceTensor::from_host_vec(&backend, input.clone(), shape)
            .expect("Failed to create GPU tensor");

        rope.apply_q_device(&mut gpu_tensor, &position_ids, num_heads)
            .expect("GPU RoPE failed");

        let gpu_result = gpu_tensor.to_host_vec().expect("Failed to copy from GPU");

        // At least some values should change from input
        let mut changed = false;
        for (i, (&inp, &out)) in input.iter().zip(gpu_result.iter()).enumerate() {
            if (inp - out).abs() > 1e-6 {
                changed = true;
                break;
            }
        }
        assert!(changed, "RoPE should modify values at non-zero position");
    }

    /// Test larger sequence
    #[test]
    fn test_rope_gpu_seq_len_8() {
        let config = RopeConfig::new(4, 16);
        let rope = Rope::new(config);

        // 8 tokens, head_dim=4, 1 head
        let input: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let position_ids: Vec<usize> = (0..8).collect();
        let num_heads = 1;

        // CPU reference
        let mut cpu_result = input.clone();
        rope.apply_q(&mut cpu_result, &position_ids, num_heads)
            .expect("CPU RoPE failed");

        // GPU run
        let backend = get_backend_or_skip();

        let shape = crate::loader::TensorShape::from_dims(&[8, 1, 4]);
        let mut gpu_tensor = DeviceTensor::from_host_vec(&backend, input.clone(), shape)
            .expect("Failed to create GPU tensor");

        rope.apply_q_device(&mut gpu_tensor, &position_ids, num_heads)
            .expect("GPU RoPE failed");

        let gpu_result = gpu_tensor.to_host_vec().expect("Failed to copy from GPU");

        // Compare all elements
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "RoPE mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
    }

    /// Test RoPE with long context positions (ROPE-03)
    ///
    /// Verifies position IDs beyond 2048 work correctly on GPU.
    /// Tests positions 2048, 3000, and 4095 with max_seq_len=4096.
    #[test]
    fn test_rope_gpu_long_context_positions() {
        // Long context config: max_seq_len=4096
        let config = RopeConfig::new(8, 4096);
        let rope = Rope::new(config);

        // Test positions beyond 2048
        let test_positions = vec![2048, 3000, 4095];
        let num_heads = 1;

        // Input for each position: 8 elements
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        for &pos in &test_positions {
            let position_ids = vec![pos];

            // CPU reference
            let mut cpu_result = input.clone();
            rope.apply_q(&mut cpu_result, &position_ids, num_heads)
                .expect("CPU RoPE failed");

            // GPU run
            let backend = get_backend_or_skip();

            let shape = crate::loader::TensorShape::from_dims(&[1, 1, 8]);
            let mut gpu_tensor = DeviceTensor::from_host_vec(&backend, input.clone(), shape)
                .expect("Failed to create GPU tensor");

            rope.apply_q_device(&mut gpu_tensor, &position_ids, num_heads)
                .expect("GPU RoPE failed");

            let gpu_result = gpu_tensor.to_host_vec().expect("Failed to copy from GPU");

            // Compare element-wise
            assert_eq!(cpu_result.len(), gpu_result.len());
            for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
                let diff = (cpu_val - gpu_val).abs();
                assert!(
                    diff < TEST_TOLERANCE,
                    "RoPE long context (pos={}) mismatch at {}: CPU={}, GPU={}, diff={}",
                    pos,
                    i,
                    cpu_val,
                    gpu_val,
                    diff
                );
            }
        }
    }

    /// Test multi-head independent rotation (ROPE-02)
    ///
    /// Verifies each head receives correct independent rotation with no cross-head contamination.
    /// Uses num_heads=8, head_dim=8, seq_len=4.
    #[test]
    fn test_rope_gpu_multi_head_independent_rotation() {
        let num_heads = 8;
        let head_dim = 8;
        let seq_len = 4;

        let config = RopeConfig::new(head_dim, 16);
        let rope = Rope::new(config);

        // Create input where each head has distinct values for clarity
        // Layout: [seq_len, num_heads, head_dim]
        let mut input = vec![0.0f32; seq_len * num_heads * head_dim];
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = s * num_heads * head_dim + h * head_dim + d;
                    // Use unique value per-head to verify independence
                    input[idx] = (h * 100 + d) as f32;
                }
            }
        }

        let position_ids: Vec<usize> = (0..seq_len).collect();

        // CPU reference
        let mut cpu_result = input.clone();
        rope.apply_q(&mut cpu_result, &position_ids, num_heads)
            .expect("CPU RoPE failed");

        // GPU run
        let backend = get_backend_or_skip();

        let shape = crate::loader::TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut gpu_tensor = DeviceTensor::from_host_vec(&backend, input.clone(), shape)
            .expect("Failed to create GPU tensor");

        rope.apply_q_device(&mut gpu_tensor, &position_ids, num_heads)
            .expect("GPU RoPE failed");

        let gpu_result = gpu_tensor.to_host_vec().expect("Failed to copy from GPU");

        // Compare all elements
        assert_eq!(cpu_result.len(), gpu_result.len());
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "RoPE multi-head mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }

        // Additional verification: head 0 output != head 1 output at same position
        // This proves heads are independently rotated
        let head_0_start = 0 * head_dim; // Position 0, head 0
        let head_1_start = 1 * head_dim; // Position 0, head 1

        let head_0_different = &gpu_result[head_0_start..head_0_start + head_dim];
        let head_1_different = &gpu_result[head_1_start..head_1_start + head_dim];

        // At least some elements should differ between heads
        let mut heads_differ = false;
        for (h0_val, h1_val) in head_0_different.iter().zip(head_1_different.iter()) {
            if (h0_val - h1_val).abs() > 1e-6 {
                heads_differ = true;
                break;
            }
        }
        assert!(heads_differ, "Head 0 and Head 1 should produce different rotations");

        // Verify rotation varies by position within each head
        let pos_0_head_0 = &gpu_result[0 * num_heads * head_dim + 0 * head_dim..0 * num_heads * head_dim + 0 * head_dim + head_dim];
        let pos_1_head_0 = &gpu_result[1 * num_heads * head_dim + 0 * head_dim..1 * num_heads * head_dim + 0 * head_dim + head_dim];

        let mut position_affects_rotation = false;
        for (p0_val, p1_val) in pos_0_head_0.iter().zip(pos_1_head_0.iter()) {
            if (p0_val - p1_val).abs() > 1e-6 {
                position_affects_rotation = true;
                break;
            }
        }
        assert!(position_affects_rotation, "Rotation should vary by position within head");
    }
}
