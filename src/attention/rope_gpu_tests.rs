//! GPU RoPE kernel tests - CPU vs GPU comparison
//!
//! Tests verify that GPU RoPE implementation matches CPU reference.

#[cfg(feature = "rocm")]
#[cfg(test)]
mod phase2_rope_tests {
    use crate::attention::rope::{Rope, RopeConfig};
    use crate::backend::DeviceTensor;

    const TEST_TOLERANCE: f32 = 1e-5;

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
        let backend = crate::backend::HipBackend::new().expect("Failed to create HIP backend");

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
        let backend = crate::backend::HipBackend::new().expect("Failed to create HIP backend");

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
        let backend = crate::backend::HipBackend::new().expect("Failed to create HIP backend");

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
        let backend = crate::backend::HipBackend::new().expect("Failed to create HIP backend");

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
        let backend = crate::backend::HipBackend::new().expect("Failed to create HIP backend");

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
}
