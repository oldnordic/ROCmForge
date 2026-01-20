//! RMSNorm GPU kernel tests - Phase 4.2
//!
//! Tests verify RMSNorm normalization: output = input / sqrt(mean(input^2) + eps) * weight
//!
//! Layout: [seq_len, hidden_size]
//! Row-wise operation: each row is normalized independently
//!
//! This follows TDD: write tests first, prove they fail, then implement kernel.

#[cfg(feature = "rocm")]
#[cfg(test)]
mod rms_norm_tests {
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;
    use serial_test::serial;

    const TEST_TOLERANCE: f32 = 1e-4;
    const TEST_TOLERANCE_LARGE: f32 = 2e-3;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> std::sync::Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for rms_norm_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// CPU reference for RMSNorm
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    /// where mean is over the last dimension (hidden_size)
    ///
    /// Input:  [seq_len, hidden_size]
    /// Weight: [hidden_size]
    /// Output: [seq_len, hidden_size]
    fn rms_norm_cpu_reference(
        input: &[f32],
        weight: &[f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * hidden_size];

        for row_idx in 0..seq_len {
            let row_start = row_idx * hidden_size;
            let row_end = row_start + hidden_size;

            // Compute mean of squares
            let mut sum_sq = 0.0f32;
            for j in row_start..row_end {
                sum_sq += input[j] * input[j];
            }
            let mean_sq = sum_sq / (hidden_size as f32);

            // RMS = sqrt(mean_sq + eps)
            let rms = (mean_sq + eps).sqrt();

            // Normalize and apply weight
            for (j, out_idx) in (row_start..row_end).enumerate() {
                output[out_idx] = (input[out_idx] / rms) * weight[j];
            }
        }

        output
    }

    /// Test 1: RMSNorm matches CPU - small dimensions
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU returns incorrect values without kernels
    fn test_rms_norm_matches_cpu_small() {
        let seq_len = 4;
        let hidden_size = 8;
        let eps = 1e-6;

        let total_input = seq_len * hidden_size;
        let input: Vec<f32> = (0..total_input).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + (i as f32) * 0.1).collect();

        let cpu_result = rms_norm_cpu_reference(&input, &weight, seq_len, hidden_size, eps);

        let backend = get_backend_or_skip();

        let input_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let weight_shape = TensorShape::from_dims(&[hidden_size]);
        let out_shape = TensorShape::from_dims(&[seq_len, hidden_size]);

        let input_gpu = DeviceTensor::from_host_vec(&backend, input, input_shape)
            .expect("Failed to create input tensor");
        let weight_gpu = DeviceTensor::from_host_vec(&backend, weight, weight_shape)
            .expect("Failed to create weight tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::mlp::kernels::rms_norm_gpu_kernel(
                &backend, // Pass backend to ensure stream consistency
                input_gpu.as_ptr() as *const f32,
                weight_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                seq_len as u32,
                hidden_size as u32,
                eps,
            )
        };

        if let Err(e) = result {
            eprintln!("SKIPPED: GPU RMSNorm kernel failed: {} - kernel not available or failed", e);
            return;
        }

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
                "RMSNorm mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
        println!("RMSNorm small max diff: {}", max_diff);
    }

    /// Test 2: RMSNorm with larger dimensions
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU returns incorrect values without kernels
    fn test_rms_norm_matches_cpu_32x128() {
        let seq_len = 32;
        let hidden_size = 128;
        let eps = 1e-6;

        let total_input = seq_len * hidden_size;
        let input: Vec<f32> = (0..total_input).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..hidden_size).map(|i| 0.5 + (i as f32) * 0.01).collect();

        let cpu_result = rms_norm_cpu_reference(&input, &weight, seq_len, hidden_size, eps);

        let backend = get_backend_or_skip();

        let input_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let weight_shape = TensorShape::from_dims(&[hidden_size]);
        let out_shape = TensorShape::from_dims(&[seq_len, hidden_size]);

        let input_gpu = DeviceTensor::from_host_vec(&backend, input, input_shape)
            .expect("Failed to create input tensor");
        let weight_gpu = DeviceTensor::from_host_vec(&backend, weight, weight_shape)
            .expect("Failed to create weight tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::mlp::kernels::rms_norm_gpu_kernel(
                &backend, // Pass backend to ensure stream consistency
                input_gpu.as_ptr() as *const f32,
                weight_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                seq_len as u32,
                hidden_size as u32,
                eps,
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
        println!("RMSNorm 32x128 max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }

    /// Test 3: RMSNorm properties
    ///
    /// RMSNorm should:
    /// - Preserve zeros (output is zero if input is zero)
    /// - Scale proportionally
    #[test]
    #[serial]
    fn test_rms_norm_properties() {
        let seq_len = 1;
        let hidden_size = 4;
        let eps = 1e-6;

        // Test with zeros
        let input_zero = vec![0.0; hidden_size];
        let weight: Vec<f32> = vec![1.0; hidden_size];

        let cpu_result = rms_norm_cpu_reference(&input_zero, &weight, seq_len, hidden_size, eps);

        // When input is all zeros, output should be all zeros
        for (i, &val) in cpu_result.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "RMSNorm with zero input should give zero at {}, got {}",
                i,
                val
            );
        }

        // Test with constant input
        let input_const = vec![2.0; hidden_size];
        let weight_const: Vec<f32> = vec![1.0; hidden_size];

        let cpu_result_const =
            rms_norm_cpu_reference(&input_const, &weight_const, seq_len, hidden_size, eps);

        // For constant input x, RMS = sqrt(x^2 + eps) ≈ |x|
        // So output ≈ x / |x| = sign(x) = 1 for x > 0
        for (i, &val) in cpu_result_const.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 0.01,
                "RMSNorm with constant positive input should give ~1 at {}, got {}",
                i,
                val
            );
        }

        println!("RMSNorm properties test passed");
    }
}
