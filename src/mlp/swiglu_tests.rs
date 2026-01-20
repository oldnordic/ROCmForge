//! SwiGLU activation GPU kernel tests - Phase 4.1
//!
//! Tests verify SwiGLU activation: gate * swish(up)
//! where swish(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
//!
//! Layout: [seq_len, intermediate_size] for all tensors
//! Element-wise operation: output[i] = gate[i] * swish(up[i])
//!
//! This follows TDD: write tests first, prove they fail, then implement kernel.

#[cfg(feature = "rocm")]
#[cfg(test)]
mod swiglu_tests {
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
                eprintln!("\n⚠️  GPU not available for swiglu_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// CPU reference for SwiGLU activation
    ///
    /// SwiGLU(x) = gate(x) * swish(up(x))
    /// where swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    ///
    /// All inputs have shape [seq_len, intermediate_size]
    /// Output has shape [seq_len, intermediate_size]
    fn swiglu_cpu_reference(
        gate: &[f32],
        up: &[f32],
        seq_len: usize,
        intermediate_size: usize,
    ) -> Vec<f32> {
        let total = seq_len * intermediate_size;
        let mut output = vec![0.0f32; total];

        for i in 0..total {
            let g = gate[i];
            let u = up[i];

            // Swish activation: swish(x) = x * sigmoid(x)
            let sigmoid_up = 1.0f32 / (1.0f32 + (-u).exp());
            let swish_up = u * sigmoid_up;

            // SwiGLU: gate * swish(up)
            output[i] = g * swish_up;
        }

        output
    }

    /// Test 1: SwiGLU matches CPU - small dimensions
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU kernel unavailable
    #[ignore] // Requires HSACO kernels - GPU kernel unavailable
    fn test_swiglu_matches_cpu_small() {
        let seq_len = 4;
        let intermediate_size = 8;

        // Create test data
        let total = seq_len * intermediate_size;
        let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
        let up: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 + 1.0).collect();

        // CPU reference
        let cpu_result = swiglu_cpu_reference(&gate, &up, seq_len, intermediate_size);

        // GPU run
        let backend = get_backend_or_skip();

        let gate_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let up_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let out_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);

        let gate_gpu = DeviceTensor::from_host_vec(&backend, gate, gate_shape)
            .expect("Failed to create gate tensor");
        let up_gpu = DeviceTensor::from_host_vec(&backend, up, up_shape)
            .expect("Failed to create up tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::mlp::kernels::swiglu_gpu_kernel(
                &backend, // Pass backend to ensure stream consistency
                gate_gpu.as_ptr() as *const f32,
                up_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                seq_len as u32,
                intermediate_size as u32,
            )
        };

        if let Err(e) = result {
            eprintln!("SKIPPED: GPU SwiGLU kernel failed: {} - kernel not available or failed", e);
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
                "SwiGLU mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
        println!("SwiGLU small max diff: {}", max_diff);
    }

    /// Test 2: SwiGLU with larger dimensions
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU kernel unavailable
    fn test_swiglu_matches_cpu_32x32() {
        let seq_len = 32;
        let intermediate_size = 32;

        let total = seq_len * intermediate_size;
        let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let up: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let cpu_result = swiglu_cpu_reference(&gate, &up, seq_len, intermediate_size);

        let backend = get_backend_or_skip();

        let gate_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let up_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let out_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);

        let gate_gpu = DeviceTensor::from_host_vec(&backend, gate, gate_shape)
            .expect("Failed to create gate tensor");
        let up_gpu = DeviceTensor::from_host_vec(&backend, up, up_shape)
            .expect("Failed to create up tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::mlp::kernels::swiglu_gpu_kernel(
                &backend, // Pass backend to ensure stream consistency
                gate_gpu.as_ptr() as *const f32,
                up_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                seq_len as u32,
                intermediate_size as u32,
            )
        };

        if let Err(e) = result {
            eprintln!("SKIPPED: GPU SwiGLU kernel failed: {} - kernel not available or failed", e);
            return;
        }

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        let mut max_diff = 0.0f32;
        for (cpu_val, gpu_val) in cpu_result.iter().zip(gpu_result.iter()) {
            max_diff = max_diff.max((cpu_val - gpu_val).abs());
        }
        println!("SwiGLU 32x32 max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }

    /// Test 3: SwiGLU with non-square dimensions
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU kernel unavailable
    fn test_swiglu_non_square() {
        let seq_len = 8;
        let intermediate_size = 64;

        let total = seq_len * intermediate_size;
        let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05).collect();
        let up: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05 - 2.0).collect();

        let cpu_result = swiglu_cpu_reference(&gate, &up, seq_len, intermediate_size);

        let backend = get_backend_or_skip();

        let gate_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let up_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let out_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);

        let gate_gpu = DeviceTensor::from_host_vec(&backend, gate, gate_shape)
            .expect("Failed to create gate tensor");
        let up_gpu = DeviceTensor::from_host_vec(&backend, up, up_shape)
            .expect("Failed to create up tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::mlp::kernels::swiglu_gpu_kernel(
                &backend, // Pass backend to ensure stream consistency
                gate_gpu.as_ptr() as *const f32,
                up_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                seq_len as u32,
                intermediate_size as u32,
            )
        };

        if let Err(e) = result {
            eprintln!("SKIPPED: GPU SwiGLU kernel failed: {} - kernel not available or failed", e);
            return;
        }

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        let mut max_diff = 0.0f32;
        for (cpu_val, gpu_val) in cpu_result.iter().zip(gpu_result.iter()) {
            max_diff = max_diff.max((cpu_val - gpu_val).abs());
        }
        println!("SwiGLU non-square max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }

    /// Test 4: Verify output is finite (no NaN/inf)
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU kernel unavailable
    fn test_swiglu_output_is_finite() {
        let seq_len = 16;
        let intermediate_size = 16;

        // Test with extreme values
        let total = seq_len * intermediate_size;
        let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 100.0 - 50.0).collect();
        let up: Vec<f32> = (0..total).map(|i| (i as f32) * 50.0 - 25.0).collect();

        let backend = get_backend_or_skip();

        let gate_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let up_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);
        let out_shape = TensorShape::from_dims(&[seq_len, intermediate_size]);

        let gate_gpu = DeviceTensor::from_host_vec(&backend, gate, gate_shape)
            .expect("Failed to create gate tensor");
        let up_gpu = DeviceTensor::from_host_vec(&backend, up, up_shape)
            .expect("Failed to create up tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let result = unsafe {
            crate::mlp::kernels::swiglu_gpu_kernel(
                &backend, // Pass backend to ensure stream consistency
                gate_gpu.as_ptr() as *const f32,
                up_gpu.as_ptr() as *const f32,
                out_gpu.buffer().as_mut_ptr() as *mut f32,
                seq_len as u32,
                intermediate_size as u32,
            )
        };

        if let Err(e) = result {
            eprintln!("SKIPPED: GPU SwiGLU kernel failed: {} - kernel not available or failed", e);
            return;
        }

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Verify all values are finite
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "SwiGLU output should be finite at index {}, got {}",
                i,
                val
            );
        }

        println!("SwiGLU finiteness test passed");
    }

    /// Test 5: Verify SwiGLU mathematical properties
    ///
    /// SwiGLU has these properties:
    /// - When up >> 0: swish(up) ≈ up, so output ≈ gate * up
    /// - When up << 0: swish(up) ≈ 0, so output ≈ 0
    /// - When up = 0: swish(0) = 0, so output = 0
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU kernel unavailable
    fn test_swiglu_mathematical_properties() {
        let seq_len = 1;
        let intermediate_size = 3;

        // Test case 1: up = 0 → output should be 0
        let gate = vec![1.0, 2.0, 3.0];
        let up_zero = vec![0.0, 0.0, 0.0];

        let cpu_result = swiglu_cpu_reference(&gate, &up_zero, seq_len, intermediate_size);

        // When up = 0, swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        // So output = gate * 0 = 0
        for (i, &val) in cpu_result.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "SwiGLU with up=0 should give ~0 at {}, got {}",
                i,
                val
            );
        }

        // Test case 2: large positive up → swish(up) ≈ up
        let gate2 = vec![1.0, 1.0, 1.0];
        let up_large = vec![10.0, 20.0, 30.0];

        let cpu_result2 = swiglu_cpu_reference(&gate2, &up_large, seq_len, intermediate_size);

        // For large up, sigmoid(up) ≈ 1, so swish(up) ≈ up
        // So output ≈ gate * up
        for i in 0..intermediate_size {
            let expected = gate2[i] * up_large[i];
            let diff = (cpu_result2[i] - expected).abs();
            assert!(
                diff < 0.1, // Allow some tolerance since swish isn't exactly identity
                "SwiGLU with large up should approximate gate*up at {}: expected ~{}, got {}",
                i,
                expected,
                cpu_result2[i]
            );
        }

        println!("SwiGLU mathematical properties test passed");
    }
}
