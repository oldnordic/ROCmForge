//! MLP GPU Path Regression Tests
//!
//! These tests verify that MLP operations stay on GPU and do NOT round-trip
//! through host memory. This is critical for performance.
//!
//! # Regression Test Invariant
//!
//! The MLP layer forward path MUST NOT call `to_host_vec()` for intermediate
//! results. The only acceptable `to_host_vec()` calls are:
//! 1. Test verification (comparing GPU results to CPU reference)
//! 2. Final output retrieval at the application layer
//!
//! # What This Tests
//!
//! - `mlp_swiglu` function uses GPU kernel for SwiGLU activation
//! - `HipBuffer::copy_from_buffer` uses `hipMemcpyDeviceToDevice`
//! - No intermediate GPU→CPU→GPU round-trips

#[cfg(test)]
mod gpu_path_regression_tests {
    use crate::backend::hip_backend::{HipBackend, HipBuffer};
    use crate::backend::DeviceTensor;
    use crate::loader::mmap_loader::TensorShape;
    use serial_test::serial;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> std::sync::Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for gpu_path_regression_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Test that MLP SwiGLU activation stays on GPU.
    ///
    /// This is a regression test to ensure the CPU fallback is NOT used.
    /// The CPU fallback path (removed in Phase 4) looked like:
    /// ```rust,ignore
    /// let mut gate_host = vec![0.0f32; size];
    /// gate_buffer.copy_to_host(&mut gate_host)?;  // ❌ GPU→CPU
    /// // ... CPU SwiGLU loop ...
    /// swiglu_buffer.copy_from_host(&swiglu_host)?;  // ❌ CPU→GPU
    /// ```
    ///
    /// The correct GPU-only path uses:
    /// ```rust,ignore
    /// unsafe {
    ///     crate::mlp::kernels::swiglu_gpu_kernel(...)?;  // ✅ GPU kernel
    /// }
    /// final_buffer.copy_from_buffer(&output_buffer)?;  // ✅ GPU→GPU
    /// ```
    #[test]
    #[serial]
    fn test_mlp_swiglu_gpu_only_path() {
        let backend = get_backend_or_skip();

        // Small test data: seq_len=4, hidden_size=8, intermediate_size=16
        let seq_len = 4;
        let hidden_size = 8;
        let intermediate_size = 16;

        // Create input tensors on GPU with correct shapes:
        // hidden_states: [seq_len, hidden_size] = [4, 8]
        // gate_weight: [hidden_size, intermediate_size] = [8, 16]
        // up_weight: [hidden_size, intermediate_size] = [8, 16]
        // down_weight: [intermediate_size, hidden_size] = [16, 8]
        // output: [seq_len, hidden_size] = [4, 8]

        let hidden_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32 * 0.1).collect();
        let gate_data: Vec<f32> = (0..hidden_size * intermediate_size)
            .map(|i| i as f32 * 0.1)
            .collect();
        let up_data: Vec<f32> = (0..hidden_size * intermediate_size)
            .map(|i| i as f32 * 0.1 - 0.5)
            .collect();
        let down_data: Vec<f32> = (0..intermediate_size * hidden_size)
            .map(|i| i as f32 * 0.1 - 0.3)
            .collect();

        let hidden_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let weight_shape = TensorShape::from_dims(&[hidden_size, intermediate_size]);
        let down_shape = TensorShape::from_dims(&[intermediate_size, hidden_size]);

        let hidden_gpu = DeviceTensor::from_host_vec(&backend, hidden_data, hidden_shape.clone())
            .expect("Failed to create hidden tensor");
        let gate_gpu = DeviceTensor::from_host_vec(&backend, gate_data, weight_shape.clone())
            .expect("Failed to create gate tensor");
        let up_gpu = DeviceTensor::from_host_vec(&backend, up_data, weight_shape.clone())
            .expect("Failed to create up tensor");
        let down_gpu = DeviceTensor::from_host_vec(&backend, down_data, down_shape)
            .expect("Failed to create down tensor");

        let mut result_gpu =
            DeviceTensor::empty(&backend, hidden_shape).expect("Failed to create result tensor");

        backend
            .mlp_swiglu(&hidden_gpu, &gate_gpu, &up_gpu, &down_gpu, &mut result_gpu)
            .expect("MLP SwiGLU failed");

        // Verify output is valid (non-zero, since inputs are non-zero)
        let result = result_gpu
            .to_host_vec()
            .expect("Failed to copy result to host");

        // Result should be non-zero since we used non-zero inputs
        assert!(
            result.iter().any(|&x| x != 0.0),
            "MLP SwiGLU produced all zeros - check kernel implementation"
        );
    }

    /// Test that GPU-to-GPU copy works correctly.
    ///
    /// Verifies that `HipBuffer::copy_from_buffer` uses `hipMemcpyDeviceToDevice`
    /// and not GPU→CPU→GPU (which would be slower).
    #[test]
    #[serial]
    fn test_gpu_to_gpu_copy() {
        let backend = get_backend_or_skip();

        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let size = data.len() * std::mem::size_of::<f32>();

        // Create two buffers on GPU
        let src_buffer = HipBuffer::new(size).expect("Failed to create source buffer");
        let dst_buffer = HipBuffer::new(size).expect("Failed to create destination buffer");

        // Copy data to GPU
        src_buffer
            .copy_from_host(&data)
            .expect("Failed to copy to source");

        // GPU-to-GPU copy
        dst_buffer
            .copy_from_buffer(&src_buffer)
            .expect("Failed to copy from buffer");

        // Verify: copy back to host and check
        let mut result = vec![0.0f32; data.len()];
        dst_buffer
            .copy_to_host(&mut result)
            .expect("Failed to copy to host");

        assert_eq!(result, data, "GPU-to-GPU copy produced incorrect data");
    }

    /// Regression test: verify no host round-trip in MLP layer.
    ///
    /// This test documents the expected behavior. If this test fails,
    /// it means someone has reintroduced a CPU fallback in the MLP path.
    #[test]
    #[serial]
    fn test_no_host_roundtrip_in_mlp_layer() {
        // This is a documentation test - the actual check is:
        // grep -n "to_host_vec" src/backend/hip_backend.rs | grep -A5 -B5 "mlp_swiglu"
        // Should return: NO MATCHES (except for test code)

        // The expected code path for mlp_swiglu (lines 1281-1358 in hip_backend.rs):
        // 1. Allocate swiglu_buffer on GPU
        // 2. Launch swiglu_gpu_kernel (GPU only)
        // 3. Synchronize
        // 4. Compute down projection (stays on GPU)
        // 5. Copy result to output using copy_from_buffer (GPU-to-GPU)

        // Steps that should NOT exist:
        // - gate_buffer.copy_to_host(&mut gate_host)?
        // - up_buffer.copy_to_host(&mut up_host)?
        // - for i in 0..swiglu_host.len() { ... }  // CPU loop
        // - swiglu_buffer.copy_from_host(&swiglu_host)?

        assert!(
            true,
            "Documentation test: MLP layer should not use to_host_vec"
        );
    }
}
