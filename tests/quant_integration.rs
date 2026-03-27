//! Integration tier tests for GPU quantization kernels.
//!
//! These tests will verify end-to-end quantization functionality:
//! - FFI calls to HIP kernels
//! - Memory allocation and transfer
//! - Correctness of quantization/dequantization
//! - Performance vs CPU baseline
//!
//! **NOTE**: These are placeholders for Phase 2 implementation.
//! Actual quantization kernels (Q4_K, Q8_0, etc.) will be implemented in future phases.
//!
//! Run with: cargo test --test quant_integration --features gpu

#![cfg(feature = "gpu")]

use serial_test::serial;

/// Test Q4_K quantization kernel
///
/// Verifies:
/// 1. Allocate GPU buffers for input weights and output quantized blocks
/// 2. Call Q4_K quantization FFI function
/// 3. Copy quantized blocks back to CPU
/// 4. Verify dequantized values match original within tolerance
#[test]
#[serial]
fn test_q4_k_quantization() {
    use rocmforge::gpu::{detect, GpuDevice, GpuQuant, GpuBuffer, QK_K, Q4_K_BLOCK_SIZE};

    // Require GPU for this test
    let caps = detect().expect("GPU required for quantization test");
    println!("Testing Q4_K quantization on: {}", caps.device_name);

    // Initialize device
    let device = GpuDevice::init(caps.device_id)
        .expect("Failed to initialize GPU device");

    // Initialize quantization context
    let gpu_quant = GpuQuant::new(device)
        .expect("Failed to initialize GpuQuant");

    // Prepare test data (256 elements = 1 Q4_K block)
    let n = QK_K;
    let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    // Allocate GPU buffers using GpuBuffer (RAII)
    let d_input = GpuBuffer::alloc(n * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n / QK_K) * Q4_K_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output = GpuBuffer::alloc(n * std::mem::size_of::<f32>())
        .expect("Failed to allocate output buffer");

    // Copy input to GPU (cast to &[u8] for copy_from_host)
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>()
        )
    };
    // GpuBuffer::copy_from_host expects &mut self, so we need to make it mutable
    let mut d_input = d_input;
    d_input.copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize on GPU (cast *mut u8 to *const f32 and *mut u8)
    gpu_quant.quantize(
        d_input.as_ptr() as *const f32,
        d_quantized.as_ptr(),
        n
    )
        .expect("Failed to quantize on GPU");

    // Dequantize on GPU
    gpu_quant.dequantize(
        d_quantized.as_ptr(),
        d_output.as_ptr() as *mut f32,
        n
    )
        .expect("Failed to dequantize on GPU");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    d_output.copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            output_bytes.as_ptr() as *const f32,
            n
        ).to_vec()
    };

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify accuracy - Q4_K has ~6-7 bits of precision
    // Allow for larger error due to 4-bit quantization
    let tolerance = 0.5; // Relaxed tolerance for 4-bit
    let mut max_error = 0.0f32;
    for (i, (orig, dequant)) in input_data.iter().zip(output_data.iter()).enumerate() {
        let error = (orig - dequant).abs();
        max_error = max_error.max(error);
        if error > tolerance {
            panic!(
                "Large quantization error at index {}: orig={}, dequant={}, error={}",
                i, orig, dequant, error
            );
        }
    }

    println!("Q4_K quantization max error: {}", max_error);
    assert!(max_error < tolerance, "Max error {} exceeds tolerance {}", max_error, tolerance);

    // Test verify_accuracy function
    println!("Testing verify_accuracy function...");
    let caps2 = detect().expect("GPU required");
    let device2 = GpuDevice::init(caps2.device_id).expect("Failed to init device");
    let gpu_quant2 = GpuQuant::new(device2).expect("Failed to init GpuQuant");

    let d_input2 = GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate");
    let d_quantized2 = GpuBuffer::alloc((n / QK_K) * Q4_K_BLOCK_SIZE).expect("Failed to allocate");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>()
        )
    };
    let mut d_input2 = d_input2;
    d_input2.copy_from_host(input_bytes).expect("Failed to copy input");

    gpu_quant2.quantize(
        d_input2.as_ptr() as *const f32,
        d_quantized2.as_ptr(),
        n
    )
        .expect("Failed to quantize");

    let (max_err, mse, rel_err) = gpu_quant2.verify_accuracy(
        d_input2.as_ptr() as *const f32,
        d_quantized2.as_ptr(),
        n
    ).expect("Failed to verify accuracy");

    println!("Verification: max_error={}, mse={}, relative_error={}", max_err, mse, rel_err);

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_input2);
    drop(d_quantized2);

    // Verification should show reasonable accuracy
    assert!(max_err < 1.0, "Max verification error {} too large", max_err);
}

/// Placeholder: Test Q8_0 quantization kernel
///
/// **To be implemented in Phase 2**:
/// 1. Allocate GPU buffers for input weights and output quantized blocks
/// 2. Call Q8_0 quantization FFI function
/// 3. Copy quantized blocks back to CPU
/// 4. Verify dequantized values match original within tolerance
#[test]
#[serial]
#[ignore = "Q8_0 kernel not implemented - Phase 2"]
fn test_q8_0_quantization() {
    // TODO: Implement in Phase 2
    // Steps:
    // - Prepare test weight data (e.g., random floats)
    // - Call hip_q8_0_quantize() FFI function
    // - Dequantize and verify accuracy
    // - Compare with CPU Q8_0 implementation
    unimplemented!("Q8_0 quantization kernel - Phase 2");
}

/// Placeholder: Test Q4_K matrix-vector multiplication
///
/// **To be implemented in Phase 2**:
/// 1. Quantize weights to Q4_K format on GPU
/// 2. Allocate GPU buffer for input vector
/// 3. Call Q4_K matmul kernel
/// 4. Verify output matches CPU gemv_q4_k_q8_k
#[test]
#[serial]
#[ignore = "Q4_K matmul not implemented - Phase 2"]
fn test_q4_k_matmul() {
    // TODO: Implement in Phase 2
    // Steps:
    // - Quantize a small weight matrix to Q4_K
    // - Prepare input vector
    // - Call hip_gemv_q4_k_q8_k() FFI function
    // - Verify output matches CPU implementation
    // - Test various matrix sizes (powers of 2, non-powers of 2)
    unimplemented!("Q4_K matmul kernel - Phase 2");
}

/// Placeholder: Test quantized GEMM (Q4_K × Q8_K)
///
/// **To be implemented in Phase 3 or 4**:
/// 1. Allocate quantized weight matrices (Q4_K, Q8_K)
/// 2. Call batched GEMM kernel
/// 3. Verify output correctness
/// 4. Benchmark vs CPU implementation
#[test]
#[serial]
#[ignore = "Quantized GEMM not implemented - Phase 3/4"]
fn test_quantized_gemm() {
    // TODO: Implement in Phase 3 or 4
    // Steps:
    // - Prepare quantized matrices
    // - Call hip_gemm_q4_k_q8_k() FFI function
    // - Verify output matches CPU
    // - Measure speedup over CPU
    unimplemented!("Quantized GEMM - Phase 3/4");
}

/// Placeholder: Test VRAM usage during quantization
///
/// **To be implemented in Phase 2**:
/// 1. Record baseline VRAM usage
/// 2. Allocate large buffers on GPU
/// 3. Run quantization
/// 4. Verify VRAM returned after cleanup
/// 5. Check for memory leaks
#[test]
#[serial]
#[ignore = "VRAM tracking not implemented - Phase 2"]
fn test_vram_during_quantization() {
    // TODO: Implement in Phase 2
    // Steps:
    // - Use rocm-smi to check VRAM before
    // - Allocate quantization buffers
    // - Run kernel
    // - Free buffers
    // - Verify VRAM returned to baseline
    unimplemented!("VRAM tracking - Phase 2");
}

/// Placeholder: Test concurrent quantization operations
///
/// **To be implemented in Phase 3 or 4**:
/// 1. Launch multiple quantization kernels concurrently
/// 2. Verify all complete successfully
/// 3. Verify outputs are correct
/// 4. Test with HIP streams
#[test]
#[serial]
#[ignore = "Concurrent operations not implemented - Phase 3/4"]
fn test_concurrent_quantization() {
    // TODO: Implement in Phase 3 or 4
    // Steps:
    // - Create multiple HIP streams
    // - Launch kernels on different streams
    // - Use hipStreamSynchronize to wait
    // - Verify all outputs correct
    unimplemented!("Concurrent operations - Phase 3/4");
}

/// Placeholder: Benchmark quantization vs CPU
///
/// **To be implemented in Phase 2 or 3**:
/// 1. Time GPU quantization
/// 2. Time CPU quantization (same data)
/// 3. Report speedup
/// 4. Verify correctness maintained
#[test]
#[serial]
#[ignore = "Benchmark not implemented - Phase 2/3"]
fn benchmark_quantization() {
    // TODO: Implement in Phase 2 or 3
    // Steps:
    // - Prepare large weight matrix (e.g., 4096x4096)
    // - Time GPU quantization (with warmup)
    // - Time CPU quantization
    // - Calculate speedup
    // - Verify dequantized accuracy
    unimplemented!("Benchmark - Phase 2/3");
}

/// Example test structure for future implementation
///
/// This shows the structure for testing a quantization kernel
/// once implemented. Remove this test when adding real integration tests.
#[test]
#[serial]
fn example_test_structure() {
    // This is an example of how future integration tests should be structured

    // 1. Setup: Allocate test data
    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    // 2. Allocate GPU buffers (when kernels are implemented)
    // let d_input = allocate_gpu_buffer(&test_data)?;
    // let d_output = allocate_gpu_buffer(size)?;

    // 3. Launch kernel (when kernels are implemented)
    // unsafe {
    //     hip_quantize_kernel<<<grid, block, 0, stream>>>(
    //         d_input.ptr, d_output.ptr, size
    //     );
    // }

    // 4. Copy result back to CPU (when kernels are implemented)
    // let mut result = vec![0.0f32; size];
    // copy_from_gpu(&mut result, &d_output)?;

    // 5. Verify correctness
    // assert_close(&test_data, &result, 1e-4);

    // 6. Cleanup (RAII should handle this automatically)
    // - GPU buffers dropped automatically
    // - VRAM freed

    // For now, just verify test structure compiles
    assert_eq!(test_data.len(), 4);
}
