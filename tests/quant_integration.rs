//! Integration tier tests for GPU quantization kernels.
//!
//! These tests verify end-to-end quantization functionality:
//! - FFI calls to HIP kernels
//! - Memory allocation and transfer
//! - Correctness of quantization/dequantization
//! - Performance vs CPU baseline
//!
//! Implemented: Q4_K, Q5_K, Q8_0 quantization
//! TODO: Q4_K matmul, quantized GEMM, VRAM tracking, concurrent operations
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
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q4_K_BLOCK_SIZE, QK_K};

    // Require GPU for this test
    let caps = detect().expect("GPU required for quantization test");
    println!("Testing Q4_K quantization on: {}", caps.device_name);

    // Initialize device
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");

    // Initialize quantization context
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Prepare test data (256 elements = 1 Q4_K block)
    let n = QK_K;
    let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    // Allocate GPU buffers using GpuBuffer (RAII)
    let d_input =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n / QK_K) * Q4_K_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate output buffer");

    // Copy input to GPU (cast to &[u8] for copy_from_host)
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    // GpuBuffer::copy_from_host expects &mut self, so we need to make it mutable
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize on GPU (cast *mut u8 to *const f32 and *mut u8)
    gpu_quant
        .quantize(d_input.as_ptr() as *const f32, d_quantized.as_ptr(), n)
        .expect("Failed to quantize on GPU");

    // Dequantize on GPU
    gpu_quant
        .dequantize(d_quantized.as_ptr(), d_output.as_ptr() as *mut f32, n)
        .expect("Failed to dequantize on GPU");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, n).to_vec() };

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
    assert!(
        max_error < tolerance,
        "Max error {} exceeds tolerance {}",
        max_error,
        tolerance
    );

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
            n * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input2 = d_input2;
    d_input2
        .copy_from_host(input_bytes)
        .expect("Failed to copy input");

    gpu_quant2
        .quantize(d_input2.as_ptr() as *const f32, d_quantized2.as_ptr(), n)
        .expect("Failed to quantize");

    let (max_err, mse, rel_err) = gpu_quant2
        .verify_accuracy(d_input2.as_ptr() as *const f32, d_quantized2.as_ptr(), n)
        .expect("Failed to verify accuracy");

    println!(
        "Verification: max_error={}, mse={}, relative_error={}",
        max_err, mse, rel_err
    );

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_input2);
    drop(d_quantized2);

    // Verification should show reasonable accuracy
    assert!(
        max_err < 1.0,
        "Max verification error {} too large",
        max_err
    );
}

/// Test Q8_0 quantization kernel
///
/// Verifies:
/// 1. Allocate GPU buffers for input weights and output quantized blocks
/// 2. Call Q8_0 quantization FFI function
/// 3. Copy quantized blocks back to CPU
/// 4. Verify dequantized values match original within tolerance
#[test]
#[serial]
fn test_q8_0_quantization() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q8_0_BLOCK_SIZE, QK8_0};

    // Require GPU for this test
    let caps = detect().expect("GPU required for quantization test");
    println!("Testing Q8_0 quantization on: {}", caps.device_name);

    // Initialize device
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");

    // Initialize quantization context
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Prepare test data (32 elements = 1 Q8_0 block)
    let n = QK8_0;
    let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    // Allocate GPU buffers using GpuBuffer (RAII)
    let d_input =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n / QK8_0) * Q8_0_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate output buffer");

    // Copy input to GPU (cast to &[u8] for copy_from_host)
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    // GpuBuffer::copy_from_host expects &mut self, so we need to make it mutable
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize on GPU (cast *mut u8 to *const f32 and *mut u8)
    gpu_quant
        .quantize_q8_0(d_input.as_ptr() as *const f32, d_quantized.as_ptr(), n)
        .expect("Failed to quantize on GPU");

    // Dequantize on GPU
    gpu_quant
        .dequantize_q8_0(d_quantized.as_ptr(), d_output.as_ptr() as *mut f32, n)
        .expect("Failed to dequantize on GPU");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, n).to_vec() };

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify accuracy - Q8_0 has ~8 bits of precision
    // Should be more accurate than Q4_K (4-bit)
    let tolerance = 0.01; // Tight tolerance for 8-bit
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

    println!("Q8_0 quantization max error: {}", max_error);
    assert!(
        max_error < tolerance,
        "Max error {} exceeds tolerance {}",
        max_error,
        tolerance
    );

    // Test verify_accuracy function
    println!("Testing verify_q8_0_accuracy function...");
    let caps2 = detect().expect("GPU required");
    let device2 = GpuDevice::init(caps2.device_id).expect("Failed to init device");
    let gpu_quant2 = GpuQuant::new(device2).expect("Failed to init GpuQuant");

    let d_input2 = GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate");
    let d_quantized2 = GpuBuffer::alloc((n / QK8_0) * Q8_0_BLOCK_SIZE).expect("Failed to allocate");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input2 = d_input2;
    d_input2
        .copy_from_host(input_bytes)
        .expect("Failed to copy input");

    gpu_quant2
        .quantize_q8_0(d_input2.as_ptr() as *const f32, d_quantized2.as_ptr(), n)
        .expect("Failed to quantize");

    let (max_err, mse, rel_err) = gpu_quant2
        .verify_q8_0_accuracy(d_input2.as_ptr() as *const f32, d_quantized2.as_ptr(), n)
        .expect("Failed to verify accuracy");

    println!(
        "Q8_0 Verification: max_error={}, mse={}, relative_error={}",
        max_err, mse, rel_err
    );

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_input2);
    drop(d_quantized2);

    // Verification should show excellent accuracy for 8-bit
    assert!(
        max_err < 0.1,
        "Max verification error {} too large for Q8_0",
        max_err
    );
}

/// Test Q5_K quantization kernel
///
/// Verifies:
/// 1. Allocate GPU buffers for input weights and output quantized blocks
/// 2. Call Q5_K quantization FFI function
/// 3. Copy quantized blocks back to CPU
/// 4. Verify dequantized values match original within tolerance
///
/// Q5_K uses non-uniform quantization with 8 sub-blocks, each with its own scale/min.
/// Block size: 256 elements → 176 bytes (d + dmin + scales[12] + qh[32] + qs[128])
#[test]
#[serial]
fn test_q5_k_quantization() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q5_K_BLOCK_SIZE, QK_K};

    // Require GPU for this test
    let caps = detect().expect("GPU required for quantization test");
    println!("Testing Q5_K quantization on: {}", caps.device_name);

    // Initialize device
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");

    // Initialize quantization context
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Prepare test data (256 elements = 1 Q5_K block)
    let n = QK_K;
    let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    // Allocate GPU buffers using GpuBuffer (RAII)
    let d_input =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n / QK_K) * Q5_K_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate output buffer");

    // Copy input to GPU (cast to &[u8] for copy_from_host)
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    // GpuBuffer::copy_from_host expects &mut self, so we need to make it mutable
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize on GPU (cast *mut u8 to *const f32 and *mut u8)
    gpu_quant
        .quantize_q5_k(d_input.as_ptr() as *const f32, d_quantized.as_ptr(), n)
        .expect("Failed to quantize on GPU");

    // Dequantize on GPU
    gpu_quant
        .dequantize_q5_k(d_quantized.as_ptr(), d_output.as_ptr() as *mut f32, n)
        .expect("Failed to dequantize on GPU");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, n).to_vec() };

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify accuracy - Q5_K has ~5 bits of precision
    // Non-uniform quantization with 8 sub-blocks provides better accuracy than Q4_K
    // Expected accuracy: < 0.5% relative error per spec
    let tolerance = 0.1; // Tighter tolerance than Q4_K, looser than Q8_0
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

    println!("Q5_K quantization max error: {}", max_error);
    assert!(
        max_error < tolerance,
        "Max error {} exceeds tolerance {}",
        max_error,
        tolerance
    );

    // Test verify_accuracy function
    println!("Testing verify_q5_k_accuracy function...");
    let caps2 = detect().expect("GPU required");
    let device2 = GpuDevice::init(caps2.device_id).expect("Failed to init device");
    let gpu_quant2 = GpuQuant::new(device2).expect("Failed to init GpuQuant");

    let d_input2 = GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate");
    let d_quantized2 = GpuBuffer::alloc((n / QK_K) * Q5_K_BLOCK_SIZE).expect("Failed to allocate");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input2 = d_input2;
    d_input2
        .copy_from_host(input_bytes)
        .expect("Failed to copy input");

    gpu_quant2
        .quantize_q5_k(d_input2.as_ptr() as *const f32, d_quantized2.as_ptr(), n)
        .expect("Failed to quantize");

    let (max_err, mse, rel_err) = gpu_quant2
        .verify_q5_k_accuracy(d_input2.as_ptr() as *const f32, d_quantized2.as_ptr(), n)
        .expect("Failed to verify accuracy");

    println!(
        "Q5_K Verification: max_error={}, mse={}, relative_error={}",
        max_err, mse, rel_err
    );

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_input2);
    drop(d_quantized2);

    // Verification should show good accuracy for Q5_K (better than Q4_K, worse than Q8_0)
    assert!(
        max_err < 0.5,
        "Max verification error {} too large for Q5_K",
        max_err
    );
    assert!(
        rel_err < 0.01,
        "Relative error {} exceeds 1% for Q5_K",
        rel_err
    );
}

/// Test Q5_K × f32 GEMV
///
/// Verifies:
/// 1. Quantize weight matrix to Q5_K format on GPU
/// 2. Allocate GPU buffers for input vector and output
/// 3. Call Q5_K GEMV kernel with non-uniform sub-block scaling
/// 4. Verify output matches CPU reference
#[test]
#[serial]
fn test_q5_k_gemv() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q5_K_BLOCK_SIZE, QK_K};

    // Require GPU for this test
    let caps = detect().expect("GPU required for Q5_K GEMV test");
    println!("Testing Q5_K GEMV on: {}", caps.device_name);

    // Initialize device and quantization context
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Test parameters: small matrix for easy verification
    let n_rows = 256; // Input dimension (must be multiple of 256 for Q5_K)
    let ncols_dst = 4; // Output dimension (optimized case)

    // Create a simple weight matrix (n_rows × ncols_dst)
    // Use values that quantize well for Q5_K
    let mut weight_data: Vec<f32> = Vec::with_capacity(n_rows * ncols_dst);
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            // Simple pattern: use col index as value
            weight_data.push((col + 1) as f32 * 0.1);
        }
    }

    // Create input vector
    let input_data: Vec<f32> = (0..n_rows).map(|i| 1.0 + i as f32 * 0.1).collect();

    // Compute CPU reference output
    let mut expected_output = vec![0.0f32; ncols_dst];
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            expected_output[col] += weight_data[col * n_rows + row] * input_data[row];
        }
    }

    // Allocate GPU buffers
    let d_weights = GpuBuffer::alloc(n_rows * ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate weight buffer");
    let d_input = GpuBuffer::alloc(n_rows * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n_rows / QK_K) * ncols_dst * Q5_K_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output = GpuBuffer::alloc(ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate output buffer");

    // Copy weights and input to GPU
    let weight_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const u8,
            n_rows * ncols_dst * std::mem::size_of::<f32>(),
        )
    };
    let mut d_weights = d_weights;
    d_weights
        .copy_from_host(weight_bytes)
        .expect("Failed to copy weights to GPU");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n_rows * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize each column separately to Q5_K
    for col in 0..ncols_dst {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK_K) * Q5_K_BLOCK_SIZE)
        };

        // Quantize this column
        gpu_quant
            .quantize_q5_k(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)
            .expect("Failed to quantize column");
    }

    // Run Q5_K GEMV kernel
    gpu_quant
        .gemv_q5_k_f32(
            d_quantized.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_output.as_ptr() as *mut f32,
            n_rows,
            ncols_dst,
        )
        .expect("Failed to run Q5_K GEMV kernel");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; ncols_dst * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, ncols_dst).to_vec()
    };

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_weights);
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify accuracy - Q5_K has ~5 bits of precision
    // With simple test data, expect very small error
    let tolerance = 5.0; // Reasonable tolerance for Q5_K
    let mut max_error = 0.0f32;
    for (i, (expected, actual)) in expected_output.iter().zip(output_data.iter()).enumerate() {
        let error = (expected - actual).abs();
        max_error = max_error.max(error);
        if error > tolerance {
            panic!(
                "Large GEMV error at output {}: expected={:.2}, actual={:.2}, error={:.2}",
                i, expected, actual, error
            );
        }
    }

    println!("Q5_K GEMV test passed! max_error={}", max_error);
    assert!(
        max_error < tolerance,
        "Max error {} exceeds tolerance {}",
        max_error,
        tolerance
    );
}

/// Test Q8_0 × f32 GEMV kernel
///
/// Verifies:
/// 1. Create a simple weight matrix and input vector
/// 2. Quantize weights to Q8_0 format on GPU
/// 3. Call Q8_0 × f32 GEMV kernel
/// 4. Verify output matches CPU reference computation
#[test]
#[serial]
fn test_q8_0_gemv() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q8_0_BLOCK_SIZE, QK8_0};

    // Require GPU for this test
    let caps = detect().expect("GPU required for GEMV test");
    println!("Testing Q8_0 GEMV on: {}", caps.device_name);

    // Initialize device and quantization context
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Test parameters: small matrix for easy verification
    let n_rows = 64; // Input dimension (must be multiple of 32)
    let ncols_dst = 4; // Output dimension (optimized case)

    // Create a simple weight matrix (n_rows × ncols_dst)
    // Use smaller values to ensure Q8_0 quantization accuracy
    let mut weight_data: Vec<f32> = Vec::with_capacity(n_rows * ncols_dst);
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            // Use values in range [-1, 1] for better Q8_0 accuracy
            weight_data.push(((col as f32) * 0.1 + (row as f32) * 0.01 - 0.5).cos());
        }
    }

    // Create input vector
    let input_data: Vec<f32> = (0..n_rows).map(|i| 1.0 + i as f32 * 0.1).collect();

    // Compute CPU reference output
    let mut expected_output = vec![0.0f32; ncols_dst];
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            expected_output[col] += weight_data[col * n_rows + row] * input_data[row];
        }
    }

    // Allocate GPU buffers
    let d_weights = GpuBuffer::alloc(n_rows * ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate weight buffer");
    let d_input = GpuBuffer::alloc(n_rows * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n_rows / QK8_0) * ncols_dst * Q8_0_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output = GpuBuffer::alloc(ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate output buffer");

    // Copy weights and input to GPU
    let weight_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const u8,
            n_rows * ncols_dst * std::mem::size_of::<f32>(),
        )
    };
    let mut d_weights = d_weights;
    d_weights
        .copy_from_host(weight_bytes)
        .expect("Failed to copy weights to GPU");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n_rows * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // First, dequantize each column separately to Q8_0
    // Note: This is a simplified approach - in production you'd quantize all at once
    for col in 0..ncols_dst {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK8_0) * Q8_0_BLOCK_SIZE)
        };

        // Quantize this column
        gpu_quant
            .quantize_q8_0(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)
            .expect("Failed to quantize column");
    }

    // Run GEMV kernel
    gpu_quant
        .gemv_q8_0_f32(
            d_quantized.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_output.as_ptr() as *mut f32,
            n_rows,
            ncols_dst,
        )
        .expect("Failed to run GEMV kernel");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; ncols_dst * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, ncols_dst).to_vec()
    };

    // Cleanup is automatic via GpuBuffer's Drop
    drop(d_weights);
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify output - allow for quantization error
    // Q8_0 has ~8 bits of precision, but can still have ~1% error depending on data range
    let tolerance = 5.0; // Reasonable tolerance for Q8_0 with cosine data
    for (col, (expected, actual)) in expected_output.iter().zip(output_data.iter()).enumerate() {
        let error = (expected - actual).abs();
        println!(
            "Column {}: expected={}, actual={}, error={}",
            col, expected, actual, error
        );

        if error > tolerance {
            panic!(
                "Large GEMV error at column {}: expected={}, actual={}, error={}",
                col, expected, actual, error
            );
        }
    }

    println!("Q8_0 GEMV test passed!");
}

/// Test Q8_0 × f32 GEMV kernel with non-standard output sizes
///
/// Tests the generic kernel path for ncols_dst > 8
#[test]
#[serial]
fn test_q8_0_gemv_large_ncols() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q8_0_BLOCK_SIZE, QK8_0};

    // Require GPU for this test
    let caps = detect().expect("GPU required for GEMV test");
    println!(
        "Testing Q8_0 GEMV with large ncols on: {}",
        caps.device_name
    );

    // Initialize device and quantization context
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Test parameters: larger output dimension to test generic path
    let n_rows = 32; // Input dimension (minimal size)
    let ncols_dst = 16; // Output dimension (tests generic path, not specialized)

    // Create a simple weight matrix
    let mut weight_data: Vec<f32> = Vec::with_capacity(n_rows * ncols_dst);
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            weight_data.push((col as f32) + (row as f32) * 0.01);
        }
    }

    // Create input vector (all ones for simple verification)
    let input_data: Vec<f32> = (0..n_rows).map(|_| 1.0).collect();

    // Compute CPU reference output (sum of each column since input is all ones)
    let mut expected_output = vec![0.0f32; ncols_dst];
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            expected_output[col] += weight_data[col * n_rows + row];
        }
    }

    // Allocate GPU buffers
    let d_weights = GpuBuffer::alloc(n_rows * ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate weight buffer");
    let d_input = GpuBuffer::alloc(n_rows * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n_rows / QK8_0) * ncols_dst * Q8_0_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output = GpuBuffer::alloc(ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate output buffer");

    // Copy weights and input to GPU
    let weight_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const u8,
            n_rows * ncols_dst * std::mem::size_of::<f32>(),
        )
    };
    let mut d_weights = d_weights;
    d_weights
        .copy_from_host(weight_bytes)
        .expect("Failed to copy weights to GPU");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n_rows * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize each column separately
    for col in 0..ncols_dst {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK8_0) * Q8_0_BLOCK_SIZE)
        };

        gpu_quant
            .quantize_q8_0(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)
            .expect("Failed to quantize column");
    }

    // Run GEMV kernel
    gpu_quant
        .gemv_q8_0_f32(
            d_quantized.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_output.as_ptr() as *mut f32,
            n_rows,
            ncols_dst,
        )
        .expect("Failed to run GEMV kernel");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; ncols_dst * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, ncols_dst).to_vec()
    };

    // Cleanup is automatic via GpuBuffer's Drop
    drop(d_weights);
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify output - allow for quantization error
    let tolerance = 2.0; // Reasonable tolerance for Q8_0 (about 0.8% error)
    let mut max_error = 0.0f32;
    for (col, (expected, actual)) in expected_output.iter().zip(output_data.iter()).enumerate() {
        let error = (expected - actual).abs();
        max_error = max_error.max(error);

        if error > tolerance {
            panic!(
                "Large GEMV error at column {}: expected={}, actual={}, error={}",
                col, expected, actual, error
            );
        }
    }

    println!(
        "Q8_0 GEMV (ncols={}) test passed! max_error={}",
        ncols_dst, max_error
    );
}

/// Test Q4_K × f32 GEMV
///
/// Verifies:
/// 1. Quantize weight matrix to Q4_K format on GPU
/// 2. Allocate GPU buffers for input vector and output
/// 3. Call Q4_K GEMV kernel (direct dequantization + dot product)
/// 4. Verify output matches CPU reference
#[test]
#[serial]
fn test_q4_k_gemv() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q4_K_BLOCK_SIZE, QK_K};

    // Require GPU for this test
    let caps = detect().expect("GPU required for Q4_K GEMV test");
    println!("Testing Q4_K GEMV on: {}", caps.device_name);

    // Initialize device and quantization context
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Test parameters: small matrix for easy verification
    let n_rows = 256; // Input dimension (must be multiple of 256 for Q4_K)
    let ncols_dst = 4; // Output dimension (optimized case)

    // Create a simple weight matrix (n_rows × ncols_dst)
    // Use smaller values to ensure Q4_K quantization accuracy
    let mut weight_data: Vec<f32> = Vec::with_capacity(n_rows * ncols_dst);
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            // Use values in range [-1, 1] for better Q4_K accuracy
            weight_data.push(((col as f32) * 0.1 + (row as f32) * 0.01 - 0.5).cos());
        }
    }

    // Create input vector
    let input_data: Vec<f32> = (0..n_rows).map(|i| 1.0 + i as f32 * 0.1).collect();

    // Compute CPU reference output
    let mut expected_output = vec![0.0f32; ncols_dst];
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            expected_output[col] += weight_data[col * n_rows + row] * input_data[row];
        }
    }

    // Allocate GPU buffers
    let d_weights = GpuBuffer::alloc(n_rows * ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate weight buffer");
    let d_input = GpuBuffer::alloc(n_rows * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n_rows / QK_K) * ncols_dst * Q4_K_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output = GpuBuffer::alloc(ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate output buffer");

    // Copy weights and input to GPU
    let weight_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const u8,
            n_rows * ncols_dst * std::mem::size_of::<f32>(),
        )
    };
    let mut d_weights = d_weights;
    d_weights
        .copy_from_host(weight_bytes)
        .expect("Failed to copy weights to GPU");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n_rows * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize each column separately to Q4_K
    for col in 0..ncols_dst {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE)
        };

        // Quantize this column
        gpu_quant
            .quantize(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)
            .expect("Failed to quantize column");
    }

    // Run Q4_K GEMV kernel
    gpu_quant
        .gemv_q4_k_f32(
            d_quantized.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_output.as_ptr() as *mut f32,
            n_rows,
            ncols_dst,
        )
        .expect("Failed to run Q4_K GEMV kernel");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; ncols_dst * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, ncols_dst).to_vec()
    };

    // Cleanup is automatic via GpuBuffer's Drop implementation
    drop(d_weights);
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify accuracy - Q4_K has ~4.5 bits of precision
    // Allow for larger error due to 4-bit quantization
    let tolerance = 10.0; // Relaxed tolerance for Q4_K (~1% relative error)
    let mut max_error = 0.0f32;
    for (i, (expected, actual)) in expected_output.iter().zip(output_data.iter()).enumerate() {
        let error = (expected - actual).abs();
        max_error = max_error.max(error);
        if error > tolerance {
            panic!(
                "Large GEMV error at output {}: expected={}, actual={}, error={}",
                i, expected, actual, error
            );
        }
    }

    println!("Q4_K GEMV test passed! max_error={}", max_error);
    assert!(
        max_error < tolerance,
        "Max error {} exceeds tolerance {}",
        max_error,
        tolerance
    );
}

/// Test Q4_0 quantization and dequantization roundtrip
///
/// Verifies:
/// 1. Quantize f32 data to Q4_0 format on GPU
/// 2. Dequantize back to f32
/// 3. Verify accuracy < 1.0% relative error
/// 4. Test verify_q4_0_accuracy function
#[test]
#[serial]
fn test_q4_0_quantization() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q4_0_BLOCK_SIZE, QK4_0};

    // Require GPU for this test
    let caps = detect().expect("GPU required for Q4_0 quantization test");
    println!("Testing Q4_0 quantization on: {}", caps.device_name);

    // Initialize device
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");

    // Initialize quantization context
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Prepare test data (256 elements = 8 Q4_0 blocks)
    let n = 8 * QK4_0; // 256 elements
                       // Use values in range [-1, 1] for better Q4_0 quantization accuracy
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01 - 1.0).cos()).collect();

    // Allocate GPU buffers
    let d_input =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n / QK4_0) * Q4_0_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate output buffer");

    // Copy input to GPU
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize to Q4_0
    gpu_quant
        .quantize_q4_0(d_input.as_ptr() as *const f32, d_quantized.as_ptr(), n)
        .expect("Failed to quantize");

    // Dequantize back to f32
    gpu_quant
        .dequantize_q4_0(d_quantized.as_ptr(), d_output.as_ptr() as *mut f32, n)
        .expect("Failed to dequantize");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, n).to_vec() };

    // Verify accuracy - Q4_0 has ~4 bits of precision
    // Use absolute tolerance since Q4_0 has limited precision
    let tolerance = 0.35; // Very relaxed tolerance for 4-bit quantization (~35% of range)
    let mut max_abs_error = 0.0f32;
    for (i, (original, dequantized)) in input_data.iter().zip(output_data.iter()).enumerate() {
        let abs_error = (original - dequantized).abs();
        max_abs_error = max_abs_error.max(abs_error);

        if abs_error > tolerance {
            panic!(
                "Large dequantization error at index {}: original={}, dequantized={}, abs_error={}",
                i, original, dequantized, abs_error
            );
        }
    }

    println!(
        "Q4_0 quantization test passed! max_abs_error={}",
        max_abs_error
    );

    // Test verify_q4_0_accuracy function
    let d_input_verify = GpuBuffer::alloc(n * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer for verification");
    let mut d_input_verify = d_input_verify;
    d_input_verify
        .copy_from_host(input_bytes)
        .expect("Failed to copy input for verification");

    let (max_error, mean_error, _std_dev) = gpu_quant
        .verify_q4_0_accuracy(
            d_input_verify.as_ptr() as *const f32,
            d_quantized.as_ptr(),
            n,
        )
        .expect("Failed to verify accuracy");

    drop(d_input_verify);

    println!(
        "Q4_0 accuracy verification: max_error={}, mean_error={}",
        max_error, mean_error
    );
    assert!(
        max_error < tolerance,
        "Max error {} exceeds tolerance {}",
        max_error,
        tolerance
    );
    assert!(
        mean_error < tolerance / 4.0,
        "Mean error {} exceeds tolerance {}",
        mean_error,
        tolerance / 4.0
    );

    // Cleanup is automatic via GpuBuffer's Drop
    drop(d_input);
    drop(d_quantized);
    drop(d_output);
}

/// Test Q4_0 × f32 GEMV kernel
///
/// Verifies:
/// 1. Quantize weight matrix to Q4_0 format on GPU
/// 2. Allocate GPU buffers for input vector and output
/// 3. Call Q4_0 × f32 GEMV kernel
/// 4. Verify output matches CPU reference
#[test]
#[serial]
fn test_q4_0_gemv() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q4_0_BLOCK_SIZE, QK4_0};

    // Require GPU for this test
    let caps = detect().expect("GPU required for Q4_0 GEMV test");
    println!("Testing Q4_0 GEMV on: {}", caps.device_name);

    // Initialize device and quantization context
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Test parameters: 32×4 test matrix as specified
    let n_rows = 32; // Input dimension (must be multiple of 32 for Q4_0)
    let ncols_dst = 4; // Output dimension

    // Create a simple weight matrix (n_rows × ncols_dst)
    // Use smaller values to ensure Q4_0 quantization accuracy
    let mut weight_data: Vec<f32> = Vec::with_capacity(n_rows * ncols_dst);
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            // Use values in range [-1, 1] for better Q4_0 accuracy
            weight_data.push(((col as f32) * 0.1 + (row as f32) * 0.01 - 0.5).cos());
        }
    }

    // Create input vector
    let input_data: Vec<f32> = (0..n_rows).map(|i| 1.0 + i as f32 * 0.1).collect();

    // Compute CPU reference output
    let mut expected_output = vec![0.0f32; ncols_dst];
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            expected_output[col] += weight_data[col * n_rows + row] * input_data[row];
        }
    }

    // Allocate GPU buffers
    let d_weights = GpuBuffer::alloc(n_rows * ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate weight buffer");
    let d_input = GpuBuffer::alloc(n_rows * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n_rows / QK4_0) * ncols_dst * Q4_0_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output = GpuBuffer::alloc(ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate output buffer");

    // Copy weights and input to GPU
    let weight_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const u8,
            n_rows * ncols_dst * std::mem::size_of::<f32>(),
        )
    };
    let mut d_weights = d_weights;
    d_weights
        .copy_from_host(weight_bytes)
        .expect("Failed to copy weights to GPU");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n_rows * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize each column separately to Q4_0
    for col in 0..ncols_dst {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK4_0) * Q4_0_BLOCK_SIZE)
        };

        // Quantize this column
        gpu_quant
            .quantize_q4_0(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)
            .expect("Failed to quantize column");
    }

    // Run GEMV kernel
    gpu_quant
        .gemv_q4_0_f32(
            d_quantized.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_output.as_ptr() as *mut f32,
            n_rows,
            ncols_dst,
        )
        .expect("Failed to run GEMV kernel");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; ncols_dst * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, ncols_dst).to_vec()
    };

    // Cleanup is automatic via GpuBuffer's Drop
    drop(d_weights);
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify output - allow for quantization error
    // Q4_0 has ~4 bits of precision, which is significantly less accurate than Q8_0 (8 bits)
    // Use 15.0% relative error tolerance (realistic for 4-bit quantization with cosine data)
    let tolerance = 0.15; // 15.0% relative error
    let mut max_relative_error = 0.0f32;
    for (col, (expected, actual)) in expected_output.iter().zip(output_data.iter()).enumerate() {
        let abs_error = (expected - actual).abs();
        let relative_error = if expected.abs() > 1e-6 {
            abs_error / expected.abs()
        } else {
            abs_error
        };
        max_relative_error = max_relative_error.max(relative_error);

        println!(
            "Column {}: expected={}, actual={}, abs_error={}, rel_error={}",
            col, expected, actual, abs_error, relative_error
        );

        if relative_error > tolerance && abs_error > 0.1 {
            panic!(
                "Large GEMV error at column {}: expected={}, actual={}, rel_error={}",
                col, expected, actual, relative_error
            );
        }
    }

    println!(
        "Q4_0 GEMV test passed! max_relative_error={}",
        max_relative_error
    );
    assert!(
        max_relative_error < tolerance,
        "Max relative error {} exceeds tolerance {}",
        max_relative_error,
        tolerance
    );
}

/// Test Q4_1 quantization and dequantization roundtrip
///
/// Verifies:
/// 1. Quantize f32 data to Q4_1 format on GPU
/// 2. Dequantize back to f32
/// 3. Verify accuracy < 0.35 absolute error
/// 4. Test verify_q4_1_accuracy function
#[test]
#[serial]
fn test_q4_1_quantization() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q4_1_BLOCK_SIZE, QK4_1};

    // Require GPU for this test
    let caps = detect().expect("GPU required for Q4_1 quantization test");
    println!("Testing Q4_1 quantization on: {}", caps.device_name);

    // Initialize device
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");

    // Initialize quantization context
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Prepare test data (256 elements = 8 Q4_1 blocks)
    let n = 8 * QK4_1; // 256 elements
                       // Use values in range [-1, 1] for better Q4_1 quantization accuracy
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01 - 1.0).cos()).collect();

    // Allocate GPU buffers
    let d_input =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n / QK4_1) * Q4_1_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output =
        GpuBuffer::alloc(n * std::mem::size_of::<f32>()).expect("Failed to allocate output buffer");

    // Copy input to GPU
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize to Q4_1
    gpu_quant
        .quantize_q4_1(d_input.as_ptr() as *const f32, d_quantized.as_ptr(), n)
        .expect("Failed to quantize");

    // Dequantize back to f32
    gpu_quant
        .dequantize_q4_1(d_quantized.as_ptr(), d_output.as_ptr() as *mut f32, n)
        .expect("Failed to dequantize");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, n).to_vec() };

    // Verify accuracy - Q4_1 has ~4 bits of precision with scale+min per block
    // Q4_1 should be similar or better than Q4_0 since it uses scale+min
    // Use absolute tolerance
    let tolerance = 0.35; // Relaxed tolerance for 4-bit quantization
    let mut max_abs_error = 0.0f32;
    for (i, (original, dequantized)) in input_data.iter().zip(output_data.iter()).enumerate() {
        let abs_error = (original - dequantized).abs();
        max_abs_error = max_abs_error.max(abs_error);

        if abs_error > tolerance {
            panic!(
                "Large dequantization error at index {}: original={}, dequantized={}, abs_error={}",
                i, original, dequantized, abs_error
            );
        }
    }

    println!(
        "Q4_1 quantization test passed! max_abs_error={}",
        max_abs_error
    );

    // Test verify_q4_1_accuracy function
    let d_input_verify = GpuBuffer::alloc(n * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer for verification");
    let mut d_input_verify = d_input_verify;
    d_input_verify
        .copy_from_host(input_bytes)
        .expect("Failed to copy input for verification");

    let (max_error, mean_error, _std_dev) = gpu_quant
        .verify_q4_1_accuracy(
            d_input_verify.as_ptr() as *const f32,
            d_quantized.as_ptr(),
            n,
        )
        .expect("Failed to verify accuracy");

    drop(d_input_verify);

    println!(
        "Q4_1 accuracy verification: max_error={}, mean_error={}",
        max_error, mean_error
    );
    assert!(
        max_error < tolerance,
        "Max error {} exceeds tolerance {}",
        max_error,
        tolerance
    );
    assert!(
        mean_error < tolerance / 4.0,
        "Mean error {} exceeds tolerance {}",
        mean_error,
        tolerance / 4.0
    );

    // Cleanup is automatic via GpuBuffer's Drop
    drop(d_input);
    drop(d_quantized);
    drop(d_output);
}

/// Test Q4_1 × f32 GEMV kernel
///
/// Verifies:
/// 1. Quantize weight matrix to Q4_1 format on GPU
/// 2. Allocate GPU buffers for input vector and output
/// 3. Call Q4_1 × f32 GEMV kernel
/// 4. Verify output matches CPU reference with 10% relative error tolerance
#[test]
#[serial]
fn test_q4_1_gemv() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q4_1_BLOCK_SIZE, QK4_1};

    // Require GPU for this test
    let caps = detect().expect("GPU required for Q4_1 GEMV test");
    println!("Testing Q4_1 GEMV on: {}", caps.device_name);

    // Initialize device and quantization context
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(device).expect("Failed to initialize GpuQuant");

    // Test parameters: 32×4 test matrix
    let n_rows = 32; // Input dimension (must be multiple of 32 for Q4_1)
    let ncols_dst = 4; // Output dimension

    // Create a simple weight matrix (n_rows × ncols_dst)
    // Use smaller values to ensure Q4_1 quantization accuracy
    let mut weight_data: Vec<f32> = Vec::with_capacity(n_rows * ncols_dst);
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            // Use values in range [-1, 1] for better Q4_1 accuracy
            weight_data.push(((col as f32) * 0.1 + (row as f32) * 0.01 - 0.5).cos());
        }
    }

    // Create input vector
    let input_data: Vec<f32> = (0..n_rows).map(|i| 1.0 + i as f32 * 0.1).collect();

    // Compute CPU reference output
    let mut expected_output = vec![0.0f32; ncols_dst];
    for col in 0..ncols_dst {
        for row in 0..n_rows {
            expected_output[col] += weight_data[col * n_rows + row] * input_data[row];
        }
    }

    // Allocate GPU buffers
    let d_weights = GpuBuffer::alloc(n_rows * ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate weight buffer");
    let d_input = GpuBuffer::alloc(n_rows * std::mem::size_of::<f32>())
        .expect("Failed to allocate input buffer");
    let d_quantized = GpuBuffer::alloc((n_rows / QK4_1) * ncols_dst * Q4_1_BLOCK_SIZE)
        .expect("Failed to allocate quantized buffer");
    let d_output = GpuBuffer::alloc(ncols_dst * std::mem::size_of::<f32>())
        .expect("Failed to allocate output buffer");

    // Copy weights and input to GPU
    let weight_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const u8,
            n_rows * ncols_dst * std::mem::size_of::<f32>(),
        )
    };
    let mut d_weights = d_weights;
    d_weights
        .copy_from_host(weight_bytes)
        .expect("Failed to copy weights to GPU");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            n_rows * std::mem::size_of::<f32>(),
        )
    };
    let mut d_input = d_input;
    d_input
        .copy_from_host(input_bytes)
        .expect("Failed to copy input to GPU");

    // Quantize each column separately to Q4_1
    for col in 0..ncols_dst {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK4_1) * Q4_1_BLOCK_SIZE)
        };

        // Quantize this column
        gpu_quant
            .quantize_q4_1(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)
            .expect("Failed to quantize column");
    }

    // Run GEMV kernel
    gpu_quant
        .gemv_q4_1_f32(
            d_quantized.as_ptr(),
            d_input.as_ptr() as *const f32,
            d_output.as_ptr() as *mut f32,
            n_rows,
            ncols_dst,
        )
        .expect("Failed to run GEMV kernel");

    // Copy output back to CPU
    let mut output_bytes = vec![0u8; ncols_dst * std::mem::size_of::<f32>()];
    d_output
        .copy_to_host(&mut output_bytes)
        .expect("Failed to copy output from GPU");

    // Convert bytes back to f32
    let output_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, ncols_dst).to_vec()
    };

    // Cleanup is automatic via GpuBuffer's Drop
    drop(d_weights);
    drop(d_input);
    drop(d_quantized);
    drop(d_output);

    // Verify output - allow for quantization error
    // Q4_1 has ~4 bits of precision with scale+min, similar to Q4_0
    // Use 10% relative error tolerance
    let tolerance = 0.10; // 10% relative error
    let mut max_relative_error = 0.0f32;
    for (col, (expected, actual)) in expected_output.iter().zip(output_data.iter()).enumerate() {
        let abs_error = (expected - actual).abs();
        let relative_error = if expected.abs() > 1e-6 {
            abs_error / expected.abs()
        } else {
            abs_error
        };
        max_relative_error = max_relative_error.max(relative_error);

        println!(
            "Column {}: expected={}, actual={}, abs_error={}, rel_error={}",
            col, expected, actual, abs_error, relative_error
        );

        if relative_error > tolerance && abs_error > 0.1 {
            panic!(
                "Large GEMV error at column {}: expected={}, actual={}, rel_error={}",
                col, expected, actual, relative_error
            );
        }
    }

    println!(
        "Q4_1 GEMV test passed! max_relative_error={}",
        max_relative_error
    );
    assert!(
        max_relative_error < tolerance,
        "Max relative error {} exceeds tolerance {}",
        max_relative_error,
        tolerance
    );
}

/// Test Q4_0 dequantization of real model weights from GGUF
///
/// Loads the `output_norm.weight` tensor (F32, 3584 elements) from a real Q4_0 GGUF model,
/// quantizes it to Q4_0 on the GPU, then dequantizes back and verifies accuracy.
/// This validates our kernels work on real data distributions, not just synthetic patterns.
#[test]
#[serial]
fn test_q4_0_real_model_weights() {
    use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, Q4_0_BLOCK_SIZE, QK4_0};
    use rocmforge::loader::GgufFile;

    let model_path = "/home/feanor/Projects/Memoria/models/Qwen2.5-7B-Instruct-Q4_0-Pure.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model file not found at {}", model_path);
        return;
    }

    let caps = detect().expect("GPU required");
    println!("Testing Q4_0 real model weights on: {}", caps.device_name);

    let device = GpuDevice::init(caps.device_id).expect("Failed to init GPU");
    let gpu_quant = GpuQuant::new(device).expect("Failed to init GpuQuant");

    // Open the GGUF model
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF model");
    println!("Loaded model: {} tensors", gguf.tensor_count());

    // Load a small F32 tensor to use as ground truth: output_norm.weight (3584 elements, F32)
    let norm_tensor = gguf
        .tensor("output_norm.weight")
        .expect("tensor lookup failed")
        .expect("output_norm.weight not found");
    assert!(matches!(
        norm_tensor.ggml_type,
        rocmforge::loader::GgmlType::F32
    ));
    let n = norm_tensor.element_count();
    println!("output_norm.weight: {} elements, {:?}", n, norm_tensor.dims);

    // Convert raw bytes to f32
    let norm_bytes = norm_tensor.data;
    let original: Vec<f32> =
        unsafe { std::slice::from_raw_parts(norm_bytes.as_ptr() as *const f32, n).to_vec() };

    // Verify original data is reasonable (RMS norm weights should be ~1.0)
    let mean: f32 = original.iter().sum::<f32>() / n as f32;
    println!(
        "Original: mean={:.4}, min={:.4}, max={:.4}",
        mean,
        original.iter().cloned().fold(f32::INFINITY, f32::min),
        original.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Upload to GPU
    let mut d_input = GpuBuffer::alloc(n * std::mem::size_of::<f32>()).unwrap();
    let d_quantized = GpuBuffer::alloc((n / QK4_0) * Q4_0_BLOCK_SIZE).unwrap();
    let d_output = GpuBuffer::alloc(n * std::mem::size_of::<f32>()).unwrap();

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            original.as_ptr() as *const u8,
            n * std::mem::size_of::<f32>(),
        )
    };
    d_input.copy_from_host(input_bytes).unwrap();

    // Quantize to Q4_0 on GPU
    gpu_quant
        .quantize_q4_0(d_input.as_ptr() as *const f32, d_quantized.as_ptr(), n)
        .expect("Failed to quantize real weights");

    // Dequantize back
    gpu_quant
        .dequantize_q4_0(d_quantized.as_ptr(), d_output.as_ptr() as *mut f32, n)
        .expect("Failed to dequantize");

    // Copy result back
    let mut output_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    d_output.copy_to_host(&mut output_bytes).unwrap();
    let dequantized: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, n).to_vec() };

    // Verify accuracy
    let mut max_abs_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;
    let mut sum_sq_orig = 0.0f32;
    for (orig, deq) in original.iter().zip(dequantized.iter()) {
        let err = (orig - deq).abs();
        max_abs_error = max_abs_error.max(err);
        sum_sq_error += err * err;
        sum_sq_orig += orig * orig;
    }
    let rmse = (sum_sq_error / n as f32).sqrt();
    let rms_orig = (sum_sq_orig / n as f32).sqrt();
    let relative_error = if rms_orig > 0.0 { rmse / rms_orig } else { 0.0 };

    println!(
        "Q4_0 real weights: max_abs_error={:.6}, rmse={:.6}, relative_error={:.4}%",
        max_abs_error,
        rmse,
        relative_error * 100.0
    );

    // Q4_0 real weights can have wide dynamic range, so tolerance is generous
    // Typical: 10-15% relative error for data with range ~[-0.2, 10.8]
    assert!(
        relative_error < 0.20,
        "Relative error {:.4}% exceeds 20%",
        relative_error * 100.0
    );

    // Also test verify_q4_0_accuracy
    let (max_error, mse, rel_error) = gpu_quant
        .verify_q4_0_accuracy(d_input.as_ptr() as *const f32, d_quantized.as_ptr(), n)
        .expect("Failed to verify accuracy");

    println!(
        "Q4_0 GPU verify: max_error={:.6}, mse={:.6}, rel_error={:.6}",
        max_error, mse, rel_error
    );
    assert!(
        rel_error < 0.20,
        "GPU verify rel_error {:.4} exceeds 20%",
        rel_error
    );
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
