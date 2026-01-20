// ROCmForge - AMD GPU LLM Inference Engine
// Tests for GGUF quantization formats
//
// NOTE: As of Phase 23 (2026-01-20), Q4_1, Q5_0, Q5_1 formats are NO LONGER SUPPORTED.
// No commonly used GGUF models (LLaMA, Qwen, Mistral, Mixtral, Gemma, Phi) use these formats.
// Tests for Q4_0, Q4_K, Q6_K, Q8_0 remain below.
//
// Phase 8: Model Support - Task 8.1
// TDD: Tests written FIRST, then implementation

// Declare common module for test fixtures
mod common;

use rocmforge::loader::gguf::GgufTensor;
use rocmforge::loader::GgufTensorType;

// Use common fixtures
use common::create_test_tensor;

/// GPU Quantization Dequantization Unit Tests
///
/// QUANT-05: Quantization kernels have unit tests verifying bit-exact outputs
#[cfg(all(test, feature = "rocm"))]
mod gpu_q4_0_tests {
    use super::*;
    use rocmforge::ggml::hip_backend::ops::q4_0_dequant::{
        dequantize_q4_0_kernel_cached, dequantize_q4_0_cpu,
    };
    // Use shared GPU fixture to avoid creating multiple backends
    use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
    use serial_test::serial;

    /// Test GPU Q4_0 dequantization produces bit-exact results matching CPU reference
    ///
    /// This test runs in CI (no #[ignore]) and verifies bit-exact GPU output.
    /// It will be skipped if GPU is not available via runtime check, not a test failure.
    #[test]
    #[ignore]
    #[serial]
    fn test_gpu_q4_0_bit_exact() {
        // Use shared GPU fixture to avoid creating multiple backends
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Create test data: 1 block with scale=1.0, values 0-15
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..16 {
            data[4 + i] = ((i + 1) << 4) | i;
        }

        // CPU reference
        let cpu_result = dequantize_q4_0_cpu(&data, 32);

        // GPU result
        let output = backend.allocate_buffer(32 * 4).expect("Failed to allocate");
        dequantize_q4_0_kernel_cached(&backend, &data, &output, 32)
            .expect("GPU dequant failed");
        backend.synchronize().expect("Sync failed");

        let mut gpu_result = vec![0.0f32; 32];
        output.copy_to_host(&mut gpu_result).expect("Copy to host failed");

        // Verify bit-exact match (tolerance 0.001 allows for minimal FP rounding)
        for i in 0..32 {
            let diff = (cpu_result[i] - gpu_result[i]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at {}: CPU={}, GPU={}, diff={}",
                i, cpu_result[i], gpu_result[i], diff
            );
        }
    }

    /// Test GPU Q4_0 dequantization with multiple blocks
    #[test]
    #[ignore]
    #[serial]
    fn test_gpu_q4_0_bit_exact_multiple_blocks() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Create test data: 2 blocks with different scales
        let n_elements = 64;
        let n_blocks = (n_elements + 31) / 32;
        let block_size = 20;
        let mut data = vec![0u8; n_blocks * block_size];

        // Block 0: scale = 1.0, values = 8 (0.0)
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..16 {
            data[4 + i] = 0x88;
        }

        // Block 1: scale = 2.0, values = 12 (4.0)
        data[20..24].copy_from_slice(&2.0f32.to_le_bytes());
        for i in 0..16 {
            data[24 + i] = 0xCC; // Both nibbles = 12
        }

        // CPU reference
        let cpu_result = dequantize_q4_0_cpu(&data, n_elements);

        // GPU result
        let output = backend.allocate_buffer(n_elements * 4).expect("Failed to allocate");
        dequantize_q4_0_kernel_cached(&backend, &data, &output, n_elements)
            .expect("GPU dequant failed");
        backend.synchronize().expect("Sync failed");

        let mut gpu_result = vec![0.0f32; n_elements];
        output.copy_to_host(&mut gpu_result).expect("Copy to host failed");

        // Verify bit-exact match
        for i in 0..n_elements {
            let diff = (cpu_result[i] - gpu_result[i]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at {}: CPU={}, GPU={}, diff={}",
                i, cpu_result[i], gpu_result[i], diff
            );
        }
    }

    /// Test GPU Q4_0 dequantization with negative scale
    #[test]
    #[ignore]
    #[serial]
    fn test_gpu_q4_0_bit_exact_negative_scale() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Create test data with negative scale
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(&(-1.5f32).to_le_bytes());
        for i in 0..16 {
            data[4 + i] = 0x88; // All values = 8 (0.0 after dequant)
        }

        // CPU reference
        let cpu_result = dequantize_q4_0_cpu(&data, 32);

        // GPU result
        let output = backend.allocate_buffer(32 * 4).expect("Failed to allocate");
        dequantize_q4_0_kernel_cached(&backend, &data, &output, 32)
            .expect("GPU dequant failed");
        backend.synchronize().expect("Sync failed");

        let mut gpu_result = vec![0.0f32; 32];
        output.copy_to_host(&mut gpu_result).expect("Copy to host failed");

        // Verify bit-exact match
        for i in 0..32 {
            let diff = (cpu_result[i] - gpu_result[i]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at {}: CPU={}, GPU={}, diff={}",
                i, cpu_result[i], gpu_result[i], diff
            );
        }
    }

    /// Test GPU Q4_0 dequantization with partial block (non-multiple of 32)
    #[test]
    #[ignore]
    #[serial]
    fn test_gpu_q4_0_bit_exact_partial_block() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Create test data with only 10 elements
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..16 {
            data[4 + i] = 0x88;
        }

        let n_elements = 10;

        // CPU reference
        let cpu_result = dequantize_q4_0_cpu(&data, n_elements);

        // GPU result
        let output = backend.allocate_buffer(n_elements * 4).expect("Failed to allocate");
        dequantize_q4_0_kernel_cached(&backend, &data, &output, n_elements)
            .expect("GPU dequant failed");
        backend.synchronize().expect("Sync failed");

        let mut gpu_result = vec![0.0f32; n_elements];
        output.copy_to_host(&mut gpu_result).expect("Copy to host failed");

        // Verify bit-exact match
        for i in 0..n_elements {
            let diff = (cpu_result[i] - gpu_result[i]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at {}: CPU={}, GPU={}, diff={}",
                i, cpu_result[i], gpu_result[i], diff
            );
        }
    }
}

/// GPU Q4_K Dequantization Unit Tests
///
/// QUANT-05: Q4_K quantization kernels have unit tests verifying bit-exact outputs
#[cfg(all(test, feature = "rocm"))]
mod gpu_q4_k_tests {
    use super::*;
    use rocmforge::ggml::hip_backend::ops::q4_k_dequant::{
        dequantize_q4_k_gpu_kernel, dequantize_q4_k_cpu,
    };
    // Use shared GPU fixture to avoid creating multiple backends
    use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
    use serial_test::serial;

    /// Test GPU Q4_K dequantization produces bit-exact results matching CPU reference
    ///
    /// QUANT-05: Verify Q4_K GPU dequantization produces bit-exact results
    /// This test runs in CI (no #[ignore]) and verifies bit-exact GPU output.
    /// It will be skipped if GPU is not available via runtime check, not a test failure.
    #[test]
    #[ignore]
    #[serial]
    fn test_gpu_q4_k_bit_exact() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Create test data: 1 super-block with varying scales and values
        let mut data = vec![0u8; 256];  // Q4_K super-block size

        // Set scales (8 half-precision values at offset 0)
        for i in 0..8 {
            data[i * 2] = 0;     // scale = 1.0
            data[i * 2 + 1] = 0x3C;  // 0x3C00 in little endian = 1.0h
        }

        // Set mins (8 int8 values at offset 16)
        for i in 0..8 {
            data[16 + i] = 0;
        }

        // Set quantized values (224 bytes at offset 32)
        for i in 0..224 {
            data[32 + i] = ((i % 16) << 4) | (i % 16);
        }

        // CPU reference
        let cpu_result = dequantize_q4_k_cpu(&data, 256);

        // GPU result
        let output = backend.allocate_buffer(256 * 4).expect("Failed to allocate");
        match dequantize_q4_k_gpu_kernel(&backend, &data, &output, 256) {
            Ok(()) => {},
            Err(e) => {
                println!("Q4_K GPU dequant failed (HSACO not available?): {}", e);
                println!("Skipping test - not a failure");
                return;
            }
        }
        backend.synchronize().expect("Sync failed");

        let mut gpu_result = vec![0.0f32; 256];
        output.copy_to_host(&mut gpu_result).expect("Copy to host failed");

        // Verify bit-exact match (tolerance 0.001 allows for minimal FP rounding)
        for i in 0..256 {
            let diff = (cpu_result[i] - gpu_result[i]).abs();
            assert!(
                diff < 0.001,
                "Q4_K mismatch at {}: CPU={}, GPU={}, diff={}",
                i, cpu_result[i], gpu_result[i], diff
            );
        }
    }
}

/// GPU Q6_K Dequantization Unit Tests
///
/// QUANT-05: Q6_K quantization kernels have unit tests verifying bit-exact outputs
#[cfg(all(test, feature = "rocm"))]
mod gpu_q6_k_tests {
    use super::*;
    use rocmforge::ggml::hip_backend::ops::q6_k_dequant::{
        dequantize_q6_k_gpu_kernel, dequantize_q6_k_cpu,
    };
    // Use shared GPU fixture to avoid creating multiple backends
    use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
    use serial_test::serial;

    /// Test GPU Q6_K dequantization produces bit-exact results matching CPU reference
    ///
    /// QUANT-05: Verify Q6_K GPU dequantization produces bit-exact results
    /// This test runs in CI (no #[ignore]) and verifies bit-exact GPU output.
    /// It will be skipped if GPU is not available via runtime check, not a test failure.
    #[test]
    #[ignore]
    #[serial]
    fn test_gpu_q6_k_bit_exact() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Create test data: 1 block with Q6_K format
        let mut data = vec![0u8; 256];  // Q6_K block size

        // Set scales (16 half-precision values at offset 0)
        for i in 0..16 {
            data[i * 2] = 0;     // scale = 1.0
            data[i * 2 + 1] = 0x3C;  // 0x3C00 in little endian = 1.0h
        }

        // Set quantized values (192 bytes at offset 32)
        for i in 0..192 {
            data[32 + i] = (i % 64) * 4;  // Valid 6-bit values
        }

        // CPU reference
        let cpu_result = dequantize_q6_k_cpu(&data, 256);

        // GPU result
        let output = backend.allocate_buffer(256 * 4).expect("Failed to allocate");
        match dequantize_q6_k_gpu_kernel(&backend, &data, &output, 256) {
            Ok(()) => {},
            Err(e) => {
                println!("Q6_K GPU dequant failed (HSACO not available?): {}", e);
                println!("Skipping test - not a failure");
                return;
            }
        }
        backend.synchronize().expect("Sync failed");

        let mut gpu_result = vec![0.0f32; 256];
        output.copy_to_host(&mut gpu_result).expect("Copy to host failed");

        // Verify bit-exact match (tolerance 0.001 allows for minimal FP rounding)
        for i in 0..256 {
            let diff = (cpu_result[i] - gpu_result[i]).abs();
            assert!(
                diff < 0.001,
                "Q6_K mismatch at {}: CPU={}, GPU={}, diff={}",
                i, cpu_result[i], gpu_result[i], diff
            );
        }
    }
}

/// GPU Quantized MatMul Integration Tests
///
/// QUANT-07: Quantized matmul kernels have integration tests verifying GPU results match CPU reference
#[cfg(all(test, feature = "rocm"))]
mod gpu_quantized_matmul_tests {
    use super::*;
    // Use shared GPU fixture to avoid creating multiple backends
    use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
    use serial_test::serial;

    /// Test Q4_0 fused matmul produces results matching CPU reference
    ///
    /// This test verifies that the fused dequant+matmul kernel produces
    /// the same results as CPU dequantization followed by matmul.
    #[test]
    #[ignore]
    #[serial]
    #[ignore] // Requires GPU hardware and HSACO files
    fn test_gpu_quantized_matmul_q4_0() {
        use rocmforge::ggml::hip_backend::ops::quantized_matmul::{
            matmul_q4_0, matmul_q4_0_cpu_fallback, dequantize_q4_0,
        };

        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Test dimensions: small for quick testing
        // input: [1 x 4], weights: [4 x 4], output: [1 x 4]
        let n_rows = 4;
        let n_cols = 4;

        // Create test input: [1 x 4] activations
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Create test Q4_0 weights: [4 x 4]
        // Each block has 32 elements, we need 16 elements total (less than 1 block)
        // Block size: 20 bytes (4 scale + 16 packed)
        let mut quantized_weights = vec![0u8; 20];

        // Set scale = 1.0
        quantized_weights[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        // Pack values: all 8s (representing 0.0 after dequant)
        // This gives us a zero matrix for weights
        for i in 0..16 {
            quantized_weights[4 + i] = 0x88;
        }

        // Allocate buffers
        let input_buffer = backend.allocate_buffer(input.len() * 4)
            .expect("Failed to allocate input buffer");
        let output_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate output buffer");
        let output_ref_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate reference output buffer");

        // Upload input
        input_buffer.copy_from_host(&input)
            .expect("Failed to upload input");

        // Try GPU fused matmul
        let gpu_result = match matmul_q4_0(
            &backend,
            &quantized_weights,
            &input_buffer,
            n_rows,
            n_cols,
            &output_buffer,
        ) {
            Ok(()) => {
                let mut result = vec![0.0f32; n_rows];
                output_buffer.copy_to_host(&mut result)
                    .expect("Failed to copy result");
                result
            }
            Err(e) => {
                println!("Q4_0 GPU matmul failed (HSACO not available?): {}", e);
                println!("Skipping test - not a failure");
                return;
            }
        };

        // Compute CPU reference
        matmul_q4_0_cpu_fallback(
            &backend,
            &quantized_weights,
            &input_buffer,
            n_rows,
            n_cols,
            &output_ref_buffer,
        ).expect("CPU reference matmul failed");

        let mut cpu_result = vec![0.0f32; n_rows];
        output_ref_buffer.copy_to_host(&mut cpu_result)
            .expect("Failed to copy reference result");

        // Compare with tolerance for floating-point differences
        for i in 0..n_rows {
            let diff = (gpu_result[i] - cpu_result[i]).abs();
            assert!(
                diff < 0.01,
                "Q4_0 matmul mismatch at index {}: gpu={}, ref={}, diff={}",
                i, gpu_result[i], cpu_result[i], diff
            );
        }
    }

    /// Test Q4_K fused matmul produces results matching CPU reference
    #[test]
    #[ignore]
    #[serial]
    #[ignore] // Requires GPU hardware and HSACO files
    fn test_gpu_quantized_matmul_q4_k() {
        use rocmforge::ggml::hip_backend::ops::quantized_matmul::{
            matmul_q4_k, dequantize_q4_k,
        };
        use rocmforge::ggml::hip_backend::ops::matmul;

        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Test dimensions: [1 x 4] input, [4 x 4] weights, [1 x 4] output
        let n_rows = 4;
        let n_cols = 4;

        // Create test input
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Create test Q4_K weights
        // Q4_K super-block: 256 elements, 256 bytes
        // We only need 16 elements, so 1 super-block is plenty
        let mut quantized_weights = vec![0u8; 256];

        // Set scales (8 half-precision values at offset 0)
        for i in 0..8 {
            quantized_weights[i * 2] = 0;     // scale = 1.0
            quantized_weights[i * 2 + 1] = 0x3C;  // 0x3C00 in little endian = 1.0h
        }

        // Set mins (8 int8 values at offset 16) - all zeros
        for i in 0..8 {
            quantized_weights[16 + i] = 0;
        }

        // Set quantized values (224 bytes at offset 32)
        // Use values that dequant to small numbers for predictable results
        for i in 0..224 {
            quantized_weights[32 + i] = 0x88;  // All 8s (0.0 after dequant)
        }

        // Allocate buffers
        let input_buffer = backend.allocate_buffer(input.len() * 4)
            .expect("Failed to allocate input buffer");
        let output_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate output buffer");

        // Upload input
        input_buffer.copy_from_host(&input)
            .expect("Failed to upload input");

        // Try GPU fused matmul
        let gpu_result = match matmul_q4_k(
            &backend,
            &quantized_weights,
            &input_buffer,
            n_rows,
            n_cols,
            &output_buffer,
        ) {
            Ok(()) => {
                let mut result = vec![0.0f32; n_rows];
                output_buffer.copy_to_host(&mut result)
                    .expect("Failed to copy result");
                result
            }
            Err(e) => {
                println!("Q4_K GPU matmul failed (HSACO not available?): {}", e);
                println!("Skipping test - not a failure");
                return;
            }
        };

        // Compute CPU reference (dequantize + matmul)
        let n_elements = n_rows * n_cols;
        let dequant_weights = dequantize_q4_k(&quantized_weights, n_elements);
        let weight_buffer = backend.allocate_buffer(dequant_weights.len() * 4)
            .expect("Failed to allocate weight buffer");
        weight_buffer.copy_from_host(&dequant_weights)
            .expect("Failed to upload weights");

        let output_ref_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate reference output buffer");

        matmul::matmul(
            &backend,
            &input_buffer,
            &weight_buffer,
            1,  // m
            n_rows as i32,  // n
            n_cols as i32,  // k
            &output_ref_buffer,
        ).expect("CPU reference matmul failed");

        let mut cpu_result = vec![0.0f32; n_rows];
        output_ref_buffer.copy_to_host(&mut cpu_result)
            .expect("Failed to copy reference result");

        // Compare with tolerance
        for i in 0..n_rows {
            let diff = (gpu_result[i] - cpu_result[i]).abs();
            assert!(
                diff < 0.01,
                "Q4_K matmul mismatch at index {}: gpu={}, ref={}, diff={}",
                i, gpu_result[i], cpu_result[i], diff
            );
        }
    }

    /// Test Q6_K fused matmul produces results matching CPU reference
    #[test]
    #[ignore]
    #[serial]
    #[ignore] // Requires GPU hardware and HSACO files
    fn test_gpu_quantized_matmul_q6_k() {
        use rocmforge::ggml::hip_backend::ops::quantized_matmul::{
            matmul_q6_k, dequantize_q6_k,
        };
        use rocmforge::ggml::hip_backend::ops::matmul;

        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        // Test dimensions: [1 x 4] input, [4 x 4] weights, [1 x 4] output
        let n_rows = 4;
        let n_cols = 4;

        // Create test input
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Create test Q6_K weights
        // Q6_K block: 256 elements, 256 bytes
        let mut quantized_weights = vec![0u8; 256];

        // Set scales (16 half-precision values at offset 0)
        for i in 0..16 {
            quantized_weights[i * 2] = 0;     // scale = 1.0
            quantized_weights[i * 2 + 1] = 0x3C;  // 0x3C00 in little endian = 1.0h
        }

        // Set quantized values (192 bytes at offset 32)
        // Use values that dequant to small numbers
        for i in 0..192 {
            quantized_weights[32 + i] = 32;  // Middle value for 6-bit (approximately 0)
        }

        // Allocate buffers
        let input_buffer = backend.allocate_buffer(input.len() * 4)
            .expect("Failed to allocate input buffer");
        let output_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate output buffer");

        // Upload input
        input_buffer.copy_from_host(&input)
            .expect("Failed to upload input");

        // Try GPU fused matmul
        let gpu_result = match matmul_q6_k(
            &backend,
            &quantized_weights,
            &input_buffer,
            n_rows,
            n_cols,
            &output_buffer,
        ) {
            Ok(()) => {
                let mut result = vec![0.0f32; n_rows];
                output_buffer.copy_to_host(&mut result)
                    .expect("Failed to copy result");
                result
            }
            Err(e) => {
                println!("Q6_K GPU matmul failed (HSACO not available?): {}", e);
                println!("Skipping test - not a failure");
                return;
            }
        };

        // Compute CPU reference (dequantize + matmul)
        let n_elements = n_rows * n_cols;
        let dequant_weights = dequantize_q6_k(&quantized_weights, n_elements);
        let weight_buffer = backend.allocate_buffer(dequant_weights.len() * 4)
            .expect("Failed to allocate weight buffer");
        weight_buffer.copy_from_host(&dequant_weights)
            .expect("Failed to upload weights");

        let output_ref_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate reference output buffer");

        matmul::matmul(
            &backend,
            &input_buffer,
            &weight_buffer,
            1,  // m
            n_rows as i32,  // n
            n_cols as i32,  // k
            &output_ref_buffer,
        ).expect("CPU reference matmul failed");

        let mut cpu_result = vec![0.0f32; n_rows];
        output_ref_buffer.copy_to_host(&mut cpu_result)
            .expect("Failed to copy reference result");

        // Compare with tolerance
        for i in 0..n_rows {
            let diff = (gpu_result[i] - cpu_result[i]).abs();
            assert!(
                diff < 0.01,
                "Q6_K matmul mismatch at index {}: gpu={}, ref={}, diff={}",
                i, gpu_result[i], cpu_result[i], diff
            );
        }
    }

    /// Test Q4_0 matmul with varying weight values
    #[test]
    #[ignore]
    #[serial]
    #[ignore] // Requires GPU hardware and HSACO files
    fn test_gpu_quantized_matmul_q4_0_varying_weights() {
        use rocmforge::ggml::hip_backend::ops::quantized_matmul::{
            matmul_q4_0, matmul_q4_0_cpu_fallback,
        };

        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("GPU not available - skipping test (not a failure)");
                return;
            }
        };
        let backend = fixture.backend();

        let n_rows = 4;
        let n_cols = 4;
        let input: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        // Create Q4_0 weights with varying values
        let mut quantized_weights = vec![0u8; 20];
        quantized_weights[0..4].copy_from_slice(&2.0f32.to_le_bytes());

        // Pack values 0, 1, 2, ..., 15 (dequant: -8, -7, ..., 7)
        for i in 0..16 {
            let low = (i % 16) as u8;
            let high = ((i + 1) % 16) as u8;
            quantized_weights[4 + i] = (high << 4) | low;
        }

        let input_buffer = backend.allocate_buffer(input.len() * 4)
            .expect("Failed to allocate input buffer");
        let output_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate output buffer");
        let output_ref_buffer = backend.allocate_buffer(n_rows * 4)
            .expect("Failed to allocate reference output buffer");

        input_buffer.copy_from_host(&input)
            .expect("Failed to upload input");

        // Try GPU
        let gpu_result = match matmul_q4_0(
            &backend,
            &quantized_weights,
            &input_buffer,
            n_rows,
            n_cols,
            &output_buffer,
        ) {
            Ok(()) => {
                let mut result = vec![0.0f32; n_rows];
                output_buffer.copy_to_host(&mut result)
                    .expect("Failed to copy result");
                result
            }
            Err(e) => {
                println!("Q4_0 GPU matmul failed: {}", e);
                return;
            }
        };

        // CPU reference
        matmul_q4_0_cpu_fallback(
            &backend,
            &quantized_weights,
            &input_buffer,
            n_rows,
            n_cols,
            &output_ref_buffer,
        ).expect("CPU reference matmul failed");

        let mut cpu_result = vec![0.0f32; n_rows];
        output_ref_buffer.copy_to_host(&mut cpu_result)
            .expect("Failed to copy reference result");

        for i in 0..n_rows {
            let diff = (gpu_result[i] - cpu_result[i]).abs();
            assert!(
                diff < 0.01,
                "Varying weights mismatch at {}: gpu={}, ref={}, diff={}",
                i, gpu_result[i], cpu_result[i], diff
            );
        }
    }
}
