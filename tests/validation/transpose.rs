//! Transpose validation tests
//!
//! This module validates that the transpose kernel works for the
//! actual failing tensor size [896, 151936] from Qwen2.5-0.5B.

use rocmforge::backend::HipBackend;
use rocmforge::backend::hip_backend::DeviceTensor;
use rocmforge::kernels::transpose::transpose_tensor;
use rocmforge::loader::TensorShape;
use serial_test::serial;

/// Test: Transpose the exact failing tensor size [896, 151936]
///
/// This is the actual failing case from qwen2.5-0.5b.gguf model loading.
/// The original bug: block=(64,64,1) = 4096 threads exceeded maxThreadsPerBlock=1024.
/// After fix: block=(32,32,1) = 1024 threads, within limit.
///
/// This test EXHAUSTIVELY verifies ALL elements against CPU reference.
#[test]
#[serial]
fn test_transpose_896_151936_exhaustive() {
    // Skip if GPU not available
    let backend = match HipBackend::new_checked() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("\n=== SKIP: test_transpose_896_151936_exhaustive ===");
            eprintln!("Reason: GPU not available - {}", e);
            eprintln!("=== END SKIP ===\n");
            return;
        }
    };

    // Skip if driver has known bug: max_threads_dim[1] == 0
    let limits = backend.limits();
    if limits.max_threads_dim[1] == 0 || limits.max_threads_dim[2] == 0 {
        eprintln!("\n=== SKIP: test_transpose_896_151936_exhaustive ===");
        eprintln!("Reason: HIP driver reports invalid maxThreadsDim (axis limit 0): {:?}", limits.max_threads_dim);
        eprintln!("This is a known driver bug - transpose requires 2D blocks");
        eprintln!("=== END SKIP ===\n");
        return;
    }

    // Qwen2.5-0.5b embedding weight dimensions
    // hidden_size = 896, vocab_size = 151936
    let hidden_size = 896usize;
    let vocab_size = 151936usize;
    let elem_count = hidden_size * vocab_size;

    eprintln!("\n=== test_transpose_896_151936_exhaustive ===");
    eprintln!("Tensor shape: [{}, {}] ({} elements)", hidden_size, vocab_size, elem_count);
    eprintln!("Testing transpose: [hidden_size, vocab_size] -> [vocab_size, hidden_size]");
    eprintln!("This may take a while for exhaustive verification...\n");

    // Create test data: sequential values for verification
    let input_data: Vec<f32> = (0..elem_count).map(|i| i as f32).collect();

    // Create input tensor on GPU
    let input_shape = TensorShape::from_dims(&[hidden_size, vocab_size]);
    let input_buffer = backend
        .allocate_buffer(elem_count * 4)
        .expect("Failed to allocate input buffer");
    backend
        .copy_to_device(&input_buffer, &input_data)
        .expect("Failed to copy input to device");
    let input_tensor = DeviceTensor {
        buffer: input_buffer,
        shape: input_shape,
    };

    // Perform transpose
    let result = transpose_tensor(&backend, &input_tensor);

    match result {
        Ok(transposed) => {
            eprintln!("Transpose kernel executed successfully!");
            eprintln!("Output shape: {:?}", transposed.shape().dims());

            // Verify shape is correctly transposed
            assert_eq!(
                transposed.shape().dims(),
                &[vocab_size, hidden_size],
                "Transposed shape should be [vocab_size, hidden_size]"
            );

            // Copy result back for exhaustive verification
            let mut output_data = vec![0.0f32; elem_count];
            backend
                .copy_from_device_safe(&transposed.buffer, &mut output_data)
                .expect("Failed to copy output from device");

            eprintln!("Starting exhaustive element verification...");
            eprintln!("Checking all {} elements...", elem_count);

            let mut errors_found = 0;
            const MAX_ERRORS_TO_PRINT: usize = 10;

            // EXHAUSTIVE verification: Check ALL elements
            // Verify transpose correctness: output[vocab_idx, hidden_idx] == input[hidden_idx, vocab_idx]
            for vocab_idx in 0..vocab_size {
                for hidden_idx in 0..hidden_size {
                    let input_idx = hidden_idx * vocab_size + vocab_idx;
                    let output_idx = vocab_idx * hidden_size + hidden_idx;

                    let expected = input_data[input_idx];
                    let actual = output_data[output_idx];

                    if actual != expected {
                        if errors_found < MAX_ERRORS_TO_PRINT {
                            eprintln!(
                                "MISMATCH at [vocab={}, hidden={}]: expected={}, actual={}",
                                vocab_idx, hidden_idx, expected, actual
                            );
                        }
                        errors_found += 1;
                    }
                }

                // Progress indicator every 10% through vocab
                if vocab_idx % (vocab_size / 10) == 0 {
                    eprintln!(
                        "Progress: {}/{} vocab vectors verified ({:.0}%)",
                        vocab_idx,
                        vocab_size,
                        (vocab_idx as f32 / vocab_size as f32) * 100.0
                    );
                }
            }

            if errors_found > 0 {
                panic!(
                    "Transpose verification FAILED: {} elements incorrect ({} shown)",
                    errors_found,
                    MAX_ERRORS_TO_PRINT.min(errors_found)
                );
            }

            eprintln!("\n=== SUCCESS ===");
            eprintln!("All {} elements verified correctly!", elem_count);
            eprintln!("No hipErrorInvalidValue from transpose kernel");
            eprintln!("===================\n");
        }
        Err(rocmforge::backend::hip_backend::HipError::KernelLoadFailed(msg)) => {
            eprintln!("\n=== SKIP: test_transpose_896_151936_exhaustive ===");
            eprintln!("Reason: Kernel load failed - {}", msg);
            eprintln!("Kernel may not be compiled. Run with compiled HSACO files.");
            eprintln!("=== END SKIP ===\n");
        }
        Err(e) => {
            panic!("Transpose failed for [896, 151936]: {:?}", e);
        }
    }
}

/// Test: Transpose with direct kernel invocation
///
/// This bypasses the transpose_tensor() convenience wrapper
/// to validate the kernel works in isolation.
#[test]
#[serial]
fn test_transpose_896_151936_standalone() {
    // Skip if GPU not available
    let backend = match HipBackend::new_checked() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("\n=== SKIP: test_transpose_896_151936_standalone ===");
            eprintln!("Reason: GPU not available - {}", e);
            eprintln!("=== END SKIP ===\n");
            return;
        }
    };

    // Skip if driver has known bug: max_threads_dim[1] == 0
    let limits = backend.limits();
    if limits.max_threads_dim[1] == 0 || limits.max_threads_dim[2] == 0 {
        eprintln!("\n=== SKIP: test_transpose_896_151936_standalone ===");
        eprintln!("Reason: HIP driver reports invalid maxThreadsDim (axis limit 0): {:?}", limits.max_threads_dim);
        eprintln!("This is a known driver bug - transpose requires 2D blocks");
        eprintln!("=== END SKIP ===\n");
        return;
    }

    let hidden_size = 896usize;
    let vocab_size = 151936usize;
    let elem_count = hidden_size * vocab_size;

    eprintln!("\n=== test_transpose_896_151936_standalone ===");
    eprintln!("Using direct TransposeKernel invocation");

    // Create test data
    let input_data: Vec<f32> = (0..elem_count).map(|i| i as f32).collect();

    // Create input tensor
    let input_shape = TensorShape::from_dims(&[hidden_size, vocab_size]);
    let input_buffer = backend
        .allocate_buffer(elem_count * 4)
        .expect("Failed to allocate input buffer");
    backend
        .copy_to_device(&input_buffer, &input_data)
        .expect("Failed to copy input to device");
    let input_tensor = DeviceTensor {
        buffer: input_buffer,
        shape: input_shape,
    };

    // Use TransposeKernel directly (backend is already Arc<HipBackend>)
    use rocmforge::kernels::transpose::TransposeKernel;
    let mut kernel = TransposeKernel::new(backend.clone());

    match kernel.transpose(&input_tensor) {
        Ok(transposed) => {
            assert_eq!(
                transposed.shape().dims(),
                &[vocab_size, hidden_size],
                "Transposed shape should be [vocab_size, hidden_size]"
            );

            // Spot-check corners
            let mut output_data = vec![0.0f32; elem_count];
            backend
                .copy_from_device_safe(&transposed.buffer, &mut output_data)
                .expect("Failed to copy output from device");

            assert_eq!(
                output_data[0], input_data[0],
                "Top-left corner should match"
            );
            assert_eq!(
                output_data[elem_count - 1],
                input_data[elem_count - 1],
                "Bottom-right corner should match"
            );

            eprintln!("Standalone kernel invocation: SUCCESS");
        }
        Err(rocmforge::backend::hip_backend::HipError::KernelLoadFailed(msg)) => {
            eprintln!("\n=== SKIP: test_transpose_896_151936_standalone ===");
            eprintln!("Reason: Kernel load failed - {}", msg);
            eprintln!("=== END SKIP ===\n");
        }
        Err(e) => {
            panic!("Standalone transpose failed: {:?}", e);
        }
    }
}
