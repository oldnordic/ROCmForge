//! hipBLAS and matrix multiplication tests for ROCmForge
//! Tests GPU matmul against CPU reference implementation

use anyhow::Context;

// GPU test imports - only available when rocm feature is enabled
#[cfg(feature = "rocm")]
use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
#[cfg(feature = "rocm")]
use serial_test::serial;

use rocmforge::backend::hip_backend::{HipBackend, HipBuffer};
use rocmforge::backend::hip_blas::HipBlasHandle;
use rocmforge::tensor::matmul::{cpu_matmul_f32, matmul_f32};

/// Validate matrix multiplication dimensions
///
/// Returns Ok(()) if dimensions are valid for matmul C = A @ B
/// where A is (m, k) and B is (k, n), producing C (m, n)
fn validate_matmul_dims(
    m: i32,
    k: i32,
    n: i32,
    a_size: usize,
    b_size: usize,
    c_size: usize,
) -> anyhow::Result<()> {
    // Check A dimensions
    let expected_a_size = (m * k) as usize;
    if a_size != expected_a_size {
        return Err(anyhow::anyhow!(
            "Matrix A size mismatch: expected {} elements (m={} * k={}), got {}",
            expected_a_size, m, k, a_size
        ));
    }

    // Check B dimensions
    let expected_b_size = (k * n) as usize;
    if b_size != expected_b_size {
        return Err(anyhow::anyhow!(
            "Matrix B size mismatch: expected {} elements (k={} * n={}), got {}",
            expected_b_size, k, n, b_size
        ));
    }

    // Check C dimensions
    let expected_c_size = (m * n) as usize;
    if c_size != expected_c_size {
        return Err(anyhow::anyhow!(
            "Matrix C size mismatch: expected {} elements (m={} * n={}), got {}",
            expected_c_size, m, n, c_size
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // CPU-only tests (don't need GPU)
    #[test]
    fn test_hip_blas_handle_creation_and_drop() -> anyhow::Result<()> {
        // Test that we can create and destroy a hipBLAS handle
        let handle = HipBlasHandle::new()
            .context("Failed to create hipBLAS handle")?;

        assert!(!handle.as_ptr().is_null(), "Handle should not be null");

        // Handle should be destroyed when dropped
        Ok(())
    }

    // GPU tests - use shared fixture and run serially
    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn test_hipblas_sgemm_simple() -> anyhow::Result<()> {
        // Use shared GPU fixture to avoid creating multiple backends
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        let handle = HipBlasHandle::new()
            .context("Failed to create hipBLAS handle")?;

        // Create tiny 1x1 matrices
        let a = vec![2.0f32];
        let b = vec![3.0f32];

        let gpu_a = HipBuffer::new(1 * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix A")?;
        let gpu_b = HipBuffer::new(1 * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix B")?;

        gpu_a.copy_from_host(&a)
            .context("Failed to copy matrix A to GPU")?;
        gpu_b.copy_from_host(&b)
            .context("Failed to copy matrix B to GPU")?;

        // Validate dimensions before matmul
        let m = 1;
        let k = 1;
        let n = 1;
        validate_matmul_dims(m, k, n, a.len(), b.len(), 1)
            .context("Dimension validation failed")?;

        // Simple 1x1 * 1x1 = 1x1 matrix multiplication
        let gpu_c = matmul_f32(backend, &handle, &gpu_a, &gpu_b, m, k, n)
            .context("Simple 1x1 matmul failed")?;

        let mut host_result = vec![0.0f32; 1];
        backend.copy_from_device_safe(&gpu_c, &mut host_result)
            .context("Failed to copy result from GPU")?;

        assert!(
            (host_result[0] - 6.0).abs() < 1e-6,
            "1x1 matmul: 2*3=6, got {}",
            host_result[0]
        );

        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn test_hipblas_sgemm_simple() {
        eprintln!("SKIP: test_hipblas_sgemm_simple requires 'rocm' feature");
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn test_gpu_matmul_matches_cpu_small() -> anyhow::Result<()> {
        // Use shared GPU fixture to avoid creating multiple backends
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Test 2x2 * 2x2 case
        let m = 2;
        let n = 2;
        let k = 2;

        // Create simple deterministic matrices
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 row-major: [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 row-major: [[5,6],[7,8]]

        // Validate dimensions
        validate_matmul_dims(m, k, n, a.len(), b.len(), (m * n) as usize)
            .context("Dimension validation failed")?;

        // Expected result: [[19,22],[43,50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        // Compute CPU reference
        let cpu_result = cpu_matmul_f32(&a, &b, m as usize, n as usize, k as usize);

        // Verify CPU computation
        assert_eq!(cpu_result.len(), expected.len());
        for (i, &val) in cpu_result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "CPU matmul element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }

        // Test GPU matmul
        let handle = HipBlasHandle::new()
            .context("Failed to create hipBLAS handle")?;

        let gpu_a = HipBuffer::new((m * k) as usize * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix A")?;
        let gpu_b = HipBuffer::new((k * n) as usize * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix B")?;

        // Copy data to GPU
        gpu_a.copy_from_host(&a)
            .context("Failed to copy matrix A to GPU")?;
        gpu_b.copy_from_host(&b)
            .context("Failed to copy matrix B to GPU")?;

        // Perform GPU matmul
        let gpu_c = matmul_f32(backend, &handle, &gpu_a, &gpu_b, m as i32, n as i32, k as i32)
            .context("GPU matmul operation failed")?;

        // Copy result back from GPU
        let mut gpu_result = vec![0.0f32; (m * n) as usize];
        backend.copy_from_device_safe(&gpu_c, &mut gpu_result)
            .context("Failed to copy result from GPU")?;

        // Compare GPU vs CPU results
        assert_eq!(gpu_result.len(), expected.len());
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "GPU matmul element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }

        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn test_gpu_matmul_matches_cpu_small() {
        eprintln!("SKIP: test_gpu_matmul_matches_cpu_small requires 'rocm' feature");
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn test_gpu_matmul_larger_matrix() -> anyhow::Result<()> {
        // Use shared GPU fixture to avoid creating multiple backends
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Test 4x3 * 3x2 case
        let m = 4;
        let n = 2;
        let k = 3;

        // Create structured deterministic matrices
        let a = vec![
            1.0, 2.0, 3.0, // Row 0
            4.0, 5.0, 6.0, // Row 1
            7.0, 8.0, 9.0, // Row 2
            10.0, 11.0, 12.0, // Row 3
        ];

        let b = vec![
            13.0, 14.0, // Row 0
            15.0, 16.0, // Row 1
            17.0, 18.0, // Row 2
        ];

        // Compute CPU reference
        let cpu_result = cpu_matmul_f32(&a, &b, m as usize, n as usize, k as usize);

        // Verify dimensions
        assert_eq!(cpu_result.len(), (m * n) as usize);

        // Test GPU matmul
        let handle = HipBlasHandle::new()
            .context("Failed to create hipBLAS handle")?;

        let gpu_a = HipBuffer::new((m * k) as usize * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix A")?;
        let gpu_b = HipBuffer::new((k * n) as usize * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix B")?;

        // Copy data to GPU
        gpu_a.copy_from_host(&a)
            .context("Failed to copy matrix A to GPU")?;
        gpu_b.copy_from_host(&b)
            .context("Failed to copy matrix B to GPU")?;

        // Perform GPU matmul
        let gpu_c = matmul_f32(backend, &handle, &gpu_a, &gpu_b, m as i32, n as i32, k as i32)
            .context("GPU matmul operation failed")?;

        // Copy result back from GPU
        let mut gpu_result = vec![0.0f32; (m * n) as usize];
        backend.copy_from_device_safe(&gpu_c, &mut gpu_result)
            .context("Failed to copy result from GPU")?;

        // Compare GPU vs CPU results
        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (&cpu_val, &gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            assert!(
                (gpu_val - cpu_val).abs() < 1e-6,
                "GPU matmul element {} mismatch: expected {}, got {}",
                i,
                cpu_val,
                gpu_val
            );
        }

        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn test_gpu_matmul_larger_matrix() {
        eprintln!("SKIP: test_gpu_matmul_larger_matrix requires 'rocm' feature");
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn test_matmul_invalid_dims_error() -> anyhow::Result<()> {
        // Use shared GPU fixture to avoid creating multiple backends
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Test dimension mismatch: 2x3 * 4x2 (k=3 vs k'=4)
        let m = 2;
        let n = 2;
        let k = 3;
        let k_prime = 4; // Mismatched inner dimension

        let handle = HipBlasHandle::new()
            .context("Failed to create hipBLAS handle")?;

        let gpu_a = HipBuffer::new((m * k) as usize * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix A")?;
        let gpu_b = HipBuffer::new((k_prime * n) as usize * std::mem::size_of::<f32>())
            .context("Failed to allocate GPU buffer for matrix B")?;

        // Validate dimensions - should detect mismatch
        let validation_result = validate_matmul_dims(
            m as i32,
            k as i32,
            n as i32,
            (m * k) as usize,
            (k_prime * n) as usize,
            (m * n) as usize,
        );

        assert!(
            validation_result.is_err(),
            "Dimension validation should detect k dimension mismatch: k={} vs k'={}",
            k,
            k_prime
        );

        // This should fail due to dimension mismatch
        let result = matmul_f32(backend, &handle, &gpu_a, &gpu_b, m as i32, n as i32, k as i32);

        // Expect error due to dimension mismatch
        match result {
            Ok(_) => panic!("Expected dimension mismatch error"),
            Err(_) => (), // Expected
        }

        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn test_matmul_invalid_dims_error() {
        eprintln!("SKIP: test_matmul_invalid_dims_error requires 'rocm' feature");
    }

    /// TDD TEST: GGML matmul wrapper with synchronization
    ///
    /// This test verifies that the GGML matmul wrapper properly synchronizes
    /// between hipBLAS operations (on custom stream) and hipMemcpy (on default stream).
    ///
    /// BUG: Without synchronization, copy_from_buffer may read incomplete data.
    /// FIX: Add backend.synchronize() after matmul_f32 before copy_from_buffer.
    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn test_ggml_matmul_wrapper_synchronization() {
        use rocmforge::backend::HipBackend;
        use rocmforge::ggml::hip_backend::ops::matmul::matmul as ggml_matmul;

        // Use shared GPU fixture to avoid creating multiple backends
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");
        let backend = fixture.backend();

        // Test 2x2 * 2x2 case
        let m = 2;
        let n = 2;
        let k = 2;

        // Create test matrices
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 row-major: [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 row-major: [[5,6],[7,8]]

        // Expected result: [[19,22],[43,50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        // Allocate GPU buffers
        let gpu_a = HipBuffer::new((m * k) as usize * std::mem::size_of::<f32>())
            .expect("Failed to allocate GPU buffer for A");
        let gpu_b = HipBuffer::new((k * n) as usize * std::mem::size_of::<f32>())
            .expect("Failed to allocate GPU buffer for B");
        let gpu_output = HipBuffer::new((m * n) as usize * std::mem::size_of::<f32>())
            .expect("Failed to allocate GPU buffer for output");

        // Copy input data to GPU
        // Explicitly type the slices to ensure T = f32
        let a_slice: &[f32] = &a;
        let b_slice: &[f32] = &b;
        gpu_a.copy_from_host(a_slice).expect("Failed to copy A to GPU");
        gpu_b.copy_from_host(b_slice).expect("Failed to copy B to GPU");

        // Perform GGML matmul (this should synchronize properly)
        ggml_matmul(backend, &gpu_a, &gpu_b, m as i32, n as i32, k as i32, &gpu_output)
            .expect("GGML matmul failed");

        // Copy result back from GPU
        let mut result = vec![0.0f32; (m * n) as usize];
        backend.copy_from_device_safe(&gpu_output, &mut result).expect("Failed to copy result from GPU");

        // Verify result matches expected
        assert_eq!(result.len(), expected.len(), "Result length mismatch");
        for (i, (&actual, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-4,
                "Element {} mismatch: expected {}, got {}",
                i,
                exp,
                actual
            );
        }
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn test_ggml_matmul_wrapper_synchronization() {
        eprintln!("SKIP: test_ggml_matmul_wrapper_synchronization requires 'rocm' feature");
    }
}
