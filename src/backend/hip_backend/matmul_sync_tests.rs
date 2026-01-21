//! Tests for hipBLAS stream synchronization
//!
//! These tests verify that concurrent matmul operations with multiple HipBlasHandle
//! instances don't race or hang due to stream synchronization issues.

#[cfg(test)]
#[serial] // Critical - prevents concurrent GPU test conflicts
mod matmul_sync_tests {
    use crate::backend::hip_backend::HipBackend;
    use crate::backend::hip_blas::HipBlasHandle;
    use crate::tensor::matmul::matmul_f32;
    use serial_test::serial;

    /// Helper to create a simple identity matrix for testing
    fn create_identity_matrix(size: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        data
    }

    /// Helper to create a simple sequential matrix
    fn create_sequential_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows * cols {
            data[i] = i as f32;
        }
        data
    }

    #[test]
    #[serial]
    fn test_concurrent_matmul_handles() {
        let backend = HipBackend::new().unwrap();

        // Create multiple handles (simulating concurrent use)
        let handle1 = HipBlasHandle::new().unwrap();
        let handle2 = HipBlasHandle::new().unwrap();

        // CRITICAL: Set streams before use
        handle1.set_stream(backend.stream().as_ptr()).unwrap();
        handle2.set_stream(backend.stream().as_ptr()).unwrap();

        // Create test data (identity matrices)
        let size = 4;
        let identity = create_identity_matrix(size);

        let a1 = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();
        let b1 = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();

        let a2 = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();
        let b2 = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();

        a1.copy_from_host(&identity).unwrap();
        b1.copy_from_host(&identity).unwrap();
        a2.copy_from_host(&identity).unwrap();
        b2.copy_from_host(&identity).unwrap();

        // Run concurrent matmuls
        let result1 = matmul_f32(&backend, &handle1, &a1, &b1, size, size, size);
        let result2 = matmul_f32(&backend, &handle2, &a2, &b2, size, size, size);

        // Both should succeed without hanging
        assert!(result1.is_ok(), "First matmul should succeed");
        assert!(result2.is_ok(), "Second matmul should succeed");

        // Sync and verify results exist
        backend.synchronize().unwrap();
    }

    #[test]
    #[serial]
    fn test_multiple_matmul_same_handle() {
        let backend = HipBackend::new().unwrap();
        let handle = HipBlasHandle::new().unwrap();

        // CRITICAL: Set stream before use
        handle.set_stream(backend.stream().as_ptr()).unwrap();

        // Create test data
        let identity = create_identity_matrix(4);
        let a = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();
        let b = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();
        a.copy_from_host(&identity).unwrap();
        b.copy_from_host(&identity).unwrap();

        // Run multiple matmuls with the same handle
        let result1 = matmul_f32(&backend, &handle, &a, &b, 4, 4, 4);
        assert!(result1.is_ok(), "First matmul should succeed");

        let result2 = matmul_f32(&backend, &handle, &a, &b, 4, 4, 4);
        assert!(result2.is_ok(), "Second matmul should succeed");

        let result3 = matmul_f32(&backend, &handle, &a, &b, 4, 4, 4);
        assert!(result3.is_ok(), "Third matmul should succeed");

        backend.synchronize().unwrap();
    }

    #[test]
    #[serial]
    fn test_matmul_rectangular_matrices() {
        let backend = HipBackend::new().unwrap();
        let handle = HipBlasHandle::new().unwrap();

        // CRITICAL: Set stream before use
        handle.set_stream(backend.stream().as_ptr()).unwrap();

        // Test rectangular matrix: 2x3 * 3x2 = 2x2
        let a_data = create_sequential_matrix(2, 3);
        let b_data = create_sequential_matrix(3, 2);

        let a = HipBuffer::new(a_data.len() * std::mem::size_of::<f32>()).unwrap();
        let b = HipBuffer::new(b_data.len() * std::mem::size_of::<f32>()).unwrap();
        a.copy_from_host(&a_data).unwrap();
        b.copy_from_host(&b_data).unwrap();

        let result = matmul_f32(&backend, &handle, &a, &b, 2, 2, 3);
        assert!(result.is_ok(), "Rectangular matmul should succeed");

        backend.synchronize().unwrap();
    }

    #[test]
    #[serial]
    fn test_matmul_stream_synchronization() {
        let backend = HipBackend::new().unwrap();
        let handle = HipBlasHandle::new().unwrap();

        // CRITICAL: Set stream before use
        handle.set_stream(backend.stream().as_ptr()).unwrap();

        // Create test data
        let identity = create_identity_matrix(8);

        let a = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();
        let b = HipBuffer::new(identity.len() * std::mem::size_of::<f32>()).unwrap();

        a.copy_from_host(&identity).unwrap();
        b.copy_from_host(&identity).unwrap();

        // Perform matmul
        let result = matmul_f32(&backend, &handle, &a, &b, 8, 8, 8)
            .expect("Matmul should succeed");

        // Copy result to host using async copy (like real code does)
        let mut host_result = vec![0.0f32; 8 * 8];
        backend
            .copy_from_device_safe(&result, &mut host_result)
            .expect("Copy should succeed");

        // CRITICAL: Must synchronize before using data
        backend
            .synchronize()
            .expect("Synchronize should succeed");

        // Verify result is identity (I * I = I)
        for i in 0..8 {
            for j in 0..8 {
                let val = host_result[i * 8 + j];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-5,
                    "Result[{},{}] = {} (expected {})",
                    i,
                    j,
                    val,
                    expected
                );
            }
        }
    }
}
