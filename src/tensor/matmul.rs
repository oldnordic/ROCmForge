//! Matrix multiplication operations for ROCmForge
//! Provides GPU-accelerated matrix multiplication using hipBLAS

use crate::backend::{
    hip_backend::{HipBuffer, HipError},
    hip_blas::{sgemm, HipBlasError, HipBlasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T},
};
use thiserror::Error;

/// Matrix multiplication error types
#[derive(Error, Debug)]
pub enum MatmulError {
    #[error("hipBLAS error: {0}")]
    HipBlasError(#[from] HipBlasError),
    #[error("HIP error: {0}")]
    HipError(#[from] HipError),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Buffer size mismatch: {0}")]
    BufferSizeError(String),
}

pub type MatmulResult<T> = Result<T, MatmulError>;

/// GPU matrix multiplication: C = A * B
///
/// Performs matrix multiplication using hipBLAS SGEMM (single precision).
/// All matrices are stored in row-major order.
///
/// # Stream Synchronization
///
/// CRITICAL: This function requires that `handle` be associated with the backend's stream
/// via `set_stream()` before calling. The handle is passed in from outside, so this
/// function CANNOT automatically set the stream. Callers MUST ensure:
///
/// ```rust
/// handle.set_stream(backend.stream().as_ptr())?;
/// matmul_f32(&backend, &handle, &a, &b, m, n, k)?;
/// ```
///
/// See docs/STREAM_SYNCHRONIZATION.md for details on why this is critical.
///
/// Arguments:
/// - backend: HIP backend (provides stream access)
/// - handle: hipBLAS handle for operations (MUST have set_stream called)
/// - a: GPU buffer containing matrix A (m×k)
/// - b: GPU buffer containing matrix B (k×n)
/// - m: number of rows in A and C
/// - n: number of columns in B and C
/// - k: number of columns in A and rows in B
///
/// Returns:
/// - GPU buffer containing matrix C (m×n)
///
/// # Panics
///
/// This function will panic if the handle is not properly configured with the backend's stream,
/// as hipBLAS operations may occur on the wrong stream, causing synchronization issues.
pub fn matmul_f32(
    backend: &crate::backend::HipBackend,
    handle: &HipBlasHandle,
    a: &HipBuffer,
    b: &HipBuffer,
    m: i32,
    n: i32,
    k: i32,
) -> MatmulResult<HipBuffer> {
    // Validate dimensions
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(MatmulError::DimensionMismatch(format!(
            "Invalid dimensions: m={}, n={}, k={} (all must be positive)",
            m, n, k
        )));
    }

    // Additional validation: ensure dimensions don't exceed reasonable limits
    if m > 10000 || n > 10000 || k > 10000 {
        return Err(MatmulError::DimensionMismatch(format!(
            "Dimensions too large: m={}, n={}, k={} (max 10000)",
            m, n, k
        )));
    }

    // Validate buffer sizes match expected dimensions
    // For matrix multiplication A (m×k) * B (k×n) = C (m×n)
    // Buffer A should have m*k elements, Buffer B should have k*n elements
    let expected_a_size = (m * k) as usize;
    let expected_b_size = (k * n) as usize;

    let actual_a_size = a.size() / std::mem::size_of::<f32>();
    let actual_b_size = b.size() / std::mem::size_of::<f32>();

    if actual_a_size != expected_a_size {
        return Err(MatmulError::BufferSizeError(format!(
            "Buffer A size mismatch: expected {} elements, have {}",
            expected_a_size, actual_a_size
        )));
    }

    if actual_b_size != expected_b_size {
        return Err(MatmulError::BufferSizeError(format!(
            "Buffer B size mismatch: expected {} elements, have {}",
            expected_b_size, actual_b_size
        )));
    }

    // Check buffer sizes (in bytes)
    let a_size = (m * k) as usize * std::mem::size_of::<f32>();
    let b_size = (k * n) as usize * std::mem::size_of::<f32>();
    let c_size = (m * n) as usize * std::mem::size_of::<f32>();

    if a.size() < a_size {
        return Err(MatmulError::BufferSizeError(format!(
            "Buffer A too small: need {} bytes, have {}",
            a_size,
            a.size()
        )));
    }

    if b.size() < b_size {
        return Err(MatmulError::BufferSizeError(format!(
            "Buffer B too small: need {} bytes, have {}",
            b_size,
            b.size()
        )));
    }

    // Allocate output buffer C
    let c = HipBuffer::new(c_size)?;

    // For hipBLAS, we need to handle both square and rectangular matrices correctly.
    // The key insight: hipBLAS expects column-major data.

    if m == n && n == k {
        // For square matrices, use the transpose approach that works
        sgemm(
            handle,
            HIPBLAS_OP_T, // Transpose A (treat as A^T)
            HIPBLAS_OP_T, // Transpose B (treat as B^T)
            n,            // Result dimensions: C^T is n×m
            m,
            k,
            1.0,                      // alpha = 1.0
            a.as_ptr() as *const f32, // A comes first in A^T * B^T
            m,                        // lda = m (rows of original A = columns of A^T)
            b.as_ptr() as *const f32, // B comes second in A^T * B^T
            k,                        // ldb = k (columns of original B = rows of B^T)
            0.0,                      // beta = 0.0
            c.as_ptr() as *mut f32,
            m, // ldc = m (rows of original A = columns of C^T)
        )?;
    } else {
        // For rectangular matrices, pre-transpose inputs to column-major format
        // Transpose A (m×k row-major to k×m column-major)
        let mut a_col_major = vec![0.0f32; (m * k) as usize];
        let mut a_host = vec![0.0f32; (m * k) as usize];
        backend.copy_from_device_safe(a, &mut a_host)?;
        for i in 0..m as usize {
            for j in 0..k as usize {
                a_col_major[j * m as usize + i] = a_host[i * k as usize + j];
            }
        }

        // Transpose B (k×n row-major to n×k column-major)
        let mut b_col_major = vec![0.0f32; (k * n) as usize];
        let mut b_host = vec![0.0f32; (k * n) as usize];
        backend.copy_from_device_safe(b, &mut b_host)?;
        for i in 0..k as usize {
            for j in 0..n as usize {
                b_col_major[j * k as usize + i] = b_host[i * n as usize + j];
            }
        }

        // Copy transposed data to GPU
        let gpu_a_col = HipBuffer::new(a_col_major.len() * std::mem::size_of::<f32>())?;
        let gpu_b_col = HipBuffer::new(b_col_major.len() * std::mem::size_of::<f32>())?;

        gpu_a_col.copy_from_host(&a_col_major)?;
        gpu_b_col.copy_from_host(&b_col_major)?;

        // Use standard hipBLAS with column-major data
        sgemm(
            handle,
            HIPBLAS_OP_N, // No transpose - data is already column-major
            HIPBLAS_OP_N, // No transpose - data is already column-major
            m,            // m (rows of A/C)
            n,            // n (cols of B/C)
            k,            // k (cols of A / rows of B)
            1.0,
            gpu_a_col.as_ptr() as *const f32,
            m, // lda = m (for column-major)
            gpu_b_col.as_ptr() as *const f32,
            k, // ldb = k (for column-major)
            0.0,
            c.as_ptr() as *mut f32,
            m, // ldc = m (for column-major)
        )?;
    }

    // The result is in column-major order. We need to transpose it to row-major.
    let mut host_col_major = vec![0.0f32; (m * n) as usize];
    backend.copy_from_device_safe(&c, &mut host_col_major)?;

    let mut host_row_major = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            host_row_major[i * n as usize + j] = host_col_major[j * m as usize + i];
        }
    }

    let c_row_major = HipBuffer::new(host_row_major.len() * std::mem::size_of::<f32>())?;
    c_row_major.copy_from_host(&host_row_major)?;

    Ok(c_row_major)
}

/// CPU reference matrix multiplication for testing
/// Simple O(m*n*k) implementation for validation
pub fn cpu_matmul_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_cpu_matmul_simple() {
        // Test 2x2 * 2x2 case
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]

        let result = cpu_matmul_f32(&a, &b, 2, 2, 2);
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // [[19,22],[43,50]]

        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "Element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    #[serial]
    fn test_cpu_matmul_rectangular() {
        // Test 2x3 * 3x2 case
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [[7,8],[9,10],[11,12]]

        let result = cpu_matmul_f32(&a, &b, 2, 2, 3);
        let expected = vec![58.0, 64.0, 139.0, 154.0]; // [[58,64],[139,154]]

        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "Element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    #[serial]
    fn test_matmul_dimension_validation() {
        let backend = crate::backend::HipBackend::new().unwrap();
        let handle = HipBlasHandle::new().unwrap();
        let a = HipBuffer::new(4).unwrap();
        let b = HipBuffer::new(4).unwrap();

        // Test invalid dimensions
        let result = matmul_f32(&backend, &handle, &a, &b, 0, 2, 2);
        assert!(result.is_err(), "m=0 should fail");

        let result = matmul_f32(&backend, &handle, &a, &b, 2, 0, 2);
        assert!(result.is_err(), "n=0 should fail");

        let result = matmul_f32(&backend, &handle, &a, &b, 2, 2, 0);
        assert!(result.is_err(), "k=0 should fail");
    }
}
