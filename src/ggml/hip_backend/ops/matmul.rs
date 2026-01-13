//! HIP matmul implementation using hipBLAS helper.

use crate::backend::{hip_blas::HipBlasHandle, HipBackend, HipError, HipResult, HipBuffer};
use crate::tensor::matmul::matmul_f32;

pub fn matmul(
    backend: &HipBackend,
    a: &HipBuffer,
    b: &HipBuffer,
    m: i32,
    n: i32,
    k: i32,
    output: &HipBuffer,
) -> HipResult<()> {
    let handle = HipBlasHandle::new()
        .map_err(|e| HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e)))?;
    handle
        .set_stream(backend.stream().as_ptr())
        .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

    let result = matmul_f32(&handle, a, b, m, n, k)
        .map_err(|e| HipError::GenericError(format!("matmul_f32 failed: {}", e)))?;
    output.copy_from_buffer(&result)
}
