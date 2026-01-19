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
    eprintln!(">>> matmul_wrapper: ENTRY m={} n={} k={}", m, n, k);

    let handle = HipBlasHandle::new()
        .map_err(|e| HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e)))?;
    eprintln!(">>> matmul_wrapper: hipBLAS handle created");

    handle
        .set_stream(backend.stream().as_ptr())
        .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;
    eprintln!(">>> matmul_wrapper: hipBLAS stream set");

    let result = matmul_f32(backend, &handle, a, b, m, n, k)
        .map_err(|e| HipError::GenericError(format!("matmul_f32 failed: {}", e)))?;
    eprintln!(">>> matmul_wrapper: matmul_f32 complete");

    // PHASE 01 FIX: Use stream-aware copy to avoid synchronization issues
    //
    // BUG: matmul_f32 queues hipBLAS operations on backend.stream() (custom stream).
    //      copy_from_buffer uses hipMemcpy on default stream (NULL stream).
    //      Without synchronization, memcpy may read incomplete data → HANG.
    //
    // FIX: Use copy_from_buffer_with_stream with backend.stream(), then synchronize.
    //      This ensures the copy is properly ordered with hipBLAS operations.
    eprintln!(">>> matmul_wrapper: About to copy from buffer using stream-aware copy...");
    output.copy_from_buffer_with_stream(&result, backend.stream().as_ptr())
        .map_err(|e| HipError::GenericError(format!("Failed to copy result buffer: {}", e)))?;
    eprintln!(">>> matmul_wrapper: Stream-aware copy complete");

    eprintln!(">>> matmul_wrapper: About to synchronize backend...");
    backend.synchronize()
        .map_err(|e| HipError::GenericError(format!("Failed to synchronize after matmul: {}", e)))?;
    eprintln!(">>> matmul_wrapper: Backend synchronization complete, EXIT");

    Ok(())
}
