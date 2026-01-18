//! HIP add/scale ops using hipBLAS.

use crate::backend::{hip_blas, hip_blas::HipBlasHandle, HipBackend, HipError, HipResult, HipBuffer};

fn set_stream(handle: &HipBlasHandle, backend: &HipBackend) -> HipResult<()> {
    handle
        .set_stream(backend.stream().as_ptr())
        .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))
}

pub fn add(
    backend: &HipBackend,
    a: &HipBuffer,
    b: &HipBuffer,
    output: &HipBuffer,
) -> HipResult<()> {
    if a.size() != b.size() || a.size() != output.size() {
        return Err(HipError::GenericError(
            "Add requires equal-sized buffers".to_string(),
        ));
    }

    // output = a
    // Use stream-aware copy for consistency with backend.stream()
    output.copy_from_buffer_with_stream(a, backend.stream().as_ptr())?;

    // output += b
    let handle = HipBlasHandle::new()
        .map_err(|e| HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e)))?;
    set_stream(&handle, backend)?;

    let n = (output.size() / std::mem::size_of::<f32>()) as i32;
    hip_blas::saxpy(
        &handle,
        n,
        1.0f32,
        b.as_ptr() as *const f32,
        1,
        output.as_ptr() as *mut f32,
        1,
    )
    .map_err(|e| HipError::GenericError(format!("hipBLAS saxpy failed: {}", e)))?;
    Ok(())
}

pub fn scale(
    backend: &HipBackend,
    input: &HipBuffer,
    factor: f32,
    output: &HipBuffer,
) -> HipResult<()> {
    if input.size() != output.size() {
        return Err(HipError::GenericError(
            "Scale requires equal-sized buffers".to_string(),
        ));
    }

    // Use stream-aware copy for consistency with backend.stream()
    output.copy_from_buffer_with_stream(input, backend.stream().as_ptr())?;

    let handle = HipBlasHandle::new()
        .map_err(|e| HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e)))?;
    set_stream(&handle, backend)?;

    let n = (output.size() / std::mem::size_of::<f32>()) as i32;
    hip_blas::sscal(
        &handle,
        n,
        factor,
        output.as_ptr() as *mut f32,
        1,
    )
    .map_err(|e| HipError::GenericError(format!("hipBLAS sscal failed: {}", e)))?;
    Ok(())
}
