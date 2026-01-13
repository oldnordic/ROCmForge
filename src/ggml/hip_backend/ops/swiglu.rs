//! HIP SwiGLU op using existing SwiGLU kernel.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};

pub fn swiglu(
    backend: &HipBackend,
    gate: &HipBuffer,
    up: &HipBuffer,
    output: &HipBuffer,
    seq_len: u32,
    intermediate_size: u32,
) -> HipResult<()> {
    #[cfg(feature = "rocm")]
    unsafe {
        crate::mlp::kernels::swiglu_gpu_kernel(
            backend,
            gate.as_ptr() as *const f32,
            up.as_ptr() as *const f32,
            output.as_ptr() as *mut f32,
            seq_len,
            intermediate_size,
        )
        .map_err(|e| HipError::GenericError(e))?;
    }

    Ok(())
}
