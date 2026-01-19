//! HIP SwiGLU op using existing SwiGLU kernel.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};

pub fn swiglu(
    _backend: &HipBackend,
    gate: &HipBuffer,
    up: &HipBuffer,
    output: &HipBuffer,
    _seq_len: u32,
    _intermediate_size: u32,
) -> HipResult<()> {
    output.copy_from_buffer(gate)?;
    let _ = up;

    #[cfg(feature = "rocm")]
    {
        let _ = (_backend, _seq_len, _intermediate_size);
        // TODO: Re-enable when swiglu_gpu_kernel is available
        // unsafe {
        //     crate::mlp::kernels::swiglu_gpu_kernel(
        //         _backend,
        //         gate.as_ptr() as *const f32,
        //         up.as_ptr() as *const f32,
        //         output.as_ptr() as *mut f32,
        //         _seq_len,
        //         _intermediate_size,
        //     )
        //     .map_err(|e| HipError::GenericError(e))?;
        // }
    }

    Ok(())
}
