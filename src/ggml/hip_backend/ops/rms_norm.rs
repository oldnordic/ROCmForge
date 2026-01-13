//! HIP RMSNorm op using existing RMSNorm kernel.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};

pub fn rms_norm(
    backend: &HipBackend,
    input: &HipBuffer,
    weight: &HipBuffer,
    output: &HipBuffer,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
) -> HipResult<()> {
    output.copy_from_buffer(input)?;

    #[cfg(feature = "rocm")]
    unsafe {
        crate::mlp::kernels::rms_norm_gpu_kernel(
            backend,
            output.as_ptr() as *const f32,
            weight.as_ptr() as *const f32,
            output.as_ptr() as *mut f32,
            seq_len,
            hidden_size,
            eps,
        )
        .map_err(|e| HipError::GenericError(e))?;
    }

    Ok(())
}
