//! HIP RMSNorm op using existing RMSNorm kernel.

use crate::backend::{HipBackend, HipResult, HipBuffer};

pub fn rms_norm(
    _backend: &HipBackend,
    input: &HipBuffer,
    _weight: &HipBuffer,
    output: &HipBuffer,
    _seq_len: u32,
    _hidden_size: u32,
    _eps: f32,
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
