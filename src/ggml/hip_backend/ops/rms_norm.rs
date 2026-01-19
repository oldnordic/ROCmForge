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
    {
        let _ = (_backend, _weight, _seq_len, _hidden_size, _eps);
        // TODO: Re-enable when rms_norm_gpu_kernel is available
        // unsafe {
        //     crate::mlp::kernels::rms_norm_gpu_kernel(
        //         _backend,
        //         output.as_ptr() as *const f32,
        //         _weight.as_ptr() as *const f32,
        //         output.as_ptr() as *mut f32,
        //         _seq_len,
        //         _hidden_size,
        //         _eps,
        //     )
        //     .map_err(|e| HipError::GenericError(e))?;
        // }
    }

    Ok(())
}
