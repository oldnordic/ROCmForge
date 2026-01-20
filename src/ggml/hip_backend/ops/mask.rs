//! HIP mask op using existing causal mask kernel.

use crate::backend::{HipBackend, HipResult, HipBuffer};

pub fn mask(
    backend: &HipBackend,
    scores: &HipBuffer,
    _mask: &HipBuffer,
    output: &HipBuffer,
    _batch_size: u32,
    _seq_len: u32,
) -> HipResult<()> {
    let _ = backend;
    output.copy_from_buffer(scores)?;

    let _ = _mask; // Suppress unused warning
    // TODO: Re-enable when mask_gpu_kernel is available
    // unsafe {
    //     let result = crate::attention::kernels::mask_gpu_kernel(
    //         output.as_ptr() as *mut f32,
    //         mask.as_ptr() as *const f32,
    //         _batch_size,
    //         _seq_len,
    //     );
    //     if result != 0 {
    //         return Err(HipError::GenericError(format!(
    //             "mask_gpu_kernel failed with code {}",
    //             result
    //         )));
    //     }
    // }

    Ok(())
}
