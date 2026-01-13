//! HIP mask op using existing causal mask kernel.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};

pub fn mask(
    backend: &HipBackend,
    scores: &HipBuffer,
    mask: &HipBuffer,
    output: &HipBuffer,
    batch_size: u32,
    seq_len: u32,
) -> HipResult<()> {
    let _ = backend;
    output.copy_from_buffer(scores)?;

    #[cfg(feature = "rocm")]
    unsafe {
        let result = crate::attention::kernels::mask_gpu_kernel(
            output.as_ptr() as *mut f32,
            mask.as_ptr() as *const f32,
            batch_size,
            seq_len,
        );
        if result != 0 {
            return Err(HipError::GenericError(format!(
                "mask_gpu_kernel failed with code {}",
                result
            )));
        }
    }

    Ok(())
}
