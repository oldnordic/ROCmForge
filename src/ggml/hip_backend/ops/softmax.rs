//! HIP softmax op using existing softmax kernel.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};

pub fn softmax(
    backend: &HipBackend,
    input: &HipBuffer,
    output: &HipBuffer,
    batch_size: u32,
    seq_len: u32,
) -> HipResult<()> {
    let _ = backend;
    output.copy_from_buffer(input)?;

    #[cfg(feature = "rocm")]
    unsafe {
        let result = crate::attention::kernels::softmax_gpu_kernel(
            output.as_ptr() as *mut f32,
            batch_size,
            seq_len,
        );
        if result != 0 {
            return Err(HipError::GenericError(format!(
                "softmax_gpu_kernel failed with code {}",
                result
            )));
        }
    }

    Ok(())
}
