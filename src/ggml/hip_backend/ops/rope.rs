//! HIP RoPE op using existing RoPE kernel.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};

pub fn rope(
    backend: &HipBackend,
    input: &HipBuffer,
    cos: &HipBuffer,
    sin: &HipBuffer,
    output: &HipBuffer,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> HipResult<()> {
    output.copy_from_buffer(input)?;

    #[cfg(feature = "rocm")]
    unsafe {
        let result = crate::attention::kernels::rope_gpu_kernel(
            output.as_ptr() as *mut f32,
            cos.as_ptr() as *const f32,
            sin.as_ptr() as *const f32,
            seq_len,
            num_heads,
            head_dim,
        );
        if result != 0 {
            return Err(HipError::GenericError(format!(
                "rope_gpu_kernel failed with code {}",
                result
            )));
        }
    }

    Ok(())
}
