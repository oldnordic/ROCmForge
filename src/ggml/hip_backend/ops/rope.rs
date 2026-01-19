//! HIP RoPE op using existing RoPE kernel.

use crate::backend::{HipBackend, HipResult, HipBuffer};

pub fn rope(
    _backend: &HipBackend,
    input: &HipBuffer,
    _cos: &HipBuffer,
    _sin: &HipBuffer,
    output: &HipBuffer,
    _seq_len: u32,
    _num_heads: u32,
    _head_dim: u32,
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
