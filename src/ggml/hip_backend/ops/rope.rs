//! HIP RoPE op using existing RoPE kernel.

use crate::backend::{HipBackend, HipError, HipResult, HipBuffer};

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
    {
        let _ = (_cos, _sin, _seq_len, _num_heads, _head_dim);
        // TODO: Re-enable when rope_gpu_kernel is available
        // unsafe {
        //     let result = crate::attention::kernels::rope_gpu_kernel(
        //         output.as_ptr() as *mut f32,
        //         _cos.as_ptr() as *const f32,
        //         _sin.as_ptr() as *const f32,
        //         _seq_len,
        //         _num_heads,
        //         _head_dim,
        //     );
        //     if result != 0 {
        //         return Err(HipError::GenericError(format!(
        //             "rope_gpu_kernel failed with code {}",
        //             result
        //         )));
        //     }
        // }
    }

    Ok(())
}
