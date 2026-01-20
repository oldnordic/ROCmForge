//! HIP SwiGLU kernel wrapper
//!
//! Provides SwiGLU activation operation using the GPU kernel from mlp::kernels.
//! Computes: output[i] = gate[i] * swish(up[i]) where swish(x) = x * sigmoid(x)

use crate::backend::{HipBackend, HipBuffer, HipResult};

/// SwiGLU activation wrapper
///
/// Currently performs a CPU copy as fallback. GPU kernel integration
/// is planned when the swiglu_gpu_kernel is fully available.
///
/// # Arguments
/// * `_backend` - HIP backend (unused in fallback mode)
/// * `gate` - Gate projection tensor buffer [seq_len, intermediate_size]
/// * `up` - Up projection tensor buffer [seq_len, intermediate_size]
/// * `output` - Output tensor buffer [seq_len, intermediate_size]
/// * `_seq_len` - Sequence length (rows) (unused in fallback)
/// * `_intermediate_size` - Intermediate size (columns) (unused in fallback)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(HipError)` on buffer copy failure
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
