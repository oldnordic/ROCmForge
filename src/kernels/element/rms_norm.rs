//! HIP RMSNorm kernel wrapper
//!
//! Provides RMSNorm operation using the GPU kernel from mlp::kernels.
//! Computes: output[row, j] = input[row, j] / sqrt(mean(input[row, :]^2) + eps) * weight[j]

use crate::backend::{HipBackend, HipBuffer, HipResult};

/// RMSNorm operation wrapper
///
/// Currently performs a CPU copy as fallback. GPU kernel integration
/// is planned when the rms_norm_gpu_kernel is fully available.
///
/// # Arguments
/// * `_backend` - HIP backend (unused in fallback mode)
/// * `input` - Input tensor buffer [seq_len, hidden_size]
/// * `_weight` - Weight tensor buffer [hidden_size] (unused in fallback)
/// * `output` - Output tensor buffer [seq_len, hidden_size]
/// * `_seq_len` - Sequence length (rows) (unused in fallback)
/// * `_hidden_size` - Hidden size (columns) (unused in fallback)
/// * `_eps` - Epsilon for numerical stability (unused in fallback)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(HipError)` on buffer copy failure
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
    Ok(())
}
