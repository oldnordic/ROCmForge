use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::forward::cpu_embed_token;
use crate::cpu::weights::CpuModelWeights;
use crate::gpu::cache::GpuForwardScratch;
use crate::gpu::device::GpuDevice;
use crate::gpu::error::GpuResult;
use crate::gpu::ffi;
use crate::gpu::kernels::elementwise::embed_q8_0_token;
use crate::gpu::weights::GpuModelWeights;
use crate::loader::GgmlType;

use super::utils::validate_token_embedding_layout;

/// Embed one token, preferring a native GPU path and falling back to CPU upload.
pub fn gpu_embed_token_hybrid(
    device: &GpuDevice,
    token_id: u32,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    match gpu_weights.token_emb_meta.wtype {
        GgmlType::Q8_0 => {
            validate_token_embedding_layout(&gpu_weights.token_emb_meta, config)?;
            embed_q8_0_token(
                gpu_weights.token_emb.as_ptr(),
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                token_id,
            )
        }
        _ => {
            // CPU embed into pinned buffer
            cpu_embed_token(
                token_id,
                cpu_weights,
                &mut scratch.input_hidden_pinned.as_slice_mut::<f32>()[..h],
                config,
            );
            // Async upload
            unsafe {
                ffi::hip_memcpy_h2d_async(
                    scratch.hidden.as_ptr(),
                    scratch.input_hidden_pinned.as_ptr(),
                    h * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }
            Ok(())
        }
    }
}
