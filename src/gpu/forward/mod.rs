//! Hybrid GPU decode forward path.
//!
//! This is the smallest end-to-end GPU runner for the current kernel set:
//! most layer math stays on HIP, while unsupported embedding and logits paths
//! may still use the existing CPU implementation.

mod decode;
mod embed;
mod layer;
mod logits;
mod utils;

pub use embed::gpu_embed_token_hybrid;
pub use layer::gpu_layer_forward_hybrid;
pub use utils::GpuLogitsMode;

use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::weights::CpuModelWeights;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ops::{gpu_dispatch_gemv_on_stream, gpu_dispatch_rms_norm};
use crate::gpu::weights::GpuModelWeights;

/// Full decode forward pass using GPU kernels plus targeted CPU fallbacks.
pub fn gpu_full_forward_hybrid(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
    logits_mode: GpuLogitsMode,
) -> GpuResult<Option<u32>> {
    if matches!(logits_mode, GpuLogitsMode::GreedyArgmax) {
        if let Some(token) =
            decode::gpu_try_full_greedy_decode_graph(device, gpu_weights, kv, scratch, pos, config)?
        {
            return Ok(Some(token));
        }
    }

    for layer_idx in 0..config.num_layers {
        layer::gpu_layer_forward_hybrid(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            pos,
            config,
        )?;

        // CRITICAL: Synchronize between layers to prevent buffer reuse race condition
        // All layers share the same scratch buffers (hidden, normed, q, k, v, etc.)
        // Without synchronization, layer N+1 can start writing to these buffers before
        // layer N's kernels finish reading from them, causing corruption.
        device.synchronize()?;
    }

    if matches!(logits_mode, GpuLogitsMode::Skip) {
        return Ok(None);
    }

    let h = config.hidden_size;
    let v = config.vocab_size;
    let gpu_result = match logits_mode {
        GpuLogitsMode::DownloadToHost => {
            let res = (|| -> GpuResult<()> {
                gpu_dispatch_rms_norm(
                    device,
                    scratch.hidden.as_ptr() as *const f32,
                    gpu_weights.output_norm.as_ptr() as *const f32,
                    scratch.normed.as_ptr() as *mut f32,
                    h,
                    config.rms_norm_eps,
                    device.stream(),
                )?;
                gpu_dispatch_gemv_on_stream(
                    device,
                    &gpu_weights.lm_head,
                    &gpu_weights.lm_head_meta,
                    scratch.normed.as_ptr() as *const f32,
                    scratch.logits.as_ptr() as *mut f32,
                    v,
                    h,
                    device.stream(),
                )?;
                utils::download_f32(&scratch.logits, &mut host_scratch.logits[..v])
            })();
            res.map(|_| None)
        }
        GpuLogitsMode::GreedyArgmax => {
            logits::gpu_try_greedy_decode_graph(device, gpu_weights, scratch, config).map(Some)
        }
        GpuLogitsMode::Skip => Ok(None),
    };

    match gpu_result {
        Ok(result) => Ok(result),
        Err(GpuError::InvalidWeightLayout { .. })
        | Err(GpuError::UnsupportedWeightType { .. })
        | Err(GpuError::UnsupportedOperation { .. }) => {
            gpu_dispatch_rms_norm(
                device,
                scratch.hidden.as_ptr() as *const f32,
                gpu_weights.output_norm.as_ptr() as *const f32,
                scratch.normed.as_ptr() as *mut f32,
                h,
                config.rms_norm_eps,
                device.stream(),
            )?;
            utils::cpu_fallback_gemv(
                "lm_head",
                &cpu_weights.lm_head,
                &cpu_weights.lm_head_meta,
                &scratch.normed,
                &mut host_scratch.normed,
                &mut host_scratch.logits,
                v,
                h,
                &mut host_scratch.q8_scratch,
            )?;
            match logits_mode {
                GpuLogitsMode::DownloadToHost => Ok(None),
                GpuLogitsMode::GreedyArgmax => Ok(Some(crate::cpu::sampler::cpu_sample_greedy(
                    &host_scratch.logits[..v],
                ))),
                GpuLogitsMode::Skip => Ok(None),
            }
        }
        Err(err) => Err(err),
    }
}
