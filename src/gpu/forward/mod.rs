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
use crate::gpu::ops::{
    gpu_dispatch_gemv_on_stream, gpu_dispatch_mpo_apply_on_stream, gpu_dispatch_rms_norm,
    gpu_dispatch_sparse_csr_gemv_on_stream,
};
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
            // Sync/scatter the newly written token at `pos` to the paged cache for all layers
            for layer_idx in 0..config.num_layers {
                kv.scatter_to_paged(layer_idx, pos, 1)?;
            }
            return Ok(Some(token));
        }
    }

    for layer_idx in 0..config.num_layers {
        layer::gpu_layer_forward_hybrid(
            device,
            gpu_weights.layer(layer_idx),
            Some(cpu_weights.layer(layer_idx)),
            kv,
            scratch,
            Some(host_scratch),
            layer_idx,
            pos,
            config,
        )?;

        // CRITICAL: Synchronize between layers to prevent buffer reuse race condition
        // All layers share the same scratch buffers (hidden, normed, q, k, v, etc.)
        // Without synchronization, layer N+1 can start writing to these buffers before
        // layer N's kernels finish reading from them, causing corruption.
        device.synchronize()?;

        let mut check_hidden = vec![0.0f32; config.hidden_size];
        utils::download_f32(&scratch.hidden, &mut check_hidden)?;
        for (idx, &val) in check_hidden.iter().enumerate() {
            if val.is_nan() {
                return Err(GpuError::InvalidOperation {
                    message: format!(
                        "NaN detected in scratch.hidden after layer {} at index {}",
                        layer_idx, idx
                    ),
                });
            }
        }

        // Scatter the newly written token at `pos` to the paged cache
        kv.scatter_to_paged(layer_idx, pos, 1)?;
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
                let mut check_normed = vec![0.0f32; h];
                utils::download_f32(&scratch.normed, &mut check_normed)?;
                for (idx, &val) in check_normed.iter().enumerate() {
                    if val.is_nan() {
                        return Err(GpuError::InvalidOperation {
                            message: format!(
                                "NaN detected in scratch.normed (after output norm) at index {}",
                                idx
                            ),
                        });
                    }
                }
                if let Some(dense) = gpu_weights.lm_head.as_dense() {
                    gpu_dispatch_gemv_on_stream(
                        device,
                        dense,
                        &gpu_weights.lm_head_meta,
                        scratch.normed.as_ptr() as *const f32,
                        scratch.logits.as_ptr() as *mut f32,
                        v,
                        h,
                        device.stream(),
                    )?;
                } else if let Some(sparse) = gpu_weights.lm_head.as_sparse_csr() {
                    gpu_dispatch_sparse_csr_gemv_on_stream(
                        device,
                        sparse,
                        scratch.normed.as_ptr() as *const f32,
                        scratch.logits.as_ptr() as *mut f32,
                        v,
                        h,
                        device.stream(),
                    )?;
                } else if let Some(mpo) = gpu_weights.lm_head.as_mpo() {
                    gpu_dispatch_mpo_apply_on_stream(
                        device,
                        mpo,
                        scratch.normed.as_ptr() as *const f32,
                        scratch.logits.as_ptr() as *mut f32,
                        v,
                        h,
                        device.stream(),
                    )?;
                } else {
                    return Err(GpuError::InvalidWeightLayout {
                        tensor: "lm_head".to_string(),
                        dims: gpu_weights.lm_head_meta.dims.clone(),
                        reason: "LM head is neither dense, sparse CSR, nor MPO".to_string(),
                    });
                }
                utils::download_f32(&scratch.logits, &mut host_scratch.logits[..v])?;
                for (idx, &val) in host_scratch.logits[..v].iter().enumerate() {
                    if val.is_nan() {
                        return Err(GpuError::InvalidOperation {
                            message: format!(
                                "NaN detected in logits (after lm_head GEMV) at index {}",
                                idx
                            ),
                        });
                    }
                }
                Ok(())
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
