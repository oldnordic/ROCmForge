//! Hybrid GPU decode forward path.
//!
//! This is the smallest end-to-end GPU runner for the current kernel set:
//! most layer math stays on HIP, while unsupported embedding and logits paths
//! may still use the existing CPU implementation.

mod decode;
mod embed;
mod layer;
mod logits;
mod ple;
mod utils;

pub use embed::gpu_embed_token_hybrid;
pub use layer::gpu_layer_forward_hybrid;
pub use ple::gpu_compute_ple_inputs_on_stream;
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
    token_id: u32,
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

    // Run all layers back-to-back on the same HIP stream.  Kernels and the
    // per-layer paged-cache scatter are asynchronous, so the CPU queues the
    // entire token instead of waiting after each layer.
    for layer_idx in 0..config.num_layers {
        // Extract shared PLE buffers from model-level weights for Gemma4
        let shared_ple_token_emb = gpu_weights.per_layer_token_emb.as_ref();
        let shared_ple_model_proj = gpu_weights.per_layer_model_proj.as_ref().and_then(|t| t.as_dense());
        let shared_ple_proj_norm = gpu_weights.per_layer_proj_norm.as_ref();

        layer::gpu_layer_forward_hybrid(
            device,
            gpu_weights.layer(layer_idx),
            Some(cpu_weights.layer(layer_idx)),
            kv,
            scratch,
            Some(host_scratch),
            layer_idx,
            pos,
            token_id,
            config,
            shared_ple_token_emb,
            shared_ple_model_proj,
            shared_ple_proj_norm,
        )?;

        // Move the just-written token from the contiguous working view to the
        // paged KV cache on the same stream; no host synchronization here.
        kv.scatter_to_paged_on_stream(layer_idx, pos, 1, device.stream())?;
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
