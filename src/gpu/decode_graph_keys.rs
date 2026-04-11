//! Decode graph key construction and pointer-tag hashing.
//!
//! Keeping this separate from the execution path makes the forward runner
//! easier to reason about and reduces accidental coupling between graph
//! replay policy and kernel launch logic.

use super::cache::GpuKvCache;
use super::device::GpuDevice;
use super::error::GpuResult;
use super::forward::GpuLogitsMode;
use super::graph::{DecodeGraphKey, DecodeGraphScope};
use super::safety::{
    experimental_ffn_fastpath_enabled, experimental_gpu_kernels_enabled,
    experimental_q8_activation_fastpath_enabled,
};
use super::weights::GpuModelWeights;
use crate::config::ModelConfig;

pub(crate) fn gpu_greedy_logits_graph_key(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    config: &ModelConfig,
) -> DecodeGraphKey {
    DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        config,
        GpuLogitsMode::GreedyArgmax,
        gpu_weights.output_norm.as_ptr() as usize,
        gpu_weights.lm_head.as_ptr() as usize,
        gpu_weights.lm_head_meta.wtype,
        gpu_weights.lm_head_meta.role,
    )
    .with_feature_flags_tag(decode_graph_feature_flags_tag())
    .with_decode_scope(DecodeGraphScope::GreedyTail)
}

pub(crate) fn gpu_full_decode_graph_key(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &GpuKvCache,
    config: &ModelConfig,
) -> GpuResult<DecodeGraphKey> {
    Ok(DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        config,
        GpuLogitsMode::GreedyArgmax,
        gpu_weights.output_norm.as_ptr() as usize,
        gpu_weights.lm_head.as_ptr() as usize,
        gpu_weights.lm_head_meta.wtype,
        gpu_weights.lm_head_meta.role,
    )
    .with_feature_flags_tag(decode_graph_feature_flags_tag())
    .with_decode_scope(DecodeGraphScope::FullGreedyDecode)
    .with_layer_weights_binding_tag(gpu_model_weights_binding_tag(gpu_weights))
    .with_kv_binding_tag(gpu_kv_binding_tag(kv)?))
}

fn decode_graph_feature_flags_tag() -> u8 {
    let mut tag = 0u8;
    if experimental_gpu_kernels_enabled() {
        tag |= 1 << 0;
    }
    if experimental_q8_activation_fastpath_enabled() {
        tag |= 1 << 1;
    }
    if experimental_ffn_fastpath_enabled() {
        tag |= 1 << 2;
    }
    tag
}

fn gpu_model_weights_binding_tag(gpu_weights: &GpuModelWeights) -> u64 {
    gpu_weights.binding_tag()
}

fn gpu_kv_binding_tag(kv: &GpuKvCache) -> GpuResult<u64> {
    Ok(kv.binding_tag())
}
