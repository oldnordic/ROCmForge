//! HIP graph helpers for repeated decode work.
//!
//! This module keeps raw graph handles behind RAII wrappers and provides a
//! topology key that later decode replay can use for safe cache invalidation.

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi;
use super::forward::GpuLogitsMode;
use super::weights::TensorRole;
use crate::config::ModelConfig;
use crate::loader::GgmlType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeGraphScope {
    GreedyTail,
    FullGreedyDecode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeGraphKey {
    device_id: i32,
    warp_size: usize,
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rope_neox: bool,
    use_attention_bias: bool,
    logits_mode_tag: u8,
    scope_tag: u8,
    output_norm_ptr: usize,
    lm_head_ptr: usize,
    lm_head_wtype_tag: u8,
    lm_head_role_tag: u8,
    feature_flags_tag: u8,
    layer_weights_binding_tag: u64,
    kv_binding_tag: u64,
}

impl DecodeGraphKey {
    pub fn for_decode(
        device: &GpuDevice,
        config: &ModelConfig,
        logits_mode: GpuLogitsMode,
    ) -> Self {
        Self::from_parts(device.device_id(), device.warp_size(), config, logits_mode)
    }

    pub fn from_parts(
        device_id: i32,
        warp_size: usize,
        config: &ModelConfig,
        logits_mode: GpuLogitsMode,
    ) -> Self {
        Self::from_parts_with_bindings(
            device_id,
            warp_size,
            config,
            logits_mode,
            0,
            0,
            GgmlType::F32,
            TensorRole::Generic,
        )
    }

    pub fn from_parts_with_bindings(
        device_id: i32,
        warp_size: usize,
        config: &ModelConfig,
        logits_mode: GpuLogitsMode,
        output_norm_ptr: usize,
        lm_head_ptr: usize,
        lm_head_wtype: GgmlType,
        lm_head_role: TensorRole,
    ) -> Self {
        Self {
            device_id,
            warp_size,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            rope_neox: config.rope_neox,
            use_attention_bias: config.use_attention_bias,
            logits_mode_tag: logits_mode_tag(logits_mode),
            scope_tag: decode_graph_scope_tag(DecodeGraphScope::GreedyTail),
            output_norm_ptr,
            lm_head_ptr,
            lm_head_wtype_tag: ggml_type_tag(lm_head_wtype),
            lm_head_role_tag: tensor_role_tag(lm_head_role),
            feature_flags_tag: 0,
            layer_weights_binding_tag: 0,
            kv_binding_tag: 0,
        }
    }

    pub fn with_decode_scope(mut self, scope: DecodeGraphScope) -> Self {
        self.scope_tag = decode_graph_scope_tag(scope);
        self
    }

    pub fn with_layer_weights_binding_tag(mut self, tag: u64) -> Self {
        self.layer_weights_binding_tag = tag;
        self
    }

    pub fn with_kv_binding_tag(mut self, tag: u64) -> Self {
        self.kv_binding_tag = tag;
        self
    }

    pub fn with_feature_flags_tag(mut self, tag: u8) -> Self {
        self.feature_flags_tag = tag;
        self
    }

    pub fn scope(self) -> DecodeGraphScope {
        match self.scope_tag {
            0 => DecodeGraphScope::GreedyTail,
            1 => DecodeGraphScope::FullGreedyDecode,
            _ => DecodeGraphScope::GreedyTail,
        }
    }
}

fn logits_mode_tag(mode: GpuLogitsMode) -> u8 {
    match mode {
        GpuLogitsMode::Skip => 0,
        GpuLogitsMode::DownloadToHost => 1,
        GpuLogitsMode::GreedyArgmax => 2,
    }
}

fn ggml_type_tag(wtype: GgmlType) -> u8 {
    wtype as u8
}

fn tensor_role_tag(role: TensorRole) -> u8 {
    match role {
        TensorRole::Generic => 0,
        TensorRole::LmHead => 1,
        TensorRole::TiedLmHead => 2,
    }
}

fn decode_graph_scope_tag(scope: DecodeGraphScope) -> u8 {
    match scope {
        DecodeGraphScope::GreedyTail => 0,
        DecodeGraphScope::FullGreedyDecode => 1,
    }
}

pub struct HipGraph {
    raw: ffi::hipGraph_t,
}

impl HipGraph {
    pub fn from_raw(raw: ffi::hipGraph_t) -> Self {
        Self { raw }
    }

    pub fn as_raw(&self) -> ffi::hipGraph_t {
        self.raw
    }

    pub fn nodes(&self) -> GpuResult<Vec<ffi::hipGraphNode_t>> {
        ffi::hip_graph_get_nodes(self.raw)
    }

    pub fn kernel_nodes(&self) -> GpuResult<Vec<ffi::hipGraphNode_t>> {
        let mut kernel_nodes = Vec::new();
        for node in self.nodes()? {
            if ffi::hip_graph_node_get_type(node)? == ffi::hipGraphNodeType::hipGraphNodeTypeKernel
            {
                kernel_nodes.push(node);
            }
        }
        Ok(kernel_nodes)
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = ffi::hip_graph_destroy(self.raw);
        }
    }
}

pub struct HipGraphExec {
    raw: ffi::hipGraphExec_t,
}

impl HipGraphExec {
    pub fn from_graph(graph: &HipGraph) -> GpuResult<Self> {
        ffi::hip_graph_instantiate(graph.as_raw()).map(|raw| Self { raw })
    }

    pub fn launch(&self, stream: ffi::hipStream_t) -> GpuResult<()> {
        ffi::hip_graph_launch(self.raw, stream)
    }

    pub fn set_kernel_node_params(
        &self,
        node: ffi::hipGraphNode_t,
        params: &ffi::hipKernelNodeParams,
    ) -> GpuResult<()> {
        ffi::hip_graph_exec_kernel_node_set_params(self.raw, node, params)
    }

    pub fn update(&self, graph: &HipGraph) -> GpuResult<bool> {
        let result = ffi::hip_graph_exec_update(self.raw, graph.as_raw())?;
        Ok(matches!(
            result,
            ffi::hipGraphExecUpdateResult::hipGraphExecUpdateSuccess
        ))
    }
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = ffi::hip_graph_exec_destroy(self.raw);
        }
    }
}

pub struct CapturedDecodeGraph {
    exec: HipGraphExec,
    graph: HipGraph,
    key: DecodeGraphKey,
    kernel_nodes: Vec<ffi::hipGraphNode_t>,
}

impl CapturedDecodeGraph {
    pub fn from_captured_graph(graph: HipGraph, key: DecodeGraphKey) -> GpuResult<Self> {
        let kernel_nodes = graph.kernel_nodes()?;
        let exec = HipGraphExec::from_graph(&graph)?;
        Ok(Self {
            exec,
            graph,
            key,
            kernel_nodes,
        })
    }

    pub fn key(&self) -> DecodeGraphKey {
        self.key
    }

    pub fn matches_key(&self, key: DecodeGraphKey) -> bool {
        self.key == key
    }

    pub fn launch(&self, stream: ffi::hipStream_t) -> GpuResult<()> {
        self.exec.launch(stream)
    }

    pub fn update(&self, new_graph: &HipGraph) -> GpuResult<bool> {
        self.exec.update(new_graph)
    }

    pub fn kernel_nodes(&self) -> &[ffi::hipGraphNode_t] {
        &self.kernel_nodes
    }

    pub fn graph(&self) -> &HipGraph {
        &self.graph
    }

    pub fn kernel_node_params(&self, index: usize) -> GpuResult<ffi::hipKernelNodeParams> {
        let node = self.kernel_node(index)?;
        ffi::hip_graph_kernel_node_get_params(node)
    }

    pub fn set_kernel_node_params(
        &self,
        index: usize,
        params: &ffi::hipKernelNodeParams,
    ) -> GpuResult<()> {
        let node = self.kernel_node(index)?;
        self.exec.set_kernel_node_params(node, params)
    }

    fn kernel_node(&self, index: usize) -> GpuResult<ffi::hipGraphNode_t> {
        self.kernel_nodes
            .get(index)
            .copied()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!(
                    "captured decode graph has no kernel node at index {}",
                    index
                ),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AttentionLayout, TensorNameRegistry, TensorNamingScheme};

    fn make_test_config() -> ModelConfig {
        ModelConfig {
            num_layers: 24,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 2048,
            hidden_size: 896,
            num_heads: 14,
            intermediate_size: 4864,
            vocab_size: 151936,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: true,
            attention_layout: AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
        }
    }

    #[test]
    fn decode_graph_key_changes_with_logits_mode() {
        let config = make_test_config();
        let greedy = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::GreedyArgmax);
        let host = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::DownloadToHost);

        assert_ne!(greedy, host);
    }

    #[test]
    fn decode_graph_key_changes_with_device_properties() {
        let config = make_test_config();
        let wave32 = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::GreedyArgmax);
        let wave64 = DecodeGraphKey::from_parts(0, 64, &config, GpuLogitsMode::GreedyArgmax);

        assert_ne!(wave32, wave64);
    }

    #[test]
    fn decode_graph_key_changes_with_bound_tensors() {
        let config = make_test_config();
        let base = DecodeGraphKey::from_parts_with_bindings(
            0,
            32,
            &config,
            GpuLogitsMode::GreedyArgmax,
            0x1000,
            0x2000,
            GgmlType::Q8_0,
            TensorRole::LmHead,
        );
        let different_weights = DecodeGraphKey::from_parts_with_bindings(
            0,
            32,
            &config,
            GpuLogitsMode::GreedyArgmax,
            0x1000,
            0x3000,
            GgmlType::Q8_0,
            TensorRole::LmHead,
        );

        assert_ne!(base, different_weights);
    }

    #[test]
    fn decode_graph_key_changes_with_scope() {
        let config = make_test_config();
        let tail = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::GreedyArgmax)
            .with_decode_scope(DecodeGraphScope::GreedyTail);
        let full = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::GreedyArgmax)
            .with_decode_scope(DecodeGraphScope::FullGreedyDecode);

        assert_ne!(tail, full);
    }

    #[test]
    fn decode_graph_key_changes_with_feature_flags() {
        let config = make_test_config();
        let baseline = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::GreedyArgmax);
        let q8_fastpath = baseline.with_feature_flags_tag(0b10);

        assert_ne!(baseline, q8_fastpath);
    }
}
