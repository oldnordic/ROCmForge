use super::super::error::{GpuError, GpuResult};
use super::super::vram_budget::active_or_default_device_id;
use super::buffer::GpuBuffer;
use super::metadata::WeightMeta;
use crate::config::ModelConfig;
use crate::loader::{GgufFile, RfmFile};

#[path = "layer/estimate.rs"]
mod estimate;
#[path = "layer/load_gguf.rs"]
mod load_gguf;
#[path = "layer/load_rfm.rs"]
mod load_rfm;
#[path = "layer/support.rs"]
mod support;
pub use self::support::{
    CpuCompressedExperts, CpuMpoExperts, GpuMoeWeights, GpuMpoWeights, GpuShortconvWeights,
    GpuSparseCsrWeights, GpuSsmWeights, SvdCorrection,
};

// ── GPU Layer Type ──────────────────────────────────────────────────────────────────

/// Architecture classification for a single GPU layer.
///
/// Determined at load time from which weight sets are present in the checkpoint.
/// The forward path matches on this enum instead of probing individual Option fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLayerType {
    /// Standard attention with separate Q/K/V weights
    Attention,
    /// Attention with fused QKV input projection (Qwen35-style)
    AttentionFusedQkv,
    /// SSM (state-space model) layer (Qwen35 hybrid)
    Ssm,
    /// Shortconv (depthwise causal conv1d) layer (LFM2-style)
    Shortconv,
}

impl GpuLayerType {
    /// Determine layer type from presence/absence of weight sets.
    ///
    /// This is the single source of truth for the dispatch table.
    /// Callers (GGUF loader, RFM loader, tests) must derive the three
    /// booleans from the checkpoint and then use this constructor.
    pub fn from_weights_present(
        has_ssm: bool,
        is_attention_layer: bool,
        has_fused_qkv: bool,
    ) -> Self {
        if has_ssm {
            GpuLayerType::Ssm
        } else if !is_attention_layer {
            GpuLayerType::Shortconv
        } else if has_fused_qkv {
            GpuLayerType::AttentionFusedQkv
        } else {
            GpuLayerType::Attention
        }
    }
}

// ── GPU Layer Weights ─────────────────────────────────────────────────────────────

/// Weights for a single transformer layer, stored in VRAM.
///
/// All weight tensors are stored in their native quantized format.
/// GPU kernels dequantize during inference.
#[derive(Debug)]
pub struct GpuLayerWeights {
    /// RMS norm weights for attention (always F32)
    pub attn_norm: GpuBuffer,
    /// Query projection weights (quantized)
    pub attn_q: GpuBuffer,
    pub attn_q_meta: WeightMeta,
    pub attn_q_svd: Option<SvdCorrection>,
    /// Optional Q RMSNorm for Qwen35 full-attention layers.
    pub attn_q_norm: Option<GpuBuffer>,
    /// Query bias (optional, always F32 if present)
    pub attn_q_bias: Option<GpuBuffer>,
    /// Key projection weights (quantized)
    pub attn_k: GpuBuffer,
    pub attn_k_meta: WeightMeta,
    pub attn_k_svd: Option<SvdCorrection>,
    /// Optional K RMSNorm for Qwen35 full-attention layers.
    pub attn_k_norm: Option<GpuBuffer>,
    /// Key bias (optional)
    pub attn_k_bias: Option<GpuBuffer>,
    /// Value projection weights (quantized)
    pub attn_v: GpuBuffer,
    pub attn_v_meta: WeightMeta,
    pub attn_v_svd: Option<SvdCorrection>,
    /// Value bias (optional)
    pub attn_v_bias: Option<GpuBuffer>,
    /// Fused QKV projection used by Qwen35-style hybrid layers.
    pub attn_qkv: Option<GpuBuffer>,
    pub attn_qkv_meta: Option<WeightMeta>,
    pub attn_qkv_svd: Option<SvdCorrection>,
    /// Attention gate projection used by Qwen35 hybrid attention/SSM mixing.
    pub attn_gate: Option<GpuBuffer>,
    pub attn_gate_meta: Option<WeightMeta>,
    pub attn_gate_svd: Option<SvdCorrection>,
    /// SSM tensors used by Qwen35 hybrid layers.
    pub ssm: Option<GpuSsmWeights>,
    /// Whether this layer uses attention (true) or shortconv (false).
    /// **Deprecated:** prefer `layer_type` for dispatch decisions.
    pub is_attention_layer: bool,
    /// Architecture classification used by the forward path.
    pub layer_type: GpuLayerType,
    /// Shortconv tensors used by LFM2 hybrid layers.
    pub shortconv: Option<GpuShortconvWeights>,
    /// Attention output projection (quantized)
    pub attn_o: GpuBuffer,
    pub attn_o_meta: WeightMeta,
    pub attn_o_svd: Option<SvdCorrection>,
    /// RMS norm weights for FFN (always F32)
    pub ffn_norm: GpuBuffer,
    /// FFN gate projection (SwiGLU gate) (quantized) — None for standard FFN (Phi-3)
    pub ffn_gate: Option<GpuBuffer>,
    pub ffn_gate_meta: Option<WeightMeta>,
    pub ffn_gate_svd: Option<SvdCorrection>,
    /// FFN up projection (quantized)
    pub ffn_up: GpuBuffer,
    pub ffn_up_meta: WeightMeta,
    pub ffn_up_svd: Option<SvdCorrection>,
    /// Optional decode-friendly interleaved Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved: Option<GpuBuffer>,
    /// Optional decode-friendly 4-column tiled Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved_tile4: Option<GpuBuffer>,
    /// FFN down projection (quantized)
    pub ffn_down: GpuBuffer,
    pub ffn_down_meta: WeightMeta,
    pub ffn_down_svd: Option<SvdCorrection>,
    /// Optional MoE router and shared-expert weights.
    pub moe: Option<GpuMoeWeights>,
    /// Optional sparse CSR variants for FFN weights.
    pub ffn_gate_sparse: Option<GpuSparseCsrWeights>,
    pub ffn_up_sparse: Option<GpuSparseCsrWeights>,
    pub ffn_down_sparse: Option<GpuSparseCsrWeights>,
    /// Optional MPO variants for FFN weights.
    pub ffn_gate_mpo: Option<GpuMpoWeights>,
    pub ffn_up_mpo: Option<GpuMpoWeights>,
    pub ffn_down_mpo: Option<GpuMpoWeights>,
    /// CPU-resident per-expert MPO-compressed weights (None for non-MoE or uncompressed models).
    /// Uploaded one expert at a time to GpuExpertScratch during decode dispatch.
    pub ffn_gate_mpo_experts: Option<CpuMpoExperts>,
    pub ffn_up_mpo_experts: Option<CpuMpoExperts>,
    pub ffn_down_mpo_experts: Option<CpuMpoExperts>,
    /// CPU-resident per-expert SVD+sparse weights (None for non-MoE or uncompressed models).
    /// Uploaded one expert at a time to GpuExpertScratch during decode dispatch.
    pub ffn_gate_compressed: Option<CpuCompressedExperts>,
    pub ffn_up_compressed: Option<CpuCompressedExperts>,
    pub ffn_down_compressed: Option<CpuCompressedExperts>,
}

pub(super) fn try_load_sparse_csr(
    file: &RfmFile,
    name: &str,
    device_id: i32,
) -> GpuResult<Option<GpuSparseCsrWeights>> {
    support::try_load_sparse_csr(file, name, device_id)
}

pub(super) fn try_load_mpo(
    file: &RfmFile,
    name: &str,
    device_id: i32,
) -> GpuResult<Option<GpuMpoWeights>> {
    support::try_load_mpo(file, name, device_id)
}

impl GpuLayerWeights {
    /// Load a single layer's weights from GGUF file into GPU memory.
    ///
    /// Returns error if any allocation or transfer fails.
    /// On error, all allocated memory is freed via Drop.
    pub fn load(file: &GgufFile, layer: usize, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_for_device(file, layer, config, active_or_default_device_id())
    }

    pub fn load_for_device(
        file: &GgufFile,
        layer: usize,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
        load_gguf::load_for_device(file, layer, config, device_id)
    }

    /// Load a single layer's weights from an RFM model file into GPU memory.
    pub fn load_rfm(file: &RfmFile, layer: usize, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_rfm_for_device(file, layer, config, active_or_default_device_id())
    }

    pub fn load_rfm_for_device(
        file: &RfmFile,
        layer: usize,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
        load_rfm::load_for_device(file, layer, config, device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::GpuLayerType;

    #[test]
    fn layer_type_ssm_overrides_everything() {
        assert_eq!(
            GpuLayerType::from_weights_present(true, true, true),
            GpuLayerType::Ssm
        );
        assert_eq!(
            GpuLayerType::from_weights_present(true, false, false),
            GpuLayerType::Ssm
        );
    }

    #[test]
    fn layer_type_shortconv_when_not_attention() {
        assert_eq!(
            GpuLayerType::from_weights_present(false, false, false),
            GpuLayerType::Shortconv
        );
        assert_eq!(
            GpuLayerType::from_weights_present(false, false, true),
            GpuLayerType::Shortconv
        );
    }

    #[test]
    fn layer_type_fused_qkv_when_attention_and_fused() {
        assert_eq!(
            GpuLayerType::from_weights_present(false, true, true),
            GpuLayerType::AttentionFusedQkv
        );
    }

    #[test]
    fn layer_type_standard_attention_fallback() {
        assert_eq!(
            GpuLayerType::from_weights_present(false, true, false),
            GpuLayerType::Attention
        );
    }

    #[test]
    fn layer_type_qwen35_hybrid_dispatch_table() {
        // Qwen3.5-style hybrid: first N layers are SSM, remaining are AttentionFusedQkv
        let num_layers = 40usize;
        let ssm_layers = 28usize; // e.g. 28 SSM + 12 attention
        let types: Vec<GpuLayerType> = (0..num_layers)
            .map(|i| {
                let has_ssm = i < ssm_layers;
                let is_attention = true; // Qwen35 layers are all attention layers
                let has_fused_qkv = !has_ssm; // non-SSM layers use fused QKV
                GpuLayerType::from_weights_present(has_ssm, is_attention, has_fused_qkv)
            })
            .collect();

        // First 28 layers should be Ssm
        for i in 0..ssm_layers {
            assert_eq!(
                types[i],
                GpuLayerType::Ssm,
                "Layer {} expected Ssm in Qwen35 hybrid",
                i
            );
        }
        // Remaining 12 layers should be AttentionFusedQkv
        for i in ssm_layers..num_layers {
            assert_eq!(
                types[i],
                GpuLayerType::AttentionFusedQkv,
                "Layer {} expected AttentionFusedQkv in Qwen35 hybrid",
                i
            );
        }
    }

    #[test]
    fn layer_type_lfm2_hybrid_dispatch_table() {
        // LFM2-style hybrid: first N layers are Shortconv, remaining are Attention
        let num_layers = 32usize;
        let shortconv_layers = 8usize; // e.g. 8 shortconv + 24 attention
        let types: Vec<GpuLayerType> = (0..num_layers)
            .map(|i| {
                let has_ssm = false;
                let is_attention = i >= shortconv_layers;
                let has_fused_qkv = false;
                GpuLayerType::from_weights_present(has_ssm, is_attention, has_fused_qkv)
            })
            .collect();

        // First 8 layers should be Shortconv
        for i in 0..shortconv_layers {
            assert_eq!(
                types[i],
                GpuLayerType::Shortconv,
                "Layer {} expected Shortconv in LFM2 hybrid",
                i
            );
        }
        // Remaining 24 layers should be Attention
        for i in shortconv_layers..num_layers {
            assert_eq!(
                types[i],
                GpuLayerType::Attention,
                "Layer {} expected Attention in LFM2 hybrid",
                i
            );
        }
    }
}
