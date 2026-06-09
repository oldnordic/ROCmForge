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
    CpuCompressedExperts, GpuMoeWeights, GpuMpoWeights, GpuSparseCsrWeights, GpuSsmWeights,
    SvdCorrection,
};

// ── GPU Layer Weights ─────────────────────────────────────────────────────────────

/// Weights for a single transformer layer, stored in VRAM.
///
/// All weight tensors are stored in their native quantized format.
/// GPU kernels dequantize during inference.
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
    /// Attention output projection (quantized)
    pub attn_o: GpuBuffer,
    pub attn_o_meta: WeightMeta,
    pub attn_o_svd: Option<SvdCorrection>,
    /// RMS norm weights for FFN (always F32)
    pub ffn_norm: GpuBuffer,
    /// FFN gate projection (SwiGLU gate) (quantized)
    pub ffn_gate: GpuBuffer,
    pub ffn_gate_meta: WeightMeta,
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
