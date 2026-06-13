//! Inference hotpath router.
//!
//! Inspects loaded model metadata and selects the optimal inference path.
//! The router runs AFTER the VRAM manager pre-flight check and BEFORE any
//! scratch buffer allocation.

use crate::config::{AttentionLayout, ModelConfig};
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::vram_budget::VramSession;
use crate::gpu::weights::{GpuModelWeights, WeightMeta};
use crate::loader::GgmlType;

// ── Hotpath Capabilities ─────────────────────────────────────────────────────────

/// Classification of model format for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    Rfm,
}

/// Detailed classification of quantization for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationClass {
    /// All weights are Q4_0 (highly optimized batched kernels available).
    PureQ4_0,
    /// All weights are Q4_1.
    PureQ4_1,
    /// All weights are Q8_0.
    PureQ8_0,
    /// Mixed quantization types or unsupported types.
    MixedOrOther,
}

/// Detected characteristics and capabilities of a loaded model.
///
/// Built from `GpuModelWeights` inspection. This is the input to path selection.
#[derive(Debug, Clone)]
pub struct HotpathCapabilities {
    pub format: ModelFormat,
    pub architecture: String,
    pub attention_layout: AttentionLayout,
    pub quant_class: QuantizationClass,
    pub has_svd: bool,
    pub has_sparse: bool,
    pub has_mpo: bool,
    pub has_moe: bool,
    pub has_ssm: bool,
    pub has_shortconv: bool,
    pub num_layers: usize,
    /// Whether the model is eligible for HIP graph capture.
    pub is_graph_eligible: bool,
    /// Whether the model is eligible for batched prefill kernels.
    pub is_prefill_eligible: bool,
    
    // Sizing hints for VRAM checks
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl HotpathCapabilities {
    /// Build capabilities by inspecting loaded GPU weights.
    pub fn from_weights(weights: &GpuModelWeights, config: &ModelConfig) -> Self {
        let mut has_svd = false;
        let mut has_sparse = false;
        let mut has_mpo = false;
        let mut has_moe = false;
        let mut has_ssm = false;
        let mut has_shortconv = false;
        
        let mut all_q4_0 = true;
        let mut all_q4_1 = true;
        let mut all_q8_0 = true;

        for layer in &weights.layers {
            if layer.attn_q_svd.is_some() || layer.attn_k_svd.is_some() || layer.attn_v_svd.is_some() ||
               layer.attn_o_svd.is_some() || layer.ffn_gate_svd.is_some() || layer.ffn_up_svd.is_some() || 
               layer.ffn_down_svd.is_some() {
                has_svd = true;
            }
            if layer.ffn_gate_sparse.is_some() || layer.ffn_up_sparse.is_some() || layer.ffn_down_sparse.is_some() {
                has_sparse = true;
            }
            if layer.ffn_gate_mpo.is_some() || layer.ffn_up_mpo.is_some() || layer.ffn_down_mpo.is_some() {
                has_mpo = true;
            }
            if layer.moe.is_some() { has_moe = true; }
            if layer.ssm.is_some() { has_ssm = true; }
            if layer.shortconv.is_some() { has_shortconv = true; }

            // Quantization check (all projection weights)
            let q_types = [
                layer.attn_q_meta.wtype, layer.attn_k_meta.wtype, layer.attn_v_meta.wtype,
                layer.attn_o_meta.wtype, layer.ffn_up_meta.wtype, layer.ffn_down_meta.wtype
            ];
            for &t in &q_types {
                if t != GgmlType::Q4_0 { all_q4_0 = false; }
                if t != GgmlType::Q4_1 { all_q4_1 = false; }
                if t != GgmlType::Q8_0 { all_q8_0 = false; }
            }
            if let Some(ref m) = layer.ffn_gate_meta {
                if m.wtype != GgmlType::Q4_0 { all_q4_0 = false; }
                if m.wtype != GgmlType::Q4_1 { all_q4_1 = false; }
                if m.wtype != GgmlType::Q8_0 { all_q8_0 = false; }
            }
        }

        let quant_class = if all_q4_0 {
            QuantizationClass::PureQ4_0
        } else if all_q4_1 {
            QuantizationClass::PureQ4_1
        } else if all_q8_0 {
            QuantizationClass::PureQ8_0
        } else {
            QuantizationClass::MixedOrOther
        };

        // Graph eligibility
        let is_graph_eligible = !has_sparse && !has_mpo;
        
        // Prefill eligibility: requires batched kernels (Q4_0/Q4_1)
        let is_prefill_eligible = (quant_class == QuantizationClass::PureQ4_0 || quant_class == QuantizationClass::PureQ4_1) && !has_sparse && !has_mpo;

        Self {
            format: ModelFormat::Gguf,
            architecture: config.architecture.clone(),
            attention_layout: config.attention_layout,
            quant_class,
            has_svd,
            has_sparse,
            has_mpo,
            has_moe,
            has_ssm,
            has_shortconv,
            num_layers: config.num_layers,
            is_graph_eligible,
            is_prefill_eligible,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        }
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!("arch={}", self.architecture));
        parts.push(format!("quant={:?}", self.quant_class));
        if self.has_svd { parts.push("svd".to_string()); }
        if self.has_sparse { parts.push("sparse".to_string()); }
        if self.has_mpo { parts.push("mpo".to_string()); }
        if self.has_moe { parts.push("moe".to_string()); }
        if self.has_ssm { parts.push("ssm".to_string()); }
        if self.has_shortconv { parts.push("shortconv".to_string()); }
        parts.join(", ")
    }
}

// Keep alias for backward compatibility during migration
pub type ModelProfile = HotpathCapabilities;

// ── Inference Path ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum InferencePath {
    /// Batched prefill with optimized kernels.
    BatchedPrefill {
        max_seq_len: usize,
    },
    /// Token-by-token decode-style processing.
    DecodeStyle,
    /// SVD-optimized path.
    SvdOptimized,
    /// CPU fallback.
    CpuFallback {
        reason: String,
    },
}

impl std::fmt::Display for InferencePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferencePath::BatchedPrefill { max_seq_len } => write!(f, "BatchedPrefill(max_seq={})", max_seq_len),
            InferencePath::DecodeStyle => write!(f, "DecodeStyle"),
            InferencePath::SvdOptimized => write!(f, "SvdOptimized"),
            InferencePath::CpuFallback { reason } => write!(f, "CpuFallback({})", reason),
        }
    }
}

// ── Router ───────────────────────────────────────────────────────────────────────

pub fn select_path(
    capabilities: &HotpathCapabilities,
    prompt_len: usize,
    _vram_session: &VramSession,
) -> InferencePath {
    if capabilities.has_sparse || capabilities.has_mpo {
        return InferencePath::DecodeStyle;
    }

    if capabilities.has_moe || (capabilities.has_ssm && prompt_len > 1) {
        return InferencePath::DecodeStyle;
    }

    if capabilities.has_svd {
        return InferencePath::SvdOptimized;
    }

    if capabilities.is_prefill_eligible && prompt_len > 1 {
        const MAX_BATCHED_SEQ: usize = 512;
        if prompt_len <= MAX_BATCHED_SEQ {
            return InferencePath::BatchedPrefill {
                max_seq_len: MAX_BATCHED_SEQ,
            };
        }
    }

    InferencePath::DecodeStyle
}

pub fn check_path_vram(
    path: &InferencePath,
    config: &ModelConfig,
    prompt_len: usize,
    vram_session: &VramSession,
) -> GpuResult<()> {
    match path {
        InferencePath::BatchedPrefill { .. } => {
            let prefill_bytes =
                super::cache::GpuPrefillScratch::estimate_total_bytes(config, prompt_len);
            let reserve_bytes = 5 * 1024 * 1024 * 1024usize; // 5 GB desktop reserve
            let required = prefill_bytes.saturating_add(reserve_bytes);
            if vram_session.startup_free < required {
                return Err(GpuError::OutOfMemory {
                    requested: required,
                    available: vram_session.startup_free,
                    hint: format!(
                        "Batched prefill needs {:.1} GB scratch + 5 GB reserve, but only {:.1} GB free. \
                         Try decode-style path or reduce prompt length.",
                        prefill_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                        vram_session.startup_free as f64 / (1024.0 * 1024.0 * 1024.0)
                    ),
                });
            }
            Ok(())
        }
        InferencePath::DecodeStyle | InferencePath::SvdOptimized => {
            // These use GpuForwardScratch which was already checked in VramSession::check_fits.
            Ok(())
        }
        InferencePath::CpuFallback { .. } => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_path_prefill() {
        let mut caps = HotpathCapabilities {
            format: ModelFormat::Gguf,
            architecture: "llama".to_string(),
            attention_layout: AttentionLayout::SplitQkv,
            quant_class: QuantizationClass::PureQ4_0,
            has_svd: false,
            has_sparse: false,
            has_mpo: false,
            has_moe: false,
            has_ssm: false,
            has_shortconv: false,
            num_layers: 12,
            is_graph_eligible: true,
            is_prefill_eligible: true,
            hidden_size: 256,
            intermediate_size: 768,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
        };
        
        let vram = VramSession::mock();
        
        // Long prompt -> BatchedPrefill
        let path = select_path(&caps, 10, &vram);
        assert!(matches!(path, InferencePath::BatchedPrefill { .. }));

        // Single token -> DecodeStyle
        let path = select_path(&caps, 1, &vram);
        assert!(matches!(path, InferencePath::DecodeStyle));
    }
}
