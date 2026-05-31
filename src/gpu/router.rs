//! Inference hotpath router.
//!
//! Inspects loaded model metadata and selects the optimal inference path.
//! The router runs AFTER the VRAM manager pre-flight check and BEFORE any
//! scratch buffer allocation.
//!
//! Design goals:
//! - Single decision point for path selection
//! - Model-profile-driven routing (not ad-hoc checks in main.rs)
//! - Easy to add new paths without touching main.rs
//! - Clear fallback chains when a path fails

use crate::config::ModelConfig;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::vram_budget::VramSession;
use crate::gpu::weights::{GpuModelWeights, WeightMeta};
use crate::loader::GgmlType;

// ── Model Profile ────────────────────────────────────────────────────────────────

/// Detected characteristics of a loaded model.
///
/// Built from `GpuModelWeights` inspection. This is the input to path selection.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Quantization type across all attention weights.
    pub attention_quant: QuantizationType,
    /// Whether any layer has SVD-corrected weights.
    pub has_svd: bool,
    /// Whether any layer has sparse CSR weights.
    pub has_sparse: bool,
    /// Whether any layer has MPO-compressed weights.
    pub has_mpo: bool,
    /// Whether any layer has MoE routing.
    pub has_moe: bool,
    /// Whether any layer has SSM (Mamba-style) state.
    pub has_ssm: bool,
    /// Architecture string from config.
    pub architecture: String,
    /// Number of layers.
    pub num_layers: usize,
}

/// Classification of quantization type for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// All attention weights are Q4_0 (batched kernels available).
    Q4_0,
    /// Mixed or other quantization (use decode-style path).
    Other,
}

impl ModelProfile {
    /// Build a profile by inspecting loaded GPU weights.
    pub fn from_weights(weights: &GpuModelWeights, config: &ModelConfig) -> Self {
        let mut has_svd = false;
        let mut has_sparse = false;
        let mut has_mpo = false;
        let mut has_moe = false;
        let mut has_ssm = false;
        let mut all_q4_0 = true;

        for layer in &weights.layers {
            // SVD check
            if layer.attn_q_svd.is_some()
                || layer.attn_k_svd.is_some()
                || layer.attn_v_svd.is_some()
                || layer.attn_o_svd.is_some()
                || layer.ffn_gate_svd.is_some()
                || layer.ffn_up_svd.is_some()
                || layer.ffn_down_svd.is_some()
            {
                has_svd = true;
            }

            // Sparse check
            if layer.ffn_gate_sparse.is_some()
                || layer.ffn_up_sparse.is_some()
                || layer.ffn_down_sparse.is_some()
            {
                has_sparse = true;
            }

            // MPO check
            if layer.ffn_gate_mpo.is_some()
                || layer.ffn_up_mpo.is_some()
                || layer.ffn_down_mpo.is_some()
            {
                has_mpo = true;
            }

            // MoE check
            if layer.moe.is_some() {
                has_moe = true;
            }

            // SSM check
            if layer.ssm.is_some() {
                has_ssm = true;
            }

            // Quantization check (attention weights only)
            if layer.attn_qkv_meta.is_some() {
                // Fused QKV means not standard split Q4_0
                all_q4_0 = false;
            } else if layer.attn_q_meta.wtype != GgmlType::Q4_0
                || layer.attn_k_meta.wtype != GgmlType::Q4_0
                || layer.attn_v_meta.wtype != GgmlType::Q4_0
                || layer.attn_o_meta.wtype != GgmlType::Q4_0
            {
                all_q4_0 = false;
            }
        }

        Self {
            attention_quant: if all_q4_0 {
                QuantizationType::Q4_0
            } else {
                QuantizationType::Other
            },
            has_svd,
            has_sparse,
            has_mpo,
            has_moe,
            has_ssm,
            architecture: config.architecture.clone(),
            num_layers: config.num_layers,
        }
    }

    /// Human-readable summary for startup logging.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!("arch={}", self.architecture));
        parts.push(format!(
            "quant={}",
            match self.attention_quant {
                QuantizationType::Q4_0 => "Q4_0",
                QuantizationType::Other => "mixed",
            }
        ));
        if self.has_svd {
            parts.push("svd".to_string());
        }
        if self.has_sparse {
            parts.push("sparse".to_string());
        }
        if self.has_mpo {
            parts.push("mpo".to_string());
        }
        if self.has_moe {
            parts.push("moe".to_string());
        }
        if self.has_ssm {
            parts.push("ssm".to_string());
        }
        parts.join(", ")
    }
}

// ── Inference Path ───────────────────────────────────────────────────────────────

/// Selected inference path for a model.
///
/// Each variant carries the context needed to execute that path.
#[derive(Debug, Clone)]
pub enum InferencePath {
    /// Batched prefill with Q4_0 kernels.
    /// Fastest path for standard transformer models with Q4_0 attention.
    BatchedPrefill {
        /// Maximum sequence length supported by batched kernels.
        max_seq_len: usize,
    },
    /// Token-by-token decode-style processing.
    /// Universal fallback that works with any quantization type.
    DecodeStyle,
    /// SVD-optimized path (when stable).
    /// Uses SVD correction kernels for attention projections.
    SvdOptimized,
    /// CPU fallback for incompatible or unsafe models.
    CpuFallback {
        /// Reason for CPU fallback.
        reason: String,
    },
}

impl std::fmt::Display for InferencePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferencePath::BatchedPrefill { max_seq_len } => {
                write!(f, "BatchedPrefill(max_seq={})", max_seq_len)
            }
            InferencePath::DecodeStyle => write!(f, "DecodeStyle"),
            InferencePath::SvdOptimized => write!(f, "SvdOptimized"),
            InferencePath::CpuFallback { reason } => write!(f, "CpuFallback({})", reason),
        }
    }
}

// ── Router ───────────────────────────────────────────────────────────────────────

/// Select the optimal inference path based on model profile and runtime constraints.
///
/// This is the single decision point. All path selection logic lives here.
///
/// # Arguments
/// * `profile` — Detected model characteristics.
/// * `prompt_len` — Number of tokens in the prompt.
/// * `vram_session` — Current VRAM state (for headroom checks).
///
/// # Returns
/// The selected `InferencePath`.
pub fn select_path(
    profile: &ModelProfile,
    prompt_len: usize,
    _vram_session: &VramSession,
) -> InferencePath {
    // Safety: experimental kernels (sparse, MPO) are gated at the dispatch level,
    // but we also avoid routing to paths that would use them.
    if profile.has_sparse || profile.has_mpo {
        // Sparse/MPO models always use decode-style until kernels are proven stable.
        return InferencePath::DecodeStyle;
    }

    // MoE and SSM models don't have batched prefill kernels yet.
    if profile.has_moe || profile.has_ssm {
        return InferencePath::DecodeStyle;
    }

    // SVD models: use optimized path by default.
    if profile.has_svd {
        return InferencePath::SvdOptimized;
    }

    // Standard transformer with Q4_0 attention: batched prefill if prompt is multi-token.
    if profile.attention_quant == QuantizationType::Q4_0 && prompt_len > 1 {
        const MAX_BATCHED_SEQ: usize = 512;
        if prompt_len <= MAX_BATCHED_SEQ {
            return InferencePath::BatchedPrefill {
                max_seq_len: MAX_BATCHED_SEQ,
            };
        }
    }

    // Single-token prompts or non-Q4_0 models: decode-style.
    InferencePath::DecodeStyle
}

/// Check if a path can be executed with available VRAM.
///
/// Returns `Ok(())` if the path fits, or an error with a descriptive message.
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

// ── Tests ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_vram_session(free_gb: f64) -> VramSession {
        VramSession {
            device_id: 0,
            total: (20.0 * 1024.0 * 1024.0 * 1024.0) as usize,
            startup_free: (free_gb * 1024.0 * 1024.0 * 1024.0) as usize,
            already_used: 0,
            desktop_reserved: (4.0 * 1024.0 * 1024.0 * 1024.0) as usize,
            inference_budget: ((free_gb - 4.0) * 1024.0 * 1024.0 * 1024.0) as usize,
        }
    }

    #[test]
    fn q4_0_multi_token_selects_batched() {
        let profile = ModelProfile {
            attention_quant: QuantizationType::Q4_0,
            has_svd: false,
            has_sparse: false,
            has_mpo: false,
            has_moe: false,
            has_ssm: false,
            architecture: "llama".to_string(),
            num_layers: 16,
        };
        let vram = fake_vram_session(16.0);
        let path = select_path(&profile, 10, &vram);
        assert!(matches!(path, InferencePath::BatchedPrefill { .. }));
    }

    #[test]
    fn single_token_selects_decode() {
        let profile = ModelProfile {
            attention_quant: QuantizationType::Q4_0,
            has_svd: false,
            has_sparse: false,
            has_mpo: false,
            has_moe: false,
            has_ssm: false,
            architecture: "llama".to_string(),
            num_layers: 16,
        };
        let vram = fake_vram_session(16.0);
        let path = select_path(&profile, 1, &vram);
        assert!(matches!(path, InferencePath::DecodeStyle));
    }

    #[test]
    fn sparse_always_selects_decode() {
        let profile = ModelProfile {
            attention_quant: QuantizationType::Q4_0,
            has_svd: false,
            has_sparse: true,
            has_mpo: false,
            has_moe: false,
            has_ssm: false,
            architecture: "llama".to_string(),
            num_layers: 16,
        };
        let vram = fake_vram_session(16.0);
        let path = select_path(&profile, 10, &vram);
        assert!(matches!(path, InferencePath::DecodeStyle));
    }

    #[test]
    fn mpo_always_selects_decode() {
        let profile = ModelProfile {
            attention_quant: QuantizationType::Q4_0,
            has_svd: false,
            has_sparse: false,
            has_mpo: true,
            has_moe: false,
            has_ssm: false,
            architecture: "llama".to_string(),
            num_layers: 16,
        };
        let vram = fake_vram_session(16.0);
        let path = select_path(&profile, 10, &vram);
        assert!(matches!(path, InferencePath::DecodeStyle));
    }

    #[test]
    fn svd_always_selects_svd_optimized() {
        let profile = ModelProfile {
            attention_quant: QuantizationType::Q4_0,
            has_svd: true,
            has_sparse: false,
            has_mpo: false,
            has_moe: false,
            has_ssm: false,
            architecture: "llama".to_string(),
            num_layers: 16,
        };
        let vram = fake_vram_session(16.0);
        let path = select_path(&profile, 10, &vram);
        assert!(matches!(path, InferencePath::SvdOptimized));
    }

    #[test]
    fn mixed_quant_selects_decode() {
        let profile = ModelProfile {
            attention_quant: QuantizationType::Other,
            has_svd: false,
            has_sparse: false,
            has_mpo: false,
            has_moe: false,
            has_ssm: false,
            architecture: "qwen2".to_string(),
            num_layers: 24,
        };
        let vram = fake_vram_session(16.0);
        let path = select_path(&profile, 10, &vram);
        assert!(matches!(path, InferencePath::DecodeStyle));
    }
}
