//! Semantic role of a weight tensor.
//!
//! Drives kernel selection, layout normalization, and transpose decisions
//! on both CPU and GPU paths. Adding a new architecture only requires tagging
//! its tensors with the correct roles — forward paths dispatch generically.
//!
//! This module lives in `config` (not `gpu`) so that the CPU weight-loading
//! path can use the same role taxonomy without depending on GPU code.

/// Semantic role of a weight tensor.
///
/// Roles are determined at load time from tensor names and model metadata.
/// They replace architecture-specific string matching in transpose logic
/// with a single, exhaustive enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRole {
    /// Default — no special handling
    Generic,
    /// Language-model head (logits projection)
    LmHead,
    /// Tied embedding / LM head (shared weights)
    TiedLmHead,
    // ── SSM (Qwen3.5 hybrid) ─────────────────────────────────────────────
    /// Depthwise causal conv1d kernel weights
    SsmConv1d,
    /// SSM alpha projection
    SsmAlpha,
    /// SSM beta projection
    SsmBeta,
    /// SSM output projection
    SsmOut,
    // ── Shortconv (LFM2.5) ───────────────────────────────────────────────
    /// Shortconv input projection
    ShortconvInProj,
    /// Shortconv depthwise conv kernel
    ShortconvConv,
    /// Shortconv output projection
    ShortconvOutProj,
}

impl TensorRole {
    /// Returns `true` if this role represents an LM head variant.
    pub fn is_lm_head(self) -> bool {
        matches!(self, TensorRole::LmHead | TensorRole::TiedLmHead)
    }

    /// Derive a basic role from a weight name and LM-head flags.
    ///
    /// Architecture-specific loaders (SSM, shortconv, MoE) should override
    /// the returned role when they know the tensor's semantic purpose.
    pub fn from_name(name: &str, is_lm_head: bool, is_tied: bool) -> Self {
        if is_lm_head && is_tied {
            return TensorRole::TiedLmHead;
        }
        if is_lm_head {
            return TensorRole::LmHead;
        }

        // Common tensor name patterns
        if name.contains("ssm_conv1d.weight") {
            return TensorRole::SsmConv1d;
        }
        if name.contains("ssm_alpha.weight") {
            return TensorRole::SsmAlpha;
        }
        if name.contains("ssm_beta.weight") {
            return TensorRole::SsmBeta;
        }
        if name.contains("ssm_out.weight") {
            return TensorRole::SsmOut;
        }
        if name.contains("shortconv_in_proj.weight") {
            return TensorRole::ShortconvInProj;
        }
        if name.contains("shortconv_conv.weight") {
            return TensorRole::ShortconvConv;
        }
        if name.contains("shortconv_out_proj.weight") {
            return TensorRole::ShortconvOutProj;
        }

        TensorRole::Generic
    }
}
