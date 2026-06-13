use super::tensor_names::TensorNamingScheme;
use std::collections::HashMap;
use std::sync::OnceLock;

/// How RoPE rotations are applied to head dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Consecutive pairs: (0,1),(2,3),... - LLaMA, Mistral
    Normal,
    /// Split-half pairs: (0,head_dim/2),(1,head_dim/2+1),... - Qwen2, GPT-NeoX
    NeoX,
}

/// How the Q/K/V weight tensors are laid out in the GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionLayout {
    /// Separate `blk.N.attn_q`, `blk.N.attn_k`, `blk.N.attn_v`
    SplitQkv,
    /// Single fused tensor `blk.N.attn_qkv` (Phi3, Falcon, etc.)
    FusedQkv,
}

/// How the FFN (feed-forward network) is structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnLayout {
    /// SwiGLU: gate + up → SiLU(gate) * up → down
    /// Used by LLaMA, Mistral, Qwen2, Gemma, etc.
    SwiGLU,
    /// Standard FFN: up → activation → down
    /// No separate gate projection. Used by Phi-3.
    Standard,
}

/// Hardcoded structural/behavioral differences between architecture families.
/// All numeric *values* (sizes, epsilons) still come from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelTraits {
    pub rope_style: RopeStyle,
    pub attention_layout: AttentionLayout,
    /// Whether Q/K/V projections have explicit bias terms
    pub use_attention_bias: bool,
    /// Fallback rope theta if absent from GGUF (should be in GGUF, but just in case)
    pub default_rope_theta: f32,
    /// Fallback RMS norm epsilon if absent from GGUF
    pub default_norm_eps: f32,
    /// Tensor naming convention used by this architecture
    pub tensor_naming: TensorNamingScheme,
    /// FFN structure: SwiGLU (gate+up+down) or Standard (up+down only)
    pub ffn_layout: FfnLayout,
}

static REGISTRY: OnceLock<HashMap<&'static str, ModelTraits>> = OnceLock::new();

/// Default traits for unknown architectures (LLaMA-compatible)
static DEFAULT_TRAITS: ModelTraits = ModelTraits {
    rope_style: RopeStyle::Normal,
    attention_layout: AttentionLayout::SplitQkv,
    use_attention_bias: false,
    default_rope_theta: 10000.0,
    default_norm_eps: 1e-5,
    tensor_naming: TensorNamingScheme::Gguf,
    ffn_layout: FfnLayout::SwiGLU,
};

fn registry() -> &'static HashMap<&'static str, ModelTraits> {
    REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();

        // LLaMA family - consecutive RoPE, no bias, split QKV, SwiGLU
        let llama = ModelTraits {
            rope_style: RopeStyle::Normal,
            attention_layout: AttentionLayout::SplitQkv,
            use_attention_bias: false,
            default_rope_theta: 10000.0,
            default_norm_eps: 1e-5,
            tensor_naming: TensorNamingScheme::Gguf,
            ffn_layout: FfnLayout::SwiGLU,
        };
        for arch in &["llama", "mistral", "baichuan", "internlm2", "deepseek"] {
            m.insert(*arch, llama.clone());
        }
        m.insert(
            "yi",
            ModelTraits {
                default_norm_eps: 1e-6,
                ..llama.clone()
            },
        );
        m.insert("mixtral", llama.clone()); // MoE variant, same behaviors

        // Qwen2 family - NeoX RoPE, QKV bias, split QKV, high rope theta, GGUF naming, SwiGLU
        let qwen2 = ModelTraits {
            rope_style: RopeStyle::NeoX,
            attention_layout: AttentionLayout::SplitQkv,
            use_attention_bias: true,
            default_rope_theta: 1_000_000.0,
            default_norm_eps: 1e-6,
            tensor_naming: TensorNamingScheme::Gguf,
            ffn_layout: FfnLayout::SwiGLU,
        };
        for arch in &["qwen2", "qwen2moe"] {
            m.insert(*arch, qwen2.clone());
        }

        // Qwen3 family - NeoX RoPE, QKV bias, split QKV, high rope theta, GGUF MoE naming, SwiGLU
        // Note: Qwen3 uses MoE architecture with _exps suffix for expert tensors
        let qwen3 = ModelTraits {
            rope_style: RopeStyle::NeoX,
            attention_layout: AttentionLayout::SplitQkv,
            use_attention_bias: true,
            default_rope_theta: 1_000_000.0,
            default_norm_eps: 1e-6,
            tensor_naming: TensorNamingScheme::GgufMoE,
            ffn_layout: FfnLayout::SwiGLU,
        };
        for arch in &["qwen3", "qwen3moe"] {
            m.insert(*arch, qwen3.clone());
        }

        // Qwen3.5 / Qwen3.6 fused MoE family
        let qwen35moe = ModelTraits {
            attention_layout: AttentionLayout::FusedQkv,
            ..qwen3.clone()
        };
        m.insert("qwen35moe", qwen35moe);

        // Qwen3.5 hybrid checkpoints store text attention as fused QKV and add
        // SSM/attention-gate tensors. Runtime support is handled explicitly by
        // the Qwen35 loader/forward path instead of pretending these are split
        // Q/K/V transformer blocks.
        m.insert(
            "qwen35",
            ModelTraits {
                rope_style: RopeStyle::Normal,
                attention_layout: AttentionLayout::FusedQkv,
                use_attention_bias: false,
                default_rope_theta: 10_000_000.0,
                default_norm_eps: 1e-6,
                tensor_naming: TensorNamingScheme::Gguf,
                ffn_layout: FfnLayout::SwiGLU,
            },
        );

        // LFM2.5 MoE — Liquid Foundation Model 2 MoE
        // Mixed attention/shortconv layers, MoE FFN, QK-Norm, SwiGLU
        m.insert(
            "lfm2moe",
            ModelTraits {
                rope_style: RopeStyle::NeoX,
                attention_layout: AttentionLayout::SplitQkv,
                use_attention_bias: false,
                default_rope_theta: 1_000_000.0,
                default_norm_eps: 1e-5,
                tensor_naming: TensorNamingScheme::GgufMoE,
                ffn_layout: FfnLayout::SwiGLU,
            },
        );

        // Legacy Qwen1: lower rope theta, no QK norm
        m.insert(
            "qwen",
            ModelTraits {
                default_rope_theta: 10000.0,
                use_attention_bias: true,
                ..qwen2.clone()
            },
        );

        // Phi family — standard FFN (no SwiGLU gate), fused QKV
        m.insert(
            "phi3",
            ModelTraits {
                rope_style: RopeStyle::Normal,
                attention_layout: AttentionLayout::FusedQkv,
                use_attention_bias: false,
                default_rope_theta: 10000.0,
                default_norm_eps: 1e-5,
                tensor_naming: TensorNamingScheme::Gguf,
                ffn_layout: FfnLayout::Standard,
            },
        );
        m.insert("phi2", llama.clone());

        // Gemma family
        let gemma = ModelTraits {
            rope_style: RopeStyle::Normal,
            attention_layout: AttentionLayout::SplitQkv,
            use_attention_bias: false,
            default_rope_theta: 10000.0,
            default_norm_eps: 1e-6,
            tensor_naming: TensorNamingScheme::Gguf,
            ffn_layout: FfnLayout::SwiGLU,
        };
        for arch in &["gemma", "gemma2", "gemma3"] {
            m.insert(*arch, gemma.clone());
        }

        // GLM
        m.insert(
            "glm",
            ModelTraits {
                rope_style: RopeStyle::NeoX,
                attention_layout: AttentionLayout::FusedQkv,
                use_attention_bias: false,
                default_rope_theta: 10000.0,
                default_norm_eps: 1e-5,
                tensor_naming: TensorNamingScheme::Gguf,
                ffn_layout: FfnLayout::SwiGLU,
            },
        );

        m
    })
}

impl ModelTraits {
    /// Look up traits for an architecture string, falling back to LLaMA defaults
    /// for unknown architectures rather than failing.
    pub fn for_arch(arch: &str) -> &'static ModelTraits {
        registry().get(arch).unwrap_or(&DEFAULT_TRAITS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traits_qwen2_neox() {
        let t = ModelTraits::for_arch("qwen2");
        assert_eq!(t.rope_style, RopeStyle::NeoX);
        assert!(t.use_attention_bias);
        assert_eq!(t.default_rope_theta, 1_000_000.0);
    }

    #[test]
    fn traits_llama_normal() {
        let t = ModelTraits::for_arch("llama");
        assert_eq!(t.rope_style, RopeStyle::Normal);
        assert!(!t.use_attention_bias);
    }

    #[test]
    fn traits_unknown_falls_back_to_llama() {
        let t = ModelTraits::for_arch("some_future_arch");
        let ll = ModelTraits::for_arch("llama");
        assert_eq!(t.rope_style, ll.rope_style);
        assert_eq!(t.default_rope_theta, ll.default_rope_theta);
    }

    #[test]
    fn traits_phi3_fused_qkv() {
        let t = ModelTraits::for_arch("phi3");
        assert_eq!(t.attention_layout, AttentionLayout::FusedQkv);
        assert_eq!(t.rope_style, RopeStyle::Normal);
    }

    #[test]
    fn qwen2_uses_gguf_scheme() {
        let traits = ModelTraits::for_arch("qwen2");
        assert_eq!(traits.tensor_naming, TensorNamingScheme::Gguf);
    }

    #[test]
    fn qwen3_uses_gguf_moe_scheme() {
        let traits = ModelTraits::for_arch("qwen3");
        assert_eq!(traits.tensor_naming, TensorNamingScheme::GgufMoE);
    }

    #[test]
    fn qwen3moe_uses_gguf_moe_scheme() {
        let traits = ModelTraits::for_arch("qwen3moe");
        assert_eq!(traits.tensor_naming, TensorNamingScheme::GgufMoE);
    }

    #[test]
    fn qwen35_uses_fused_qkv_and_gguf_scheme() {
        let traits = ModelTraits::for_arch("qwen35");
        assert_eq!(traits.attention_layout, AttentionLayout::FusedQkv);
        assert_eq!(traits.tensor_naming, TensorNamingScheme::Gguf);
        assert!(!traits.use_attention_bias);
    }

    #[test]
    fn unknown_arch_falls_back_to_gguf() {
        let traits = ModelTraits::for_arch("unknown_future_arch");
        assert_eq!(traits.tensor_naming, TensorNamingScheme::Gguf);
    }
}
