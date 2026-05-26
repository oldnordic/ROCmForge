use std::collections::HashMap;

/// Tensor naming convention used by GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorNamingScheme {
    /// GGUF-style: `blk.N.ffn_gate.weight`
    Gguf,
    /// HuggingFace-style: `model.layers.N.mlp.gate_proj.weight`
    HuggingFace,
    /// GGUF MoE-style: `blk.N.ffn_gate_exps.weight` (Mixture-of-Experts)
    GgufMoE,
}

/// Semantic tensor names that are resolved to actual tensor names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorName {
    // Attention weights
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    // Attention biases (optional)
    AttnQBias,
    AttnKBias,
    AttnVBias,
    // FFN weights
    FfnGate,
    FfnUp,
    FfnDown,
    // FFN MoE weights (Mixture-of-Experts)
    FfnGateExps,
    FfnUpExps,
    FfnDownExps,
    FfnGateInp,
    // Attention normalization (optional, for some MoE models)
    AttnQNorm,
    AttnKNorm,
    // Normalization
    AttnNorm,
    FfnNorm,
    // Embeddings
    TokenEmb,
    LmHead,
    // Output norm
    OutputNorm,
}

/// Registry for resolving semantic tensor names to actual tensor names
/// based on the model's naming convention.
#[derive(Debug, Clone)]
pub struct TensorNameRegistry {
    /// The naming scheme this registry uses
    pub scheme: TensorNamingScheme,
    /// Map of semantic names to format templates (use {} for layer number)
    templates: HashMap<TensorName, String>,
}

impl TensorNameRegistry {
    /// Create a registry for a specific naming scheme.
    pub fn from_scheme(scheme: &TensorNamingScheme) -> Self {
        match scheme {
            TensorNamingScheme::Gguf => Self::gguf_format(),
            TensorNamingScheme::HuggingFace => Self::huggingface_format(),
            TensorNamingScheme::GgufMoE => Self::gguf_moe_format(),
        }
    }

    /// Resolve a semantic tensor name for a specific layer.
    ///
    /// # Panics
    /// Panics if the tensor name is not in the registry (indicates bug).
    pub fn resolve(&self, name: TensorName, layer: usize) -> String {
        let template = self
            .templates
            .get(&name)
            .expect("Tensor name should be in registry");
        template.replace("{}", &layer.to_string())
    }

    /// Resolve an optional tensor name (for bias tensors that may not exist).
    ///
    /// Returns None if the naming scheme doesn't support this tensor.
    pub fn resolve_optional(&self, name: TensorName, layer: usize) -> Option<String> {
        self.templates
            .get(&name)
            .cloned()
            .map(|t| t.replace("{}", &layer.to_string()))
    }

    /// Create GGUF-format registry.
    fn gguf_format() -> Self {
        let mut templates = HashMap::new();

        // Attention weights
        templates.insert(TensorName::AttnQ, "blk.{}.attn_q.weight".to_string());
        templates.insert(TensorName::AttnK, "blk.{}.attn_k.weight".to_string());
        templates.insert(TensorName::AttnV, "blk.{}.attn_v.weight".to_string());
        templates.insert(
            TensorName::AttnOutput,
            "blk.{}.attn_output.weight".to_string(),
        );

        // Attention biases
        templates.insert(TensorName::AttnQBias, "blk.{}.attn_q.bias".to_string());
        templates.insert(TensorName::AttnKBias, "blk.{}.attn_k.bias".to_string());
        templates.insert(TensorName::AttnVBias, "blk.{}.attn_v.bias".to_string());

        // FFN weights
        templates.insert(TensorName::FfnGate, "blk.{}.ffn_gate.weight".to_string());
        templates.insert(TensorName::FfnUp, "blk.{}.ffn_up.weight".to_string());
        templates.insert(TensorName::FfnDown, "blk.{}.ffn_down.weight".to_string());

        // Normalization
        templates.insert(TensorName::AttnNorm, "blk.{}.attn_norm.weight".to_string());
        templates.insert(TensorName::FfnNorm, "blk.{}.ffn_norm.weight".to_string());

        // Embeddings (no layer number)
        templates.insert(TensorName::TokenEmb, "token_embd.weight".to_string());
        templates.insert(TensorName::LmHead, "output.weight".to_string());
        templates.insert(TensorName::OutputNorm, "output_norm.weight".to_string());

        Self {
            scheme: TensorNamingScheme::Gguf,
            templates,
        }
    }

    /// Create HuggingFace-format registry.
    fn huggingface_format() -> Self {
        let mut templates = HashMap::new();

        // Attention weights
        templates.insert(
            TensorName::AttnQ,
            "model.layers.{}.self_attn.q_proj.weight".to_string(),
        );
        templates.insert(
            TensorName::AttnK,
            "model.layers.{}.self_attn.k_proj.weight".to_string(),
        );
        templates.insert(
            TensorName::AttnV,
            "model.layers.{}.self_attn.v_proj.weight".to_string(),
        );
        templates.insert(
            TensorName::AttnOutput,
            "model.layers.{}.self_attn.o_proj.weight".to_string(),
        );

        // Attention biases
        templates.insert(
            TensorName::AttnQBias,
            "model.layers.{}.self_attn.q_proj.bias".to_string(),
        );
        templates.insert(
            TensorName::AttnKBias,
            "model.layers.{}.self_attn.k_proj.bias".to_string(),
        );
        templates.insert(
            TensorName::AttnVBias,
            "model.layers.{}.self_attn.v_proj.bias".to_string(),
        );

        // FFN weights
        templates.insert(
            TensorName::FfnGate,
            "model.layers.{}.mlp.gate_proj.weight".to_string(),
        );
        templates.insert(
            TensorName::FfnUp,
            "model.layers.{}.mlp.up_proj.weight".to_string(),
        );
        templates.insert(
            TensorName::FfnDown,
            "model.layers.{}.mlp.down_proj.weight".to_string(),
        );

        // Normalization
        templates.insert(
            TensorName::AttnNorm,
            "model.layers.{}.input_layernorm.weight".to_string(),
        );
        templates.insert(
            TensorName::FfnNorm,
            "model.layers.{}.post_attention_layernorm.weight".to_string(),
        );

        // Embeddings (no layer number)
        templates.insert(
            TensorName::TokenEmb,
            "model.embed_tokens.weight".to_string(),
        );
        templates.insert(TensorName::LmHead, "lm_head.weight".to_string());
        templates.insert(TensorName::OutputNorm, "model.norm.weight".to_string());

        Self {
            scheme: TensorNamingScheme::HuggingFace,
            templates,
        }
    }

    /// Create GGUF MoE-format registry for Mixture-of-Experts models.
    ///
    /// Uses `_exps` suffix for expert tensors and includes Q/K normalization.
    fn gguf_moe_format() -> Self {
        let mut templates = HashMap::new();

        // Attention weights
        templates.insert(TensorName::AttnQ, "blk.{}.attn_q.weight".to_string());
        templates.insert(TensorName::AttnK, "blk.{}.attn_k.weight".to_string());
        templates.insert(TensorName::AttnV, "blk.{}.attn_v.weight".to_string());
        templates.insert(
            TensorName::AttnOutput,
            "blk.{}.attn_output.weight".to_string(),
        );

        // Attention biases
        templates.insert(TensorName::AttnQBias, "blk.{}.attn_q.bias".to_string());
        templates.insert(TensorName::AttnKBias, "blk.{}.attn_k.bias".to_string());
        templates.insert(TensorName::AttnVBias, "blk.{}.attn_v.bias".to_string());

        // FFN weights - standard names (fallback, not used in qwen3moe)
        templates.insert(TensorName::FfnGate, "blk.{}.ffn_gate.weight".to_string());
        templates.insert(TensorName::FfnUp, "blk.{}.ffn_up.weight".to_string());
        templates.insert(TensorName::FfnDown, "blk.{}.ffn_down.weight".to_string());

        // FFN MoE weights - expert tensors with `_exps` suffix
        templates.insert(
            TensorName::FfnGateExps,
            "blk.{}.ffn_gate_exps.weight".to_string(),
        );
        templates.insert(
            TensorName::FfnUpExps,
            "blk.{}.ffn_up_exps.weight".to_string(),
        );
        templates.insert(
            TensorName::FfnDownExps,
            "blk.{}.ffn_down_exps.weight".to_string(),
        );
        templates.insert(
            TensorName::FfnGateInp,
            "blk.{}.ffn_gate_inp.weight".to_string(),
        );

        // Attention normalization (optional, for MoE models)
        templates.insert(
            TensorName::AttnQNorm,
            "blk.{}.attn_q_norm.weight".to_string(),
        );
        templates.insert(
            TensorName::AttnKNorm,
            "blk.{}.attn_k_norm.weight".to_string(),
        );

        // Normalization
        templates.insert(TensorName::AttnNorm, "blk.{}.attn_norm.weight".to_string());
        templates.insert(TensorName::FfnNorm, "blk.{}.ffn_norm.weight".to_string());

        // Embeddings (no layer number)
        templates.insert(TensorName::TokenEmb, "token_embd.weight".to_string());
        templates.insert(TensorName::LmHead, "output.weight".to_string());
        templates.insert(TensorName::OutputNorm, "output_norm.weight".to_string());

        Self {
            scheme: TensorNamingScheme::GgufMoE,
            templates,
        }
    }

    /// Get all expected tensor names for a layer (for debugging).
    pub fn expected_tensors(&self, layer: usize) -> Vec<String> {
        self.templates
            .keys()
            .filter(|name| {
                !matches!(
                    *name,
                    TensorName::TokenEmb | TensorName::LmHead | TensorName::OutputNorm
                )
            })
            .map(|name| self.resolve(*name, layer))
            .collect()
    }

    /// List all supported tensor names (for debugging).
    pub fn list_all_names(&self) -> Vec<&'static str> {
        vec![
            "AttnQ",
            "AttnK",
            "AttnV",
            "AttnOutput",
            "AttnQBias",
            "AttnKBias",
            "AttnVBias",
            "FfnGate",
            "FfnUp",
            "FfnDown",
            "FfnGateExps",
            "FfnUpExps",
            "FfnDownExps",
            "FfnGateInp",
            "AttnQNorm",
            "AttnKNorm",
            "AttnNorm",
            "FfnNorm",
            "TokenEmb",
            "LmHead",
            "OutputNorm",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_scheme_formats_correctly() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        assert_eq!(
            registry.resolve(TensorName::FfnGate, 0),
            "blk.0.ffn_gate.weight"
        );
        assert_eq!(
            registry.resolve(TensorName::AttnQ, 5),
            "blk.5.attn_q.weight"
        );
        assert_eq!(
            registry.resolve(TensorName::AttnNorm, 2),
            "blk.2.attn_norm.weight"
        );
    }

    #[test]
    fn huggingface_scheme_formats_correctly() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
        assert_eq!(
            registry.resolve(TensorName::FfnGate, 0),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            registry.resolve(TensorName::AttnQ, 5),
            "model.layers.5.self_attn.q_proj.weight"
        );
        assert_eq!(
            registry.resolve(TensorName::AttnNorm, 2),
            "model.layers.2.input_layernorm.weight"
        );
    }

    #[test]
    fn gguf_scheme_maps_all_tensors() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        let all_names = [
            TensorName::AttnQ,
            TensorName::AttnK,
            TensorName::AttnV,
            TensorName::AttnOutput,
            TensorName::AttnQBias,
            TensorName::AttnKBias,
            TensorName::AttnVBias,
            TensorName::FfnGate,
            TensorName::FfnUp,
            TensorName::FfnDown,
            TensorName::AttnNorm,
            TensorName::FfnNorm,
            TensorName::TokenEmb,
            TensorName::LmHead,
            TensorName::OutputNorm,
        ];
        for name in all_names {
            let result = registry.resolve(name, 0);
            assert!(
                !result.is_empty(),
                "TensorName::{:?} should produce non-empty string",
                name
            );
        }
    }

    #[test]
    fn huggingface_scheme_maps_all_tensors() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
        let all_names = [
            TensorName::AttnQ,
            TensorName::AttnK,
            TensorName::AttnV,
            TensorName::AttnOutput,
            TensorName::AttnQBias,
            TensorName::AttnKBias,
            TensorName::AttnVBias,
            TensorName::FfnGate,
            TensorName::FfnUp,
            TensorName::FfnDown,
            TensorName::AttnNorm,
            TensorName::FfnNorm,
            TensorName::TokenEmb,
            TensorName::LmHead,
            TensorName::OutputNorm,
        ];
        for name in all_names {
            let result = registry.resolve(name, 0);
            assert!(
                !result.is_empty(),
                "TensorName::{:?} should produce non-empty string",
                name
            );
        }
    }

    #[test]
    fn gguf_moe_maps_exp_tensors() {
        let moe = TensorNameRegistry::from_scheme(&TensorNamingScheme::GgufMoE);
        assert_eq!(
            moe.resolve(TensorName::FfnGateExps, 0),
            "blk.0.ffn_gate_exps.weight"
        );
        assert_eq!(
            moe.resolve(TensorName::FfnUpExps, 0),
            "blk.0.ffn_up_exps.weight"
        );
        assert_eq!(
            moe.resolve(TensorName::FfnDownExps, 0),
            "blk.0.ffn_down_exps.weight"
        );
        assert_eq!(
            moe.resolve(TensorName::FfnGateInp, 0),
            "blk.0.ffn_gate_inp.weight"
        );
        assert_eq!(
            moe.resolve(TensorName::AttnQNorm, 0),
            "blk.0.attn_q_norm.weight"
        );
        assert_eq!(
            moe.resolve(TensorName::AttnKNorm, 0),
            "blk.0.attn_k_norm.weight"
        );
    }

    #[test]
    fn optional_bias_resolution() {
        let gguf = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        assert!(gguf.resolve_optional(TensorName::AttnQBias, 0).is_some());
    }

    #[test]
    fn layer_number_formatting() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        assert!(registry.resolve(TensorName::FfnGate, 0).contains(".0."));
        assert!(registry.resolve(TensorName::FfnGate, 99).contains(".99."));
        assert!(registry.resolve(TensorName::FfnGate, 999).contains(".999."));
    }

    #[test]
    fn non_layer_tensors_format_correctly() {
        let gguf = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        assert_eq!(gguf.resolve(TensorName::TokenEmb, 0), "token_embd.weight");
        assert_eq!(
            gguf.resolve(TensorName::OutputNorm, 0),
            "output_norm.weight"
        );

        let hf = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
        assert_eq!(
            hf.resolve(TensorName::TokenEmb, 0),
            "model.embed_tokens.weight"
        );
        assert_eq!(hf.resolve(TensorName::OutputNorm, 0), "model.norm.weight");
    }
}
