//! Model architecture detection and naming patterns

use std::collections::HashSet;
use std::fmt;

use crate::backend::HipError;

/// Detected model architecture based on tensor naming patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Qwen2-style: tensors start with `blk.N.`
    Qwen2,
    /// LLaMA-style: tensors start with `transformer.layers.N.`
    LLaMA,
    /// Mistral-style: tensors start with `model.layers.N.`
    Mistral,
    /// Yi-style: tensors start with `model.layers.N.`
    ///
    /// Note: Yi uses the same tensor naming pattern as Mistral.
    /// Differentiation is done via the `general.architecture` metadata key
    /// which will be "yi" for Yi models and "mistral" for Mistral models.
    Yi,
    /// Mixtral-style: tensors start with `model.layers.N.`
    ///
    /// Note: Mixtral is a Mixture-of-Experts (MoE) architecture that uses
    /// the same tensor naming pattern as Mistral. Differentiation is done
    /// via the `general.architecture` metadata key.
    Mixtral,
}

impl Architecture {
    /// Get the layer prefix pattern for this architecture
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        match self {
            Architecture::Qwen2 => format!("blk.{}", layer_idx),
            Architecture::LLaMA => format!("transformer.layers.{}", layer_idx),
            Architecture::Mistral | Architecture::Yi | Architecture::Mixtral => {
                format!("model.layers.{}", layer_idx)
            }
        }
    }

    /// Get the architecture name for logging
    pub fn name(&self) -> &'static str {
        match self {
            Architecture::Qwen2 => "Qwen2",
            Architecture::LLaMA => "LLaMA",
            Architecture::Mistral => "Mistral",
            Architecture::Yi => "Yi",
        }
    }

    /// Detect model architecture from available tensor names
    ///
    /// Scans tensor names to identify the architecture pattern:
    /// - Qwen2: tensors start with `blk.N.`
    /// - LLaMA: tensors start with `transformer.layers.N.`
    /// - Mistral/Yi: tensors start with `model.layers.N.`
    ///
    /// Note: Mistral and Yi share the same tensor naming pattern.
    /// Differentiation is done via `general.architecture` metadata key
    /// which should be checked after detection.
    pub fn detect(tensor_names: &HashSet<String>) -> Result<Self, HipError> {
        // Check for Qwen2 pattern: blk.0.*
        let qwen2_pattern = "blk.0.";
        let has_qwen2 = tensor_names
            .iter()
            .any(|name| name.starts_with(qwen2_pattern));

        if has_qwen2 {
            println!("Detected architecture: Qwen2 (pattern: {})", qwen2_pattern);
            return Ok(Architecture::Qwen2);
        }

        // Check for LLaMA pattern: transformer.layers.0.*
        let llama_pattern = "transformer.layers.0.";
        let has_llama = tensor_names
            .iter()
            .any(|name| name.starts_with(llama_pattern));

        if has_llama {
            println!("Detected architecture: LLaMA (pattern: {})", llama_pattern);
            return Ok(Architecture::LLaMA);
        }

        // Check for Mistral pattern: model.layers.0.*
        let mistral_pattern = "model.layers.0.";
        let has_mistral = tensor_names
            .iter()
            .any(|name| name.starts_with(mistral_pattern));

        if has_mistral {
            println!(
                "Detected architecture: Mistral (pattern: {})",
                mistral_pattern
            );
            return Ok(Architecture::Mistral);
        }

        // Unknown architecture - try to provide helpful error
        let sample_tensors: Vec<_> = tensor_names
            .iter()
            .filter(|name| name.contains('.'))
            .take(10)
            .collect();

        Err(HipError::GenericError(format!(
            "Unable to detect model architecture from tensor names. \
             Expected patterns like 'blk.0.*' (Qwen2), 'transformer.layers.0.*' (LLaMA), \
             or 'model.layers.0.*' (Mistral/Yi). \
             Sample tensors found: {:?}",
            sample_tensors
        )))
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_detection() {
        let tensor_names: HashSet<String> = vec![
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_k.weight".to_string(),
            "blk.0.ffn_up.weight".to_string(),
        ]
        .into_iter()
        .collect();

        let arch = Architecture::detect(&tensor_names).unwrap();
        assert_eq!(arch, Architecture::Qwen2);
        assert_eq!(arch.name(), "Qwen2");
        assert_eq!(arch.layer_prefix(0), "blk.0");
    }

    #[test]
    fn test_llama_detection() {
        let tensor_names: HashSet<String> = vec![
            "transformer.layers.0.attention_q.weight".to_string(),
            "transformer.layers.0.attention_k.weight".to_string(),
            "transformer.layers.0.ffn_gate.weight".to_string(),
        ]
        .into_iter()
        .collect();

        let arch = Architecture::detect(&tensor_names).unwrap();
        assert_eq!(arch, Architecture::LLaMA);
        assert_eq!(arch.name(), "LLaMA");
        assert_eq!(arch.layer_prefix(0), "transformer.layers.0");
    }

    #[test]
    fn test_mistral_detection() {
        let tensor_names: HashSet<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            "model.layers.0.mlp.gate_proj.weight".to_string(),
        ]
        .into_iter()
        .collect();

        let arch = Architecture::detect(&tensor_names).unwrap();
        assert_eq!(arch, Architecture::Mistral);
        assert_eq!(arch.name(), "Mistral");
        assert_eq!(arch.layer_prefix(0), "model.layers.0");
    }

    #[test]
    fn test_yi_variant_layer_prefix() {
        // Yi shares the same tensor pattern as Mistral
        // Test that layer_prefix works correctly for Yi
        let arch = Architecture::Yi;
        assert_eq!(arch.name(), "Yi");
        assert_eq!(arch.layer_prefix(0), "model.layers.0");
        assert_eq!(arch.layer_prefix(5), "model.layers.5");
    }

    #[test]
    fn test_unknown_architecture_error() {
        let tensor_names: HashSet<String> = vec![
            "unknown.prefix.weight".to_string(),
            "another.unknown.tensor".to_string(),
        ]
        .into_iter()
        .collect();

        let result = Architecture::detect(&tensor_names);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unable to detect model architecture"));
    }
}
