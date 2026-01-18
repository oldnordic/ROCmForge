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
}

impl Architecture {
    /// Get the layer prefix pattern for this architecture
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        match self {
            Architecture::Qwen2 => format!("blk.{}", layer_idx),
            Architecture::LLaMA => format!("transformer.layers.{}", layer_idx),
            Architecture::Mistral => format!("model.layers.{}", layer_idx),
        }
    }

    /// Get the architecture name for logging
    pub fn name(&self) -> &'static str {
        match self {
            Architecture::Qwen2 => "Qwen2",
            Architecture::LLaMA => "LLaMA",
            Architecture::Mistral => "Mistral",
        }
    }

    /// Detect model architecture from available tensor names
    ///
    /// Scans tensor names to identify the architecture pattern:
    /// - Qwen2: tensors start with `blk.N.`
    /// - LLaMA: tensors start with `transformer.layers.N.`
    /// - Mistral: tensors start with `model.layers.N.`
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
             Expected patterns like 'blk.0.*', 'transformer.layers.0.*', or 'model.layers.0.*'. \
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
