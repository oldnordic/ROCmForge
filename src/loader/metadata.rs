//! GGUF metadata structures

use serde::Serialize;

/// GGUF metadata extracted from file header
#[derive(Debug, Clone, Serialize)]
pub struct GgufMetadata {
    pub architecture: String,
    pub file_type: u32,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>, // MQA/GQA support
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub use_rotary_embeddings: bool,
    #[serde(skip_serializing)]
    pub embedded_tokenizer_json: Option<String>,
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            file_type: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: None,
            hidden_size: 0,
            intermediate_size: 0,
            head_dim: 0,
            max_position_embeddings: 2048,
            vocab_size: 0,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
            embedded_tokenizer_json: None,
        }
    }
}

impl GgufMetadata {
    /// Update metadata from a key-value pair
    ///
    /// This is used during GGUF file parsing to populate metadata
    /// from the key-value pairs in the file header.
    pub fn update_from_kv(&mut self, key: &str, value: &str) {
        match key {
            "general.architecture" => self.architecture = value.to_string(),
            "general.file_type" => self.file_type = value.parse().unwrap_or(0),
            // GLM-specific keys
            "glm.n_layers" => self.num_layers = value.parse().unwrap_or(0),
            "glm.n_heads" => self.num_heads = value.parse().unwrap_or(0),
            "glm.n_embd" => self.hidden_size = value.parse().unwrap_or(0),
            "glm.intermediate_size" => self.intermediate_size = value.parse().unwrap_or(0),
            "glm.head_dim" => self.head_dim = value.parse().unwrap_or(0),
            "glm.max_position_embeddings" => {
                self.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "glm.vocab_size" => self.vocab_size = value.parse().unwrap_or(0),
            "glm.rms_norm_eps" => self.rms_norm_eps = value.parse().unwrap_or(1e-6),
            // Gemma 3-specific keys (actual keys from GGUF file)
            "gemma3.embedding_length" => self.hidden_size = value.parse().unwrap_or(0),
            "gemma3.block_count" => self.num_layers = value.parse().unwrap_or(0),
            "gemma3.feed_forward_length" => self.intermediate_size = value.parse().unwrap_or(0),
            "gemma3.attention.head_count" => self.num_heads = value.parse().unwrap_or(0),
            "gemma3.attention.head_count_kv" => {
                self.num_kv_heads = Some(value.parse().unwrap_or(0))
            }
            "gemma3.attention.key_length" => self.head_dim = value.parse().unwrap_or(0),
            "gemma3.attention.value_length" => self.head_dim = value.parse().unwrap_or(0),
            "gemma3.context_length" => {
                self.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "gemma3.attention.layer_norm_rms_epsilon" => {
                self.rms_norm_eps = value.parse().unwrap_or(1e-6)
            }
            // Qwen2-specific keys
            "qwen2.block_count" => self.num_layers = value.parse().unwrap_or(0),
            "qwen2.attention.head_count" => self.num_heads = value.parse().unwrap_or(0),
            "qwen2.embedding_length" => self.hidden_size = value.parse().unwrap_or(0),
            "qwen2.intermediate_size" => self.intermediate_size = value.parse().unwrap_or(0),
            "qwen2.rope.dimension_count" => self.head_dim = value.parse().unwrap_or(0),
            "qwen2.max_position_embeddings" => {
                self.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "qwen2.vocab_size" => self.vocab_size = value.parse().unwrap_or(0),
            // Llama-specific keys (also used by some Qwen models)
            "llama.block_count" => self.num_layers = value.parse().unwrap_or(0),
            "llama.attention.head_count" => self.num_heads = value.parse().unwrap_or(0),
            "llama.attention.head_count_kv" => {
                self.num_kv_heads = Some(value.parse().unwrap_or(0))
            }
            "llama.embedding_length" => self.hidden_size = value.parse().unwrap_or(0),
            "llama.feed_forward_length" => self.intermediate_size = value.parse().unwrap_or(0),
            "llama.rope.dimension_count" => {
                // Usually head_dim = hidden_size / num_heads, but this gives rope dimensions
                self.head_dim = value.parse().unwrap_or(0)
            }
            "llama.max_position_embeddings" => {
                self.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "llama.vocab_size" => self.vocab_size = value.parse().unwrap_or(0),
            // Mistral-specific keys
            "mistral.n_layers" | "mistral.block_count" => self.num_layers = value.parse().unwrap_or(0),
            "mistral.attention.head_count" | "mistral.n_heads" => self.num_heads = value.parse().unwrap_or(0),
            "mistral.attention.head_count_kv" => {
                self.num_kv_heads = Some(value.parse().unwrap_or(0))
            }
            "mistral.embedding_length" | "mistral.hidden_size" | "mistral.n_embd" => {
                self.hidden_size = value.parse().unwrap_or(0)
            }
            "mistral.feed_forward_length" | "mistral.intermediate_size" => {
                self.intermediate_size = value.parse().unwrap_or(0)
            }
            "mistral.attention.key_length" | "mistral.head_dim" => {
                self.head_dim = value.parse().unwrap_or(0)
            }
            "mistral.max_position_embeddings" => {
                self.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "mistral.vocab_size" => self.vocab_size = value.parse().unwrap_or(0),
            // Common RMS norm epsilon key names
            "llama.attention.layer_norm_rms_epsilon"
            | "qwen2.attention.layer_norm_rms_epsilon"
            | "qwen2.attention_norm_epsilon"
            | "mistral.attention.layer_norm_rms_epsilon"
            | "mistral.norm_eps" => {
                self.rms_norm_eps = value.parse().unwrap_or(1e-6)
            }
            // Tokenizer JSON (embedded in some models)
            "tokenizer.json" => {
                if self.embedded_tokenizer_json.is_none() {
                    self.embedded_tokenizer_json = Some(value.to_string());
                }
            }
            key if key.ends_with(".tokenizer_json") => {
                if self.embedded_tokenizer_json.is_none() {
                    self.embedded_tokenizer_json = Some(value.to_string());
                }
            }
            // Ignore unknown keys
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_metadata() {
        let meta = GgufMetadata::default();
        assert_eq!(meta.architecture, "unknown");
        assert_eq!(meta.max_position_embeddings, 2048);
        assert_eq!(meta.rms_norm_eps, 1e-6);
    }

    #[test]
    fn test_update_from_kv() {
        let mut meta = GgufMetadata::default();
        meta.update_from_kv("general.architecture", "llama");
        assert_eq!(meta.architecture, "llama");

        meta.update_from_kv("llama.block_count", "32");
        assert_eq!(meta.num_layers, 32);

        meta.update_from_kv("llama.vocab_size", "32000");
        assert_eq!(meta.vocab_size, 32000);
    }

    #[test]
    fn test_kv_heads_support() {
        let mut meta = GgufMetadata::default();
        meta.update_from_kv("llama.attention.head_count_kv", "4");
        assert_eq!(meta.num_kv_heads, Some(4));
    }

    #[test]
    fn test_mistral_metadata_parsing() {
        let mut meta = GgufMetadata::default();

        // Test all Mistral key variants
        meta.update_from_kv("mistral.block_count", "32");
        assert_eq!(meta.num_layers, 32);

        meta.update_from_kv("mistral.attention.head_count", "32");
        assert_eq!(meta.num_heads, 32);

        meta.update_from_kv("mistral.attention.head_count_kv", "8");
        assert_eq!(meta.num_kv_heads, Some(8));

        meta.update_from_kv("mistral.hidden_size", "4096");
        assert_eq!(meta.hidden_size, 4096);

        meta.update_from_kv("mistral.intermediate_size", "14336");
        assert_eq!(meta.intermediate_size, 14336);

        meta.update_from_kv("mistral.head_dim", "128");
        assert_eq!(meta.head_dim, 128);

        meta.update_from_kv("mistral.max_position_embeddings", "32768");
        assert_eq!(meta.max_position_embeddings, 32768);

        meta.update_from_kv("mistral.vocab_size", "32000");
        assert_eq!(meta.vocab_size, 32000);

        meta.update_from_kv("mistral.norm_eps", "0.00001");
        assert_eq!(meta.rms_norm_eps, 0.00001);

        // Test alternative key names
        let mut meta2 = GgufMetadata::default();
        meta2.update_from_kv("mistral.n_layers", "24");
        assert_eq!(meta2.num_layers, 24);

        meta2.update_from_kv("mistral.embedding_length", "2048");
        assert_eq!(meta2.hidden_size, 2048);

        meta2.update_from_kv("mistral.n_heads", "20");
        assert_eq!(meta2.num_heads, 20);
    }
}
