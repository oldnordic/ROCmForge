//! GGUF metadata — arch-prefix KV resolution.
//!
//! All hyperparameters are resolved dynamically from raw KV pairs:
//!   1. `<arch>.<param>`          e.g. "qwen2.block_count"
//!   2. `llama.<param>`           llama.cpp compatibility fallback
//!   3. bare `<param>`
//!
//! Adapted from Memoria — verified correct against Qwen2.5, LLaMA, Phi3.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub raw: HashMap<String, String>,
    pub architecture: String,
}

impl GgufMetadata {
    pub fn from_kv(kv: HashMap<String, String>) -> Self {
        let architecture = kv
            .get("general.architecture")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        Self {
            raw: kv,
            architecture,
        }
    }

    // ── Resolution ────────────────────────────────────────────────────────────

    pub fn get(&self, param: &str) -> Option<&str> {
        let arch_key = format!("{}.{}", self.architecture, param);
        if let Some(v) = self.raw.get(&arch_key) {
            return Some(v.as_str());
        }
        if self.architecture != "llama" {
            let llama_key = format!("llama.{}", param);
            if let Some(v) = self.raw.get(&llama_key) {
                return Some(v.as_str());
            }
        }
        self.raw.get(param).map(|s| s.as_str())
    }

    fn resolve_usize(&self, keys: &[&str]) -> Option<usize> {
        for key in keys {
            if let Some(v) = self.get(key).and_then(|s| s.parse::<usize>().ok()) {
                if v > 0 {
                    return Some(v);
                }
            }
        }
        None
    }

    fn resolve_f32(&self, keys: &[&str]) -> Option<f32> {
        for key in keys {
            if let Some(v) = self.get(key).and_then(|s| s.parse::<f32>().ok()) {
                if v > 0.0 {
                    return Some(v);
                }
            }
        }
        None
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn block_count(&self) -> usize {
        self.resolve_usize(&["block_count", "n_layers", "num_hidden_layers"])
            .unwrap_or(0)
    }

    pub fn embedding_length(&self) -> usize {
        self.resolve_usize(&["embedding_length", "hidden_size", "n_embd"])
            .unwrap_or(0)
    }

    pub fn feed_forward_length(&self) -> usize {
        self.resolve_usize(&["feed_forward_length", "intermediate_size", "ffn_dim"])
            .unwrap_or(0)
    }

    pub fn attention_head_count(&self) -> usize {
        self.resolve_usize(&["attention.head_count", "n_heads", "num_attention_heads"])
            .unwrap_or(0)
    }

    pub fn attention_head_count_kv(&self) -> Option<usize> {
        self.resolve_usize(&[
            "attention.head_count_kv",
            "n_heads_kv",
            "num_key_value_heads",
        ])
    }

    pub fn context_length(&self) -> usize {
        self.resolve_usize(&["context_length", "max_position_embeddings", "n_ctx"])
            .unwrap_or(2048)
    }

    pub fn vocab_size(&self) -> usize {
        self.resolve_usize(&["vocab_size"]).unwrap_or(0)
    }

    pub fn rms_norm_eps(&self, default: f32) -> f32 {
        self.resolve_f32(&[
            "attention.layer_norm_rms_epsilon",
            "attention_norm_epsilon",
            "norm_eps",
            "rms_norm_eps",
        ])
        .unwrap_or(default)
    }

    pub fn rope_freq_base(&self, default: f32) -> f32 {
        // Also try un-prefixed "rope.freq_base" directly in raw KV
        self.resolve_f32(&["rope.freq_base", "rope_theta"])
            .or_else(|| {
                self.raw
                    .get("rope.freq_base")
                    .and_then(|s| s.parse::<f32>().ok())
                    .filter(|&v| v > 0.0)
            })
            .unwrap_or(default)
    }

    /// Head dimension: tries explicit key first, then computes from hidden/heads.
    pub fn head_dim(&self) -> usize {
        for key in &["attention.key_length", "head_dim", "rope.dimension_count"] {
            if let Some(v) = self.resolve_usize(&[key]) {
                return v;
            }
        }
        let hidden = self.embedding_length();
        let heads = self.attention_head_count();
        if hidden > 0 && heads > 0 {
            hidden / heads
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(pairs: &[(&str, &str)]) -> GgufMetadata {
        let kv = pairs
            .iter()
            .map(|&(k, v)| (k.to_string(), v.to_string()))
            .collect();
        GgufMetadata::from_kv(kv)
    }

    #[test]
    fn arch_prefix() {
        let m = make(&[
            ("general.architecture", "qwen2"),
            ("qwen2.block_count", "28"),
        ]);
        assert_eq!(m.block_count(), 28);
    }

    #[test]
    fn llama_fallback() {
        let m = make(&[
            ("general.architecture", "newarch"),
            ("llama.block_count", "32"),
        ]);
        assert_eq!(m.block_count(), 32);
    }

    #[test]
    fn head_dim_computed() {
        let m = make(&[
            ("general.architecture", "llama"),
            ("llama.embedding_length", "4096"),
            ("llama.attention.head_count", "32"),
        ]);
        assert_eq!(m.head_dim(), 128);
    }

    #[test]
    fn rope_freq_from_kv() {
        let m = make(&[
            ("general.architecture", "qwen2"),
            ("qwen2.rope.freq_base", "1000000"),
        ]);
        assert_eq!(m.rope_freq_base(10000.0), 1_000_000.0);
    }

    #[test]
    fn kv_heads_gqa() {
        let m = make(&[
            ("general.architecture", "qwen2"),
            ("qwen2.attention.head_count_kv", "2"),
        ]);
        assert_eq!(m.attention_head_count_kv(), Some(2));
    }
}
