//! Tokenizer abstraction backed by Hugging Face `tokenizers`, with a fallback.

use crate::loader::gguf::GgufLoader;
use crate::models::{cached_embedded_tokenizer, CachedTokenizer};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::warn;

#[derive(Clone, Default)]
pub struct TokenizerAdapter {
    inner: Option<Arc<Tokenizer>>,
}

static TOKENIZER_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static TOKENIZER_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

impl TokenizerAdapter {
    /// Create adapter from optional tokenizer JSON path.
    pub fn from_path(path: Option<&str>) -> Self {
        Self::from_spec(path, None)
    }

    /// Create adapter from either a path or embedded JSON.
    pub fn from_spec(path: Option<&str>, embedded_json: Option<&str>) -> Self {
        if let Some(path) = path {
            if let Some(inner) = Self::load_tokenizer_from_path(path) {
                return Self { inner: Some(inner) };
            }
        }

        if let Some(json) = embedded_json {
            match Tokenizer::from_str(json) {
                Ok(tok) => {
                    return Self {
                        inner: Some(Arc::new(tok)),
                    }
                }
                Err(err) => {
                    warn!("Failed to build tokenizer from embedded JSON: {}", err);
                }
            }
        }

        Self::default()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(tok) = &self.inner {
            tok.encode(text, true)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_else(|_| fallback_encode(text))
        } else {
            fallback_encode(text)
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        if let Some(tok) = &self.inner {
            tok.decode(tokens, true)
                .unwrap_or_else(|_| fallback_decode(tokens))
        } else {
            fallback_decode(tokens)
        }
    }

    pub fn decode_token(&self, token: u32) -> String {
        if let Some(tok) = &self.inner {
            tok.id_to_token(token)
                .unwrap_or_else(|| fallback_token_text(token))
        } else {
            fallback_token_text(token)
        }
    }

    fn load_tokenizer_from_path(path: &str) -> Option<Arc<Tokenizer>> {
        match Tokenizer::from_file(path) {
            Ok(tok) => Some(Arc::new(tok)),
            Err(err) => {
                warn!("Failed to load tokenizer {}: {}", path, err);
                None
            }
        }
    }
}

pub fn infer_tokenizer_path(gguf_path: &str) -> Option<String> {
    let gguf = Path::new(gguf_path);
    let dir = gguf
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(stem) = gguf.file_stem().and_then(|s| s.to_str()) {
        candidates.push(dir.join(format!("{}.tokenizer.json", stem)));
        candidates.push(dir.join(format!("{}.tokenizer.model", stem)));
        candidates.push(dir.join(format!("{}.json", stem)));
        candidates.push(dir.join(format!("{}.model", stem)));
    }

    candidates.push(dir.join("tokenizer.json"));
    candidates.push(dir.join("tokenizer.model"));

    for path in candidates {
        if path.exists() {
            return Some(path.to_string_lossy().into_owned());
        }
    }

    None
}

/// Attempt to load an embedded tokenizer JSON blob from a GGUF file.
pub fn embedded_tokenizer_from_gguf(path: &str) -> Option<CachedTokenizer> {
    if let Some(data) = cached_embedded_tokenizer(path) {
        TOKENIZER_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return Some(data);
    }
    match GgufLoader::metadata_from_file(path) {
        Ok(meta) => meta.embedded_tokenizer_json.map(|json| {
            TOKENIZER_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            CachedTokenizer {
                json,
                cached: false,
                refreshed: true,
            }
        }),
        Err(err) => {
            warn!("Failed to read embedded tokenizer from {}: {}", path, err);
            None
        }
    }
}

pub fn tokenizer_cache_counters() -> (u64, u64) {
    (
        TOKENIZER_CACHE_HITS.load(Ordering::Relaxed),
        TOKENIZER_CACHE_MISSES.load(Ordering::Relaxed),
    )
}

fn fallback_encode(text: &str) -> Vec<u32> {
    if text.trim().is_empty() {
        return vec![0];
    }
    text.split_whitespace()
        .map(|word| {
            let mut hash = 1u32;
            for byte in word.bytes() {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
            }
            hash % 32000
        })
        .collect()
}

fn fallback_decode(tokens: &[u32]) -> String {
    tokens.iter().map(|&t| fallback_token_text(t)).collect()
}

fn fallback_token_text(token: u32) -> String {
    format!(" token_{}", token)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_TOKENIZER: &str = r#"{
        "version":"1.0",
        "truncation":null,
        "padding":null,
        "added_tokens":[],
        "normalizer":null,
        "pre_tokenizer":{"type":"Whitespace"},
        "post_processor":null,
        "decoder":null,
        "model":{"type":"WordLevel","vocab":{"hello":0,"world":1,"<unk>":2},"unk_token":"<unk>"}
    }"#;

    #[test]
    fn tokenizer_from_embedded_json() {
        let adapter = TokenizerAdapter::from_spec(None, Some(SIMPLE_TOKENIZER));
        let tokens = adapter.encode("hello world");
        assert_eq!(tokens, vec![0, 1]);
        let text = adapter.decode(&tokens);
        assert!(text.contains("hello"));
    }
}
