//! Tokenizer module.
//!
//! Provides text tokenization and detokenization.  The supported formats are
//! Byte-Pair Encoding (BPE) for Qwen-style models and SentencePiece (Unigram)
//! for Gemma-style models.

mod bpe;
mod spm;

pub use bpe::{BpeTokenizer, PreTokenizerType, VocabType};
pub use spm::SpmTokenizer;

/// Common tokenizer operations.
pub trait Tokenizer {
    /// Encode text to token IDs.
    fn encode(&self, text: &str, add_special: bool) -> Vec<u32>;

    /// Decode token IDs to text.
    fn decode(&self, tokens: &[u32], skip_special: bool) -> String;

    /// Decode a single token ID to text.
    fn decode_token(&self, id: u32) -> String;

    /// Is this token an end-of-generation signal?
    fn is_eog(&self, id: u32) -> bool;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    fn bos_id(&self) -> Option<u32>;

    fn eos_id(&self) -> Option<u32>;

    fn add_bos(&self) -> bool;

    fn add_eos(&self) -> bool;
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.encode(text, add_special)
    }

    fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        self.decode(tokens, skip_special)
    }

    fn decode_token(&self, id: u32) -> String {
        self.decode_token(id)
    }

    fn is_eog(&self, id: u32) -> bool {
        self.is_eog(id)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn bos_id(&self) -> Option<u32> {
        self.bos_id()
    }

    fn eos_id(&self) -> Option<u32> {
        self.eos_id()
    }

    fn add_bos(&self) -> bool {
        self.add_bos()
    }

    fn add_eos(&self) -> bool {
        self.add_eos()
    }
}

impl Tokenizer for SpmTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.encode(text, add_special)
    }

    fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        self.decode(tokens, skip_special)
    }

    fn decode_token(&self, id: u32) -> String {
        self.decode_token(id)
    }

    fn is_eog(&self, id: u32) -> bool {
        self.is_eog(id)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn bos_id(&self) -> Option<u32> {
        self.bos_id()
    }

    fn eos_id(&self) -> Option<u32> {
        self.eos_id()
    }

    fn add_bos(&self) -> bool {
        self.add_bos()
    }

    fn add_eos(&self) -> bool {
        self.add_eos()
    }
}

/// Unified tokenizer handle.
#[derive(Clone, Debug)]
pub enum TokenizerHandle {
    Bpe(BpeTokenizer),
    Spm(SpmTokenizer),
}

impl Tokenizer for TokenizerHandle {
    fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        match self {
            TokenizerHandle::Bpe(t) => t.encode(text, add_special),
            TokenizerHandle::Spm(t) => t.encode(text, add_special),
        }
    }

    fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        match self {
            TokenizerHandle::Bpe(t) => t.decode(tokens, skip_special),
            TokenizerHandle::Spm(t) => t.decode(tokens, skip_special),
        }
    }

    fn decode_token(&self, id: u32) -> String {
        match self {
            TokenizerHandle::Bpe(t) => t.decode_token(id),
            TokenizerHandle::Spm(t) => t.decode_token(id),
        }
    }

    fn is_eog(&self, id: u32) -> bool {
        match self {
            TokenizerHandle::Bpe(t) => t.is_eog(id),
            TokenizerHandle::Spm(t) => t.is_eog(id),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            TokenizerHandle::Bpe(t) => t.vocab_size(),
            TokenizerHandle::Spm(t) => t.vocab_size(),
        }
    }

    fn bos_id(&self) -> Option<u32> {
        match self {
            TokenizerHandle::Bpe(t) => t.bos_id(),
            TokenizerHandle::Spm(t) => t.bos_id(),
        }
    }

    fn eos_id(&self) -> Option<u32> {
        match self {
            TokenizerHandle::Bpe(t) => t.eos_id(),
            TokenizerHandle::Spm(t) => t.eos_id(),
        }
    }

    fn add_bos(&self) -> bool {
        match self {
            TokenizerHandle::Bpe(t) => t.add_bos(),
            TokenizerHandle::Spm(t) => t.add_bos(),
        }
    }

    fn add_eos(&self) -> bool {
        match self {
            TokenizerHandle::Bpe(t) => t.add_eos(),
            TokenizerHandle::Spm(t) => t.add_eos(),
        }
    }
}

// Inherent forwarding methods so callers don't have to import the `Tokenizer`
// trait to use the common handle.
impl TokenizerHandle {
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        Tokenizer::encode(self, text, add_special)
    }

    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        Tokenizer::decode(self, tokens, skip_special)
    }

    pub fn decode_token(&self, id: u32) -> String {
        Tokenizer::decode_token(self, id)
    }

    pub fn is_eog(&self, id: u32) -> bool {
        Tokenizer::is_eog(self, id)
    }

    pub fn vocab_size(&self) -> usize {
        Tokenizer::vocab_size(self)
    }

    pub fn bos_id(&self) -> Option<u32> {
        Tokenizer::bos_id(self)
    }

    pub fn eos_id(&self) -> Option<u32> {
        Tokenizer::eos_id(self)
    }

    pub fn add_bos(&self) -> bool {
        Tokenizer::add_bos(self)
    }

    pub fn add_eos(&self) -> bool {
        Tokenizer::add_eos(self)
    }
}
