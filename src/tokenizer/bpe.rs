//! Byte-Pair Encoding (BPE) tokenizer.
//!
//! Supports GPT-2 style BPE (Qwen2.5) with byte-level encoding.
//! Build with `BpeTokenizer::from_gguf(tokenizer_data)`.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

// ── Pre-tokenizer regex for Qwen2 ───────────────────────────────────────────────

static REGEX_QWEN2: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
    ).unwrap()
});

// ── Internal types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct BytesKey(Vec<u8>);

impl Hash for BytesKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

pub type BpeRank = i32;

#[derive(Clone, Debug)]
struct Symbol {
    text: Vec<u8>,
    prev: i32,
    next: i32,
}

enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}

// ── Public enums ───────────────────────────────────────────────────────────────

/// Vocabulary type for the tokenizer.
///
/// Only BPE is supported in rocmforge (no SentencePiece).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VocabType {
    Bpe, // GPT-2 style byte-level BPE
}

/// Pre-tokenizer type for splitting text before BPE.
///
/// Only Qwen2 is supported in rocmforge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreTokenizerType {
    Qwen2,
}

// ── Tokenizer ───────────────────────────────────────────────────────────────────

/// BPE tokenizer for Qwen2.5 models.
///
/// Encodes text to token IDs and decodes token IDs back to text.
/// Built from GGUF tokenizer data using `from_gguf()`.
#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<BytesKey, u32>,
    merges: HashMap<(BytesKey, BytesKey), BpeRank>,
    special_tokens: HashSet<u32>,
    pre_type: PreTokenizerType,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    byte_encoder: HashMap<u8, String>,
    byte_decoder: HashMap<char, u8>,
}

impl BpeTokenizer {
    /// Build from GGUF tokenizer arrays.
    pub fn from_gguf(data: &crate::loader::TokenizerData) -> Self {
        // Determine pre-tokenizer type
        let pre_type = match data.pre.as_deref() {
            Some("qwen2") => PreTokenizerType::Qwen2,
            _ => PreTokenizerType::Qwen2, // Default to Qwen2 for rocmforge
        };

        Self::new(data.tokens.clone(), data.merges.clone(),
                 data.bos_token_id, data.eos_token_id, data.unk_token_id,
                 data.add_bos, data.add_eos, pre_type)
    }

    fn new(vocab: Vec<Vec<u8>>, merges: Vec<(Vec<u8>, Vec<u8>)>,
            bos: Option<u32>, eos: Option<u32>, unk: Option<u32>,
            add_bos: bool, add_eos: bool, pre_type: PreTokenizerType) -> Self {
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        let mut special_tokens = HashSet::new();
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(BytesKey(token.clone()), id as u32);
            if token.starts_with(b"<") && token.ends_with(b">") {
                special_tokens.insert(id as u32);
            }
        }
        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, (a, b)) in merges.iter().enumerate() {
            merge_map.insert((BytesKey(a.clone()), BytesKey(b.clone())), rank as BpeRank);
        }
        let byte_encoder = Self::build_byte_encoder();
        let byte_decoder = byte_encoder
            .iter()
            .filter_map(|(&b, s)| s.chars().next().map(|ch| (ch, b)))
            .collect();
        Self {
            vocab,
            token_to_id,
            merges: merge_map,
            special_tokens,
            pre_type,
            bos_id: bos,
            eos_id: eos,
            unk_id: unk,
            add_bos,
            add_eos,
            byte_encoder,
            byte_decoder,
        }
    }

    // ── Encode ──────────────────────────────────────────────────────────────────

    /// Encode text to token IDs.
    ///
    /// If `add_special` is true, adds BOS/EOS tokens as configured.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos_id {
                tokens.push(b);
            }
        }
        for fragment in self.split_by_special_tokens(text) {
            match fragment {
                Fragment::Text(t) => {
                    for word in self.regex_split(t) {
                        tokens.extend(self.tokenize_word(&word));
                    }
                }
                Fragment::Special(id) => tokens.push(id),
            }
        }
        if add_special && self.add_eos {
            if let Some(e) = self.eos_id {
                tokens.push(e);
            }
        }
        tokens
    }

    fn split_by_special_tokens<'a>(&'a self, text: &'a str) -> Vec<Fragment<'a>> {
        let mut fragments = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            let mut found_special = false;
            for (id, token) in self.vocab.iter().enumerate() {
                if self.special_tokens.contains(&(id as u32)) {
                    if let Ok(s) = std::str::from_utf8(token) {
                        if remaining.starts_with(s) {
                            fragments.push(Fragment::Special(id as u32));
                            remaining = &remaining[s.len()..];
                            found_special = true;
                            break;
                        }
                    }
                }
            }
            if !found_special {
                let mut text_end = remaining.len();
                for (id, token) in self.vocab.iter().enumerate() {
                    if self.special_tokens.contains(&(id as u32)) {
                        if let Ok(s) = std::str::from_utf8(token) {
                            if let Some(pos) = remaining.find(s) {
                                if pos > 0 && pos < text_end {
                                    text_end = pos;
                                }
                            }
                        }
                    }
                }
                let end = text_end.max(1).min(remaining.len());
                fragments.push(Fragment::Text(&remaining[..end]));
                remaining = &remaining[end..];
            }
        }
        fragments
    }

    fn regex_split(&self, text: &str) -> Vec<String> {
        let re = &*REGEX_QWEN2;
        let mut pieces: Vec<String> = re.find_iter(text).map(|m| m.as_str().to_string()).collect();
        // Qwen2: split punctuation+newline chunks (e.g. "?\n") into ["?", "\n"]
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces.drain(..) {
            if piece.len() > 1 && (piece.ends_with('\n') || piece.ends_with('\r')) {
                let mut cut = piece.len();
                while cut > 0 {
                    // SAFETY: cut > 0 guarantees piece[..cut] is non-empty
                    let ch = match piece[..cut].chars().next_back() {
                        Some(c) => c,
                        None => break, // defensive: should never happen when cut > 0
                    };
                    if ch == '\n' || ch == '\r' {
                        cut -= ch.len_utf8();
                    } else {
                        break;
                    }
                }
                let head = &piece[..cut];
                let tail = &piece[cut..];
                if !head.is_empty()
                    && head
                        .chars()
                        .last()
                        .map(|c| c.is_ascii_punctuation())
                        .unwrap_or(false)
                    && !tail.is_empty()
                {
                    out.push(head.to_string());
                    for ch in tail.chars() {
                        out.push(ch.to_string());
                    }
                    continue;
                }
            }
            out.push(piece);
        }
        out
    }

    fn tokenize_word(&self, word: &str) -> Vec<u32> {
        // Map raw bytes through byte-encoder alphabet
        let mut encoded = String::new();
        for b in word.bytes() {
            if let Some(s) = self.byte_encoder.get(&b) {
                encoded.push_str(s);
            } else {
                encoded.push(b as char);
            }
        }
        // Split into UTF-8 character symbols
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut offset = 0;
        let bytes = encoded.as_bytes();
        while offset < bytes.len() {
            let clen = utf8_char_len(bytes[offset]);
            let end = (offset + clen).min(bytes.len());
            symbols.push(Symbol {
                text: encoded[offset..end].as_bytes().to_vec(),
                prev: symbols.len() as i32 - 1,
                next: -1,
            });
            offset = end;
        }
        for i in 0..symbols.len() {
            if i + 1 < symbols.len() {
                symbols[i].next = (i + 1) as i32;
            }
        }
        if symbols.is_empty() {
            return Vec::new();
        }
        // BPE merge loop
        loop {
            let mut best_rank = BpeRank::MAX;
            let mut best_idx: i32 = -1;
            let mut i = 0i32;
            while i >= 0 && (i as usize) < symbols.len() {
                let next = symbols[i as usize].next;
                if next >= 0 && !symbols[next as usize].text.is_empty() {
                    if let Some(&rank) = self.merges.get(&(
                        BytesKey(symbols[i as usize].text.clone()),
                        BytesKey(symbols[next as usize].text.clone()),
                    )) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }
                i = next;
            }
            if best_idx < 0 || best_rank == BpeRank::MAX {
                break;
            }
            let left = best_idx as usize;
            let right = symbols[left].next as usize;
            let next_next = symbols[right].next;
            let mut merged =
                Vec::with_capacity(symbols[left].text.len() + symbols[right].text.len());
            merged.extend_from_slice(&symbols[left].text);
            merged.extend_from_slice(&symbols[right].text);
            symbols[left].text = merged;
            symbols[left].next = next_next;
            if next_next >= 0 {
                symbols[next_next as usize].prev = best_idx;
            }
            symbols[right].text.clear();
        }
        // Collect tokens
        let mut tokens = Vec::new();
        let mut idx = 0i32;
        while idx >= 0 && (idx as usize) < symbols.len() {
            if !symbols[idx as usize].text.is_empty() {
                let key = BytesKey(symbols[idx as usize].text.clone());
                if let Some(&id) = self.token_to_id.get(&key) {
                    tokens.push(id);
                } else {
                    // Byte fallback
                    for &b in &symbols[idx as usize].text {
                        if let Some(s) = self.byte_encoder.get(&b) {
                            let k = BytesKey(s.as_bytes().to_vec());
                            tokens.push(
                                *self
                                    .token_to_id
                                    .get(&k)
                                    .unwrap_or(&self.unk_id.unwrap_or(0)),
                            );
                        } else {
                            tokens.push(self.unk_id.unwrap_or(0));
                        }
                    }
                }
                idx = symbols[idx as usize].next;
            } else {
                idx += 1;
            }
        }
        tokens
    }

    // ── Decode ──────────────────────────────────────────────────────────────────

    /// Decode token IDs to text.
    ///
    /// If `skip_special` is true, skips special tokens in the output.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        let mut raw = Vec::new();
        for &id in tokens {
            if skip_special && self.special_tokens.contains(&id) {
                continue;
            }
            if let Some(bytes) = self.vocab.get(id as usize) {
                raw.extend_from_slice(bytes);
            }
        }
        self.decode_bytes(&String::from_utf8_lossy(&raw))
    }

    /// Decode a single token ID to text.
    pub fn decode_token(&self, id: u32) -> String {
        if let Some(bytes) = self.vocab.get(id as usize) {
            self.decode_bytes(&String::from_utf8_lossy(bytes))
        } else {
            format!("<unk_{}>", id)
        }
    }

    fn decode_bytes(&self, text: &str) -> String {
        // Qwen2 vocab contains proper UTF-8; skip the double-encoding repair
        let text = text.to_string();
        // Map byte-level BPE unicode back to original bytes
        let mut bytes = Vec::with_capacity(text.len());
        for ch in text.chars() {
            if ch == '▁' {
                bytes.push(b' ');
                continue;
            }
            if let Some(&b) = self.byte_decoder.get(&ch) {
                bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        let decoded = String::from_utf8_lossy(&bytes).to_string();
        self.convert_hex_escapes(&decoded)
    }

    fn convert_hex_escapes(&self, text: &str) -> String {
        let mut result = String::new();
        let mut i = 0;
        while i < text.len() {
            let slice = &text[i..];
            if slice.starts_with("<0x") {
                let rest = &slice[3..];
                let mut hex = String::new();
                for ch in rest.chars() {
                    if ch.is_ascii_hexdigit() && hex.len() < 2 {
                        hex.push(ch);
                    } else {
                        break;
                    }
                }
                let tail = &rest[hex.len()..];
                if !hex.is_empty() && tail.starts_with('>') {
                    if let Ok(b) = u8::from_str_radix(&hex, 16) {
                        result.push(b as char);
                        i += 3 + hex.len() + 1;
                        continue;
                    }
                }
            }
            if let Some(ch) = slice.chars().next() {
                result.push(ch);
                i += ch.len_utf8();
            } else {
                break;
            }
        }
        result
    }

    /// Is this token an end-of-generation signal?
    pub fn is_eog(&self, id: u32) -> bool {
        if Some(id) == self.eos_id {
            return true;
        }
        if let Some(bytes) = self.vocab.get(id as usize) {
            matches!(
                String::from_utf8_lossy(bytes).as_ref(),
                ""
                    | "<|eos|>"
                    | "</s>"
                    | "<|eot_id|>"
                    | "<|eom_id|>"
                    | "<|im_end|>"
                    | "<|end|>"
                    | "<end_of_turn>"
            )
        } else {
            false
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────────────

    /// Returns the vocabulary size (number of tokens).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Returns the BOS (beginning of sequence) token ID, if configured.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// Returns the EOS (end of sequence) token ID, if configured.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Returns whether BOS token should be added.
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Returns whether EOS token should be added.
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    // ── Byte encoder ────────────────────────────────────────────────────────────

    fn build_byte_encoder() -> HashMap<u8, String> {
        let mut bs: Vec<u8> = Vec::new();
        bs.extend(b'!'..=b'~');
        bs.extend(0xA1..=0xAC_u8);
        bs.extend(0xAE..=0xFF_u8);
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0u32;
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut enc = HashMap::with_capacity(256);
        for (b, c) in bs.into_iter().zip(cs) {
            if let Some(ch) = char::from_u32(c) {
                enc.insert(b, ch.to_string());
            }
        }
        enc
    }
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if (b & 0xE0) == 0xC0 {
        2
    } else if (b & 0xF0) == 0xE0 {
        3
    } else if (b & 0xF8) == 0xF0 {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tokenizer() -> BpeTokenizer {
        let vocab: Vec<Vec<u8>> = vec![
            b"h".to_vec(),
            b"e".to_vec(),
            b"l".to_vec(),
            b"o".to_vec(),
            b" ".to_vec(),
            b"he".to_vec(),
            b"ll".to_vec(),
            b"lo".to_vec(),
            b"hello".to_vec(),
            b"".to_vec(),
        ];
        let merges = vec![
            (b"h".to_vec(), b"e".to_vec()),
            (b"l".to_vec(), b"l".to_vec()),
            (b"he".to_vec(), b"ll".to_vec()),
            (b"hel".to_vec(), b"lo".to_vec()),
        ];
        BpeTokenizer::new(vocab, merges, None, Some(9), None, false, false, PreTokenizerType::Qwen2)
    }

    #[test]
    fn vocab_size() {
        assert_eq!(toy_tokenizer().vocab_size(), 10);
    }

    #[test]
    fn eos_id() {
        assert_eq!(toy_tokenizer().eos_id(), Some(9));
    }

    #[test]
    fn is_eog_by_id() {
        assert!(toy_tokenizer().is_eog(9));
        assert!(!toy_tokenizer().is_eog(0));
    }

    #[test]
    fn decode_single_token() {
        let tok = toy_tokenizer();
        let decoded = tok.decode_token(5);
        assert!(decoded.contains('h') || !decoded.is_empty());
    }

    #[test]
    fn byte_encoder_covers_all_256() {
        let enc = BpeTokenizer::build_byte_encoder();
        assert_eq!(enc.len(), 256);
    }
}
