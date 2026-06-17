//! SentencePiece (Unigram) tokenizer.
//!
//! Provides text tokenization for models such as Gemma that use the
//! SentencePiece tokenizer format stored in GGUF files.

use std::collections::{HashMap, HashSet};

const SPACE_SYMBOL: char = '\u{2581}'; // ▁

/// Tokenizer for SentencePiece vocabularies.
#[derive(Clone, Debug)]
pub struct SpmTokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<String, u32>,
    special_tokens: HashSet<u32>,
    byte_tokens: HashMap<u8, u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    /// Per-token scores used by Viterbi.  When the GGUF scores are a uniform
    /// sentinel value (e.g. Gemma4 uses -1000 for every token) we fall back to a
    /// constant negative cost so that the Viterbi search prefers longer tokens.
    token_score: Vec<f32>,
    /// Use greedy longest-match encoding instead of score-driven Viterbi.
    /// Set when the stored scores are synthetic (uniform or token-id values)
    /// instead of real SentencePiece log-probabilities.
    greedy: bool,
}

impl SpmTokenizer {
    /// Build from GGUF tokenizer arrays.
    pub fn from_gguf(data: &crate::loader::TokenizerData) -> Self {
        let mut token_to_id = HashMap::with_capacity(data.tokens.len());
        let mut special_tokens = HashSet::new();
        let mut byte_tokens = HashMap::new();
        let mut token_score = Vec::with_capacity(data.tokens.len());

        let scores_uniform = data.scores.len() == data.tokens.len()
            && data.scores.iter().all(|&s| s == data.scores[0]);
        // Some GGUF files (e.g. Ollama's gemma4:e2b) store scores as
        // 0.0, 1.0, 2.0, ... which are not real log-probabilities.  Detect that
        // score sequence and fall back to greedy longest-match encoding.
        let scores_are_token_indices = data.scores.len() == data.tokens.len()
            && data.scores.iter().enumerate().all(|(i, &s)| s == i as f32);
        let use_greedy = scores_uniform || scores_are_token_indices;

        for (id, token) in data.tokens.iter().enumerate() {
            let id = id as u32;
            if let Ok(s) = std::str::from_utf8(token) {
                token_to_id.insert(s.to_string(), id);
            }

            let ty = data.token_types.get(id as usize).copied().unwrap_or(1);
            match ty {
                1 | 6 => {
                    // Normal or byte token — not special.
                }
                _ => {
                    special_tokens.insert(id);
                }
            }

            if ty == 6 {
                if let Ok(s) = std::str::from_utf8(token) {
                    if let Some(hex) = s.strip_prefix("<0x").and_then(|s| s.strip_prefix('>')) {
                        if let Ok(b) = u8::from_str_radix(hex, 16) {
                            byte_tokens.insert(b, id);
                        }
                    }
                }
            }

            let score = if use_greedy || data.scores.len() != data.tokens.len() {
                -1.0
            } else {
                data.scores[id as usize]
            };
            token_score.push(score);
        }

        Self {
            vocab: data.tokens.clone(),
            token_to_id,
            special_tokens,
            byte_tokens,
            bos_id: data.bos_token_id,
            eos_id: data.eos_token_id,
            unk_id: data.unk_token_id,
            add_bos: data.add_bos,
            add_eos: data.add_eos,
            add_space_prefix: false,
            token_score,
            greedy: use_greedy,
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos_id {
                tokens.push(b);
            }
        }
        for fragment in self.split_by_special_tokens(text) {
            match fragment {
                Fragment::Text(t) => tokens.extend(self.encode_text(t)),
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

    fn encode_text(&self, text: &str) -> Vec<u32> {
        // SentencePiece normalizes spaces to the ▁ marker.
        let mut normalized = text.replace(' ', &SPACE_SYMBOL.to_string());
        if self.add_space_prefix && !normalized.starts_with(SPACE_SYMBOL) {
            normalized.insert(0, SPACE_SYMBOL);
        }
        if normalized.is_empty() {
            return Vec::new();
        }

        if self.greedy {
            self.encode_text_greedy(&normalized)
        } else {
            self.encode_text_viterbi(&normalized)
        }
    }

    /// Greedy longest-match segmentation.
    ///
    /// Used when the GGUF scores are synthetic rather than real
    /// log-probabilities.  At each position emit the longest token that matches,
    /// falling back to one token per UTF-8 byte for unknown characters.
    fn encode_text_greedy(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut out = Vec::new();
        let mut i = 0;
        while i < n {
            let mut best_len = 0usize;
            let mut best_id = None;
            let mut piece = String::new();
            for (j, &ch) in chars.iter().enumerate().take(n).skip(i) {
                piece.push(ch);
                if let Some(&id) = self.token_to_id.get(&piece) {
                    best_len = j - i + 1;
                    best_id = Some(id);
                }
            }
            if let Some(id) = best_id {
                out.push(id);
                i += best_len;
            } else {
                // Byte fallback for an unknown character.
                let c = chars[i];
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                for &b in bytes {
                    if let Some(&id) = self.byte_tokens.get(&b) {
                        out.push(id);
                    } else if let Some(unk) = self.unk_id {
                        out.push(unk);
                    }
                }
                i += 1;
            }
        }
        out
    }

    fn encode_text_viterbi(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut prev: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best[0] = 0.0;

        for i in 0..n {
            if best[i].is_infinite() && best[i] < 0.0 {
                continue;
            }
            let mut matched_any = false;
            for j in (i + 1)..=n {
                let piece: String = chars[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&piece) {
                    matched_any = true;
                    let score = best[i] + self.token_score[id as usize];
                    if score > best[j] {
                        best[j] = score;
                        prev[j] = Some((i, id));
                    }
                }
            }
            if !matched_any {
                // Byte fallback for an unknown character.
                let c = chars[i];
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                if let Some(&id) = self.byte_tokens.get(&bytes[0]) {
                    let score = best[i] + self.token_score[id as usize];
                    if score > best[i + 1] {
                        best[i + 1] = score;
                        prev[i + 1] = Some((i, id));
                    }
                } else if let Some(unk) = self.unk_id {
                    let score = best[i] + self.token_score[unk as usize];
                    if score > best[i + 1] {
                        best[i + 1] = score;
                        prev[i + 1] = Some((i, unk));
                    }
                }
            }
        }

        // Backtrack.
        let mut out = Vec::new();
        let mut pos = n;
        while pos > 0 {
            if let Some((start, id)) = prev[pos] {
                out.push(id);
                pos = start;
            } else {
                // Should not happen for a well-formed vocabulary; emit unk and
                // step back one character to guarantee termination.
                if let Some(unk) = self.unk_id {
                    out.push(unk);
                }
                pos = pos.saturating_sub(1);
            }
        }
        out.reverse();
        out
    }

    fn split_by_special_tokens<'a>(&'a self, text: &'a str) -> Vec<Fragment<'a>> {
        let mut fragments = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            let mut found: Option<(u32, usize)> = None;
            for &id in &self.special_tokens {
                if let Ok(s) = std::str::from_utf8(&self.vocab[id as usize]) {
                    if let Some(pos) = remaining.find(s) {
                        if pos == 0 {
                            fragments.push(Fragment::Special(id));
                            remaining = &remaining[s.len()..];
                            found = None;
                            break;
                        } else if found.map(|(_, p)| p).unwrap_or(usize::MAX) > pos {
                            found = Some((id, pos));
                        }
                    }
                }
            }
            if let Some((id, pos)) = found {
                if pos > 0 {
                    fragments.push(Fragment::Text(&remaining[..pos]));
                }
                fragments.push(Fragment::Special(id));
                remaining = &remaining[pos + self.vocab[id as usize].len()..];
            } else {
                fragments.push(Fragment::Text(remaining));
                break;
            }
        }
        fragments
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        let mut raw = Vec::new();
        for &id in tokens {
            if skip_special && self.special_tokens.contains(&id) {
                continue;
            }
            if let Some(bytes) = self.vocab.get(id as usize) {
                if self.token_type(id) == 6 {
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        if let Some(hex) = s.strip_prefix("<0x").and_then(|s| s.strip_prefix('>')) {
                            if let Ok(b) = u8::from_str_radix(hex, 16) {
                                raw.push(b);
                                continue;
                            }
                        }
                    }
                }
                raw.extend_from_slice(bytes);
            }
        }
        let text = String::from_utf8_lossy(&raw);
        text.replace(SPACE_SYMBOL, " ")
    }

    /// Decode a single token ID to text.
    pub fn decode_token(&self, id: u32) -> String {
        self.decode(&[id], false)
    }

    /// Is this token an end-of-generation signal?
    pub fn is_eog(&self, id: u32) -> bool {
        if Some(id) == self.eos_id {
            return true;
        }
        if let Some(bytes) = self.vocab.get(id as usize) {
            matches!(
                String::from_utf8_lossy(bytes).as_ref(),
                "" | "<|eos|>"
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

    fn token_type(&self, id: u32) -> i32 {
        // This is only used for byte-token detection during decode; the type
        // vector is not stored separately because byte tokens are identified by
        // their "<0xXX>" form.
        if let Some(bytes) = self.vocab.get(id as usize) {
            if let Ok(s) = std::str::from_utf8(bytes) {
                if s.starts_with("<0x") && s.ends_with('>') {
                    return 6;
                }
            }
        }
        1
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    pub fn add_eos(&self) -> bool {
        self.add_eos
    }
}

enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}
