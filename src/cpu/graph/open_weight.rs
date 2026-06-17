//! Open-weight experiment: extract base-model signals from branch
//! summaries and use them to train a small differentiable scorer.
//!
//! Two signal sources are provided:
//! - `extract_hidden_state` + `BranchValueHead`: final hidden-state vector.
//! - `extract_answer_logit_sum` + `BranchLogitScorer`: logit sum for answer
//!   words such as "good" / "bad".
//!
//! The 0.5B model weights stay frozen; only the small head/scorer is updated.
//! This keeps the experiment feasible on CPU while still updating real weights
//! from trace feedback.

use crate::config::ModelConfig;
use crate::cpu::cache::{CpuForwardScratch, CpuKvCache};
use crate::cpu::forward::cpu_prefill;
use crate::cpu::weights::CpuModelWeights;
use crate::cpu::CpuError;
use crate::tokenizer::Tokenizer;

use super::value_head::{BranchValueExample, BranchValueHead};
use super::{GraphMap, GraphSummarizer};
use std::collections::HashMap;

/// Default small KV-cache size for short branch-summary prompts.
///
/// Prompts are at most a few dozen tokens; 256 positions is plenty and avoids
/// allocating the model's full context window for every example.
pub const SHORT_PROMPT_MAX_SEQ_LEN: usize = 256;

/// Extract the final hidden state (post-final RMS norm) for a short prompt.
///
/// The returned vector has length `config.hidden_size` and represents the
/// model's representation of `prompt` at its last token.
pub fn extract_hidden_state(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    max_seq_len: usize,
) -> Result<Vec<f32>, CpuError> {
    extract_hidden_state_with_special(weights, config, tokenizer, prompt, max_seq_len, true)
}

pub fn extract_hidden_state_with_special(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    max_seq_len: usize,
    add_special: bool,
) -> Result<Vec<f32>, CpuError> {
    let tokens = tokenizer.encode(prompt, add_special);
    if tokens.is_empty() {
        return Err(CpuError::InvalidOperation(
            "extract_hidden_state: empty tokenization".to_string(),
        ));
    }
    if tokens.len() > max_seq_len {
        return Err(CpuError::InvalidOperation(format!(
            "extract_hidden_state: prompt length {} exceeds max_seq_len {}",
            tokens.len(),
            max_seq_len
        )));
    }

    let mut kv = CpuKvCache::new(config, max_seq_len);
    let mut scratch = CpuForwardScratch::new(config);
    // `cpu_prefill` writes the final normalized hidden state into
    // `scratch.normed` and the logits into `scratch.logits`. We only need the
    // hidden state here.
    cpu_prefill(&mut [], weights, &mut kv, &mut scratch, &tokens, config)?;

    Ok(scratch.normed.to_vec())
}

/// Build a one-branch version of the introspection prompt.
///
/// The text is deliberately short so the 0.5B model can be run many times
/// during the experiment.
pub fn build_branch_prompt(score: f32, divergence: f32) -> String {
    format!(
        "You are a reasoning-branch evaluator. A branch has a quality score (higher is better) and divergence from ideal (lower is better).\nBranch: score={:.4}, divergence={:.4}\nIs this branch good?",
        score, divergence
    )
}

/// Collect a hidden-state example for every branch in `maps`.
///
/// Returns `(trace_id, timestamp) -> BranchValueExample` so callers can join
/// it with `GraphTraceDataset::preference_pairs()`.
pub fn collect_branch_hidden_states(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tokenizer: &dyn Tokenizer,
    maps: &[(String, GraphMap)],
    summarizer: &GraphSummarizer,
    max_seq_len: usize,
) -> Result<HashMap<(String, u64), BranchValueExample>, CpuError> {
    let reference_score = summarizer.reference_score;
    let mut out = HashMap::new();

    for (trace_id, map) in maps {
        let scores = map.branch_scores();
        let divergence = map.divergence(reference_score);
        for (&timestamp, &score) in &scores {
            let div = divergence.get(&timestamp).copied().unwrap_or(0.0f32);
            let prompt = build_branch_prompt(score, div);
            let hidden = extract_hidden_state(weights, config, tokenizer, &prompt, max_seq_len)?;
            out.insert(
                (trace_id.clone(), timestamp),
                BranchValueExample {
                    trace_id: trace_id.clone(),
                    timestamp,
                    hidden,
                    score,
                },
            );
        }
    }

    Ok(out)
}

// ── Logit-based answer scorer ────────────────────────────────────────────────────

/// Example that records the model's preference between two answer words for a
/// single branch.
#[derive(Debug, Clone)]
pub struct BranchLogitExample {
    pub trace_id: String,
    pub timestamp: u64,
    /// Sum of logits for the positive answer word (e.g. "good").
    pub positive_logit: f32,
    /// Sum of logits for the negative answer word (e.g. "bad").
    pub negative_logit: f32,
    pub score: f32,
}

impl BranchLogitExample {
    /// Raw logit-margin signal: higher means the model leans toward the
    /// positive word.
    pub fn margin(&self) -> f32 {
        self.positive_logit - self.negative_logit
    }
}

/// Trainable scalar rescaling of a logit-margin signal.
///
/// The input is `positive_logit - negative_logit` for a branch. The scaler is
/// trained with pairwise hinge loss so that better branches receive a larger
/// output than worse branches. This is a single open weight that sits on top
/// of the frozen base model's output distribution.
#[derive(Debug, Clone, Copy)]
pub struct BranchLogitScorer {
    pub scale: f32,
}

impl BranchLogitScorer {
    pub fn new() -> Self {
        Self { scale: 1.0f32 }
    }

    pub fn predict(&self, positive_logit: f32, negative_logit: f32) -> f32 {
        self.scale * (positive_logit - negative_logit)
    }

    pub fn predict_from_example(&self, ex: &BranchLogitExample) -> f32 {
        self.predict(ex.positive_logit, ex.negative_logit)
    }

    /// Train with pairwise hinge loss.
    ///
    /// For every ordered pair `(worse, better)` where `worse.score <
    /// better.score`, if `scale * (x_w - x_b) + margin > 0` the scale is
    /// updated so that the better branch's margin grows relative to the worse.
    pub fn fit(&mut self, examples: &[BranchLogitExample], epochs: usize, lr: f32, margin: f32) {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (i, a) in examples.iter().enumerate() {
            for (j, b) in examples.iter().enumerate() {
                if i != j && a.score < b.score {
                    pairs.push((i, j));
                }
            }
        }
        if pairs.is_empty() {
            return;
        }

        for _ in 0..epochs {
            for &(worse_idx, better_idx) in &pairs {
                let x_w = examples[worse_idx].margin();
                let x_b = examples[better_idx].margin();
                let violation = self.scale * (x_w - x_b) + margin;
                if violation > 0.0f32 {
                    // d/dscale [scale*(x_w - x_b) + margin] = x_w - x_b.
                    self.scale -= lr * (x_w - x_b);
                }
            }
        }
    }
}

impl Default for BranchLogitScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a short yes/no prompt that ends with the model about to answer.
pub fn build_yes_no_prompt(score: f32) -> String {
    format!(
        "You are a reasoning-branch evaluator. A branch has a quality score (higher is better). Branch: score={:.4}. Is this branch good? Answer:",
        score
    )
}

/// Find all token IDs whose decoded text (ignoring leading/trailing whitespace)
/// matches `word`.
pub fn find_answer_token_ids(tokenizer: &dyn Tokenizer, word: &str) -> Vec<u32> {
    let target = word.to_lowercase();
    let mut ids = Vec::new();

    // First try an exact single-token encode.
    let encoded = tokenizer.encode(word, false);
    if encoded.len() == 1 {
        ids.push(encoded[0]);
    }

    // Also search the vocabulary for any token that decodes to the target word.
    for id in 0..tokenizer.vocab_size() as u32 {
        let decoded = tokenizer.decode_token(id).trim().to_lowercase();
        if decoded == target {
            ids.push(id);
        }
    }

    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Extract the sum of logits for all token IDs that match `word`.
///
/// Summing over ambiguous token forms (e.g. "good" and " good") makes the
/// signal robust to whether the tokenizer attaches a leading space.
pub fn extract_answer_logit_sum(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    word: &str,
    max_seq_len: usize,
) -> Result<f32, CpuError> {
    let tokens = tokenizer.encode(prompt, true);
    if tokens.is_empty() {
        return Err(CpuError::InvalidOperation(
            "extract_answer_logit_sum: empty tokenization".to_string(),
        ));
    }
    if tokens.len() > max_seq_len {
        return Err(CpuError::InvalidOperation(format!(
            "extract_answer_logit_sum: prompt length {} exceeds max_seq_len {}",
            tokens.len(),
            max_seq_len
        )));
    }

    let ids = find_answer_token_ids(tokenizer, word);
    if ids.is_empty() {
        return Err(CpuError::InvalidOperation(format!(
            "extract_answer_logit_sum: no token found for word '{}'",
            word
        )));
    }

    let mut kv = CpuKvCache::new(config, max_seq_len);
    let mut scratch = CpuForwardScratch::new(config);
    cpu_prefill(&mut [], weights, &mut kv, &mut scratch, &tokens, config)?;

    let logits = &*scratch.logits;
    let mut sum = 0.0f32;
    for &id in &ids {
        sum += logits.get(id as usize).copied().unwrap_or(0.0f32);
    }
    Ok(sum)
}

/// Collect a logit-margin example for every branch in `maps`.
pub fn collect_branch_logit_examples(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tokenizer: &dyn Tokenizer,
    maps: &[(String, GraphMap)],
    max_seq_len: usize,
) -> Result<HashMap<(String, u64), BranchLogitExample>, CpuError> {
    let mut out = HashMap::new();

    for (trace_id, map) in maps {
        let scores = map.branch_scores();
        for (&timestamp, &score) in &scores {
            let prompt = build_yes_no_prompt(score);
            let positive_logit =
                extract_answer_logit_sum(weights, config, tokenizer, &prompt, "yes", max_seq_len)?;
            let negative_logit =
                extract_answer_logit_sum(weights, config, tokenizer, &prompt, "no", max_seq_len)?;
            out.insert(
                (trace_id.clone(), timestamp),
                BranchLogitExample {
                    trace_id: trace_id.clone(),
                    timestamp,
                    positive_logit,
                    negative_logit,
                    score,
                },
            );
        }
    }

    Ok(out)
}

// ── Label-logit bias scorer ──────────────────────────────────────────────────────

/// Trainable per-label bias added to the base model's label-token logits.
///
/// Useful when the prompt asks the model to choose one of several labeled
/// branches ("A", "B", ...) and the answer should be a single letter. The bias
/// vector is a tiny set of open weights updated from trace preference pairs.
#[derive(Debug, Clone)]
pub struct BranchLabelBias {
    bias: HashMap<char, f32>,
}

impl BranchLabelBias {
    pub fn new() -> Self {
        Self {
            bias: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bias.is_empty()
    }

    pub fn get(&self, label: char) -> f32 {
        self.bias.get(&label).copied().unwrap_or(0.0f32)
    }

    pub fn predict(&self, label: char, logit: f32) -> f32 {
        logit + self.get(label)
    }

    /// Train with pairwise hinge loss on label logits.
    ///
    /// `examples` maps a branch label to its raw model logit and score.
    pub fn fit(&mut self, examples: &[(char, f32, f32)], epochs: usize, lr: f32, margin: f32) {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (i, (_, _, a_score)) in examples.iter().enumerate() {
            for (j, (_, _, b_score)) in examples.iter().enumerate() {
                if i != j && a_score < b_score {
                    pairs.push((i, j));
                }
            }
        }
        if pairs.is_empty() {
            return;
        }

        for _ in 0..epochs {
            for &(worse_idx, better_idx) in &pairs {
                let (w_label, w_logit, _) = examples[worse_idx];
                let (b_label, b_logit, _) = examples[better_idx];
                let w_score = w_logit + self.get(w_label);
                let b_score = b_logit + self.get(b_label);
                let violation = w_score - b_score + margin;
                if violation > 0.0f32 {
                    *self.bias.entry(b_label).or_insert(0.0f32) += lr;
                    *self.bias.entry(w_label).or_insert(0.0f32) -= lr;
                }
            }
        }
    }
}

impl Default for BranchLabelBias {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a multi-branch prompt that asks for a single-letter answer.
pub fn build_label_choice_prompt(summary: &super::GraphSummary) -> (String, Vec<(char, u64)>) {
    let mut text = String::from(
        "You are a reasoning-branch evaluator. Each branch has a quality score (higher is better).\n\n",
    );
    let mut labels: Vec<(char, u64)> = Vec::with_capacity(summary.branches.len());
    for (idx, branch) in summary.branches.iter().enumerate() {
        let label = branch_label_letter(idx);
        labels.push((label, branch.timestamp));
        text.push_str(&format!("Branch {}: score={:.4}\n", label, branch.score));
    }
    text.push_str("\nWhich branch is better? Answer with a single letter (A, B, C, ...):");
    (text, labels)
}

fn branch_label_letter(index: usize) -> char {
    (b'A' + (index % 26) as u8) as char
}

/// Train a `BranchValueHead` on branch hidden states extracted from persisted
/// `GraphMap` traces in `trace_dir` and save it to `save_path`.
///
/// Each branch is converted to the standard one-branch prompt, its final hidden
/// state is extracted with `extract_hidden_state`, and the head is trained with
/// MSE against the branch's recorded score.
pub fn train_value_head_from_trace_dir(
    trace_dir: &std::path::Path,
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tokenizer: &dyn Tokenizer,
    epochs: usize,
    lr: f32,
    save_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use super::dataset::GraphTraceDataset;

    let dataset = GraphTraceDataset::from_dir(trace_dir)?;
    let summarizer = GraphSummarizer::new(1.0);

    let maps: Vec<(String, GraphMap)> = dataset
        .traces
        .into_iter()
        .map(|(id, map)| (id, map))
        .collect();

    let hidden_states =
        collect_branch_hidden_states(weights, config, tokenizer, &maps, &summarizer, 256)?;

    let examples: Vec<BranchValueExample> = hidden_states.into_values().collect();
    if examples.is_empty() {
        return Err("No branch examples found in trace directory".into());
    }

    let hidden_size = examples[0].hidden.len();
    let mut head = BranchValueHead::new(hidden_size);
    head.fit_mse(&examples, epochs, lr);
    head.save(save_path)?;

    eprintln!(
        "Trained value head on {} branch examples, saved to {}",
        examples.len(),
        save_path.display()
    );
    Ok(())
}

/// Extract the summed logits for each label letter in `prompt`.
pub fn extract_label_logits(
    weights: &CpuModelWeights,
    config: &ModelConfig,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    labels: &[char],
    max_seq_len: usize,
) -> Result<HashMap<char, f32>, CpuError> {
    let tokens = tokenizer.encode(prompt, true);
    if tokens.is_empty() {
        return Err(CpuError::InvalidOperation(
            "extract_label_logits: empty tokenization".to_string(),
        ));
    }
    if tokens.len() > max_seq_len {
        return Err(CpuError::InvalidOperation(format!(
            "extract_label_logits: prompt length {} exceeds max_seq_len {}",
            tokens.len(),
            max_seq_len
        )));
    }

    let mut kv = CpuKvCache::new(config, max_seq_len);
    let mut scratch = CpuForwardScratch::new(config);
    cpu_prefill(&mut [], weights, &mut kv, &mut scratch, &tokens, config)?;

    let logits = &*scratch.logits;
    let mut out = HashMap::new();
    for &label in labels {
        let ids = find_answer_token_ids(tokenizer, &label.to_string());
        let mut sum = 0.0f32;
        for &id in &ids {
            sum += logits.get(id as usize).copied().unwrap_or(0.0f32);
        }
        out.insert(label, sum);
    }
    Ok(out)
}
