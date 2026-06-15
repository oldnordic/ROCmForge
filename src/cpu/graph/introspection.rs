//! Introspection interface for captured CPU graph traces.
//!
//! Turns a persisted `GraphMap` into a short, model-readable summary and parses
//! the model's structured response back into an `IntrospectionReport`.

use crate::cpu::graph::{GraphMap, Shelf};
use serde::{Deserialize, Serialize};

/// Summary of a single captured branch.
#[derive(Debug, Clone)]
pub struct BranchSummary {
    pub timestamp: u64,
    pub score: f32,
    pub divergence: f32,
}

/// Compressed summary of a captured graph session.
#[derive(Debug, Clone)]
pub struct GraphSummary {
    pub n_branches: usize,
    pub branches: Vec<BranchSummary>,
    pub final_hidden_norm: f32,
}

/// Text prompt plus a fixed-size vector encoding of the same summary.
#[derive(Debug, Clone)]
pub struct IntrospectionPrompt {
    pub text: String,
    pub vector: Vec<f32>,
    pub label_map: Vec<(String, u64)>,
}

/// Structured result parsed from a model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionReport {
    pub chosen_branch: u64,
    pub explanation: String,
}

/// Persistent annotation attached to a branch/timestamp in a `GraphMap`.
///
/// Bias is a scalar preference (higher = prefer this branch). The optional
/// `key` lets a later session match the annotation by semantic content (e.g.
/// the action name) even when timestamps differ between sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchAnnotation {
    pub timestamp: u64,
    pub bias: f32,
    pub report: Option<IntrospectionReport>,
    pub key: Option<String>,
}

/// Builds summaries and prompts from `GraphMap` snapshots.
#[derive(Debug, Clone, Copy)]
pub struct GraphSummarizer {
    /// Reference score used to compute per-branch divergence.
    pub reference_score: f32,
}

impl Default for GraphSummarizer {
    fn default() -> Self {
        Self {
            reference_score: 1.0,
        }
    }
}

impl GraphSummarizer {
    pub fn new(reference_score: f32) -> Self {
        Self { reference_score }
    }

    /// Produce a `GraphSummary` from a captured map.
    pub fn summarize(&self, map: &GraphMap) -> GraphSummary {
        let scores = map.branch_scores();
        let divergence = map.divergence(self.reference_score);

        let mut branches: Vec<BranchSummary> = scores
            .iter()
            .map(|(&ts, &score)| BranchSummary {
                timestamp: ts,
                score,
                divergence: divergence.get(&ts).copied().unwrap_or(0.0),
            })
            .collect();
        branches.sort_by_key(|b| b.timestamp);

        let final_hidden_norm = final_hidden_norm(map);

        GraphSummary {
            n_branches: branches.len(),
            branches,
            final_hidden_norm,
        }
    }

    /// Turn a summary into a short text prompt and vector encoding.
    pub fn prompt(&self, summary: &GraphSummary) -> IntrospectionPrompt {
        let mut text = String::from(
            "You are a reasoning-branch evaluator. Each branch has a quality score (higher is better) and a divergence from the ideal reference (lower is better).\n\n",
        );

        let mut label_map = Vec::with_capacity(summary.branches.len());
        for (idx, branch) in summary.branches.iter().enumerate() {
            let label = branch_label(idx);
            label_map.push((label.clone(), branch.timestamp));
            text.push_str(&format!(
                "Branch {label}: score={:.4}, divergence={:.4}\n",
                branch.score, branch.divergence
            ));
        }

        text.push_str(&format!(
            "\nFinal hidden-state norm: {:.4}\n",
            summary.final_hidden_norm
        ));
        text.push_str(
            "\nWhich branch is better? Choose exactly one branch. Reply in this exact format:\n\nCHOICE: Branch <LETTER>\nREASON: <one sentence>\n",
        );

        let vector = vectorize(summary);

        IntrospectionPrompt {
            text,
            vector,
            label_map,
        }
    }
}

impl IntrospectionPrompt {
    /// Parse a model response into an `IntrospectionReport`.
    pub fn parse_response(&self, response: &str) -> Option<IntrospectionReport> {
        let choice_line = response
            .lines()
            .map(str::trim)
            .find(|line| line.to_uppercase().starts_with("CHOICE:"))?;
        let after = choice_line
            .strip_prefix("CHOICE:")
            .or_else(|| choice_line.strip_prefix("choice:"))?;
        let choice = after.trim();

        let chosen_timestamp = self
            .label_map
            .iter()
            .find(|(label, _)| {
                label.eq_ignore_ascii_case(choice)
                    || choice.eq_ignore_ascii_case(&format!("Branch {label}"))
            })
            .map(|(_, ts)| *ts)?;

        let explanation = response
            .lines()
            .map(str::trim)
            .find(|line| line.to_uppercase().starts_with("REASON:"))
            .map(|line| {
                line.strip_prefix("REASON:")
                    .or_else(|| line.strip_prefix("reason:"))
                    .unwrap_or(line)
                    .trim()
                    .to_string()
            })
            .unwrap_or_default();

        Some(IntrospectionReport {
            chosen_branch: chosen_timestamp,
            explanation,
        })
    }
}

fn branch_label(index: usize) -> String {
    let mut label = String::new();
    let mut n = index;
    loop {
        let ch = (b'A' + (n % 26) as u8) as char;
        label.insert(0, ch);
        if n < 26 {
            break;
        }
        n /= 26;
    }
    label
}

fn final_hidden_norm(map: &GraphMap) -> f32 {
    let last_persistent = map
        .output_log
        .iter()
        .filter(|(_, _, handle)| {
            handle.shelf == Shelf::Persistent && map.arena.is_f32_handle_valid(*handle)
        })
        .max_by_key(|(ts, _, _)| *ts)
        .map(|(_, _, handle)| *handle);

    if let Some(handle) = last_persistent {
        let slice = map.arena.f32(handle);
        slice.iter().map(|x| x * x).sum::<f32>().sqrt()
    } else {
        0.0
    }
}

fn vectorize(summary: &GraphSummary) -> Vec<f32> {
    let (sum_score, min_score, max_score) = summary.branches.iter().fold(
        (0.0f32, f32::INFINITY, f32::NEG_INFINITY),
        |(sum, min, max), b| (sum + b.score, min.min(b.score), max.max(b.score)),
    );
    let mean_score = if summary.branches.is_empty() {
        0.0
    } else {
        sum_score / summary.branches.len() as f32
    };
    let mean_divergence = if summary.branches.is_empty() {
        0.0
    } else {
        summary.branches.iter().map(|b| b.divergence).sum::<f32>() / summary.branches.len() as f32
    };

    vec![
        summary.n_branches as f32,
        mean_score,
        min_score,
        max_score,
        mean_divergence,
        summary.final_hidden_norm,
    ]
}
