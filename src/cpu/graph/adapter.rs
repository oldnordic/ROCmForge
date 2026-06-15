//! Lightweight mistake-driven adapter for branch ranking.
//!
//! This is a pragmatic, CPU-only stand-in for a full LoRA adapter on the 0.5B
//! base model. It trains a tiny MLP on per-branch feature vectors derived from
//! a `GraphSummary` and optimizes a pairwise hinge loss over preference pairs.
//!
//! The goal is to demonstrate the closed loop: persisted traces -> dataset ->
//! trained adapter -> improved branch selection on held-out traces.

use crate::cpu::graph::{dataset::GraphTraceDataset, GraphMap, GraphSummarizer};
use std::collections::HashMap;

/// A tiny two-layer MLP that scores a branch feature vector.
#[derive(Debug, Clone)]
pub struct BranchAdapter {
    input_size: usize,
    hidden_size: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl BranchAdapter {
    /// Create a new adapter with deterministic small random initialization.
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let mut state = seed;
        let scale1 = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let w1: Vec<f32> = (0..input_size * hidden_size)
            .map(|_| {
                state = lcg(state);
                (lcg_f32(state) - 0.5) * scale1
            })
            .collect();
        let b1 = vec![0.0f32; hidden_size];

        let scale2 = (2.0 / (hidden_size + 1) as f32).sqrt();
        let w2: Vec<f32> = (0..hidden_size)
            .map(|_| {
                state = lcg(state);
                (lcg_f32(state) - 0.5) * scale2
            })
            .collect();
        let b2 = vec![0.0f32; 1];

        Self {
            input_size,
            hidden_size,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Forward pass returning the scalar score for `features`.
    pub fn predict(&self, features: &[f32]) -> f32 {
        if features.len() != self.input_size {
            return 0.0;
        }
        let hidden: Vec<f32> = self
            .b1
            .iter()
            .enumerate()
            .map(|(h, &bias)| {
                let sum = bias
                    + features
                        .iter()
                        .enumerate()
                        .take(self.input_size)
                        .map(|(i, &x)| x * self.w1[i * self.hidden_size + h])
                        .sum::<f32>();
                relu(sum)
            })
            .collect();
        self.b2[0]
            + hidden
                .iter()
                .zip(self.w2.iter())
                .map(|(&a, &w)| a * w)
                .sum::<f32>()
    }

    /// Train the adapter on preference pairs from `dataset` using SGD and a
    /// pairwise hinge loss.
    ///
    /// `epochs` full passes over the preference pairs are performed.
    pub fn train(
        &mut self,
        dataset: &GraphTraceDataset,
        epochs: usize,
        learning_rate: f32,
        margin: f32,
    ) {
        let summarizer = GraphSummarizer::default();
        let mut features_by_branch: HashMap<(String, u64), Vec<f32>> = HashMap::new();
        for (trace_id, map) in &dataset.traces {
            let summary = summarizer.summarize(map);
            for (timestamp, feats) in branch_features(&summary) {
                features_by_branch.insert((trace_id.clone(), timestamp), feats);
            }
        }

        let pairs = dataset.preference_pairs();
        if pairs.is_empty() {
            return;
        }

        for _ in 0..epochs {
            for pair in &pairs {
                let worse = features_by_branch
                    .get(&(pair.trace_id.clone(), pair.worse_timestamp))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0f32; self.input_size]);
                let better = features_by_branch
                    .get(&(pair.trace_id.clone(), pair.better_timestamp))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0f32; self.input_size]);

                let score_worse = self.predict(&worse);
                let score_better = self.predict(&better);
                let violation = score_worse - score_better + margin;
                if violation > 0.0 {
                    let (dw1_w, db1_w, dw2_w, db2_w) = self.backward(&worse, 1.0);
                    let (dw1_b, db1_b, dw2_b, db2_b) = self.backward(&better, -1.0);
                    self.update(&dw1_w, &db1_w, &dw2_w, &db2_w, learning_rate);
                    self.update(&dw1_b, &db1_b, &dw2_b, &db2_b, learning_rate);
                }
            }
        }
    }

    /// Rank all branches in `map` by predicted adapter score.
    ///
    /// Returns `(timestamp, score)` tuples sorted highest-first.
    pub fn rank_branches(&self, map: &GraphMap) -> Vec<(u64, f32)> {
        let summarizer = GraphSummarizer::default();
        let summary = summarizer.summarize(map);
        let mut ranked: Vec<(u64, f32)> = branch_features(&summary)
            .into_iter()
            .map(|(ts, feats)| (ts, self.predict(&feats)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Backward pass for a single input with output gradient `output_grad`.
    /// Returns gradients for w1, b1, w2, b2 in that order.
    fn backward(
        &self,
        features: &[f32],
        output_grad: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Forward caches.
        let (z1, a1): (Vec<f32>, Vec<f32>) = self
            .b1
            .iter()
            .enumerate()
            .map(|(h, &bias)| {
                let sum = bias
                    + features
                        .iter()
                        .enumerate()
                        .take(self.input_size)
                        .map(|(i, &x)| x * self.w1[i * self.hidden_size + h])
                        .sum::<f32>();
                (sum, relu(sum))
            })
            .unzip();

        // Gradients.
        let dw2: Vec<f32> = a1.iter().map(|&a| output_grad * a).collect();
        let db2 = vec![output_grad; 1];

        let da1: Vec<f32> = self.w2.iter().map(|&w| output_grad * w).collect();

        let dz1: Vec<f32> = da1
            .iter()
            .zip(z1.iter())
            .map(|(&da, &z)| da * relu_grad(z))
            .collect();

        let db1 = dz1.clone();
        let mut dw1 = vec![0.0f32; self.input_size * self.hidden_size];
        for (h, &dz) in dz1.iter().enumerate() {
            for (i, &x) in features.iter().enumerate().take(self.input_size) {
                dw1[i * self.hidden_size + h] = dz * x;
            }
        }

        (dw1, db1, dw2, db2)
    }

    fn update(&mut self, dw1: &[f32], db1: &[f32], dw2: &[f32], db2: &[f32], lr: f32) {
        for (w, d) in self.w1.iter_mut().zip(dw1.iter()) {
            *w -= lr * d;
        }
        for (b, d) in self.b1.iter_mut().zip(db1.iter()) {
            *b -= lr * d;
        }
        for (w, d) in self.w2.iter_mut().zip(dw2.iter()) {
            *w -= lr * d;
        }
        for (b, d) in self.b2.iter_mut().zip(db2.iter()) {
            *b -= lr * d;
        }
    }
}

/// Per-branch feature vector derived from a `GraphSummary`.
fn branch_features(summary: &crate::cpu::graph::GraphSummary) -> Vec<(u64, Vec<f32>)> {
    if summary.branches.is_empty() {
        return Vec::new();
    }
    let mean_score =
        summary.branches.iter().map(|b| b.score).sum::<f32>() / summary.branches.len() as f32;
    summary
        .branches
        .iter()
        .enumerate()
        .map(|(idx, branch)| {
            let feats = vec![
                branch.score,
                branch.divergence,
                branch.score - mean_score,
                summary.final_hidden_norm,
                idx as f32 / summary.branches.len() as f32,
                1.0,
            ];
            (branch.timestamp, feats)
        })
        .collect()
}

fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn relu_grad(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn lcg(state: u64) -> u64 {
    state.wrapping_mul(1103515245).wrapping_add(12345)
}

fn lcg_f32(state: u64) -> f32 {
    ((state >> 16) & 0x7fff) as f32 / 32768.0
}
