//! Training-dataset export for persisted CPU graph traces.
//!
//! A `GraphTraceDataset` reads one or more `GraphMap` snapshots and emits
//! examples in three standard fine-tuning formats:
//!
//! - **Process supervision:** a per-branch quality label.
//! - **Rejection sampling:** accepted (best) branches vs rejected branches.
//! - **Preference pairs:** ordered (worse, better) branch pairs.

use std::collections::HashMap;
use std::path::Path;

use crate::cpu::graph::{GraphMap, GraphMapError};
use serde::{Deserialize, Serialize};

/// Per-branch example for process-supervision training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessSupervisionExample {
    pub trace_id: String,
    pub timestamp: u64,
    pub score: f32,
    pub divergence: f32,
    /// Normalized quality label in `[0, 1]`.
    pub label: f32,
}

/// Accepted/rejected branch example for rejection-sampling training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RejectionSamplingExample {
    pub trace_id: String,
    pub timestamp: u64,
    pub score: f32,
    pub accepted: bool,
}

/// Preference-pair example stating that `better_timestamp` should be preferred
/// over `worse_timestamp`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreferencePair {
    pub trace_id: String,
    pub worse_timestamp: u64,
    pub better_timestamp: u64,
    pub worse_score: f32,
    pub better_score: f32,
}

/// A collection of loaded graph traces ready for dataset export.
#[derive(Debug, Clone)]
pub struct GraphTraceDataset {
    pub traces: Vec<(String, GraphMap)>,
}

impl GraphTraceDataset {
    /// Create a dataset from a single in-memory map.
    pub fn from_map(trace_id: impl Into<String>, map: GraphMap) -> Self {
        Self {
            traces: vec![(trace_id.into(), map)],
        }
    }

    /// Load every immediate subdirectory of `dir` as a `GraphMap`.
    ///
    /// The directory name is used as the trace id.
    pub fn from_dir(dir: &Path) -> Result<Self, GraphMapError> {
        let mut traces = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let map = GraphMap::load(&path)?;
                let id = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                traces.push((id, map));
            }
        }
        Ok(Self { traces })
    }

    /// Emit one process-supervision example per branch.
    ///
    /// The label is the score normalized to the `[0, 1]` range using the min
    /// and max scores observed within the same trace.
    pub fn process_supervision_examples(&self) -> Vec<ProcessSupervisionExample> {
        let mut examples = Vec::new();
        for (trace_id, map) in &self.traces {
            let scores = map.branch_scores();
            let divergences = map.divergence(1.0);
            if scores.is_empty() {
                continue;
            }
            let min = scores.values().copied().fold(f32::INFINITY, f32::min);
            let max = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            for (timestamp, score) in &scores {
                let divergence = divergences.get(timestamp).copied().unwrap_or(0.0);
                let label = if range > 1e-8 {
                    (score - min) / range
                } else {
                    0.5
                };
                examples.push(ProcessSupervisionExample {
                    trace_id: trace_id.clone(),
                    timestamp: *timestamp,
                    score: *score,
                    divergence,
                    label,
                });
            }
        }
        examples
    }

    /// Emit one accepted example for the best-scoring branch and one rejected
    /// example for every other branch in each trace.
    pub fn rejection_sampling_examples(&self) -> Vec<RejectionSamplingExample> {
        let mut examples = Vec::new();
        for (trace_id, map) in &self.traces {
            let scores = map.branch_scores();
            if scores.is_empty() {
                continue;
            }
            let best = scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ts, _)| *ts);
            for (timestamp, score) in &scores {
                examples.push(RejectionSamplingExample {
                    trace_id: trace_id.clone(),
                    timestamp: *timestamp,
                    score: *score,
                    accepted: Some(*timestamp) == best,
                });
            }
        }
        examples
    }

    /// Emit all ordered preference pairs within each trace where one branch
    /// scores strictly lower than another.
    pub fn preference_pairs(&self) -> Vec<PreferencePair> {
        let mut pairs = Vec::new();
        for (trace_id, map) in &self.traces {
            let scores: Vec<(u64, f32)> = map.branch_scores().into_iter().collect();
            for (i, (worse_ts, worse_score)) in scores.iter().enumerate() {
                for (better_ts, better_score) in scores.iter().skip(i + 1) {
                    if worse_score < better_score {
                        pairs.push(PreferencePair {
                            trace_id: trace_id.clone(),
                            worse_timestamp: *worse_ts,
                            better_timestamp: *better_ts,
                            worse_score: *worse_score,
                            better_score: *better_score,
                        });
                    } else if better_score < worse_score {
                        pairs.push(PreferencePair {
                            trace_id: trace_id.clone(),
                            worse_timestamp: *better_ts,
                            better_timestamp: *worse_ts,
                            worse_score: *better_score,
                            better_score: *worse_score,
                        });
                    }
                }
            }
        }
        pairs
    }

    /// Return a map of every `(trace_id, timestamp)` score reachable from the
    /// dataset. Useful for lossless conversion tests.
    pub fn score_index(&self) -> HashMap<(String, u64), f32> {
        let mut index = HashMap::new();
        for (trace_id, map) in &self.traces {
            for (timestamp, score) in map.branch_scores() {
                index.insert((trace_id.clone(), timestamp), score);
            }
        }
        index
    }
}
