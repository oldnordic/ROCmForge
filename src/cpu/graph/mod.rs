//! GeoGraph-backed CPU execution engine.

use crate::cpu::weights::WeightMeta;
use crate::cpu::CpuError;
#[cfg(feature = "cpu-graph")]
use crate::loader::GgmlType;

#[cfg(feature = "cpu-graph")]
pub use geographdb_core::algorithms::four_d::{GraphNode4D, TemporalEdge, TemporalWindow};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "cpu-graph")]
pub mod map;

#[cfg(feature = "cpu-graph")]
pub mod introspection;

#[cfg(feature = "cpu-graph")]
pub mod dataset;

#[cfg(feature = "cpu-graph")]
pub mod adapter;

#[cfg(feature = "cpu-graph")]
pub mod value_head;

#[cfg(feature = "cpu-graph")]
pub mod open_weight;

#[cfg(feature = "cpu-graph")]
pub use introspection::{
    BranchAnnotation, BranchSummary, GraphSummarizer, GraphSummary, IntrospectionPrompt,
    IntrospectionReport,
};

#[cfg(feature = "cpu-graph")]
pub use dataset::{
    GraphTraceDataset, PreferencePair, ProcessSupervisionExample, RejectionSamplingExample,
};

#[cfg(feature = "cpu-graph")]
pub use adapter::BranchAdapter;

#[cfg(feature = "cpu-graph")]
pub use value_head::{BranchChoiceExample, BranchChoiceHead, BranchValueExample, BranchValueHead};

#[cfg(feature = "cpu-graph")]
pub use open_weight::{
    build_branch_prompt, build_label_choice_prompt, build_yes_no_prompt,
    collect_branch_hidden_states, collect_branch_logit_examples, extract_answer_logit_sum,
    extract_hidden_state, extract_hidden_state_with_special, extract_label_logits,
    find_answer_token_ids, train_value_head_from_trace_dir, BranchLabelBias, BranchLogitExample,
    BranchLogitScorer, SHORT_PROMPT_MAX_SEQ_LEN,
};

/// Logical shelf where an arena handle lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Shelf {
    /// Weights and other immutable data captured once.
    Constants,
    /// State that survives across timestamps (hidden, KV cache).
    Persistent,
    /// Temporary scratch buffers reused within a timestamp.
    Ephemeral,
}

/// Scalar metric used by a `Score` node to evaluate a tensor against an
/// optional reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScoreMetric {
    /// Dot-product similarity normalized by both vector magnitudes.
    CosineSimilarity,
    /// Negative Euclidean distance to the reference (higher is closer).
    L2Similarity,
    /// Arithmetic mean of the tensor elements.
    MeanActivation,
    /// Negative Shannon entropy of the softmax distribution (sharper is higher).
    NegEntropy,
    /// Negative cross-entropy against the reference distribution (higher is
    /// a better match).
    CrossEntropy,
}

impl ScoreMetric {
    /// Parse a human-readable metric name into a `ScoreMetric`.
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "cosine" | "cosine-similarity" => ScoreMetric::CosineSimilarity,
            "l2" | "l2-similarity" => ScoreMetric::L2Similarity,
            "mean" | "mean-activation" => ScoreMetric::MeanActivation,
            "cross-entropy" => ScoreMetric::CrossEntropy,
            "entropy" | "neg-entropy" => ScoreMetric::NegEntropy,
            _ => ScoreMetric::NegEntropy,
        }
    }
}

#[cfg(feature = "cpu-graph")]
pub use map::{CandidateBranch, GpuTraceEntry, GraphMap, GraphMapError};

/// Stable handle to a contiguous f32 tensor inside a `CpuGraphArena`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct F32Handle {
    pub shelf: Shelf,
    pub offset: usize,
    pub len: usize,
}

impl F32Handle {
    pub fn new(shelf: Shelf, offset: usize, len: usize) -> Self {
        Self { shelf, offset, len }
    }
}

/// Stable handle to a contiguous u8 tensor inside a `CpuGraphArena`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct U8Handle {
    pub shelf: Shelf,
    pub offset: usize,
    pub len: usize,
}

impl U8Handle {
    pub fn new(shelf: Shelf, offset: usize, len: usize) -> Self {
        Self { shelf, offset, len }
    }
}

/// Snapshot of the persistent shelf at a single timestamp boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentSnapshot {
    pub f32_data: Vec<f32>,
    pub u8_data: Vec<u8>,
    pub f32_bindings: HashMap<usize, F32Handle>,
    pub u8_bindings: HashMap<usize, U8Handle>,
}

/// Stable storage arena for all tensor data referenced by a captured CPU graph.
///
/// Instead of recording raw pointer addresses that can become invalid between
/// capture and replay, every captured op stores handles (offsets) into this
/// arena.  The arena owns the bytes; handles remain valid as long as the arena
/// itself is alive.
///
/// The arena is split into three shelves:
/// - `Constants` — weights, sin/cos tables, and other immutable data.
/// - `Persistent` — hidden state and KV cache; survives across timestamps.
/// - `Ephemeral` — scratch buffers reused within a single timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuGraphArena {
    pub(crate) f32_constants: Vec<f32>,
    pub(crate) u8_constants: Vec<u8>,
    pub(crate) f32_persistent: Vec<f32>,
    pub(crate) u8_persistent: Vec<u8>,
    pub(crate) f32_ephemeral: Vec<f32>,
    pub(crate) u8_ephemeral: Vec<u8>,
    /// Maps the original caller pointer to the current f32 handle for that slice.
    /// Used by `read_back` to copy replay results back to caller-owned buffers.
    pub(crate) f32_bindings: HashMap<usize, F32Handle>,
    /// Same, for u8 slices.
    pub(crate) u8_bindings: HashMap<usize, U8Handle>,
}

impl CpuGraphArena {
    pub fn new() -> Self {
        Self {
            f32_constants: Vec::new(),
            u8_constants: Vec::new(),
            f32_persistent: Vec::new(),
            u8_persistent: Vec::new(),
            f32_ephemeral: Vec::new(),
            u8_ephemeral: Vec::new(),
            f32_bindings: HashMap::new(),
            u8_bindings: HashMap::new(),
        }
    }

    fn f32_vec(&self, shelf: Shelf) -> &[f32] {
        match shelf {
            Shelf::Constants => &self.f32_constants,
            Shelf::Persistent => &self.f32_persistent,
            Shelf::Ephemeral => &self.f32_ephemeral,
        }
    }

    fn f32_vec_mut(&mut self, shelf: Shelf) -> &mut Vec<f32> {
        match shelf {
            Shelf::Constants => &mut self.f32_constants,
            Shelf::Persistent => &mut self.f32_persistent,
            Shelf::Ephemeral => &mut self.f32_ephemeral,
        }
    }

    fn u8_vec(&self, shelf: Shelf) -> &[u8] {
        match shelf {
            Shelf::Constants => &self.u8_constants,
            Shelf::Persistent => &self.u8_persistent,
            Shelf::Ephemeral => &self.u8_ephemeral,
        }
    }

    fn u8_vec_mut(&mut self, shelf: Shelf) -> &mut Vec<u8> {
        match shelf {
            Shelf::Constants => &mut self.u8_constants,
            Shelf::Persistent => &mut self.u8_persistent,
            Shelf::Ephemeral => &mut self.u8_ephemeral,
        }
    }

    /// Allocate a new, zero-initialized f32 slot on the given shelf.
    pub fn alloc_f32(&mut self, shelf: Shelf, len: usize) -> F32Handle {
        let vec = self.f32_vec_mut(shelf);
        let offset = vec.len();
        vec.resize(offset + len, 0.0f32);
        F32Handle::new(shelf, offset, len)
    }

    /// Copy an f32 slice into the given shelf and return its handle.
    pub fn copy_f32(&mut self, shelf: Shelf, src: &[f32]) -> F32Handle {
        let handle = self.alloc_f32(shelf, src.len());
        self.f32_vec_mut(shelf)[handle.offset..handle.offset + handle.len].copy_from_slice(src);
        handle
    }

    /// Bind a caller-owned f32 slice to a shelf slot.  The slot is initialized
    /// from the current slice contents and registered for `read_back`.
    pub fn bind_f32(&mut self, shelf: Shelf, ptr: usize, src: &[f32]) -> F32Handle {
        let handle = self.copy_f32(shelf, src);
        self.f32_bindings.insert(ptr, handle);
        handle
    }

    /// Rebind an already-allocated f32 handle to a caller-owned slice for read-back.
    pub fn rebind_f32(&mut self, ptr: usize, handle: F32Handle) {
        self.f32_bindings.insert(ptr, handle);
    }

    pub fn f32(&self, handle: F32Handle) -> &[f32] {
        &self.f32_vec(handle.shelf)[handle.offset..handle.offset + handle.len]
    }

    pub fn f32_mut(&mut self, handle: F32Handle) -> &mut [f32] {
        let shelf = handle.shelf;
        &mut self.f32_vec_mut(shelf)[handle.offset..handle.offset + handle.len]
    }

    /// Check whether `handle` still points inside the current shelf data.
    pub fn is_f32_handle_valid(&self, handle: F32Handle) -> bool {
        let len = self.f32_vec(handle.shelf).len();
        handle.offset + handle.len <= len
    }

    /// Allocate a new, zero-initialized u8 slot on the given shelf.
    pub fn alloc_u8(&mut self, shelf: Shelf, len: usize) -> U8Handle {
        let vec = self.u8_vec_mut(shelf);
        let offset = vec.len();
        vec.resize(offset + len, 0u8);
        U8Handle::new(shelf, offset, len)
    }

    /// Copy a u8 slice into the given shelf and return its handle.
    pub fn copy_u8(&mut self, shelf: Shelf, src: &[u8]) -> U8Handle {
        let handle = self.alloc_u8(shelf, src.len());
        self.u8_vec_mut(shelf)[handle.offset..handle.offset + handle.len].copy_from_slice(src);
        handle
    }

    /// Bind a caller-owned u8 slice to a shelf slot.
    pub fn bind_u8(&mut self, shelf: Shelf, ptr: usize, src: &[u8]) -> U8Handle {
        let handle = self.copy_u8(shelf, src);
        self.u8_bindings.insert(ptr, handle);
        handle
    }

    pub fn u8(&self, handle: U8Handle) -> &[u8] {
        &self.u8_vec(handle.shelf)[handle.offset..handle.offset + handle.len]
    }

    pub fn u8_mut(&mut self, handle: U8Handle) -> &mut [u8] {
        let shelf = handle.shelf;
        &mut self.u8_vec_mut(shelf)[handle.offset..handle.offset + handle.len]
    }

    /// Copy all current bindings back to the original caller slices.
    ///
    /// # Safety
    /// The original slices must still be alive and have the same length as when
    /// they were bound.  After this call the caller-owned buffers reflect the
    /// final replay state.
    pub unsafe fn read_back(&self) {
        for (&ptr, &handle) in &self.f32_bindings {
            let src = self.f32(handle);
            let dst = std::slice::from_raw_parts_mut(ptr as *mut f32, handle.len);
            dst.copy_from_slice(src);
        }
        for (&ptr, &handle) in &self.u8_bindings {
            let src = self.u8(handle);
            let dst = std::slice::from_raw_parts_mut(ptr as *mut u8, handle.len);
            dst.copy_from_slice(src);
        }
    }

    /// Capture the current persistent shelf state.
    pub fn snapshot_persistent(&self) -> PersistentSnapshot {
        PersistentSnapshot {
            f32_data: self.f32_persistent.clone(),
            u8_data: self.u8_persistent.clone(),
            f32_bindings: self
                .f32_bindings
                .iter()
                .filter(|(_, h)| h.shelf == Shelf::Persistent)
                .map(|(&k, &v)| (k, v))
                .collect(),
            u8_bindings: self
                .u8_bindings
                .iter()
                .filter(|(_, h)| h.shelf == Shelf::Persistent)
                .map(|(&k, &v)| (k, v))
                .collect(),
        }
    }

    /// Restore the persistent shelf to a previously captured snapshot.
    pub fn restore_persistent(&mut self, snapshot: &PersistentSnapshot) {
        self.f32_persistent.clone_from(&snapshot.f32_data);
        self.u8_persistent.clone_from(&snapshot.u8_data);
        self.f32_bindings
            .retain(|_, h| h.shelf != Shelf::Persistent);
        self.u8_bindings.retain(|_, h| h.shelf != Shelf::Persistent);
        self.f32_bindings.extend(&snapshot.f32_bindings);
        self.u8_bindings.extend(&snapshot.u8_bindings);
    }
}

impl Default for CpuGraphArena {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU operator node types.
///
/// Each tensor is addressed by a stable handle into a `CpuGraphArena` rather
/// than a raw pointer, so the graph remains valid after the original buffers
/// are moved or dropped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CpuOpNode {
    RmsNorm {
        hidden: F32Handle,
        weight: F32Handle,
        out: F32Handle,
        n: usize,
        eps: f32,
    },
    Gemv {
        weight: U8Handle,
        weight_bytes: usize,
        input: F32Handle,
        out: F32Handle,
        scratch: Option<U8Handle>,
        m: usize,
        n: usize,
        /// Integer code for `GgmlType`; stored as a code so the node is
        /// serializable without depending on `GgmlType`'s exact enum layout.
        wtype_code: u32,
        needs_transpose: bool,
    },
    RoPE {
        x: F32Handle,
        sin: F32Handle,
        cos: F32Handle,
        out: F32Handle,
        n_heads: usize,
        head_dim: usize,
        neox: bool,
    },
    Attention {
        q: F32Handle,
        k: F32Handle,
        v: F32Handle,
        out: F32Handle,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    },
    SiLU {
        gate: F32Handle,
        up: F32Handle,
        out: F32Handle,
        h: usize,
    },
    ResidualAdd {
        a: F32Handle,
        b: F32Handle,
        out: F32Handle,
        h: usize,
    },
    /// Reduction operator that writes a single scalar score to `out`.
    Score {
        a: F32Handle,
        b: Option<F32Handle>,
        out: F32Handle,
        metric: ScoreMetric,
        n: usize,
    },
}

#[cfg(feature = "cpu-graph")]
pub struct CpuGraph {
    /// Temporal graph nodes.  Exposed so tests and tooling can inspect timestamps.
    pub nodes: Vec<GraphNode4D>,
    pub(crate) ops: Vec<CpuOpNode>,
}

#[cfg(feature = "cpu-graph")]
impl CpuGraph {
    pub(crate) fn from_parts(nodes: Vec<GraphNode4D>, ops: Vec<CpuOpNode>) -> Self {
        Self { nodes, ops }
    }
}

/// Abstract context for executing CPU operations.
/// Allows same forward code to be used for direct execution or graph capture.
#[cfg(test)]
pub mod tests;

/// Compute a scalar `metric` from tensor `a` and an optional reference `b`.
#[cfg(feature = "cpu-graph")]
pub fn compute_score(metric: ScoreMetric, a: &[f32], b: Option<&[f32]>) -> f32 {
    const EPS: f32 = 1e-8;
    match metric {
        ScoreMetric::CosineSimilarity => {
            let b = match b {
                Some(b) if b.len() == a.len() => b,
                _ => return 0.0,
            };
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = norm_a * norm_b;
            if denom < EPS {
                return 0.0;
            }
            dot / denom
        }
        ScoreMetric::L2Similarity => {
            if let Some(b) = b {
                if b.len() != a.len() {
                    return 0.0;
                }
                let dist: f32 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt();
                -dist
            } else {
                let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                -norm
            }
        }
        ScoreMetric::MeanActivation => {
            if a.is_empty() {
                0.0
            } else {
                a.iter().sum::<f32>() / a.len() as f32
            }
        }
        ScoreMetric::NegEntropy => {
            if a.is_empty() {
                return 0.0;
            }
            let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = a.iter().map(|x| (x - max).exp()).sum();
            let log_z = max + exp_sum.ln();
            a.iter()
                .map(|x| {
                    let p = (x - max).exp() / exp_sum;
                    if p > EPS {
                        // p * log(p) where log(p) = x - log_z
                        p * (x - log_z)
                    } else {
                        0.0
                    }
                })
                .sum()
        }
        ScoreMetric::CrossEntropy => {
            let b = match b {
                Some(b) if b.len() == a.len() => b,
                _ => return 0.0,
            };
            if a.is_empty() {
                return 0.0;
            }
            let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = a.iter().map(|x| (x - max).exp()).sum();
            let log_z = max + exp_sum.ln();
            a.iter().zip(b).map(|(x, y)| y * (x - log_z)).sum()
        }
    }
}

pub trait CpuExecutionContext {
    #[allow(clippy::too_many_arguments)]
    fn execute_gemv(
        &mut self,
        w: &[u8],
        meta: &WeightMeta,
        x: &[f32],
        y: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        q8_scratch: Option<&mut [u8]>,
    ) -> Result<(), CpuError>;

    fn execute_rms_norm(&mut self, x: &[f32], w: &[f32], out: &mut [f32], eps: f32);

    fn execute_rope(
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        sin: &[f32],
        cos: &[f32],
        neox: bool,
    );

    #[allow(clippy::too_many_arguments)]
    fn execute_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    );

    fn execute_silu(&mut self, gate: &[f32], up: &mut [f32]);

    fn execute_residual_add(&mut self, a: &mut [f32], b: &[f32]);
}

/// Direct execution context that runs kernels immediately.
pub struct DirectContext;

impl CpuExecutionContext for DirectContext {
    #[allow(clippy::too_many_arguments)]
    fn execute_gemv(
        &mut self,
        w: &[u8],
        meta: &WeightMeta,
        x: &[f32],
        y: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        q8_scratch: Option<&mut [u8]>,
    ) -> Result<(), CpuError> {
        crate::cpu::ops::dispatch_gemv(w, meta, x, y, out_dim, in_dim, q8_scratch)
    }

    fn execute_rms_norm(&mut self, x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
        crate::cpu::ops::rms_norm(x, w, out, eps);
    }

    fn execute_rope(
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        sin: &[f32],
        cos: &[f32],
        neox: bool,
    ) {
        crate::cpu::ops::rope(x, n_heads, head_dim, sin, cos, neox);
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        _max_seq_len: usize,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        crate::cpu::ops::flash_attn_decode(
            q,
            k,
            v,
            out,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            0,
            0.0,
            scale,
        );
    }

    fn execute_silu(&mut self, gate: &[f32], up: &mut [f32]) {
        crate::cpu::ops::silu_fuse(gate, up);
    }

    fn execute_residual_add(&mut self, a: &mut [f32], b: &[f32]) {
        crate::cpu::ops::residual_add(a, b);
    }
}

#[cfg(feature = "cpu-graph")]
pub struct CaptureContext {
    pub graph: CpuGraph,
    pub arena: CpuGraphArena,
    pub layer: usize,
    pub step: usize,
    pub timestamp: u64,
    /// The timestamp of the last op that was added.  Used to detect timestamp
    /// boundaries and snapshot the persistent shelf.
    last_timestamp: u64,
    /// History of caller-pointer -> output-handle bindings, annotated with the
    /// timestamp at which they became valid.  Used by `rebind_after_regress` to
    /// restore the arena bindings that correspond to a rolled-back temporal
    /// state.
    pub output_log: Vec<(u64, usize, F32Handle)>,
    /// Persistent-shelf snapshots keyed by timestamp.  Enables instant rollback
    /// without replaying the prefix.
    pub shelf_snapshots: HashMap<u64, PersistentSnapshot>,
    /// Recorded branch scores keyed by timestamp.
    pub score_log: Vec<(u64, ScoreMetric, f32)>,
    /// Human or model-generated annotations (bias, report, semantic key) keyed
    /// by branch timestamp.
    pub branch_annotations: HashMap<u64, BranchAnnotation>,
    /// Token-level candidates evaluated by the online reranker.
    pub candidate_branches: Vec<CandidateBranch>,
}

#[cfg(feature = "cpu-graph")]
impl CaptureContext {
    /// Convenience constructor.
    pub fn new(layer: usize, timestamp: u64) -> Self {
        Self {
            graph: CpuGraph::new(),
            arena: CpuGraphArena::new(),
            layer,
            step: 0,
            timestamp,
            last_timestamp: timestamp,
            output_log: Vec::new(),
            shelf_snapshots: HashMap::new(),
            score_log: Vec::new(),
            branch_annotations: HashMap::new(),
            candidate_branches: Vec::new(),
        }
    }

    /// Copy all replay results back to the caller-owned slices that were bound
    /// during capture.
    ///
    /// # Safety
    /// The caller must ensure all bound slices are still alive and have their
    /// original lengths.
    pub unsafe fn read_back(&self) {
        self.arena.read_back();
    }

    /// Snapshot the current persistent shelf if we have crossed a timestamp
    /// boundary since the last op.
    fn maybe_snapshot(&mut self) {
        if self.timestamp != self.last_timestamp {
            let snap = self.arena.snapshot_persistent();
            self.shelf_snapshots.insert(self.last_timestamp, snap);
            self.last_timestamp = self.timestamp;
        }
    }

    /// Roll back the graph and persistent state to `timestamp`.  This restores
    /// the persistent shelf from the snapshot taken at that boundary and fixes
    /// the caller-pointer bindings so `read_back` reflects the rolled-back
    /// state.
    pub fn regress_to(&mut self, timestamp: u64) {
        self.graph.regress(timestamp);
        if let Some(snap) = self.shelf_snapshots.get(&timestamp).cloned() {
            self.arena.restore_persistent(&snap);
        }
        self.rebind_after_regress(timestamp);
    }

    /// Restore the arena's caller-pointer bindings to the state that existed
    /// just after `timestamp`.  This must be called after `graph.regress` so
    /// that `read_back` reflects the rolled-back computation rather than the
    /// most-recently-captured branch.
    pub fn rebind_after_regress(&mut self, timestamp: u64) {
        let mut surviving: std::collections::HashMap<usize, F32Handle> =
            std::collections::HashMap::new();
        for (ts, ptr, handle) in &self.output_log {
            if *ts <= timestamp {
                surviving.insert(*ptr, *handle);
            }
        }
        for (ptr, handle) in surviving {
            self.arena.rebind_f32(ptr, handle);
        }
    }

    /// Score `input` against an optional `reference` using `metric` and record
    /// the scalar in `score_log` keyed by the current timestamp.
    ///
    /// The score is written into the ephemeral shelf so that replaying the
    /// current timestamp window reproduces it.
    pub fn score_against(
        &mut self,
        input: &[f32],
        reference: Option<&[f32]>,
        metric: ScoreMetric,
    ) -> f32 {
        self.maybe_snapshot();

        let a = self.arena.copy_f32(Shelf::Ephemeral, input);
        let b = reference.map(|r| self.arena.copy_f32(Shelf::Constants, r));
        let out = self.arena.alloc_f32(Shelf::Ephemeral, 1);

        let op = CpuOpNode::Score {
            a,
            b,
            out,
            metric,
            n: input.len(),
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;

        let score = compute_score(metric, input, reference);
        self.arena.f32_mut(out)[0] = score;
        self.score_log.push((self.timestamp, metric, score));
        score
    }

    /// Attach a bias/report annotation to the branch captured at `timestamp`.
    ///
    /// `key` is an optional semantic label (e.g. the action name) that lets a
    /// later session match this annotation even when timestamps differ.
    pub fn annotate_branch(
        &mut self,
        timestamp: u64,
        bias: f32,
        key: Option<&str>,
        report: Option<IntrospectionReport>,
    ) {
        let annotation = BranchAnnotation {
            timestamp,
            bias,
            report,
            key: key.map(|s| s.to_string()),
        };
        self.branch_annotations.insert(timestamp, annotation);
    }
}

#[cfg(feature = "cpu-graph")]
impl CpuExecutionContext for CaptureContext {
    #[allow(clippy::too_many_arguments)]
    fn execute_gemv(
        &mut self,
        w: &[u8],
        meta: &WeightMeta,
        x: &[f32],
        y: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        mut q8_scratch: Option<&mut [u8]>,
    ) -> Result<(), CpuError> {
        self.maybe_snapshot();

        let weight = self.arena.copy_u8(Shelf::Constants, w);
        let input = self.arena.copy_f32(Shelf::Ephemeral, x);
        let out = self.arena.alloc_f32(Shelf::Ephemeral, y.len());
        self.arena.rebind_f32(y.as_ptr() as usize, out);
        self.output_log
            .push((self.timestamp, y.as_ptr() as usize, out));
        let scratch = q8_scratch
            .as_ref()
            .map(|s| self.arena.copy_u8(Shelf::Ephemeral, s));

        let op = CpuOpNode::Gemv {
            weight,
            weight_bytes: w.len(),
            input,
            out,
            scratch,
            m: out_dim,
            n: in_dim,
            wtype_code: meta.wtype as u32,
            needs_transpose: meta.needs_transpose,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        let scratch_borrow = q8_scratch.as_deref_mut();
        crate::cpu::ops::dispatch_gemv(w, meta, x, y, out_dim, in_dim, scratch_borrow)?;
        self.arena.f32_mut(out).copy_from_slice(y);
        if let (Some(h), Some(s)) = (scratch, q8_scratch.as_mut()) {
            self.arena.u8_mut(h).copy_from_slice(s);
        }
        Ok(())
    }

    fn execute_rms_norm(&mut self, x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
        self.maybe_snapshot();

        let hidden = self.arena.copy_f32(Shelf::Persistent, x);
        let weight = self.arena.copy_f32(Shelf::Constants, w);
        let h_out = self.arena.alloc_f32(Shelf::Ephemeral, out.len());
        self.arena.rebind_f32(out.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, out.as_ptr() as usize, h_out));

        let op = CpuOpNode::RmsNorm {
            hidden,
            weight,
            out: h_out,
            n: x.len(),
            eps,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::rms_norm(x, w, out, eps);
        self.arena.f32_mut(h_out).copy_from_slice(out);
    }

    fn execute_rope(
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        sin: &[f32],
        cos: &[f32],
        neox: bool,
    ) {
        self.maybe_snapshot();

        let x_in = self.arena.copy_f32(Shelf::Ephemeral, x);
        let sin_h = self.arena.copy_f32(Shelf::Constants, sin);
        let cos_h = self.arena.copy_f32(Shelf::Constants, cos);
        let h_out = self.arena.alloc_f32(Shelf::Ephemeral, x.len());
        self.arena.rebind_f32(x.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, x.as_ptr() as usize, h_out));

        let op = CpuOpNode::RoPE {
            x: x_in,
            sin: sin_h,
            cos: cos_h,
            out: h_out,
            n_heads,
            head_dim,
            neox,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::rope(x, n_heads, head_dim, sin, cos, neox);
        self.arena.f32_mut(h_out).copy_from_slice(x);
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        self.maybe_snapshot();

        let q_h = self.arena.copy_f32(Shelf::Ephemeral, q);
        let k_h = self.arena.copy_f32(Shelf::Persistent, k);
        let v_h = self.arena.copy_f32(Shelf::Persistent, v);
        let h_out = self.arena.alloc_f32(Shelf::Ephemeral, out.len());
        self.arena.rebind_f32(out.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, out.as_ptr() as usize, h_out));

        let op = CpuOpNode::Attention {
            q: q_h,
            k: k_h,
            v: v_h,
            out: h_out,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        let scale = 1.0 / (head_dim as f32).sqrt();
        crate::cpu::ops::flash_attn_decode(
            q,
            k,
            v,
            out,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            0,
            0.0,
            scale,
        );
        self.arena.f32_mut(h_out).copy_from_slice(out);
    }

    fn execute_silu(&mut self, gate: &[f32], up: &mut [f32]) {
        self.maybe_snapshot();

        let gate_h = self.arena.copy_f32(Shelf::Ephemeral, gate);
        let up_in = self.arena.copy_f32(Shelf::Ephemeral, up);
        let h_out = self.arena.alloc_f32(Shelf::Ephemeral, up.len());
        self.arena.rebind_f32(up.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, up.as_ptr() as usize, h_out));

        let op = CpuOpNode::SiLU {
            gate: gate_h,
            up: up_in,
            out: h_out,
            h: gate.len(),
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::silu_fuse(gate, up);
        self.arena.f32_mut(h_out).copy_from_slice(up);
    }

    fn execute_residual_add(&mut self, a: &mut [f32], b: &[f32]) {
        self.maybe_snapshot();

        let a_in = self.arena.copy_f32(Shelf::Persistent, a);
        let b_h = self.arena.copy_f32(Shelf::Ephemeral, b);
        let h_out = self.arena.alloc_f32(Shelf::Persistent, a.len());
        self.arena.rebind_f32(a.as_ptr() as usize, h_out);
        self.output_log
            .push((self.timestamp, a.as_ptr() as usize, h_out));

        let op = CpuOpNode::ResidualAdd {
            a: a_in,
            b: b_h,
            out: h_out,
            h: a.len(),
        };
        self.graph
            .add_node(op, self.layer, self.step, self.timestamp);
        self.step += 1;
        crate::cpu::ops::residual_add(a, b);
        self.arena.f32_mut(h_out).copy_from_slice(a);
    }
}

#[cfg(feature = "cpu-graph")]
impl Default for CpuGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cpu-graph")]
impl CpuGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            ops: Vec::new(),
        }
    }

    pub fn add_node(&mut self, op: CpuOpNode, layer: usize, step: usize, timestamp: u64) {
        let id = self.ops.len() as u64;
        self.ops.push(op);

        let node = GraphNode4D {
            id,
            x: layer as f32,
            y: step as f32,
            z: 0.0,
            begin_ts: timestamp,
            // `u64::MAX` means "alive until the end of time".  `0` is reserved
            // for disabled/regressed nodes.
            end_ts: u64::MAX,
            properties: std::collections::BTreeMap::new(),
            successors: Vec::new(),
        };
        self.nodes.push(node);
    }

    /// Execute nodes valid within the given temporal window.
    pub fn execute_window(
        &self,
        arena: &mut CpuGraphArena,
        window: TemporalWindow,
    ) -> Result<(), CpuError> {
        // Treat `window.end == 0` as "until the end of time".  A node is
        // active when its (begin, end) interval overlaps the window.  A node
        // with `end_ts == 0` has been disabled by `regress` and never runs.
        let window_end = if window.end == 0 {
            u64::MAX
        } else {
            window.end
        };
        let mut active_nodes: Vec<&GraphNode4D> = self
            .nodes
            .iter()
            .filter(|n| {
                if n.end_ts == 0 {
                    return false; // disabled by regress
                }
                let node_end = if n.end_ts == u64::MAX {
                    u64::MAX
                } else {
                    n.end_ts
                };
                n.begin_ts < window_end && node_end > window.start
            })
            .collect();

        // Sort by time first, then spatial coordinates (Layer -> Step)
        active_nodes.sort_by(|a, b| {
            a.begin_ts
                .cmp(&b.begin_ts)
                .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
        });

        for node in active_nodes {
            let op = &self.ops[node.id as usize];
            self.execute_op(op, arena)?;
        }
        Ok(())
    }

    fn execute_op(&self, op: &CpuOpNode, arena: &mut CpuGraphArena) -> Result<(), CpuError> {
        // Use raw base pointers so we can form slices for multiple disjoint
        // handles without fighting the borrow checker.  All handles are
        // non-overlapping offsets allocated by the arena, so this is sound.
        let f32_const = arena.f32_constants.as_mut_ptr();
        let f32_persist = arena.f32_persistent.as_mut_ptr();
        let f32_ephem = arena.f32_ephemeral.as_mut_ptr();
        let u8_const = arena.u8_constants.as_mut_ptr();
        let u8_persist = arena.u8_persistent.as_mut_ptr();
        let u8_ephem = arena.u8_ephemeral.as_mut_ptr();

        let f32_base = |shelf: Shelf| match shelf {
            Shelf::Constants => f32_const,
            Shelf::Persistent => f32_persist,
            Shelf::Ephemeral => f32_ephem,
        };
        let u8_base = |shelf: Shelf| match shelf {
            Shelf::Constants => u8_const,
            Shelf::Persistent => u8_persist,
            Shelf::Ephemeral => u8_ephem,
        };

        let f32_slice = |h: F32Handle| unsafe {
            std::slice::from_raw_parts(f32_base(h.shelf).add(h.offset), h.len)
        };
        let f32_slice_mut = |h: F32Handle| unsafe {
            std::slice::from_raw_parts_mut(f32_base(h.shelf).add(h.offset), h.len)
        };
        let u8_slice = |h: U8Handle| unsafe {
            std::slice::from_raw_parts(u8_base(h.shelf).add(h.offset), h.len)
        };
        let u8_slice_mut = |h: U8Handle| unsafe {
            std::slice::from_raw_parts_mut(u8_base(h.shelf).add(h.offset), h.len)
        };

        match op {
            CpuOpNode::RmsNorm {
                hidden,
                weight,
                out,
                eps,
                n: _,
            } => {
                let hidden = f32_slice(*hidden);
                let weight = f32_slice(*weight);
                let out = f32_slice_mut(*out);
                crate::cpu::ops::rms_norm(hidden, weight, out, *eps);
            }
            CpuOpNode::Gemv {
                weight,
                input,
                out,
                scratch,
                m,
                n,
                wtype_code,
                needs_transpose,
                weight_bytes: _,
            } => {
                let w = u8_slice(*weight);
                let x = f32_slice(*input);
                let y = f32_slice_mut(*out);
                let q8_scratch = scratch.map(u8_slice_mut);
                let wtype = GgmlType::from_u32(*wtype_code).map_err(|_| {
                    CpuError::InvalidOperation(format!("invalid wtype code {}", wtype_code))
                })?;
                let meta = WeightMeta {
                    wtype,
                    dims: vec![*m as u64, *n as u64],
                    needs_transpose: *needs_transpose,
                    role: crate::config::TensorRole::Generic,
                    svd_k: None,
                };
                crate::cpu::ops::dispatch_gemv(w, &meta, x, y, *m, *n, q8_scratch)?;
            }
            CpuOpNode::RoPE {
                x,
                sin,
                cos,
                out,
                n_heads,
                head_dim,
                neox,
            } => {
                let x_out = f32_slice_mut(*out);
                // RoPE is an in-place rotation; copy the source values into the
                // output slot first so the kernel can read and write the same buffer.
                // If x and out share a handle this is a no-op.
                if *x != *out {
                    x_out.copy_from_slice(f32_slice(*x));
                }
                let sin = f32_slice(*sin);
                let cos = f32_slice(*cos);
                crate::cpu::ops::rope(x_out, *n_heads, *head_dim, sin, cos, *neox);
            }
            CpuOpNode::Attention {
                q,
                k,
                v,
                out,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq_len: _,
            } => {
                let q = f32_slice(*q);
                let k = f32_slice(*k);
                let v = f32_slice(*v);
                let out = f32_slice_mut(*out);
                let scale = 1.0 / (*head_dim as f32).sqrt();
                crate::cpu::ops::flash_attn_decode(
                    q,
                    k,
                    v,
                    out,
                    *seq_len,
                    *num_heads,
                    *num_kv_heads,
                    *head_dim,
                    0,
                    0.0,
                    scale,
                );
            }
            CpuOpNode::SiLU {
                gate,
                up,
                out,
                h: _,
            } => {
                let gate = f32_slice(*gate);
                let up_in = f32_slice(*up);
                let out_handle = *out;
                let out = f32_slice_mut(out_handle);
                if *up != out_handle {
                    out.copy_from_slice(up_in);
                }
                crate::cpu::ops::silu_fuse(gate, out);
            }
            CpuOpNode::ResidualAdd { a, b, out, h: _ } => {
                let a_in = f32_slice(*a);
                let out_handle = *out;
                let out = f32_slice_mut(out_handle);
                if *a != out_handle {
                    out.copy_from_slice(a_in);
                }
                let b = f32_slice(*b);
                crate::cpu::ops::residual_add(out, b);
            }
            CpuOpNode::Score {
                a,
                b,
                out,
                metric,
                n: _,
            } => {
                let a = f32_slice(*a);
                let b = b.map(f32_slice);
                let out = f32_slice_mut(*out);
                out[0] = compute_score(*metric, a, b);
            }
        }
        Ok(())
    }

    /// Invalidate all nodes captured after the given timestamp.
    ///
    /// Nodes that began after `timestamp` are disabled (`end_ts = 0`).  Nodes
    /// that were active across `timestamp` are capped so they no longer run in
    /// windows that start after `timestamp`.  Newly added nodes use
    /// `end_ts = u64::MAX` to mean "alive forever".
    pub fn regress(&mut self, timestamp: u64) {
        for node in self.nodes.iter_mut() {
            if node.begin_ts > timestamp {
                // Fully invalidate nodes that started after the regression point.
                node.end_ts = 0;
            } else if node.end_ts != 0 && node.end_ts > timestamp {
                // Cap nodes that were active across the regression point.
                node.end_ts = timestamp;
            }
        }
    }
}
