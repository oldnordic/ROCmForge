//! Persistent graph-map storage.
//!
//! A `GraphMap` is the minimal serializable snapshot of a captured CPU graph
//! session.  It uses `geographdb-core` for both the 4D graph topology and a
//! sectioned sidecar file for the arena, ops, bindings, output log, and
//! timestamp shelf snapshots.

use std::collections::HashMap;
use std::path::Path;

use crate::cpu::graph::{
    BranchAnnotation, CaptureContext, CpuGraph, CpuGraphArena, CpuOpNode, F32Handle,
    PersistentSnapshot, ScoreMetric, U8Handle,
};
use geographdb_core::algorithms::four_d::GraphNode4D;
use geographdb_core::storage::{load_graph4d, save_graph4d, SectionedStorage};
use thiserror::Error;

const ARENA_SIDECAR: &str = "arena.geodb";
const OPS_SECTION: &str = "ops";
const ARENA_F32_CONSTANTS_SECTION: &str = "arena_f32_constants";
const ARENA_U8_CONSTANTS_SECTION: &str = "arena_u8_constants";
const ARENA_F32_PERSISTENT_SECTION: &str = "arena_f32_persistent";
const ARENA_U8_PERSISTENT_SECTION: &str = "arena_u8_persistent";
const ARENA_F32_EPHEMERAL_SECTION: &str = "arena_f32_ephemeral";
const ARENA_U8_EPHEMERAL_SECTION: &str = "arena_u8_ephemeral";
const F32_BINDINGS_SECTION: &str = "f32_bindings";
const U8_BINDINGS_SECTION: &str = "u8_bindings";
const OUTPUT_LOG_SECTION: &str = "output_log";
const SHELF_SNAPSHOTS_SECTION: &str = "shelf_snapshots";
const SCORE_LOG_SECTION: &str = "score_log";
const BRANCH_ANNOTATIONS_SECTION: &str = "branch_annotations";
const META_SECTION: &str = "meta";
const GPU_TRACE_SECTION: &str = "gpu_trace";
const CANDIDATE_BRANCHES_SECTION: &str = "candidate_branches";

/// A single decode-step entry for a GPU-captured session.
///
/// Unlike the CPU `GraphMap`, the GPU trace does not record full layer-by-layer
/// operations or tensor data.  It captures the token-level trajectory and a
/// scalar branch score so that larger GPU-resident models can still produce a
/// persistent reasoning trace.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuTraceEntry {
    pub timestamp: u64,
    pub pos: usize,
    pub input_token_id: u32,
    pub sampled_token_id: u32,
    pub score: f32,
}

/// A token-level candidate evaluated by the online reranker.
///
/// Each candidate is a potential next token.  The value-head score and the
/// biased logit are recorded so the search tree can be inspected later.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CandidateBranch {
    pub parent_timestamp: u64,
    pub token_id: u32,
    pub value_score: f32,
    pub biased_logit: f32,
    pub chosen: bool,
}

#[derive(Debug, Error)]
pub enum GraphMapError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Graph storage error: {0}")]
    Geo(String),

    #[error("Serialization error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),

    #[error("Missing section: {0}")]
    MissingSection(&'static str),
}

/// Minimal metadata for a captured session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GraphMapMeta {
    layer: usize,
    timestamp: u64,
    last_timestamp: u64,
}

/// Persistent snapshot of a captured CPU graph session.
#[derive(Debug, Clone)]
pub struct GraphMap {
    pub nodes: Vec<GraphNode4D>,
    pub ops: Vec<CpuOpNode>,
    pub arena: CpuGraphArena,
    pub output_log: Vec<(u64, usize, F32Handle)>,
    pub shelf_snapshots: HashMap<u64, PersistentSnapshot>,
    pub score_log: Vec<(u64, ScoreMetric, f32)>,
    pub branch_annotations: HashMap<u64, BranchAnnotation>,
    pub gpu_trace: Vec<GpuTraceEntry>,
    pub candidate_branches: Vec<CandidateBranch>,
    pub layer: usize,
    pub timestamp: u64,
    pub last_timestamp: u64,
}

impl GraphMap {
    /// Build a map from a capture context without consuming it.
    pub fn from_context(ctx: &CaptureContext) -> Self {
        Self {
            nodes: ctx.graph.nodes.clone(),
            ops: ctx.graph.ops.clone(),
            arena: ctx.arena.clone(),
            output_log: ctx.output_log.clone(),
            shelf_snapshots: ctx.shelf_snapshots.clone(),
            score_log: ctx.score_log.clone(),
            branch_annotations: ctx.branch_annotations.clone(),
            gpu_trace: Vec::new(),
            candidate_branches: ctx.candidate_branches.clone(),
            layer: ctx.layer,
            timestamp: ctx.timestamp,
            last_timestamp: ctx.last_timestamp,
        }
    }

    /// Build a GPU-only trace map with empty CPU graph/arena fields.
    pub fn from_gpu_trace(
        score_log: Vec<(u64, ScoreMetric, f32)>,
        gpu_trace: Vec<GpuTraceEntry>,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            ops: Vec::new(),
            arena: CpuGraphArena::new(),
            output_log: Vec::new(),
            shelf_snapshots: HashMap::new(),
            score_log,
            branch_annotations: HashMap::new(),
            gpu_trace,
            candidate_branches: Vec::new(),
            layer: 0,
            timestamp: 0,
            last_timestamp: 0,
        }
    }

    /// Persist the map to `dir`.  The directory is created if necessary.
    pub fn save(&self, dir: &Path) -> Result<(), GraphMapError> {
        std::fs::create_dir_all(dir)?;

        // 1. Save 4D graph topology through geographdb-core.
        save_graph4d(&self.nodes, dir).map_err(|e| GraphMapError::Geo(e.to_string()))?;

        // 2. Sidecar file for everything else.
        let sidecar_path = dir.join(ARENA_SIDECAR);
        let mut storage = SectionedStorage::create(&sidecar_path)
            .map_err(|e| GraphMapError::Geo(e.to_string()))?;

        let ops_bytes = bincode::serialize(&self.ops)?;
        let f32_constants = bincode::serialize(&self.arena.f32_constants)?;
        let u8_constants = bincode::serialize(&self.arena.u8_constants)?;
        let f32_persistent = bincode::serialize(&self.arena.f32_persistent)?;
        let u8_persistent = bincode::serialize(&self.arena.u8_persistent)?;
        let f32_ephemeral = bincode::serialize(&self.arena.f32_ephemeral)?;
        let u8_ephemeral = bincode::serialize(&self.arena.u8_ephemeral)?;
        let f32_bindings: Vec<(usize, F32Handle)> = self
            .arena
            .f32_bindings
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        let u8_bindings: Vec<(usize, U8Handle)> = self
            .arena
            .u8_bindings
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        let output_log_bytes = bincode::serialize(&self.output_log)?;
        let shelf_snapshots_bytes = bincode::serialize(&self.shelf_snapshots)?;
        let score_log_bytes = bincode::serialize(&self.score_log)?;
        let branch_annotations_bytes = bincode::serialize(&self.branch_annotations)?;
        let gpu_trace_bytes = bincode::serialize(&self.gpu_trace)?;
        let candidate_branches_bytes = bincode::serialize(&self.candidate_branches)?;
        let meta = GraphMapMeta {
            layer: self.layer,
            timestamp: self.timestamp,
            last_timestamp: self.last_timestamp,
        };
        let meta_bytes = bincode::serialize(&meta)?;

        create_and_write(&mut storage, OPS_SECTION, &ops_bytes)?;
        create_and_write(&mut storage, ARENA_F32_CONSTANTS_SECTION, &f32_constants)?;
        create_and_write(&mut storage, ARENA_U8_CONSTANTS_SECTION, &u8_constants)?;
        create_and_write(&mut storage, ARENA_F32_PERSISTENT_SECTION, &f32_persistent)?;
        create_and_write(&mut storage, ARENA_U8_PERSISTENT_SECTION, &u8_persistent)?;
        create_and_write(&mut storage, ARENA_F32_EPHEMERAL_SECTION, &f32_ephemeral)?;
        create_and_write(&mut storage, ARENA_U8_EPHEMERAL_SECTION, &u8_ephemeral)?;
        create_and_write(
            &mut storage,
            F32_BINDINGS_SECTION,
            &bincode::serialize(&f32_bindings)?,
        )?;
        create_and_write(
            &mut storage,
            U8_BINDINGS_SECTION,
            &bincode::serialize(&u8_bindings)?,
        )?;
        create_and_write(&mut storage, OUTPUT_LOG_SECTION, &output_log_bytes)?;
        create_and_write(
            &mut storage,
            SHELF_SNAPSHOTS_SECTION,
            &shelf_snapshots_bytes,
        )?;
        create_and_write(&mut storage, SCORE_LOG_SECTION, &score_log_bytes)?;
        create_and_write(
            &mut storage,
            BRANCH_ANNOTATIONS_SECTION,
            &branch_annotations_bytes,
        )?;
        create_and_write(&mut storage, GPU_TRACE_SECTION, &gpu_trace_bytes)?;
        create_and_write(
            &mut storage,
            CANDIDATE_BRANCHES_SECTION,
            &candidate_branches_bytes,
        )?;
        create_and_write(&mut storage, META_SECTION, &meta_bytes)?;

        storage
            .flush()
            .map_err(|e| GraphMapError::Geo(e.to_string()))?;
        drop(storage);
        Ok(())
    }

    /// Load a map from `dir`.
    pub fn load(dir: &Path) -> Result<Self, GraphMapError> {
        // 1. Load 4D graph topology.
        let nodes = load_graph4d(dir).map_err(|e| GraphMapError::Geo(e.to_string()))?;

        // 2. Load sidecar.
        let sidecar_path = dir.join(ARENA_SIDECAR);
        let mut storage =
            SectionedStorage::open(&sidecar_path).map_err(|e| GraphMapError::Geo(e.to_string()))?;

        let ops: Vec<CpuOpNode> = read_section_bincode(&mut storage, OPS_SECTION)?;
        let f32_constants: Vec<f32> =
            read_section_bincode(&mut storage, ARENA_F32_CONSTANTS_SECTION)?;
        let u8_constants: Vec<u8> = read_section_bincode(&mut storage, ARENA_U8_CONSTANTS_SECTION)?;
        let f32_persistent: Vec<f32> =
            read_section_bincode(&mut storage, ARENA_F32_PERSISTENT_SECTION)?;
        let u8_persistent: Vec<u8> =
            read_section_bincode(&mut storage, ARENA_U8_PERSISTENT_SECTION)?;
        let f32_ephemeral: Vec<f32> =
            read_section_bincode(&mut storage, ARENA_F32_EPHEMERAL_SECTION)?;
        let u8_ephemeral: Vec<u8> = read_section_bincode(&mut storage, ARENA_U8_EPHEMERAL_SECTION)?;
        let f32_bindings_vec: Vec<(usize, F32Handle)> =
            read_section_bincode(&mut storage, F32_BINDINGS_SECTION)?;
        let u8_bindings_vec: Vec<(usize, U8Handle)> =
            read_section_bincode(&mut storage, U8_BINDINGS_SECTION)?;
        let output_log: Vec<(u64, usize, F32Handle)> =
            read_section_bincode(&mut storage, OUTPUT_LOG_SECTION)?;
        let shelf_snapshots: HashMap<u64, PersistentSnapshot> =
            read_section_bincode(&mut storage, SHELF_SNAPSHOTS_SECTION)?;
        let score_log: Vec<(u64, ScoreMetric, f32)> =
            read_section_bincode(&mut storage, SCORE_LOG_SECTION)?;
        let branch_annotations: HashMap<u64, BranchAnnotation> =
            read_section_bincode(&mut storage, BRANCH_ANNOTATIONS_SECTION)?;
        let gpu_trace: Vec<GpuTraceEntry> = if storage.get_section(GPU_TRACE_SECTION).is_some() {
            read_section_bincode(&mut storage, GPU_TRACE_SECTION)?
        } else {
            Vec::new()
        };
        let candidate_branches: Vec<CandidateBranch> =
            if storage.get_section(CANDIDATE_BRANCHES_SECTION).is_some() {
                read_section_bincode(&mut storage, CANDIDATE_BRANCHES_SECTION)?
            } else {
                Vec::new()
            };
        let meta: GraphMapMeta = read_section_bincode(&mut storage, META_SECTION)?;

        let f32_bindings: HashMap<usize, F32Handle> = f32_bindings_vec.into_iter().collect();
        let u8_bindings: HashMap<usize, U8Handle> = u8_bindings_vec.into_iter().collect();

        let arena = CpuGraphArena {
            f32_constants,
            u8_constants,
            f32_persistent,
            u8_persistent,
            f32_ephemeral,
            u8_ephemeral,
            f32_bindings,
            u8_bindings,
        };

        Ok(Self {
            nodes,
            ops,
            arena,
            output_log,
            shelf_snapshots,
            score_log,
            branch_annotations,
            gpu_trace,
            candidate_branches,
            layer: meta.layer,
            timestamp: meta.timestamp,
            last_timestamp: meta.last_timestamp,
        })
    }

    /// Consume the map and reconstruct a `CaptureContext` that can replay.
    pub fn into_context(self) -> CaptureContext {
        CaptureContext {
            graph: CpuGraph::from_parts(self.nodes, self.ops),
            arena: self.arena,
            layer: self.layer,
            step: 0,
            timestamp: self.timestamp,
            last_timestamp: self.last_timestamp,
            output_log: self.output_log,
            shelf_snapshots: self.shelf_snapshots,
            score_log: self.score_log,
            branch_annotations: self.branch_annotations,
            candidate_branches: self.candidate_branches,
        }
    }

    /// Return the GPU decode trace, if any.
    pub fn gpu_trace(&self) -> &[GpuTraceEntry] {
        &self.gpu_trace
    }

    /// Return the reranker candidate branches, if any.
    pub fn candidate_branches(&self) -> &[CandidateBranch] {
        &self.candidate_branches
    }

    /// Return the bias for each annotated timestamp.
    pub fn branch_biases(&self) -> HashMap<u64, f32> {
        self.branch_annotations
            .iter()
            .map(|(&ts, a)| (ts, a.bias))
            .collect()
    }

    /// Return the bias for the branch annotated with `key`, if any.
    pub fn branch_bias_by_key(&self, key: &str) -> Option<f32> {
        self.branch_annotations
            .values()
            .find(|a| a.key.as_deref() == Some(key))
            .map(|a| a.bias)
    }

    /// Aggregate all annotations by key, keeping the maximum bias per key.
    pub fn biases_by_key(&self) -> HashMap<String, f32> {
        let mut out = HashMap::new();
        for a in self.branch_annotations.values() {
            if let Some(ref key) = a.key {
                out.entry(key.clone())
                    .and_modify(|b: &mut f32| *b = b.max(a.bias))
                    .or_insert(a.bias);
            }
        }
        out
    }

    /// Attach or overwrite an annotation for a branch timestamp.
    pub fn annotate_branch(&mut self, annotation: BranchAnnotation) {
        self.branch_annotations
            .insert(annotation.timestamp, annotation);
    }

    /// Return the last recorded score for each timestamp.
    pub fn branch_scores(&self) -> HashMap<u64, f32> {
        let mut scores = HashMap::new();
        for (ts, _metric, score) in &self.score_log {
            scores.insert(*ts, *score);
        }
        scores
    }

    /// Return the absolute difference between each branch score and a reference
    /// score.  For similarity metrics the reference is typically the best
    /// attainable score (e.g. `1.0` for cosine similarity).
    pub fn divergence(&self, reference: f32) -> HashMap<u64, f32> {
        self.branch_scores()
            .into_iter()
            .map(|(ts, score)| (ts, (score - reference).abs()))
            .collect()
    }
}

fn create_and_write(
    storage: &mut SectionedStorage,
    name: &str,
    data: &[u8],
) -> Result<(), GraphMapError> {
    storage
        .create_section(name, data.len() as u64, 0)
        .map_err(|e| GraphMapError::Geo(e.to_string()))?;
    storage
        .write_section(name, data)
        .map_err(|e| GraphMapError::Geo(e.to_string()))?;
    Ok(())
}

fn read_section_bincode<T: serde::de::DeserializeOwned>(
    storage: &mut SectionedStorage,
    name: &'static str,
) -> Result<T, GraphMapError> {
    let bytes = storage
        .read_section(name)
        .map_err(|e| GraphMapError::Geo(e.to_string()))?;
    Ok(bincode::deserialize(&bytes)?)
}
