#![cfg(feature = "cpu-graph")]
//! Step 3 validation experiment: branch scoring and divergence metrics.
//!
//! Adds a `CpuOpNode::Score` operator and `CaptureContext::score_against()` so
//! that branches can be ranked without replaying them.  The experiment proves
//! that a branch whose hidden state is close to a target reference receives a
//! higher score than a branch that was pushed away from the target, and that
//! the `GraphMap` round-trip preserves both the scores and the divergence.

use rocmforge::cpu::graph::{
    CaptureContext, CpuExecutionContext, CpuGraph, CpuGraphArena, CpuOpNode, GraphMap, ScoreMetric,
    Shelf, TemporalWindow,
};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}

#[test]
fn test_score_op_executes_scalar_output() {
    let mut arena = CpuGraphArena::new();
    let a = arena.copy_f32(Shelf::Constants, &[1.0f32, 2.0, 3.0]);
    let b = arena.copy_f32(Shelf::Constants, &[1.0f32, 1.0, 1.0]);
    let out = arena.alloc_f32(Shelf::Ephemeral, 1);

    let op = CpuOpNode::Score {
        a,
        b: Some(b),
        out,
        metric: ScoreMetric::CosineSimilarity,
        n: 3,
    };

    let mut graph = CpuGraph::new();
    graph.add_node(op, 0, 0, 7);
    graph
        .execute_window(&mut arena, TemporalWindow { start: 7, end: 8 })
        .expect("score replay failed");

    let score = arena.f32(out)[0];
    let expected = cosine_similarity(&[1.0f32, 2.0, 3.0], &[1.0f32, 1.0, 1.0]);
    assert!(
        (score - expected).abs() < 1e-5,
        "score replay mismatch: got {score}, expected {expected}"
    );
}

#[test]
fn test_branch_score_ranks_correct_above_failed() {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let target = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    // Branch A: push hidden toward the target.
    let toward = vec![0.9f32; 8];
    ctx.timestamp = 1;
    ctx.execute_residual_add(&mut hidden, &toward);
    let score_a = ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    // Roll back to the pre-branch state.
    ctx.regress_to(0);

    // Branch B: push hidden away from the target.
    let away = vec![-2.0f32; 8];
    ctx.timestamp = 2;
    ctx.execute_residual_add(&mut hidden, &away);
    let score_b = ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    assert!(
        score_a > score_b,
        "branch toward target should score higher: a={score_a}, b={score_b}"
    );

    let map = GraphMap::from_context(&ctx);
    let scores = map.branch_scores();
    assert_eq!(scores.get(&1).copied(), Some(score_a));
    assert_eq!(scores.get(&2).copied(), Some(score_b));

    let divergence = map.divergence(1.0);
    assert!(
        divergence[&1] < divergence[&2],
        "branch toward target should diverge less: a={:?}, b={:?}",
        divergence.get(&1),
        divergence.get(&2)
    );
}

#[test]
fn test_graph_map_preserves_score_log() {
    let mut ctx = CaptureContext::new(0, 0);
    let input = vec![1.0f32, 2.0, 3.0];
    let reference = vec![1.0f32, 1.0, 1.0];
    ctx.score_against(&input, Some(&reference), ScoreMetric::CosineSimilarity);

    let map = GraphMap::from_context(&ctx);
    let dir = tempfile::tempdir().expect("tempdir");
    map.save(dir.path()).expect("save graph map");
    let loaded = GraphMap::load(dir.path()).expect("load graph map");

    assert_eq!(loaded.score_log, ctx.score_log);
    assert_eq!(loaded.branch_scores(), map.branch_scores());
}
