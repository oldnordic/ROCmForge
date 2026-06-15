#![cfg(feature = "cpu-graph")]
//! Step 5 validation experiment: feedback loop as graph annotations.
//!
//! A first graph session evaluates several candidate branches, scores each one,
//! and stores the score as a `branch_bias` annotation keyed by branch name in
//! the `GraphMap`. A second session loads those biases and evaluates the
//! branches in bias-descending order. The test asserts that the biased second
//! session reaches the best branch with fewer evaluations (i.e. fewer
//! rollbacks) than the default-order first session.

use rocmforge::cpu::graph::{CaptureContext, CpuExecutionContext, GraphMap, ScoreMetric, Shelf};

const DIM: usize = 8;
const BRANCHES: [&str; 4] = ["branch_a", "branch_b", "branch_c", "branch_d"];

fn initial_hidden() -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 1.0;
    v
}

fn target() -> Vec<f32> {
    vec![1.0f32; DIM]
}

fn perturbations() -> Vec<Vec<f32>> {
    vec![
        vec![-0.8f32; DIM], // bad
        vec![0.2f32; DIM],  // mediocre
        vec![0.9f32; DIM],  // good
        vec![-0.3f32; DIM], // bad
    ]
}

/// First session: evaluate every branch in fixed order, score it, and annotate
/// the graph map with the score as a bias under the branch key.
fn first_session() -> (GraphMap, String) {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden = initial_hidden();
    let target = target();
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    let perts = perturbations();
    let mut best_key = BRANCHES[0].to_string();
    let mut best_score = f32::NEG_INFINITY;

    for (idx, (pert, key)) in perts.iter().zip(BRANCHES.iter()).enumerate() {
        ctx.timestamp = (idx + 1) as u64;
        ctx.execute_residual_add(&mut hidden, pert);
        let score = ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);
        ctx.annotate_branch(ctx.timestamp, score, Some(key), None);
        if score > best_score {
            best_score = score;
            best_key = key.to_string();
        }
        ctx.regress_to(0);
    }

    (GraphMap::from_context(&ctx), best_key)
}

/// Count how many branches must be evaluated before finding `best_key` when
/// using the given evaluation order. Each branch is created in a fresh context.
fn evaluations_until_best(order: &[usize], best_key: &str) -> usize {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden = initial_hidden();
    let target = target();
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    let perts = perturbations();
    let mut count = 0;
    for &idx in order {
        count += 1;
        ctx.timestamp = count as u64;
        ctx.execute_residual_add(&mut hidden, &perts[idx]);
        let _score = ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);
        if BRANCHES[idx] == best_key {
            break;
        }
        ctx.regress_to(0);
    }
    count
}

#[test]
fn test_branch_annotations_persist_in_graph_map() {
    let (map, best_key) = first_session();
    let dir = tempfile::tempdir().expect("tempdir");
    map.save(dir.path()).expect("save graph map");
    let loaded = GraphMap::load(dir.path()).expect("load graph map");

    let biases = loaded.biases_by_key();
    assert_eq!(biases.len(), BRANCHES.len());

    // The best branch must have the highest bias after round-trip.
    let best_bias = biases[&best_key];
    for (key, bias) in &biases {
        if *key == best_key {
            continue;
        }
        assert!(
            best_bias >= *bias,
            "best branch must have the highest bias: {} ({}) vs {} ({})",
            best_key,
            best_bias,
            key,
            bias
        );
    }
}

#[test]
fn test_feedback_loop_biases_reduce_evaluations() {
    let (map, best_key) = first_session();

    // Default order evaluates branch_a, then b, c, d.
    let default_order: Vec<usize> = (0..BRANCHES.len()).collect();
    let default_count = evaluations_until_best(&default_order, &best_key);

    // Biased order uses the persisted biases to sort branches highest-first.
    let biases = map.biases_by_key();
    let mut biased_order: Vec<usize> = (0..BRANCHES.len()).collect();
    biased_order.sort_by(|&a, &b| {
        let ba = biases.get(BRANCHES[b]).copied().unwrap_or(0.0);
        let bb = biases.get(BRANCHES[a]).copied().unwrap_or(0.0);
        ba.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let biased_count = evaluations_until_best(&biased_order, &best_key);

    eprintln!(
        "default_order={:?} count={}  biased_order={:?} count={}  best={}",
        default_order, default_count, biased_order, biased_count, best_key
    );

    assert!(
        biased_count < default_count,
        "biased session must reach the best branch faster: biased={} default={}",
        biased_count,
        default_count
    );
}
