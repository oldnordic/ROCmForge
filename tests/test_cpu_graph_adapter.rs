#![cfg(feature = "cpu-graph")]
//! Step 7 validation experiment: lightweight mistake-driven adapter.
//!
//! Trains a tiny two-layer MLP (`BranchAdapter`) on preference pairs exported
//! from a `GraphTraceDataset`, then tests it on held-out traces. The adapter
//! must pick the highest-scoring branch more accurately than a random baseline.

use fastrand::Rng;
use rocmforge::cpu::graph::{
    BranchAdapter, CaptureContext, CpuExecutionContext, GraphMap, GraphTraceDataset, ScoreMetric,
    Shelf,
};

const DIM: usize = 4;

fn make_trace(rng: &mut Rng, n_branches: usize) -> GraphMap {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden = vec![0.0f32; DIM];
    hidden[0] = 1.0;
    let target: Vec<f32> = (0..DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    for idx in 0..n_branches {
        ctx.timestamp = (idx + 1) as u64;
        let scale = rng.f32() * 2.0 - 1.0;
        let perturbation = vec![scale; DIM];
        ctx.execute_residual_add(&mut hidden, &perturbation);
        let _score = ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);
        ctx.regress_to(0);
    }

    GraphMap::from_context(&ctx)
}

fn best_branch_by_score(map: &GraphMap) -> u64 {
    *map.branch_scores()
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(ts, _)| ts)
        .expect("at least one branch")
}

fn evaluate_adapter(adapter: &BranchAdapter, map: &GraphMap) -> bool {
    let ranked = adapter.rank_branches(map);
    let predicted_best = ranked[0].0;
    let true_best = best_branch_by_score(map);
    predicted_best == true_best
}

fn evaluate_random(map: &GraphMap, rng: &mut Rng) -> bool {
    let scores = map.branch_scores();
    let branches: Vec<u64> = scores.keys().copied().collect();
    let predicted = branches[rng.usize(..branches.len())];
    let true_best = best_branch_by_score(map);
    predicted == true_best
}

#[test]
fn test_adapter_trains_and_beats_random_baseline() {
    let mut rng = Rng::with_seed(12345);

    let train_maps: Vec<GraphMap> = (0..24).map(|_| make_trace(&mut rng, 4)).collect();
    let test_maps: Vec<GraphMap> = (0..12).map(|_| make_trace(&mut rng, 4)).collect();

    let train_traces: Vec<(String, GraphMap)> = train_maps
        .into_iter()
        .enumerate()
        .map(|(i, m)| (format!("train_{}", i), m))
        .collect();
    let dataset = GraphTraceDataset {
        traces: train_traces,
    };

    let mut adapter = BranchAdapter::new(6, 8, 42);
    adapter.train(&dataset, 80, 0.02, 0.5);

    let mut adapter_correct = 0usize;
    let mut random_correct = 0usize;
    for map in &test_maps {
        if evaluate_adapter(&adapter, map) {
            adapter_correct += 1;
        }
        if evaluate_random(map, &mut rng) {
            random_correct += 1;
        }
    }

    eprintln!(
        "Adapter accuracy: {}/{}  Random accuracy: {}/{}",
        adapter_correct,
        test_maps.len(),
        random_correct,
        test_maps.len()
    );

    assert!(
        adapter_correct > random_correct,
        "adapter must beat random baseline: adapter={}/{} random={}/{}",
        adapter_correct,
        test_maps.len(),
        random_correct,
        test_maps.len()
    );

    // The adapter should also be meaningfully better than random; on 4-branch
    // traces random expectation is ~25%.
    assert!(
        adapter_correct >= test_maps.len() * 2 / 3,
        "adapter should reach at least 2/3 accuracy: {}/{}",
        adapter_correct,
        test_maps.len()
    );
}
