#![cfg(feature = "cpu-graph")]
//! Step 6 validation experiment: trace dataset for training.
//!
//! Builds a `GraphTraceDataset` over persisted `GraphMap`s and verifies that
//! it can be exported into process-supervision, rejection-sampling, and
//! preference-pair formats without losing the original branch scores.

use rocmforge::cpu::graph::{
    dataset::{GraphTraceDataset, PreferencePair, ProcessSupervisionExample},
    CaptureContext, CpuExecutionContext, GraphMap, ScoreMetric, Shelf,
};

fn make_trace(_name: &str, scores: &[f32]) -> GraphMap {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden = vec![1.0f32, 0.0, 0.0, 0.0];
    let target = vec![1.0f32; 4];
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    for (idx, &scale) in scores.iter().enumerate() {
        ctx.timestamp = (idx + 1) as u64;
        let perturbation = vec![scale; 4];
        ctx.execute_residual_add(&mut hidden, &perturbation);
        let _score = ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);
        ctx.regress_to(0);
    }

    GraphMap::from_context(&ctx)
}

#[test]
fn test_dataset_process_supervision_preserves_scores() {
    let map = make_trace("trace_a", &[0.9, -0.5, 0.2]);
    let dataset = GraphTraceDataset::from_map("trace_a", map);
    let examples = dataset.process_supervision_examples();

    assert_eq!(examples.len(), 3);
    let index = dataset.score_index();
    for ex in &examples {
        let expected = index[&(ex.trace_id.clone(), ex.timestamp)];
        assert!(
            (ex.score - expected).abs() < 1e-5,
            "score mismatch at timestamp {}: {} vs {}",
            ex.timestamp,
            ex.score,
            expected
        );
        assert!(
            (0.0..=1.0).contains(&ex.label),
            "label must be normalized to [0,1]: {}",
            ex.label
        );
    }

    let mut sorted = examples.clone();
    sorted.sort_by_key(|e| e.timestamp);
    let labels: Vec<f32> = sorted.iter().map(|e| e.label).collect();
    // Timestamps 1/2/3 correspond to scales 0.9, -0.5, 0.2.
    assert!(
        labels[0] > labels[2] && labels[2] > labels[1],
        "labels must rank scores"
    );
}

#[test]
fn test_dataset_rejection_sampling_marks_best_as_accepted() {
    let map = make_trace("trace_b", &[0.1, 0.8, -0.3, 0.5]);
    let best_timestamp = *map
        .branch_scores()
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(ts, _)| ts)
        .expect("at least one branch");

    let dataset = GraphTraceDataset::from_map("trace_b", map);
    let examples = dataset.rejection_sampling_examples();

    let accepted: Vec<_> = examples.iter().filter(|e| e.accepted).collect();
    assert_eq!(
        accepted.len(),
        1,
        "exactly one branch should be accepted per trace"
    );
    assert_eq!(
        accepted[0].timestamp, best_timestamp,
        "accepted branch must have the highest score"
    );
    assert!(
        examples.iter().filter(|e| !e.accepted).count() == 3,
        "all other branches should be rejected"
    );
}

#[test]
fn test_dataset_preference_pairs_cover_all_orderings() {
    let scores = [0.1f32, 0.8, -0.3, 0.5];
    let map = make_trace("trace_c", &scores);
    let dataset = GraphTraceDataset::from_map("trace_c", map);
    let pairs = dataset.preference_pairs();

    // n*(n-1)/2 ordered pairs for 4 distinct scores = 6
    assert_eq!(pairs.len(), 6, "all score orderings should produce a pair");

    for pair in &pairs {
        assert!(
            pair.worse_score < pair.better_score,
            "worse_score must be lower than better_score"
        );
    }

    // Every unordered combination appears exactly once as a directed pair.
    let mut seen = std::collections::HashSet::new();
    for pair in &pairs {
        let key = if pair.worse_timestamp < pair.better_timestamp {
            (pair.worse_timestamp, pair.better_timestamp)
        } else {
            (pair.better_timestamp, pair.worse_timestamp)
        };
        assert!(seen.insert(key), "duplicate pair: {:?}", key);
    }
    assert_eq!(seen.len(), 6);
}

#[test]
fn test_dataset_loads_from_persisted_directory() {
    let dir = tempfile::tempdir().expect("tempdir");

    let map_a = make_trace("trace_a", &[0.9, -0.5, 0.2]);
    let map_b = make_trace("trace_b", &[0.1, 0.8, -0.3, 0.5]);

    let path_a = dir.path().join("trace_a");
    let path_b = dir.path().join("trace_b");
    map_a.save(&path_a).expect("save trace_a");
    map_b.save(&path_b).expect("save trace_b");

    let dataset = GraphTraceDataset::from_dir(dir.path()).expect("load dataset");
    assert_eq!(dataset.traces.len(), 2);

    let mut ids: Vec<_> = dataset.traces.iter().map(|(id, _)| id.clone()).collect();
    ids.sort();
    assert_eq!(ids, vec!["trace_a", "trace_b"]);

    let ps_examples = dataset.process_supervision_examples();
    assert_eq!(ps_examples.len(), 7);

    let rs_examples = dataset.rejection_sampling_examples();
    assert_eq!(rs_examples.iter().filter(|e| e.accepted).count(), 2);

    // Preference pairs: C(3,2) + C(4,2) = 3 + 6 = 9
    let pairs = dataset.preference_pairs();
    assert_eq!(pairs.len(), 9);
}

#[test]
fn test_preference_pair_struct_serde_round_trip() {
    let pair = PreferencePair {
        trace_id: "trace_x".to_string(),
        worse_timestamp: 1,
        better_timestamp: 3,
        worse_score: 0.1,
        better_score: 0.9,
    };
    let encoded = bincode::serialize(&pair).expect("serialize");
    let decoded: PreferencePair = bincode::deserialize(&encoded).expect("deserialize");
    assert_eq!(pair, decoded);
}

#[test]
fn test_process_supervision_example_serde_round_trip() {
    let ex = ProcessSupervisionExample {
        trace_id: "trace_y".to_string(),
        timestamp: 2,
        score: 0.7,
        divergence: 0.3,
        label: 0.85,
    };
    let encoded = bincode::serialize(&ex).expect("serialize");
    let decoded: ProcessSupervisionExample = bincode::deserialize(&encoded).expect("deserialize");
    assert_eq!(ex, decoded);
}
