#![cfg(feature = "cpu-graph")]
//! Fast unit test for the Step 8 value head.
//!
//! This test does not load a model. It creates synthetic hidden vectors where
//! the target score is a known linear function of the input, then verifies
//! that `BranchValueHead` learns to rank examples above a random baseline.

use fastrand::Rng;
use rocmforge::cpu::graph::{BranchValueExample, BranchValueHead};

const HIDDEN_SIZE: usize = 32;
const N_TRAIN: usize = 64;
const N_TEST: usize = 24;

/// Generate a hidden vector whose score is `score = dot(hidden, target) + 0.1 * norm`.
fn make_example(rng: &mut Rng, target: &[f32], idx: usize) -> BranchValueExample {
    let hidden: Vec<f32> = (0..HIDDEN_SIZE)
        .map(|_| rng.f32() * 2.0f32 - 1.0f32)
        .collect();
    let score = target
        .iter()
        .zip(hidden.iter())
        .map(|(&t, &h)| t * h)
        .sum::<f32>()
        + 0.1f32 * hidden.iter().map(|&h| h * h).sum::<f32>().sqrt();
    BranchValueExample {
        trace_id: format!("trace_{}", idx),
        timestamp: idx as u64,
        hidden,
        score,
    }
}

fn best_by_score(examples: &[BranchValueExample]) -> u64 {
    examples
        .iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|ex| ex.timestamp)
        .expect("non-empty examples")
}

fn evaluate_head(head: &BranchValueHead, examples: &[BranchValueExample]) -> bool {
    let predicted = examples
        .iter()
        .max_by(|a, b| {
            head.predict(&a.hidden)
                .partial_cmp(&head.predict(&b.hidden))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|ex| ex.timestamp)
        .expect("non-empty examples");
    predicted == best_by_score(examples)
}

fn evaluate_random(examples: &[BranchValueExample], rng: &mut Rng) -> bool {
    let predicted = examples[rng.usize(..examples.len())].timestamp;
    predicted == best_by_score(examples)
}

#[test]
fn test_value_head_trains_on_synthetic_hidden_states() {
    let mut rng = Rng::with_seed(2025);
    let target: Vec<f32> = (0..HIDDEN_SIZE)
        .map(|_| rng.f32() * 2.0f32 - 1.0f32)
        .collect();

    let train: Vec<BranchValueExample> = (0..N_TRAIN)
        .map(|i| make_example(&mut rng, &target, i))
        .collect();
    let test: Vec<BranchValueExample> = (N_TRAIN..N_TRAIN + N_TEST)
        .map(|i| make_example(&mut rng, &target, i))
        .collect();

    let mut head = BranchValueHead::new(HIDDEN_SIZE);
    head.fit_mse(&train, 40, 0.1f32);

    let mut trained_correct = 0usize;
    let mut random_correct = 0usize;
    for i in 0..(N_TEST / 4) {
        // Evaluate on small groups of 4 to mimic a 4-branch trace.
        let group = &test[i * 4..(i + 1) * 4];
        if evaluate_head(&head, group) {
            trained_correct += 1;
        }
        if evaluate_random(group, &mut rng) {
            random_correct += 1;
        }
    }

    eprintln!(
        "Value head accuracy: {}/{}  Random accuracy: {}/{}",
        trained_correct,
        N_TEST / 4,
        random_correct,
        N_TEST / 4
    );

    assert!(
        trained_correct > random_correct,
        "trained head must beat random baseline"
    );
    assert!(
        trained_correct >= (N_TEST / 4) * 2 / 3,
        "trained head should reach at least 2/3 accuracy: {}/{}",
        trained_correct,
        N_TEST / 4
    );
}
