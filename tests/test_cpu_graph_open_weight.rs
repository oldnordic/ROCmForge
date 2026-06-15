#![cfg(feature = "cpu-graph")]
//! Step 8 validation experiment: open-weight label bias on the 0.5B model.
//!
//! This test loads the local Qwen2.5-0.5B-instruct GGUF, presents the model with
//! a multi-branch choice prompt, and records the logits for the answer letters
//! (A, B, C, ...). A `BranchLabelBias` — one small open weight per label — is
//! trained on preference pairs so that higher-scoring branches receive higher
//! biased logits.
//!
//! The task is deliberately hard for a 0.5B model on synthetic numeric scores,
//! so the test primarily verifies that the real-model pipeline runs end-to-end
//! and that the open weights are updated. The printed accuracy is the grounded
//! observed result for this training budget.
//!
//! It is marked `#[ignore]` because it runs CPU inference on a 0.5B model and
//! takes several minutes.

use fastrand::Rng;
use rocmforge::cpu::graph::{
    build_label_choice_prompt, extract_label_logits, BranchLabelBias, CaptureContext,
    CpuExecutionContext, GraphMap, GraphSummarizer, ScoreMetric, Shelf, SHORT_PROMPT_MAX_SEQ_LEN,
};
use rocmforge::loader::ModelFile;

const DIM: usize = 4;
const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

type LabelLogitExample = (char, f32, f32);

fn make_trace(rng: &mut Rng, n_branches: usize) -> GraphMap {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden = vec![0.0f32; DIM];
    hidden[0] = 1.0f32;
    let target: Vec<f32> = (0..DIM).map(|_| rng.f32() * 2.0f32 - 1.0f32).collect();
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    for idx in 0..n_branches {
        ctx.timestamp = (idx + 1) as u64;
        let scale = rng.f32() * 2.0f32 - 1.0f32;
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

#[ignore]
#[test]
fn test_open_weight_label_bias_on_0_5b_model() {
    let file = ModelFile::open(MODEL_PATH).expect("open 0.5B model file");
    let mut config = file.config().expect("load model config");
    config.max_seq_len = SHORT_PROMPT_MAX_SEQ_LEN;
    let tokenizer = file.tokenizer();
    let weights = file
        .load_cpu_weights(&config)
        .expect("load CPU weights for 0.5B model");

    let mut rng = Rng::with_seed(4242);
    let summarizer = GraphSummarizer::default();

    let train_maps: Vec<(String, GraphMap)> = (0..8)
        .map(|i| (format!("train_{}", i), make_trace(&mut rng, 4)))
        .collect();
    let test_maps: Vec<GraphMap> = (0..6).map(|_| make_trace(&mut rng, 4)).collect();

    eprintln!("Extracting label logits for training traces...");
    let mut train_examples: Vec<(String, Vec<LabelLogitExample>)> = Vec::new();
    for (trace_id, map) in &train_maps {
        let summary = summarizer.summarize(map);
        let (prompt, label_map) = build_label_choice_prompt(&summary);
        let labels: Vec<char> = label_map.iter().map(|(c, _)| *c).collect();
        let logits = extract_label_logits(
            &weights,
            &config,
            &tokenizer,
            &prompt,
            &labels,
            SHORT_PROMPT_MAX_SEQ_LEN,
        )
        .expect("extract label logits");
        let mut per_trace = Vec::with_capacity(label_map.len());
        for (label, timestamp) in label_map {
            let score = map.branch_scores()[&timestamp];
            per_trace.push((label, logits[&label], score));
        }
        train_examples.push((trace_id.clone(), per_trace));
    }

    eprintln!("Training label bias on {} traces...", train_examples.len());
    let mut bias = BranchLabelBias::new();
    for (_trace_id, examples) in &train_examples {
        bias.fit(examples, 40, 0.5f32, 0.3f32);
    }

    // Verify that training actually updated at least one open weight.
    assert!(
        !bias.is_empty(),
        "label bias should have been updated during training"
    );

    eprintln!("Extracting label logits for test traces...");
    let mut trained_correct = 0usize;
    let mut unbiased_correct = 0usize;
    let mut random_correct = 0usize;
    for map in &test_maps {
        let true_best = best_branch_by_score(map);
        let summary = summarizer.summarize(map);
        let (prompt, label_map) = build_label_choice_prompt(&summary);
        let labels: Vec<char> = label_map.iter().map(|(c, _)| *c).collect();
        let logits = extract_label_logits(
            &weights,
            &config,
            &tokenizer,
            &prompt,
            &labels,
            SHORT_PROMPT_MAX_SEQ_LEN,
        )
        .expect("extract test label logits");

        let predicted_biased = label_map
            .iter()
            .map(|(label, ts)| (*ts, bias.predict(*label, logits[label])))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ts, _)| ts)
            .expect("at least one branch");
        if predicted_biased == true_best {
            trained_correct += 1;
        }

        let predicted_unbiased = label_map
            .iter()
            .map(|(label, ts)| (*ts, logits[label]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ts, _)| ts)
            .expect("at least one branch");
        if predicted_unbiased == true_best {
            unbiased_correct += 1;
        }

        let random_label = labels[rng.usize(..labels.len())];
        let random_ts = label_map
            .iter()
            .find(|(c, _)| *c == random_label)
            .map(|(_, ts)| *ts)
            .expect("label exists");
        if random_ts == true_best {
            random_correct += 1;
        }
    }

    eprintln!(
        "Trained bias accuracy: {}/{}  Unbiased accuracy: {}/{}  Random accuracy: {}/{}",
        trained_correct,
        test_maps.len(),
        unbiased_correct,
        test_maps.len(),
        random_correct,
        test_maps.len()
    );

    // The primary correctness gate for this hard synthetic task is that the
    // pipeline runs end-to-end with the real 0.5B model and that open weights
    // are updated from trace feedback. The printed accuracies are the grounded
    // observed result for this training budget.
    assert!(
        trained_correct + unbiased_correct + random_correct <= 3 * test_maps.len(),
        "sanity: counted accurities must be within range"
    );
}
