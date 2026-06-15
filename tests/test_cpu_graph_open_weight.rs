#![cfg(feature = "cpu-graph")]
//! Step 8 validation experiment: train an open-weight choice head on the 0.5B model.
//!
//! This test loads the local Qwen2.5-0.5B-instruct GGUF, applies the chat
//! template to the standard two-branch introspection prompt, and extracts the
//! final hidden-state vector. A `BranchChoiceHead` (a small linear binary
//! classifier) is trained on those hidden states to predict which of the two
//! branches has the higher score. The two branches have semantic descriptions
//! (toward vs away from the target direction) that the 0.5B model can already
//! discriminate in Step 4, so this measures whether a tiny open-weight head can
//! learn from the base model's frozen representations.
//!
//! It is marked `#[ignore]` because it runs CPU inference on a 0.5B model
//! dozens of times and takes several minutes.

use fastrand::Rng;
use rocmforge::cpu::graph::{
    BranchChoiceExample, BranchChoiceHead, CaptureContext, CpuExecutionContext, GraphMap,
    GraphSummarizer, ScoreMetric, Shelf, SHORT_PROMPT_MAX_SEQ_LEN,
};
use rocmforge::loader::ModelFile;

const DIM: usize = 16;
const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn make_branch_pair(rng: &mut Rng) -> (GraphMap, usize) {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden: Vec<f32> = (0..DIM).map(|_| rng.f32() - 0.5f32).collect();
    let target: Vec<f32> = (0..DIM).map(|_| rng.f32() * 2.0f32 - 1.0f32).collect();
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    // Branch A: add a small vector correlated with the target direction.
    let toward: Vec<f32> = target
        .iter()
        .map(|&t| t * rng.f32() * 0.3f32 + rng.f32() * 0.05f32)
        .collect();
    ctx.timestamp = 1;
    ctx.execute_residual_add(&mut hidden, &toward);
    ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    ctx.regress_to(0);

    // Branch B: add a vector pointing away from the target direction.
    let away: Vec<f32> = target
        .iter()
        .map(|&t| -t * rng.f32() * 0.6f32 + rng.f32() * 0.05f32)
        .collect();
    ctx.timestamp = 2;
    ctx.execute_residual_add(&mut hidden, &away);
    ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    let map = GraphMap::from_context(&ctx);
    let scores = map.branch_scores();
    let better = if scores[&1] > scores[&2] { 0 } else { 1 };
    (map, better)
}

#[ignore]
#[test]
fn test_open_weight_choice_head_on_0_5b_model() {
    let file = ModelFile::open(MODEL_PATH).expect("open 0.5B model file");
    let mut config = file.config().expect("load model config");
    config.max_seq_len = SHORT_PROMPT_MAX_SEQ_LEN;
    let tokenizer = file.tokenizer();
    let template = file.chat_template(&config, false);
    let weights = file
        .load_cpu_weights(&config)
        .expect("load CPU weights for 0.5B model");

    let mut rng = Rng::with_seed(4242);
    let summarizer = GraphSummarizer::new(1.0);

    let train_pairs: Vec<(GraphMap, usize)> = (0..12).map(|_| make_branch_pair(&mut rng)).collect();
    let test_pairs: Vec<(GraphMap, usize)> = (0..8).map(|_| make_branch_pair(&mut rng)).collect();

    eprintln!("Extracting hidden states for training pairs...");
    let mut train_examples: Vec<BranchChoiceExample> = Vec::new();
    for (map, better_label) in &train_pairs {
        let summary = summarizer.summarize(map);
        let prompt = summarizer.prompt(&summary);
        let full_text = template.apply(&prompt.text);
        let hidden = rocmforge::cpu::graph::extract_hidden_state_with_special(
            &weights,
            &config,
            &tokenizer,
            &full_text,
            SHORT_PROMPT_MAX_SEQ_LEN,
            false,
        )
        .expect("extract training hidden state");
        train_examples.push(BranchChoiceExample {
            hidden,
            label: *better_label,
        });
    }

    eprintln!(
        "Training binary choice head on {} examples...",
        train_examples.len()
    );
    let mut head = BranchChoiceHead::new(config.hidden_size, 2);
    head.fit(&train_examples, 100, 0.02f32);

    eprintln!("Extracting hidden states for test pairs...");
    let mut trained_correct = 0usize;
    let mut random_correct = 0usize;
    for (map, better_label) in &test_pairs {
        let summary = summarizer.summarize(map);
        let prompt = summarizer.prompt(&summary);
        let full_text = template.apply(&prompt.text);
        let hidden = rocmforge::cpu::graph::extract_hidden_state_with_special(
            &weights,
            &config,
            &tokenizer,
            &full_text,
            SHORT_PROMPT_MAX_SEQ_LEN,
            false,
        )
        .expect("extract test hidden state");

        let predicted = head.predict(&hidden);
        if predicted == *better_label {
            trained_correct += 1;
        }

        let random_label = rng.usize(..2);
        if random_label == *better_label {
            random_correct += 1;
        }
    }

    eprintln!(
        "Trained choice head accuracy: {}/{}  Random accuracy: {}/{}",
        trained_correct,
        test_pairs.len(),
        random_correct,
        test_pairs.len()
    );

    assert!(
        trained_correct > random_correct,
        "trained choice head must beat random baseline"
    );
    assert!(
        trained_correct >= test_pairs.len() * 3 / 4,
        "trained choice head should reach at least 3/4 accuracy: {}/{}",
        trained_correct,
        test_pairs.len()
    );
}
