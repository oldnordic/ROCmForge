#![cfg(feature = "cpu-graph")]
//! Step 4 validation experiment: introspection prompt interface.
//!
//! Builds a `GraphSummarizer` that compresses a `GraphMap` into a short text
//! prompt plus a vector encoding, asks the local 0.5B Qwen2.5 model to pick the
//! better of two branches, and parses the structured response into an
//! `IntrospectionReport`. The test asserts that the model chooses the
//! higher-scoring branch more often than random on a small held-out set.

use fastrand::Rng;
use rocmforge::config::{detect_chat_template, ChatTemplate, ModelConfig};
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    graph::{
        introspection::{GraphSummarizer, IntrospectionPrompt},
        CaptureContext, CpuExecutionContext, GraphMap, ScoreMetric, Shelf,
    },
    sampler::cpu_sample_greedy,
    weights::CpuModelWeights,
};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

struct ModelSession {
    config: ModelConfig,
    weights: CpuModelWeights,
    tokenizer: BpeTokenizer,
    template: ChatTemplate,
}

fn load_model() -> ModelSession {
    let file = GgufFile::open(MODEL_PATH).expect("open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("parse model config");
    let weights = CpuModelWeights::load(&file, &config).expect("load CPU weights");
    let tokenizer = BpeTokenizer::from_gguf(file.tokenizer_data());
    let template =
        detect_chat_template(&config.architecture, file.tokenizer_data().model.as_deref());
    ModelSession {
        config,
        weights,
        tokenizer,
        template,
    }
}

fn generate_response(
    prompt: &IntrospectionPrompt,
    session: &ModelSession,
    max_tokens: usize,
) -> String {
    let full_text = session.template.apply(&prompt.text);
    let prompt_tokens = session.tokenizer.encode(&full_text, false);
    let max_seq = prompt_tokens.len() + max_tokens;
    let mut kv = CpuKvCache::new(&session.config, max_seq);
    let mut scratch = CpuForwardScratch::new(&session.config);
    let mut hidden = vec![0.0f32; session.config.hidden_size];

    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        cpu_embed_token(token_id, &session.weights, &mut hidden, &session.config);
        cpu_full_forward(
            &mut hidden,
            &session.weights,
            &mut kv,
            &mut scratch,
            pos,
            &session.config,
        )
        .expect("prompt forward failed");
    }

    let mut next = cpu_sample_greedy(&scratch.logits);
    let mut generated = Vec::new();
    let mut pos = prompt_tokens.len();
    while !session.tokenizer.is_eog(next) && generated.len() < max_tokens {
        generated.push(next);
        cpu_embed_token(next, &session.weights, &mut hidden, &session.config);
        cpu_full_forward(
            &mut hidden,
            &session.weights,
            &mut kv,
            &mut scratch,
            pos,
            &session.config,
        )
        .expect("decode forward failed");
        pos += 1;
        next = cpu_sample_greedy(&scratch.logits);
    }

    session.tokenizer.decode(&generated, true)
}

fn make_branch_pair(rng: &mut Rng) -> (GraphMap, u64) {
    let dim = 16usize;
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden: Vec<f32> = (0..dim).map(|_| rng.f32() - 0.5).collect();
    let target: Vec<f32> = (0..dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    // Branch A: add a small vector correlated with the target direction.
    let toward: Vec<f32> = target
        .iter()
        .map(|&t| t * rng.f32() * 0.3 + rng.f32() * 0.05)
        .collect();
    ctx.timestamp = 1;
    ctx.execute_residual_add(&mut hidden, &toward);
    ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    ctx.regress_to(0);

    // Branch B: add a vector pointing away from the target direction.
    let away: Vec<f32> = target
        .iter()
        .map(|&t| -t * rng.f32() * 0.6 + rng.f32() * 0.05)
        .collect();
    ctx.timestamp = 2;
    ctx.execute_residual_add(&mut hidden, &away);
    ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    let map = GraphMap::from_context(&ctx);
    let scores = map.branch_scores();
    let better = if scores[&1] > scores[&2] { 1 } else { 2 };
    (map, better)
}

#[test]
#[serial]
fn test_introspection_prompt_parses_report() {
    let mut ctx = CaptureContext::new(0, 0);
    let mut hidden = vec![1.0f32, 0.0, 0.0];
    let target = vec![1.0f32, 1.0, 1.0];
    let ptr = hidden.as_ptr() as usize;
    ctx.arena.bind_f32(Shelf::Persistent, ptr, &hidden);

    ctx.timestamp = 1;
    ctx.execute_residual_add(&mut hidden, &[0.9f32; 3]);
    ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    ctx.regress_to(0);

    ctx.timestamp = 2;
    ctx.execute_residual_add(&mut hidden, &[-2.0f32; 3]);
    ctx.score_against(&hidden, Some(&target), ScoreMetric::CosineSimilarity);

    let map = GraphMap::from_context(&ctx);
    let summarizer = GraphSummarizer::new(1.0);
    let summary = summarizer.summarize(&map);
    let prompt = summarizer.prompt(&summary);

    assert!(!prompt.text.is_empty());
    assert_eq!(prompt.vector.len(), 6);
    assert_eq!(prompt.label_map.len(), 2);

    let fake_response = "CHOICE: Branch A\nREASON: It has the higher score.";
    let report = prompt
        .parse_response(fake_response)
        .expect("parse structured response");
    assert_eq!(report.chosen_branch, 1);
    assert!(!report.explanation.is_empty());
}

#[test]
#[serial]
#[ignore = "slow: loads 0.5B Qwen2.5 model and runs CPU inference (~10 min)"]
fn test_introspection_model_prefers_higher_scored_branch() {
    if skip_if_model_missing() {
        eprintln!(
            "Skipping introspection model test: model not found at {}",
            MODEL_PATH
        );
        return;
    }

    let session = load_model();
    let summarizer = GraphSummarizer::new(1.0);
    let mut rng = Rng::with_seed(2025);

    const PAIRS: usize = 4;
    let mut correct = 0usize;
    let mut parsed = 0usize;

    for i in 0..PAIRS {
        let (map, better_ts) = make_branch_pair(&mut rng);
        let summary = summarizer.summarize(&map);
        let prompt = summarizer.prompt(&summary);
        let response = generate_response(&prompt, &session, 24);

        eprintln!("Pair {} prompt:\n{}\n", i, prompt.text);
        eprintln!("Pair {} response:\n{}\n", i, response);

        if let Some(report) = prompt.parse_response(&response) {
            parsed += 1;
            if report.chosen_branch == better_ts {
                correct += 1;
            }
        }
    }

    eprintln!(
        "Introspection accuracy: {}/{} parsed, {}/{} correct",
        parsed, PAIRS, correct, parsed
    );

    assert!(
        parsed == PAIRS,
        "model must produce a parseable response for every pair: {}/{} parsed",
        parsed,
        PAIRS
    );
    assert!(
        correct > PAIRS / 2,
        "model must pick the higher-scoring branch more often than random: {}/{} correct",
        correct,
        PAIRS
    );
}
