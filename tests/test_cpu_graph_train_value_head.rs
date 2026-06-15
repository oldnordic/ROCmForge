#![cfg(feature = "cpu-graph")]
//! Train a process reward model on trivia prefixes.
//!
//! This test loads the local Qwen2.5-0.5B-instruct GGUF, generates several
//! temperature-sampled completions for each prompt in `eval/rerank_trivia.jsonl`,
//! labels each completion by exact-match prefix against the expected answer, and
//! creates one training example for every prefix hidden state. The resulting
//! `BranchValueHead` is a tiny process reward model: it scores a partial
//! sequence by how likely it is to lead to a correct final answer.
//!
//! The trained head is saved to `target/trivia_prm_head.bin` so the reranker
//! eval can load it via `ROCMFORGE_TEST_VALUE_HEAD_PATH`.
//!
//! Marked `#[ignore]` because it loads the 0.5B model and generates many
//! completions.

use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward, cpu_prefill},
    graph::value_head::{BranchValueExample, BranchValueHead},
    sampler::cpu_sample_top_p,
    weights::CpuModelWeights,
};
use rocmforge::loader::ModelFile;
use rocmforge::tokenizer::BpeTokenizer;
use serde::Deserialize;

const DEFAULT_MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const DATASET_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/eval/rerank_trivia.jsonl");

fn model_path() -> String {
    std::env::var("ROCMFORGE_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
}

const SAMPLES_PER_PROMPT: usize = 4;
const MAX_TOKENS_PER_COMPLETION: usize = 24;
const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.9;

#[derive(Debug, Deserialize)]
struct TriviaSample {
    prompt: String,
    expected_continuation: String,
}

fn model_exists() -> bool {
    std::path::Path::new(&model_path()).exists()
}

fn normalize(s: &str) -> String {
    s.trim().to_lowercase()
}

fn generate_completion(
    prompt_tokens: &[u32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    tok: &BpeTokenizer,
    config: &rocmforge::config::ModelConfig,
    seed: u64,
) -> (String, Vec<Vec<f32>>) {
    let mut hidden = vec![0.0f32; config.hidden_size];
    let mut pos = prompt_tokens.len();
    let mut n_generated = 0usize;
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut prefix_hiddens: Vec<Vec<f32>> = Vec::new();

    let mut next_token = cpu_sample_top_p(
        &scratch.logits[..config.vocab_size],
        TEMPERATURE,
        TOP_P,
        seed,
    );

    loop {
        if tok.is_eog(next_token) || n_generated >= MAX_TOKENS_PER_COMPLETION {
            break;
        }
        generated_tokens.push(next_token);
        n_generated += 1;

        cpu_embed_token(next_token, weights, &mut hidden, config);
        cpu_full_forward(&mut hidden, weights, kv, scratch, pos, config).ok();
        pos += 1;

        // Capture the hidden state after every generated token. The reranker can
        // score any partial prefix online, so the value head should learn a
        // process reward: is this prefix on track toward a correct answer?
        prefix_hiddens.push(scratch.normed.to_vec());

        next_token = cpu_sample_top_p(
            &scratch.logits[..config.vocab_size],
            TEMPERATURE,
            TOP_P,
            seed.wrapping_add(n_generated as u64),
        );
    }

    (tok.decode(&generated_tokens, true), prefix_hiddens)
}

#[ignore]
#[test]
fn train_process_reward_head_on_trivia_sample() -> Result<(), Box<dyn std::error::Error>> {
    if !model_exists() {
        eprintln!(
            "Skipping value-head training: model not found at {}",
            model_path()
        );
        return Ok(());
    }

    let model_path = model_path();
    let file = ModelFile::open(&model_path)?;
    let config = file.config()?;
    let tok = file.tokenizer();
    let weights = file.load_cpu_weights(&config)?;

    let dataset = std::fs::read_to_string(DATASET_PATH)?;
    let samples: Vec<TriviaSample> = dataset
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON in dataset"))
        .collect();

    let mut examples: Vec<BranchValueExample> = Vec::new();
    let mut rng_seed = 0xabcdef_u64;

    for (idx, sample) in samples.iter().enumerate() {
        let prompt = &sample.prompt;
        let prompt_tokens = tok.encode(prompt, false);
        if prompt_tokens.is_empty() {
            continue;
        }

        let max_seq = prompt_tokens.len() + MAX_TOKENS_PER_COMPLETION;
        let mut base_kv = CpuKvCache::new(&config, max_seq);
        let mut scratch = CpuForwardScratch::new(&config);
        cpu_prefill(
            &mut [],
            &weights,
            &mut base_kv,
            &mut scratch,
            &prompt_tokens,
            &config,
        )?;

        for sample_idx in 0..SAMPLES_PER_PROMPT {
            let mut kv = base_kv.clone();
            let mut sample_scratch = CpuForwardScratch::new(&config);
            sample_scratch.logits.copy_from_slice(&scratch.logits);
            sample_scratch.normed.copy_from_slice(&scratch.normed);

            let seed = rng_seed;
            rng_seed = rng_seed.wrapping_add(1);

            let (text, prefix_hiddens) = generate_completion(
                &prompt_tokens,
                &weights,
                &mut kv,
                &mut sample_scratch,
                &tok,
                &config,
                seed,
            );

            if prefix_hiddens.is_empty() {
                continue;
            }

            let correct = normalize(&text).starts_with(&normalize(&sample.expected_continuation));
            let score = if correct { 1.0f32 } else { 0.0f32 };

            for (token_pos, hidden) in prefix_hiddens.into_iter().enumerate() {
                examples.push(BranchValueExample {
                    trace_id: format!("trivia-{}-{}", idx, sample_idx),
                    timestamp: token_pos as u64,
                    hidden,
                    score,
                });
            }

            eprintln!(
                "[prompt {} sample {}] correct={} expected={:?} generated={:.60}",
                idx,
                sample_idx,
                correct,
                sample.expected_continuation,
                text.replace('\n', " ")
            );
        }
    }

    if examples.is_empty() {
        eprintln!("No training examples collected; aborting.");
        return Ok(());
    }

    let correct_count = examples.iter().filter(|e| e.score > 0.5).count();
    eprintln!(
        "Collected {} examples, {} correct ({:.1}%)",
        examples.len(),
        correct_count,
        correct_count as f32 / examples.len() as f32 * 100.0
    );

    let mut head = BranchValueHead::new(config.hidden_size);
    head.fit_mse(&examples, 40, 0.005);

    let head_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("trivia_prm_head.bin");
    head.save(&head_path)?;
    eprintln!(
        "Saved trained process-reward head to {}",
        head_path.display()
    );

    // Quick sanity check: score distribution on training examples.
    let (mut min_score, mut max_score) = (f32::INFINITY, f32::NEG_INFINITY);
    for ex in &examples {
        let pred = head.predict(&ex.hidden);
        min_score = min_score.min(pred);
        max_score = max_score.max(pred);
    }
    eprintln!(
        "Predicted score range: [{:.4}, {:.4}]",
        min_score, max_score
    );

    Ok(())
}
