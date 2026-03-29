// examples/test_q5_0_model.rs

//! Tests Q5_0 quantization support with real GGUF models.
//!
//! This verifies that models with Q5_0 quantized weights can be loaded
//! and used for inference.

use std::env;
use std::process;
use std::time::Instant;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    sampler::cpu_sample_greedy,
    weights::CpuModelWeights,
};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        eprintln!();
        eprintln!("Tests Q5_0 quantization support with a real model.");
        process::exit(1);
    }

    let model_path = &args[1];

    println!("Testing Q5_0 support with model: {}", model_path);
    println!();

    // 1. Open GGUF file
    print!("Opening GGUF file... ");
    let file = match GgufFile::open(model_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("FAILED: {}", e);
            process::exit(1);
        }
    };
    println!("OK");

    // 2. Load model configuration
    print!("Loading model config... ");
    let config = match ModelConfig::from_gguf(&file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAILED: {}", e);
            process::exit(1);
        }
    };
    println!(
        "OK ({} layers, {} vocab, {} hidden)",
        config.num_layers, config.vocab_size, config.hidden_size
    );

    // 3. Load tokenizer
    print!("Loading tokenizer... ");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    println!("OK");

    // 4. Load weights (this tests Q5_0 dequantization)
    print!("Loading weights... ");
    let t_load = Instant::now();
    let weights = match CpuModelWeights::load(&file, &config) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("FAILED: {}", e);
            process::exit(1);
        }
    };
    println!("OK in {:.1}s", t_load.elapsed().as_secs_f64());

    // 5. Run a simple inference test
    println!();
    println!("Running inference test...");

    let test_prompt = "Hello";
    let prompt_tokens = tok.encode(test_prompt, false);
    if prompt_tokens.is_empty() {
        eprintln!("ERROR: Prompt tokenized to zero tokens");
        process::exit(1);
    }
    println!(
        "  Prompt: \"{}\" -> {} tokens",
        test_prompt,
        prompt_tokens.len()
    );

    // Allocate KV cache and scratch
    let max_seq = (prompt_tokens.len() + 5).min(config.max_seq_len);
    let mut kv = CpuKvCache::new(&config, max_seq);
    let mut scratch = CpuForwardScratch::new(&config);
    let mut hidden = vec![0.0f32; config.hidden_size];

    // Embed first token
    cpu_embed_token(prompt_tokens[0], &weights, &mut hidden, &config);

    // Run prefill for remaining prompt tokens
    if prompt_tokens.len() > 1 {
        for i in 1..prompt_tokens.len() {
            cpu_full_forward(&mut hidden, &weights, &mut kv, &mut scratch, i, &config)
                .expect("Prefill forward failed");
            // Note: In real prefill we'd batch these, but this is a simple test
        }
    }

    // Generate 3 tokens
    let mut generated = Vec::new();
    for _ in 0..3 {
        cpu_full_forward(
            &mut hidden,
            &weights,
            &mut kv,
            &mut scratch,
            prompt_tokens.len() + generated.len(),
            &config,
        )
        .expect("Decode forward failed");

        let next_token = cpu_sample_greedy(&scratch.logits);
        generated.push(next_token);

        // Stop at EOS
        if Some(next_token) == tok.eos_id() {
            break;
        }

        // Embed the next token for the next iteration
        cpu_embed_token(next_token, &weights, &mut hidden, &config);
    }

    // Decode output
    let output_text = tok.decode(&generated, false);
    println!(
        "  Generated: \"{}\" ({} tokens)",
        output_text,
        generated.len()
    );

    println!();
    println!("Q5_0 support: OK (all tests passed)");
}
