//! End-to-end CPU loader and inference integration tests for .rfm model format.
//!
//! Validates that converting a real GGUF model to RFM and executing CPU forward passes
//! produces mathematically identical logits to loading and running the GGUF model.

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    weights::CpuModelWeights,
};
use rocmforge::loader::{GgufFile, RfmFile};
use rocmforge::tokenizer::BpeTokenizer;
use std::path::Path;
use std::process::Command;

const CANDIDATE_PATHS: &[&str] = &[
    "/home/feanor/Projects/llama.cpp/models/llama3.2-1b-instruct-q4_0.gguf",
    "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf",
];

fn find_available_model() -> Option<&'static str> {
    CANDIDATE_PATHS
        .iter()
        .copied()
        .find(|&path| Path::new(path).exists())
}

#[test]
fn test_gguf_to_rfm_cpu_inference_equivalence() {
    let model_path = match find_available_model() {
        Some(path) => path,
        None => {
            eprintln!("Skipping end-to-end equivalence test: No valid GGUF candidate found.");
            return;
        }
    };

    println!("[Test] Selected GGUF candidate for test: {}", model_path);

    // Step 1: Create a unique temporary output path for the .rfm file
    let temp_dir = std::env::temp_dir();
    let rfm_filename = format!(
        "test_model_{}.rfm",
        Path::new(model_path).file_stem().unwrap().to_str().unwrap()
    );
    let rfm_path = temp_dir.join(rfm_filename);

    // Clean up any stale file from prior runs
    if rfm_path.exists() {
        let _ = std::fs::remove_file(&rfm_path);
    }

    println!("[Test] Converting GGUF model to RFM format...");

    // Step 2: Execute rocmforge-convert CLI tool directly (bypassing Cargo build locks) with 1 layer limit
    let status = Command::new("target/debug/rocmforge-convert")
        .args([model_path, rfm_path.to_str().expect("invariant: rfm_path is valid UTF-8"), "--max-layers", "1"])
        .status()
        .expect("Failed to invoke rocmforge-convert");

    assert!(status.success(), "rocmforge-convert execution failed");
    assert!(rfm_path.exists(), "Converted RFM file was not created");

    println!("[Test] Loading GGUF and RFM models...");

    // Step 3: Load the GGUF model
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF model");
    let mut gguf_config = ModelConfig::from_gguf(&gguf).expect("Failed to parse GGUF config");

    // Step 4: Load the converted RFM model
    let rfm = RfmFile::open(&rfm_path).expect("Failed to open RFM model");
    let rfm_config = ModelConfig::from_rfm(&rfm.metadata).expect("Failed to parse RFM config");

    // Align GGUF config's number of layers to match the truncated RFM model to speed up test execution
    gguf_config.num_layers = rfm_config.num_layers;

    // Verify config parameters match exactly
    assert_eq!(gguf_config.num_layers, rfm_config.num_layers);
    assert_eq!(gguf_config.hidden_size, rfm_config.hidden_size);
    assert_eq!(gguf_config.num_heads, rfm_config.num_heads);
    assert_eq!(gguf_config.num_kv_heads, rfm_config.num_kv_heads);
    assert_eq!(gguf_config.head_dim, rfm_config.head_dim);
    assert_eq!(gguf_config.intermediate_size, rfm_config.intermediate_size);
    assert_eq!(gguf_config.vocab_size, rfm_config.vocab_size);
    assert_eq!(gguf_config.max_seq_len, rfm_config.max_seq_len);
    assert_eq!(gguf_config.rms_norm_eps, rfm_config.rms_norm_eps);
    assert_eq!(gguf_config.rope_theta, rfm_config.rope_theta);
    assert_eq!(gguf_config.rope_neox, rfm_config.rope_neox);
    assert_eq!(
        gguf_config.use_attention_bias,
        rfm_config.use_attention_bias
    );
    assert_eq!(gguf_config.architecture, rfm_config.architecture);

    println!("[Test] Configuration match verified.");

    // Step 5: Load weights into CPU memory
    let gguf_weights =
        CpuModelWeights::load(&gguf, &gguf_config).expect("Failed to load GGUF weights");
    let rfm_weights =
        CpuModelWeights::load_rfm(&rfm, &rfm_config).expect("Failed to load RFM weights");

    // Step 6: Initialize Tokenizers
    let gguf_tok = BpeTokenizer::from_gguf(gguf.tokenizer_data());
    let rfm_tok = BpeTokenizer::from_rfm(&rfm.metadata);

    // Verify tokenizers encode a prompt identically
    let prompt = "Paris is";
    let gguf_tokens = gguf_tok.encode(prompt, false);
    let rfm_tokens = rfm_tok.encode(prompt, false);
    assert_eq!(gguf_tokens, rfm_tokens, "Tokenizer outputs diverged!");

    println!("[Test] Running parallel CPU inference passes to compare logits...");

    // Step 7: Allocate separate CPU inference states
    let mut gguf_kv = CpuKvCache::new(&gguf_config, gguf_tokens.len());
    let mut gguf_scratch = CpuForwardScratch::new(&gguf_config);
    let mut gguf_hidden = vec![0.0f32; gguf_config.hidden_size];

    let mut rfm_kv = CpuKvCache::new(&rfm_config, rfm_tokens.len());
    let mut rfm_scratch = CpuForwardScratch::new(&rfm_config);
    let mut rfm_hidden = vec![0.0f32; rfm_config.hidden_size];

    // Token-by-token forward pass comparison
    for (pos, &token_id) in gguf_tokens.iter().enumerate() {
        // Run GGUF step
        cpu_embed_token(token_id, &gguf_weights, &mut gguf_hidden, &gguf_config);
        cpu_full_forward(
            &mut gguf_hidden,
            &gguf_weights,
            &mut gguf_kv,
            &mut gguf_scratch,
            pos,
            &gguf_config,
        )
        .expect("GGUF CPU forward step failed");

        // Run RFM step
        cpu_embed_token(token_id, &rfm_weights, &mut rfm_hidden, &rfm_config);
        cpu_full_forward(
            &mut rfm_hidden,
            &rfm_weights,
            &mut rfm_kv,
            &mut rfm_scratch,
            pos,
            &rfm_config,
        )
        .expect("RFM CPU forward step failed");

        // Compare logits
        assert_eq!(
            gguf_scratch.logits.len(),
            rfm_scratch.logits.len(),
            "Logits length mismatch at pos {}",
            pos
        );

        let mut max_diff = 0.0f32;
        for (i, (gguf_val, rfm_val)) in gguf_scratch
            .logits
            .iter()
            .zip(rfm_scratch.logits.iter())
            .enumerate()
        {
            let diff = (gguf_val - rfm_val).abs();
            max_diff = max_diff.max(diff);
            if diff > 1e-5 {
                panic!(
                    "Logits mismatch at pos {}, index {}: GGUF = {}, RFM = {}, diff = {}",
                    pos, i, gguf_val, rfm_val, diff
                );
            }
        }

        println!(
            "  Token position {}/{} matches perfectly (max diff: {:.2e})",
            pos + 1,
            gguf_tokens.len(),
            max_diff
        );
    }

    println!("[Test] End-to-end CPU inference equivalence test PASSED successfully!");

    // Step 8: Clean up temporary file
    let _ = std::fs::remove_file(&rfm_path);
}
