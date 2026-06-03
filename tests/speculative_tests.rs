#![cfg(feature = "gpu")]
#![allow(warnings)]

//! Integration correctness tests for Speculative Decoding.

mod common;

use rocmforge::gpu::{GpuDevice, SpeculativeEngine};
use serial_test::serial;
use std::path::PathBuf;

#[test]
#[serial]
fn test_speculative_engine_instantiation_and_verification() {
    require_gpu!();

    let model_path = "/home/feanor/Projects/rocmforge/llama3.2-1b-instruct-q4_0.rfm";
    let path = std::path::Path::new(model_path);
    if !path.exists() {
        eprintln!(
            "Skipping speculative test: LLaMA 1B RFM not found at {}",
            model_path
        );
        return;
    }

    let device = GpuDevice::init(0).expect("Failed to initialize GPU");

    // Co-load the same model as both Target and Draft models for deterministic verification!
    // Allocate caches for up to 256 sequence length, prompt length 32
    let mut engine = SpeculativeEngine::new(&device, model_path, model_path, 256, 32)
        .expect("Failed to construct SpeculativeEngine");

    // Verify cache non-overlapping bounds, configurations and VRAM allocations
    assert_eq!(
        engine.target_config.hidden_size,
        engine.draft_config.hidden_size
    );
    assert!(engine.target_kv.vram_bytes() > 0);
    assert!(engine.draft_kv.vram_bytes() > 0);

    // Act: Run a short draft sequence of 4 tokens starting at position 10
    // We draft starting at position 10, using token ID 104 as the input
    let draft_tokens = engine
        .draft_tokens(&device, 10, 4, 104)
        .expect("Failed to draft speculative tokens");
    assert_eq!(draft_tokens.len(), 4);

    // Verify: Run target verification pass over the 4 drafted tokens
    // last_verified_token is 104 (at pos 9), so start_pos is 10
    let (accepted_tokens, num_accepted) = engine
        .verify_tokens(&device, 10, &draft_tokens, 104)
        .expect("Failed to verify speculative tokens");

    println!("DIAGNOSTIC: Draft tokens = {:?}", draft_tokens);
    println!("DIAGNOSTIC: Accepted tokens = {:?}", accepted_tokens);
    println!("DIAGNOSTIC: Num accepted = {}", num_accepted);

    // Since Target and Draft models are identical, all 4 tokens must be accepted!
    assert_eq!(num_accepted, 4);
    assert_eq!(accepted_tokens.len(), 5); // 4 accepted + 1 next target token
    assert_eq!(&accepted_tokens[0..4], &draft_tokens[..]);
}

#[test]
#[serial]
fn test_speculative_orchestrator_generate() {
    require_gpu!();

    let model_path = "/home/feanor/Projects/rocmforge/llama3.2-1b-instruct-q4_0.rfm";
    let path = std::path::Path::new(model_path);
    if !path.exists() {
        eprintln!(
            "Skipping speculative generation test: LLaMA 1B RFM not found at {}",
            model_path
        );
        return;
    }

    let device = GpuDevice::init(0).expect("Failed to initialize GPU");

    // Co-load the same model as both Target and Draft models for deterministic verification!
    let mut engine = SpeculativeEngine::new(&device, model_path, model_path, 256, 32)
        .expect("Failed to construct SpeculativeEngine");

    let file = rocmforge::loader::ModelFile::open(model_path).expect("Failed to open model file");
    let tokenizer = file.tokenizer();

    let orchestrator = rocmforge::gpu::SpeculativeOrchestrator::new(4)
        .expect("Failed to create SpeculativeOrchestrator");

    // Run generation on a simple prompt
    let prompt = "Hello";
    let prompt_tokens = tokenizer.encode(prompt, false);

    let (text, tokens_len) = orchestrator
        .generate(&device, &mut engine, &tokenizer, &prompt_tokens, 10)
        .expect("Failed to generate tokens speculatively");

    println!(
        "SPECULATIVE GENERATION RESULT: '{}' ({} tokens)",
        text, tokens_len
    );
    assert!(tokens_len > 0);
    assert!(!text.is_empty());
}
