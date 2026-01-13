//! Test for inference loop spawning race condition (P0 Bug)
//!
//! This test verifies that the inference loop is properly spawned and ready
//! to process requests before submit_request() is called.
//!
//! Root cause: CLI uses `engine.run_inference_loop().await` without external
//! `tokio::spawn()`, which can create a race condition where requests are
//! submitted before the inference loop task is actually running.
//!
//! Expected behavior: HTTP server pattern with external `tokio::spawn()`
//! should be used to ensure the inference loop is scheduled immediately.

use rocmforge::engine::{EngineConfig, InferenceEngine};
use std::sync::Arc;
use std::time::Duration;

/// Test the BROKEN CLI pattern (no external tokio::spawn)
///
/// This demonstrates the race condition: the inference loop task might not
/// be scheduled yet when submit_request() is called.
#[tokio::test]
async fn test_cli_pattern_broken_no_external_spawn() {
    let gguf_path = std::path::Path::new("tests/data/tiny_model.gguf");
    if !gguf_path.exists() {
        eprintln!("Skipping test: tiny_model.gguf missing");
        return;
    }

    let mut engine = match InferenceEngine::new(EngineConfig::default()) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Skipping: failed to init inference engine: {err}");
            return;
        }
    };

    if let Err(err) = engine.load_gguf_model(gguf_path).await {
        eprintln!("Skipping: failed to load GGUF: {err}");
        return;
    }

    let engine = Arc::new(engine);
    engine.start().await.expect("start failed");

    // BROKEN CLI PATTERN: No external tokio::spawn()
    // This matches src/bin/rocmforge_cli.rs:540
    engine.run_inference_loop().await;

    // Immediately submit a request - might race with inference loop startup
    let prompt_tokens = vec![1u32, 2u32, 3u32];
    let request_id = match engine
        .submit_request(prompt_tokens.clone(), 2, 1.0, 40, 0.9)
        .await
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("ERROR: submit_request failed immediately after engine start: {err}");
            panic!("submit_request should not fail immediately after engine start");
        }
    };

    // Wait a bit and check if request was processed
    tokio::time::sleep(Duration::from_millis(100)).await;

    let status = engine.get_request_status(request_id).await.unwrap();
    // With the broken pattern, this might fail because inference loop isn't running
    if status.is_none() {
        panic!("Race condition detected: request disappeared (inference loop not running)");
    }

    // Check if request is being processed
    // If inference loop isn't running, request will stay in pending state forever
    let mut processed = false;
    for _ in 0..10 {
        let status = engine.get_request_status(request_id).await.unwrap();
        if let Some(s) = status {
            if !s.generated_tokens.is_empty() {
                processed = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    if !processed {
        panic!("Race condition: inference loop did not process the request (pattern is broken)");
    }
}

/// Test the STABLE HTTP server pattern (with external tokio::spawn)
///
/// This should always work because external tokio::spawn() ensures the
/// inference loop task is scheduled immediately.
#[tokio::test]
async fn test_http_server_pattern_stable_with_external_spawn() {
    let gguf_path = std::path::Path::new("tests/data/tiny_model.gguf");
    if !gguf_path.exists() {
        eprintln!("Skipping test: tiny_model.gguf missing");
        return;
    }

    let mut engine = match InferenceEngine::new(EngineConfig::default()) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Skipping: failed to init inference engine: {err}");
            return;
        }
    };

    if let Err(err) = engine.load_gguf_model(gguf_path).await {
        eprintln!("Skipping: failed to load GGUF: {err}");
        return;
    }

    let engine = Arc::new(engine);
    engine.start().await.expect("start failed");

    // STABLE HTTP SERVER PATTERN: External tokio::spawn()
    // This matches src/http/server.rs:551-557
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        let _ = engine_clone.run_inference_loop().await;
    });

    // Immediately submit a request - inference loop should be ready
    let prompt_tokens = vec![1u32, 2u32, 3u32];
    let request_id = match engine
        .submit_request(prompt_tokens.clone(), 2, 1.0, 40, 0.9)
        .await
    {
        Ok(id) => id,
        Err(err) => {
            panic!("submit_request failed: {err}");
        }
    };

    // Wait a bit and check if request was processed
    tokio::time::sleep(Duration::from_millis(100)).await;

    let status = engine.get_request_status(request_id).await.unwrap();
    assert!(status.is_some(), "Request should exist");

    // Check if request is being processed
    let mut processed = false;
    for _ in 0..10 {
        let status = engine.get_request_status(request_id).await.unwrap();
        if let Some(s) = status {
            if !s.generated_tokens.is_empty() {
                processed = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    assert!(processed, "Inference loop should process the request");
}

/// Comparison test: run both patterns and verify HTTP server pattern works
#[tokio::test]
async fn test_compare_spawn_patterns() {
    let gguf_path = std::path::Path::new("tests/data/tiny_model.gguf");
    if !gguf_path.exists() {
        eprintln!("Skipping test: tiny_model.gguf missing");
        return;
    }

    // Test HTTP server pattern (should always work)
    {
        let mut engine = InferenceEngine::new(EngineConfig::default()).unwrap();
        engine.load_gguf_model(gguf_path).await.unwrap();
        let engine = Arc::new(engine);
        engine.start().await.unwrap();

        let engine_clone = engine.clone();
        tokio::spawn(async move {
            let _ = engine_clone.run_inference_loop().await;
        });

        let prompt_tokens = vec![1u32, 2u32, 3u32];
        let request_id = engine
            .submit_request(prompt_tokens, 2, 1.0, 40, 0.9)
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        let status = engine.get_request_status(request_id).await.unwrap();
        assert!(status.is_some(), "HTTP pattern: Request should exist");
    }

    // Test CLI pattern (may fail due to race condition)
    {
        let mut engine = InferenceEngine::new(EngineConfig::default()).unwrap();
        engine.load_gguf_model(gguf_path).await.unwrap();
        let engine = Arc::new(engine);
        engine.start().await.unwrap();

        // CLI pattern - no external spawn
        engine.run_inference_loop().await;

        let prompt_tokens = vec![1u32, 2u32, 3u32];
        let request_id = engine
            .submit_request(prompt_tokens, 2, 1.0, 40, 0.9)
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        let status = engine.get_request_status(request_id).await.unwrap();
        // This assertion might fail due to race condition
        assert!(
            status.is_some(),
            "CLI pattern: Request should exist (may fail)"
        );
    }
}
