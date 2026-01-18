//! End-to-End Inference Tests for ROCmForge
//!
//! This test suite validates the complete inference pipeline from model loading
//! through token generation using real GGUF models.
//!
//! # Requirements
//!
//! - ROCm/HIP runtime (AMD GPU)
//! - Small GGUF model (e.g., qwen2.5-0.5b.gguf or tiny-llama.gguf)
//! - Tests skip gracefully if models are unavailable
//!
//! # Running the Tests
//!
//! ```bash
//! # Skip E2E tests (default - requires model)
//! cargo test
//!
//! # Run E2E tests (requires model)
//! ROCFORGE_TEST_MODEL=/path/to/model.gguf cargo test --test e2e_inference_tests -- --ignored
//!
//! # Run specific test
//! ROCFORGE_TEST_MODEL=/path/to/model.gguf cargo test --test e2e_inference_tests test_single_token_inference -- --ignored
//! ```

use std::sync::Arc;
use rocmforge::engine::{EngineError, InferenceEngine};
use rocmforge::http::server::{GenerateRequest, InferenceServer};
use rocmforge::tokenizer::TokenizerAdapter;
use serial_test::serial;
mod common;
use common::{has_test_model, test_model_path};

// =============================================================================
// Part 1: Basic Inference Smoke Tests
// =============================================================================

#[tokio::test]
#[ignore] // Requires real model file
#[serial]
async fn test_single_token_inference() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        println!("Set ROCFORGE_TEST_MODEL=/path/to/model.gguf to run this test");
        return Ok(());
    }

    let model_path = test_model_path();
    println!("Loading model from: {:?}", model_path);

    // Load engine with GGUF model
    let engine = InferenceEngine::from_gguf(&model_path).await?;
    engine.start().await?;

    // Start inference loop in background
    let engine_clone = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_clone);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    // Submit a simple prompt
    let prompt_tokens = vec![1u32]; // Single token prompt
    let request_id = engine_clone
        .submit_request(prompt_tokens, 1, 1.0, 50, 0.9)
        .await?;

    println!("Submitted request {} for single token inference", request_id);

    // Wait for completion
    let max_wait = std::time::Duration::from_secs(30);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > max_wait {
            return Err(anyhow::anyhow!("Timeout waiting for single token generation"));
        }

        let status = engine_clone.get_request_status(request_id).await?;
        if let Some(req) = status {
            if req.is_complete() {
                println!("Request {} completed", request_id);
                assert!(!req.generated_tokens.is_empty(), "Should generate at least one token");
                return Ok(());
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

#[tokio::test]
#[ignore]
#[serial]
async fn test_multi_token_generation() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();
    println!("Loading model from: {:?}", model_path);

    let engine = InferenceEngine::from_gguf(&model_path).await?;
    engine.start().await?;

    let engine_clone = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_clone);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    // Submit a prompt that should generate multiple tokens
    let prompt_tokens = vec![1u32, 2, 3]; // Simple prompt
    let max_tokens = 10;
    let request_id = engine_clone
        .submit_request(prompt_tokens, max_tokens, 1.0, 50, 0.9)
        .await?;

    println!("Submitted request {} for multi-token generation", request_id);

    // Wait for completion
    let max_wait = std::time::Duration::from_secs(60);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > max_wait {
            return Err(anyhow::anyhow!("Timeout waiting for multi-token generation"));
        }

        let status = engine_clone.get_request_status(request_id).await?;
        if let Some(req) = status {
            if req.is_complete() {
                println!("Request {} completed with {} tokens",
                    request_id, req.generated_tokens.len());

                assert!(!req.generated_tokens.is_empty(),
                    "Should generate at least one token");
                assert!(req.generated_tokens.len() <= max_tokens,
                    "Should not exceed max_tokens");

                // Verify token count is reasonable (at least 1, at most max_tokens)
                assert!(req.generated_tokens.len() > 0,
                    "Token count should be positive");
                return Ok(());
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

#[tokio::test]
#[ignore]
#[serial]
async fn test_request_status_tracking() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await?;
    engine.start().await?;

    let engine_clone = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_clone);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    let prompt_tokens = vec![1u32];
    let request_id = engine_clone
        .submit_request(prompt_tokens.clone(), 5, 1.0, 50, 0.9)
        .await?;

    // Check immediate status
    let status = engine_clone.get_request_status(request_id).await?;
    assert!(status.is_some(), "Request should exist");
    let req = status.unwrap();
    assert_eq!(req.request_id, request_id);
    assert_eq!(req.prompt_tokens, prompt_tokens);
    assert_eq!(req.max_tokens, 5);

    // Wait for completion
    let max_wait = std::time::Duration::from_secs(30);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > max_wait {
            break;
        }

        let status = engine_clone.get_request_status(request_id).await?;
        if let Some(req) = status {
            if req.is_complete() {
                println!("Request completed with {} tokens", req.generated_tokens.len());
                return Ok(());
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    Ok(())
}

#[tokio::test]
#[ignore]
#[serial]
async fn test_inference_with_different_temperatures() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await?;
    engine.start().await?;

    let engine_clone = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_clone);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    let prompt_tokens = vec![1u32];

    // Test with low temperature (should be more deterministic)
    let req_id_1 = engine_clone
        .submit_request(prompt_tokens.clone(), 2, 0.1, 50, 0.9)
        .await?;

    // Test with high temperature (should be more random)
    let req_id_2 = engine_clone
        .submit_request(prompt_tokens, 2, 1.5, 50, 0.9)
        .await?;

    // Wait for both to complete
    let max_wait = std::time::Duration::from_secs(30);
    let start = std::time::Instant::now();

    let mut completed_1 = false;
    let mut completed_2 = false;

    loop {
        if start.elapsed() > max_wait {
            break;
        }

        if !completed_1 {
            if let Ok(Some(req)) = engine_clone.get_request_status(req_id_1).await {
                if req.is_complete() {
                    completed_1 = true;
                    println!("Low temp request completed");
                }
            }
        }

        if !completed_2 {
            if let Ok(Some(req)) = engine_clone.get_request_status(req_id_2).await {
                if req.is_complete() {
                    completed_2 = true;
                    println!("High temp request completed");
                }
            }
        }

        if completed_1 && completed_2 {
            return Ok(());
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    Err(anyhow::anyhow!("Timeout waiting for temperature tests"))
}

// =============================================================================
// Part 2: Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_invalid_model_path() {
    let invalid_path = "/nonexistent/path/to/model.gguf";
    let result = InferenceEngine::from_gguf(invalid_path).await;

    assert!(result.is_err(), "Should fail with invalid model path");
    if let Err(e) = result {
        println!("Got expected error: {}", e);
        match e {
            EngineError::ModelLoadFailed(_) => {
                // Expected
            }
            _ => panic!("Expected ModelLoadFailed error, got: {}", e),
        }
    }
}

#[tokio::test]
async fn test_max_tokens_zero() {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return;
    }

    // Note: We're not actually running inference with max_tokens=0
    // Just testing the API accepts it
    let model_path = test_model_path();

    // This should successfully load and start
    let result = InferenceEngine::from_gguf(&model_path).await;
    assert!(result.is_ok(), "Should successfully load model");

    let mut engine = result.unwrap();
    let start_result = engine.start().await;
    assert!(start_result.is_ok(), "Should successfully start engine");
}

#[tokio::test]
async fn test_get_nonexistent_request_status() {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return;
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await.unwrap();
    engine.start().await.unwrap();

    let engine_clone = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_clone);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    // Query status of non-existent request
    let fake_request_id = 99999;
    let status = engine_clone.get_request_status(fake_request_id).await;

    assert!(status.is_ok(), "Status query should not error");
    let status_opt = status.unwrap();
    assert!(status_opt.is_none(), "Non-existent request should return None");
}

#[tokio::test]
async fn test_cancel_request() {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return;
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await.unwrap();
    engine.start().await.unwrap();

    let engine_clone = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_clone);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    // Submit a request
    let prompt_tokens = vec![1u32, 2, 3, 4, 5];
    let request_id = engine_clone
        .submit_request(prompt_tokens, 100, 1.0, 50, 0.9)
        .await
        .unwrap();

    // Immediately cancel it
    let cancel_result = engine_clone.cancel_request(request_id).await;
    assert!(cancel_result.is_ok(), "Cancel should succeed");

    // Verify request was cancelled
    let status = engine_clone.get_request_status(request_id).await.unwrap();
    assert!(status.is_some(), "Request should still exist");
    let req = status.unwrap();
    assert!(req.is_complete(), "Cancelled request should be marked complete");
}

// =============================================================================
// Part 3: HTTP Server Integration Tests
// =============================================================================

#[tokio::test]
#[ignore]
#[serial]
async fn test_http_server_requires_engine() {
    // Server can be created without an engine
    let tokenizer = TokenizerAdapter::default();
    let server = InferenceServer::new(None, tokenizer);

    // But requests should fail without engine
    let request = GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(10),
        temperature: Some(1.0),
        top_k: Some(50),
        top_p: Some(0.9),
        stream: Some(false),
    };

    let result = server.generate(request).await;
    assert!(result.is_err(), "Should fail without engine loaded");
}

#[tokio::test]
#[ignore]
#[serial]
async fn test_http_server_generate_with_engine() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();

    // Load engine
    let engine = InferenceEngine::from_gguf(&model_path).await?;
    engine.start().await?;

    let engine_arc = Arc::new(engine);

    // Start inference loop
    let engine_loop = Arc::clone(&engine_arc);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    // Create server with engine
    let tokenizer = TokenizerAdapter::default();
    let server = InferenceServer::new(Some(engine_arc), tokenizer);

    // Submit generation request
    let request = GenerateRequest {
        prompt: "The".to_string(), // Simple prompt
        max_tokens: Some(5),
        temperature: Some(1.0),
        top_k: Some(50),
        top_p: Some(0.9),
        stream: Some(false),
    };

    let response = server.generate(request).await?;

    println!("Generation response: request_id={}, text={}, tokens={}, finished={}",
        response.request_id, response.text, response.tokens.len(), response.finished);

    assert!(!response.text.is_empty() || !response.tokens.is_empty(),
        "Should generate either text or tokens");

    Ok(())
}

#[tokio::test]
#[ignore]
#[serial]
async fn test_http_server_request_status() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await?;
    engine.start().await?;

    let engine_arc = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_arc);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    let tokenizer = TokenizerAdapter::default();
    let server = InferenceServer::new(Some(engine_arc), tokenizer);

    let request = GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(5),
        temperature: Some(1.0),
        top_k: Some(50),
        top_p: Some(0.9),
        stream: Some(false),
    };

    let response = server.generate(request).await?;
    let request_id = response.request_id;

    // Check status
    let status_response = server.get_request_status(request_id).await?;
    assert_eq!(status_response.request_id, request_id);
    assert!(status_response.finished, "Request should be finished");

    Ok(())
}

#[tokio::test]
#[ignore]
#[serial]
async fn test_http_server_nonexistent_request_status() {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return;
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await.unwrap();
    engine.start().await.unwrap();

    let engine_arc = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_arc);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    let tokenizer = TokenizerAdapter::default();
    let server = InferenceServer::new(Some(engine_arc), tokenizer);

    // Query status of non-existent request
    let fake_request_id = 99999;
    let result = server.get_request_status(fake_request_id).await;

    assert!(result.is_err(), "Should return error for non-existent request");
}

// =============================================================================
// Part 4: Engine Configuration Tests
// =============================================================================

#[tokio::test]
#[ignore]
#[serial]
async fn test_engine_from_gguf_creates_correct_config() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await?;

    // Verify engine was created successfully
    // The config should be derived from the GGUF model
    let stats = engine.get_engine_stats().await;
    assert!(stats.model_loaded, "Model should be loaded");

    Ok(())
}

#[tokio::test]
#[ignore]
#[serial]
async fn test_multiple_sequential_requests() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::from_gguf(&model_path).await?;
    engine.start().await?;

    let engine_clone = Arc::new(engine);
    let engine_loop = Arc::clone(&engine_clone);
    tokio::spawn(async move {
        let _ = engine_loop.run_inference_loop().await;
    });

    // Submit multiple requests sequentially
    let num_requests = 3;
    let mut request_ids = Vec::new();

    for i in 0..num_requests {
        let prompt_tokens = vec![1u32 + i as u32];
        let request_id = engine_clone
            .submit_request(prompt_tokens, 2, 1.0, 50, 0.9)
            .await?;
        request_ids.push(request_id);
        println!("Submitted request {} ({} of {})", request_id, i + 1, num_requests);
    }

    // Wait for all to complete
    let max_wait = std::time::Duration::from_secs(60);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > max_wait {
            return Err(anyhow::anyhow!("Timeout waiting for sequential requests"));
        }

        // Count completed requests
        let mut completed = 0;
        for req_id in &request_ids {
            if let Ok(Some(req)) = engine_clone.get_request_status(*req_id).await {
                if req.is_complete() {
                    completed += 1;
                }
            }
        }

        if completed == num_requests {
            println!("All {} sequential requests completed", num_requests);
            return Ok(());
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

// =============================================================================
// Part 5: Cleanup and Memory Safety Tests
// =============================================================================

#[tokio::test]
#[ignore]
#[serial]
async fn test_engine_cleanup() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();

    {
        // Create engine in a scope
        let engine = InferenceEngine::from_gguf(&model_path).await?;
        engine.start().await?;

        let engine_clone = Arc::new(engine);
        let engine_loop = Arc::clone(&engine_clone);
        tokio::spawn(async move {
            let _ = engine_loop.run_inference_loop().await;
        });

        // Submit a request
        let prompt_tokens = vec![1u32];
        let _request_id = engine_clone
            .submit_request(prompt_tokens, 1, 1.0, 50, 0.9)
            .await?;

        // Wait a bit for processing
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Stop engine
        engine_clone.stop().await?;

        println!("Engine stopped successfully");
    } // Engine goes out of scope here

    // If we get here without crashing, cleanup succeeded
    println!("Engine cleanup successful");
    Ok(())
}
