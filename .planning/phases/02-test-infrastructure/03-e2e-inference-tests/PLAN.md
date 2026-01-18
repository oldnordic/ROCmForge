# Plan 02-03: Add End-to-End Inference Tests

**Phase**: 02 - Test Infrastructure
**Status**: Pending
**Complexity**: High
**Estimated Time**: 4-6 hours

---

## Problem Statement

The codebase lacks comprehensive end-to-end inference tests. While individual components (matmul, attention, KV cache) have tests, there is no test that validates the full request → response pipeline from HTTP request to generated tokens.

**Current Gap**:
- No full inference flow test (prompt → tokens)
- No HTTP server integration test
- No streaming response validation
- No multi-turn conversation test

---

## What Needs Testing

### 1. Single-Token Inference (Smoke Test)
- Load a real GGUF model
- Submit a simple prompt
- Verify at least one token is generated
- Verify no crashes or panics

### 2. Multi-Token Generation
- Generate multiple tokens (e.g., "Hello, world!")
- Verify token sequence makes sense
- Verify KV cache is correctly updated

### 3. Streaming Response
- Test SSE streaming format
- Verify chunks are delivered incrementally
- Verify final chunk marks completion

### 4. Batch Processing
- Submit multiple requests concurrently
- Verify all complete successfully
- Verify no resource leaks

### 5. HTTP API Integration
- POST /v1/chat/completions endpoint
- Verify request/response format
- Verify OpenAI-compatible API

### 6. Error Handling
- Invalid model path
- Invalid prompt format
- GPU OOM simulation (if possible)

---

## Implementation Plan

### Task 1: Create Test Model Fixture

**File**: `tests/common/mod.rs` (or create `tests/e2e_common.rs`)

```rust
/// Path to a small test model (e.g., tiny-llama or similar)
/// Users can override via ROCFORGE_TEST_MODEL env var
pub fn test_model_path() -> PathBuf {
    env::var("ROCFORGE_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/models/tiny-llama.gguf"))
}

/// Check if test model is available (skip tests if not)
pub fn has_test_model() -> bool {
    test_model_path().exists()
}
```

### Task 2: Write Basic Inference Smoke Test

**File**: `tests/e2e_inference_tests.rs` (create new file)

```rust
#[tokio::test]
#[ignore] // Requires real model file
async fn test_single_token_inference() -> anyhow::Result<()> {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::new(model_path).await?;

    // Generate single token
    let prompt = "The";
    let result = engine.generate(prompt, 1).await?;

    assert!(!result.text.is_empty());
    assert!(result.tokens_generated == 1);

    Ok(())
}
```

### Task 3: Write Multi-Token Generation Test

```rust
#[tokio::test]
#[ignore]
async fn test_multi_token_generation() -> anyhow::Result<()> {
    if !has_test_model() {
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::new(model_path).await?;

    let prompt = "The capital of France is";
    let result = engine.generate(prompt, 10).await?;

    assert!(!result.text.is_empty());
    assert!(result.tokens_generated <= 10);
    assert!(result.text.contains("Paris") || result.text.len() > 0);

    Ok(())
}
```

### Task 4: Write Streaming Response Test

```rust
#[tokio::test]
#[ignore]
async fn test_streaming_generation() -> anyhow::Result<()> {
    if !has_test_model() {
        return Ok(());
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::new(model_path).await?;

    let prompt = "Count from 1 to 5:";
    let mut stream = engine.generate_streaming(prompt, 20).await?;

    let mut full_text = String::new();
    let mut chunk_count = 0;

    while let Some(chunk) = stream.next().await {
        full_text.push_str(&chunk.text);
        chunk_count += 1;
    }

    assert!(chunk_count > 1, "Should receive multiple chunks");
    assert!(!full_text.is_empty());

    Ok(())
}
```

### Task 5: Write HTTP Server Integration Test

```rust
#[tokio::test]
#[ignore]
async fn test_http_chat_completions() -> anyhow::Result<()> {
    if !has_test_model() {
        return Ok(());
    }

    let model_path = test_model_path();
    let addr = start_test_server(model_path).await?;
    let client = reqwest::Client::new();

    let request = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    });

    let response = client
        .post(format!("http://{}/v1/chat/completions", addr))
        .json(&request)
        .send()
        .await?;

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await?;
    let choices = body["choices"].as_array();
    assert!(choices.is_some());
    assert!(choices.unwrap().len() > 0);

    Ok(())
}
```

### Task 6: Write Error Handling Tests

```rust
#[tokio::test]
async fn test_invalid_model_path() {
    let invalid_path = PathBuf::from("/nonexistent/model.gguf");
    let result = InferenceEngine::new(invalid_path).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_empty_prompt() {
    if !has_test_model() {
        return;
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::new(model_path).await.unwrap();

    let result = engine.generate("", 10).await;

    // Should either error or handle gracefully
    // Verify behavior matches expectations
}

#[tokio::test]
async fn test_max_tokens_zero() {
    if !has_test_model() {
        return;
    }

    let model_path = test_model_path();
    let engine = InferenceEngine::new(model_path).await.unwrap();

    let result = engine.generate("Hello", 0).await;

    // Should return empty result (no tokens generated)
    assert!(result.is_ok());
    assert!(result.unwrap().tokens_generated == 0);
}
```

---

## Testing Strategy

1. **Use `#[ignore]` by default**: E2E tests require real models and are slow
2. **Model availability check**: Skip tests if `ROCFORGE_TEST_MODEL` not set
3. **Parallel execution**: Design tests to run concurrently (different models)
4. **Cleanup**: Ensure engines are properly dropped to prevent GPU leaks

---

## Dependencies

**No Dependencies**: Can run in parallel with 02-01, 02-02, 02-04

**Affects**: None (test-only)

---

## Definition of Done

- [ ] New file: `tests/e2e_inference_tests.rs` created
- [ ] 6+ test categories implemented
- [ ] All tests use `#[ignore]` attribute (opt-in via `cargo test --ignored`)
- [ ] Tests handle missing model gracefully
- [ ] HTTP server test validates OpenAI-compatible API
- [ ] Proper cleanup to prevent GPU leaks
- [ ] Documentation on how to run E2E tests

---

## Notes

- E2E tests are slow and require real models - use `#[ignore]`
- Document environment variable: `ROCFORGE_TEST_MODEL=/path/to/model.gguf`
- Consider adding a tiny synthetic model to repo for CI
- Streaming test requires async stream handling
- HTTP test requires `reqwest` or similar HTTP client

---

## Running the Tests

```bash
# Skip E2E tests (default)
cargo test

# Run E2E tests (requires model)
cargo test --ignored --test e2e_inference_tests

# With custom model path
ROCFORGE_TEST_MODEL=/models/tiny-llama.gguf cargo test --ignored
```

---

*Plan: 02-03*
*Created: 2026-01-18*
