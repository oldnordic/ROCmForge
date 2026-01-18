# End-to-End (E2E) Inference Tests

This document describes how to run the E2E inference tests for ROCmForge.

## Overview

The E2E tests (`tests/e2e_inference_tests.rs`) validate the complete inference pipeline:
- Model loading from real GGUF files
- Token generation (single and multi-token)
- HTTP server integration
- Error handling
- Request tracking and cancellation

## Prerequisites

### Hardware Requirements
- AMD GPU with ROCm support
- Sufficient GPU memory for model (typically 2-4GB for small models)

### Software Requirements
- ROCm runtime installed
- `rocm-smi` command should work
- `amdhip64` library in `LD_LIBRARY_PATH`

### Model Requirements
You need a GGUF model file. Recommended small models:
- `qwen2.5-0.5b.gguf` - Tiny Chinese/English model (~500MB)
- `tiny-llama.gguf` - Small test model
- `bge-small-en-v1.5.Q8_0.gguf` - Embedding model

Download from HuggingFace or other sources.

## Running the Tests

### Skip E2E Tests (Default)

E2E tests are marked with `#[ignore]` and won't run by default:

```bash
cargo test
```

This skips E2E tests because they require:
1. A real GGUF model file
2. AMD GPU with ROCm
3. More time to run (30-60 seconds per test)

### Run E2E Tests (Requires Model)

Set the `ROCFORGE_TEST_MODEL` environment variable to your model path:

```bash
# Run all E2E tests
ROCFORGE_TEST_MODEL=/path/to/model.gguf cargo test --test e2e_inference_tests -- --ignored

# Run specific E2E test
ROCFORGE_TEST_MODEL=/path/to/model.gguf cargo test --test e2e_inference_tests test_single_token_inference -- --ignored

# Run with verbose output
ROCFORGE_TEST_MODEL=/path/to/model.gguf cargo test --test e2e_inference_tests -- --ignored --nocapture
```

### Test Organization

Tests are organized into 5 parts:

1. **Basic Inference Smoke Tests** (4 tests)
   - `test_single_token_inference` - Single token generation
   - `test_multi_token_generation` - Multi-token generation
   - `test_request_status_tracking` - Request progress tracking
   - `test_inference_with_different_temperatures` - Temperature effects

2. **Error Handling Tests** (4 tests)
   - `test_invalid_model_path` - Bad path error handling
   - `test_max_tokens_zero` - Edge case
   - `test_get_nonexistent_request_status` - Query safety
   - `test_cancel_request` - Request cancellation

3. **HTTP Server Integration Tests** (4 tests)
   - `test_http_server_requires_engine` - Server without engine fails
   - `test_http_server_generate_with_engine` - Full HTTP flow
   - `test_http_server_request_status` - Status endpoint
   - `test_http_server_nonexistent_request_status` - Error handling

4. **Engine Configuration Tests** (2 tests)
   - `test_engine_from_gguf_creates_correct_config` - Config validation
   - `test_multiple_sequential_requests` - Sequential handling

5. **Cleanup and Memory Safety Tests** (1 test)
   - `test_engine_cleanup` - Resource cleanup verification

## Troubleshooting

### Test Skipped: "no test model available"

**Problem**: Tests skip because `ROCFORGE_TEST_MODEL` is not set or file doesn't exist.

**Solution**:
```bash
# Check if model path is correct
ls -la $ROCFORGE_TEST_MODEL

# Set the environment variable
export ROCFORGE_TEST_MODEL=/path/to/your/model.gguf

# Verify it's set
echo $ROCFORGE_TEST_MODEL
```

### GPU Initialization Error

**Problem**: GPU not available or ROCm not configured.

**Solution**:
```bash
# Check GPU status
rocm-smi

# Check if amdhip64 is in library path
echo $LD_LIBRARY_PATH | grep -o amd64

# Add ROCm libraries to path if needed
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Timeout Errors

**Problem**: Tests timeout after 30-60 seconds.

**Possible causes**:
1. Model is too large for GPU memory
2. GPU is busy with other tasks
3. ROCm driver issue

**Solution**:
- Use a smaller model (e.g., qwen2.5-0.5b instead of 7b)
- Close other GPU applications
- Restart ROCm services if needed

### Compilation Errors

**Problem**: Tests don't compile.

**Solution**:
```bash
# Clean build
cargo clean

# Rebuild
cargo build --tests
```

## Continuous Integration

For CI/CD pipelines, you can:

1. **Skip E2E tests by default** (recommended)
   ```bash
   cargo test
   ```

2. **Run E2E tests in a separate job** with GPU runner
   ```yaml
   # Example GitHub Actions
   - name: Run E2E tests
     env:
       ROCFORGE_TEST_MODEL: /models/test-model.gguf
     run: cargo test --test e2e_inference_tests -- --ignored
   ```

3. **Use a cached test model** to avoid downloading every run

## Test Execution Time

| Test | Approx. Time |
|------|--------------|
| Single token inference | 2-5 seconds |
| Multi-token generation | 5-10 seconds |
| Temperature test | 10-15 seconds |
| HTTP server tests | 5-15 seconds each |
| Sequential requests | 10-20 seconds |
| **Total** | **~60-120 seconds** |

## Adding New E2E Tests

When adding new E2E tests:

1. **Mark with `#[ignore]`** to skip by default
2. **Check model availability** at the start:
   ```rust
   if !has_test_model() {
       println!("Skipping: no test model available");
       return Ok(());
   }
   ```
3. **Use `#[serial]`** attribute for GPU safety (one test at a time)
4. **Set reasonable timeouts** (30-60 seconds for most tests)
5. **Handle model loading errors** gracefully

## References

- Plan: `.planning/phases/02-test-infrastructure/03-e2e-inference-tests/PLAN.md`
- Test file: `tests/e2e_inference_tests.rs`
- Common fixtures: `tests/common/mod.rs`

## Support

For issues with E2E tests:
1. Check the troubleshooting section above
2. Review test output with `--nocapture` flag
3. Verify model file is valid GGUF format
4. Check GPU availability with `rocm-smi`
