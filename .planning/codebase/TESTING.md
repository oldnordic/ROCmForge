# Testing Patterns

**Analysis Date:** 2026-01-20

## Test Framework

**Runner:**
- Built-in `cargo test` (Rust's test framework)
- `serial_test` crate for sequential GPU tests

**Assertion Library:**
- Standard `assert!`, `assert_eq!`, `assert_matches!` macros
- `proptest` for property-based testing

**Run Commands:**
```bash
cargo test                              # Run all tests
cargo test --test <test_name>           # Run specific test file
cargo test -- --ignored                 # Run ignored tests
cargo test -- --nocapture               # Show println output
ROCFORGE_TEST_MODEL=/path/to/model.gguf cargo test --test e2e_inference_tests -- --ignored
```

**Config:**
- No separate test config file
- Test configuration via environment variables:
  - `ROCFORGE_TEST_MODEL` - Path to test model for E2E tests

## Test File Organization

**Location:**
- Co-located tests in `src/` using `#[cfg(test)]` modules
- Integration tests in `tests/` directory
- Common test helpers in `tests/common/`

**Naming:**
- Module tests: `*_tests.rs` (e.g., `causal_mask_tests.rs`, `kernel_tests.rs`)
- Integration tests: `<topic>_tests.rs` (e.g., `simple_model_tests.rs`, `e2e_inference_tests.rs`)
- Benchmarks: `<topic>_bench.rs` in `benches/`

**Structure:**
```
src/
  attention/
    kernels.rs
    kernel_tests.rs          # Co-located GPU kernel tests
    causal_mask_tests.rs
  model/
    simple_transformer.rs
    position_embedding_tests.rs

tests/
  common/
    tempfile_helpers.rs      # Shared test utilities
  simple_model_tests.rs      # Integration tests
  e2e_inference_tests.rs     # End-to-end tests
  attention_gpu_tests.rs
```

## Test Structure

**Suite Organization:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test body
    }

    #[test]
    #[serial]  // For GPU tests that can't run in parallel
    fn test_gpu_operation() {
        // Test requiring exclusive GPU access
    }

    #[test]
    #[ignore]  // Requires real model file
    fn test_with_real_model() {
        // Test skipped by default
    }
}
```

**Patterns:**

**Setup pattern:**
```rust
fn get_backend_or_skip() -> Arc<HipBackend> {
    match HipBackend::new_checked() {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("GPU not available: {}", e);
            panic!("GPU_SKIP");  // Graceful skip pattern
        }
    }
}
```

**Teardown pattern:**
- Rust's Drop handles resource cleanup
- GPU memory leak checking via `assert_no_leak()`:
  ```rust
  fixture.assert_no_leak(5);  // 5% tolerance
  ```

**Assertion pattern:**
```rust
// Standard assertions
assert_eq!(expected, actual);
assert!(condition, "Context: {}", value);

// Tolerance for floating point
assert!((expected - actual).abs() < 1e-4, "Diff too large");

// Pattern matching
assert!(matches!(result, Ok(_)));
assert!(matches!(err, SamplerError::InvalidTemperature(_)));
```

## Mocking

**Framework:** `mockall` (in dev-dependencies)

**Patterns:**
- Limited use observed - mostly integration-style testing
- Test fixtures use real GPU when available

**What to Mock:**
- File I/O for some loader tests
- HTTP responses (if testing server components)

**What NOT to Mock:**
- GPU kernels (use real hardware or skip)
- Core numerical operations
- Attention mechanisms

## Fixtures and Factories

**Test Data:**

```rust
// Seeded random for reproducibility
let data = Tensor::random_seeded(size, 42);

// Test fixtures module (tests/common/)
pub fn create_temp_file() -> anyhow::Result<tempfile::NamedTempFile>

// Common patterns
let q = vec![1.0f32; seq_len * dim];
let k = vec![0.5f32; seq_len * dim];
```

**Location:**
- `tests/common/tempfile_helpers.rs` - File/tempdir helpers
- `src/backend/gpu_test_common.rs` - GPU test fixture

**GPU Test Fixture:**
```rust
// Global shared GPU fixture
pub static GPU_FIXTURE: Lazy<Option<GpuTestFixture>> = Lazy::new(|| {
    if !HipBackend::gpu_available() {
        return None;
    }
    match GpuTestFixture::new() {
        Ok(fixture) => Some(fixture),
        Err(e) => None,
    }
});
```

## Coverage

**Requirements:** No enforced coverage target

**View Coverage:**
- No coverage tool configuration detected
- Use `cargo test -- --no-run` for compilation check

**Test counts (approximate):**
- 305 tests in `src/` directory
- 27 tests marked `#[ignore]`
- 27 ignored tests (require GPU/HSACO)

## Test Types

**Unit Tests:**
- Co-located in `src/` using `#[cfg(test)]`
- Test individual functions and methods
- Example: `test_softmax_basic`, `test_linear_forward`

**Integration Tests:**
- In `tests/` directory
- Test module interactions
- Example: `tests/simple_model_tests.rs`

**E2E Tests:**
- `tests/e2e_inference_tests.rs`
- Require real GGUF model file
- Marked `#[ignore]` by default
- Run with: `ROCFORGE_TEST_MODEL=/path/to/model.gguf cargo test --test e2e_inference_tests -- --ignored`

**Property Tests:**
- Using `proptest` crate
- Example in `src/sampler/sampler.rs`:
  ```rust
  proptest! {
      #[test]
      fn test_sampling_properties(
          logits in prop::collection::vec(-10.0f32..10.0f32, 1..100),
          temperature in 0.1f32..2.0f32,
          // ...
      ) {
          // Property verification
      }
  }
  ```

**GPU Tests:**
- Require `#[cfg(feature = "rocm")]` - only run with ROCm feature
- Use `#[serial]` attribute from `serial_test` crate
- Graceful skip pattern:
  ```rust
  fn get_backend_or_skip() -> Arc<HipBackend> {
      match HipBackend::new_checked() {
          Ok(backend) => backend,
          Err(e) => {
              eprintln!("GPU not available: {}", e);
              panic!("GPU_SKIP");
          }
      }
  }
  ```

## Common Patterns

**Async Testing:**
```rust
#[tokio::test]
async fn test_async_operation() {
    // Use tokio::test for async tests
    let result = engine.submit_request(...).await?;
    assert!(result.is_ok());
}
```

**Error Testing:**
```rust
#[test]
fn test_invalid_input_returns_error() {
    let result = sampler.sample(&[]);
    assert!(result.is_err());
    assert!(matches!(result, Err(SamplerError::EmptyLogits)));
}

#[test]
fn test_config_validation() {
    let config = SamplingConfig::new(0.0, 50, 0.9);  // Invalid temp
    assert!(config.is_err());
    assert!(matches!(config, Err(SamplerError::InvalidTemperature(0.0))));
}
```

**Numerical Accuracy Testing:**
```rust
const TEST_TOLERANCE: f32 = 1e-5;

// CPU vs GPU comparison
for (i, (&cpu_val, &gpu_val)) in cpu_iter.zip(gpu_iter).enumerate() {
    let diff = (cpu_val - gpu_val).abs();
    assert!(diff < TEST_TOLERANCE,
        "Mismatch at {}: CPU={}, GPU={}, diff={}",
        i, cpu_val, gpu_val, diff);
}

// Check for finite values
assert!(output.iter().all(|x| x.is_finite()),
    "All outputs should be finite");
```

**Determinism Testing:**
```rust
#[test]
fn test_model_deterministic_with_same_seed() {
    let model1 = SimpleModel::new(20, 4, 1, 3, ModelBackend::Cpu, 999);
    let model2 = SimpleModel::new(20, 4, 1, 3, ModelBackend::Cpu, 999);

    let output1 = model1.forward(&input).unwrap();
    let output2 = model2.forward(&input).unwrap();

    for (i, (&v1, &v2)) in output1.iter().zip(output2.iter()).enumerate() {
        assert!((v1 - v2).abs() < 1e-6,
            "Output at index {} differs: {} vs {}", i, v1, v2);
    }
}
```

**Skip Pattern for Missing Resources:**
```rust
#[test]
#[ignore]  // Requires real model
fn test_with_real_model() {
    if !has_test_model() {
        println!("Skipping: no test model available");
        return;
    }
    // Actual test...
}
```

**HSACO Kernel Testing:**
```rust
let result = unsafe {
    crate::attention::kernels::some_kernel(
        ptr,
        param1,
        param2,
    )
};

if let Err(e) = result {
    eprintln!("SKIPPED: GPU kernel failed: {} - HSACO not loaded", e);
    return;  // Skip if kernel not compiled
}
```

---

*Testing analysis: 2026-01-20*
