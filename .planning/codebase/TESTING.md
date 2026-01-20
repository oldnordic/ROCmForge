# Testing Patterns

**Analysis Date:** 2026-01-20

## Test Framework

**Runner:**
- `cargo test` (built-in Rust test framework)
- Config: `Makefile` targets for test execution

**Assertion Library:**
- Built-in `assert!`, `assert_eq!`, `assert_matches!`
- Custom matchers for error types: `matches!(result, Err(ErrorType))`

**Feature-gated testing:**
```toml
[features]
default = []
rocm = []
simd = []
```

**Run Commands:**
```bash
# Run all tests with serial execution (GPU requirement)
make test           # cargo test --features rocm --lib -- --test-threads=1

# Verbose output
make test-verbose   # cargo test --features rocm --lib -- --test-threads=1 --nocapture

# Library unit tests only
make test-lib       # cargo test --features rocm --lib -- --test-threads=1

# Documentation tests
make test-docs      # cargo test --doc

# Check without running
make check          # cargo check --features rocm

# Lint
make clippy         # cargo clippy --features rocm -- -D warnings

# Format
make fmt            # cargo fmt
```

**Serial test execution:**
- GPU tests require `--test-threads=1` due to shared device state
- `serial_test` crate for test serialization:
```rust
#[test]
#[serial]
fn test_gpu_operation() { ... }
```

## Test File Organization

**Location:**
- **Co-located**: Tests in `#[cfg(test)]` modules within source files (preferred for unit tests)
- **Separate**: Integration tests in `tests/` directory at project root
- **Benchmark**: `benches/` directory for Criterion benchmarks

**Naming:**
- Inline test modules: `mod tests`, `mod kernel_tests`, `mod *_tests.rs`
- Test files: `*_tests.rs` suffix (e.g., `flash_attention_tests.rs`)
- Integration tests: `*_integration_tests.rs` for integration tests
- Phase-tagged: `phase*N_*_tests.rs` for development phase tracking

**Structure:**
```
tests/
├── common/
│   ├── mod.rs                # Shared test utilities
│   ├── fixtures.rs           # Reusable test fixtures
│   └── tempfile_helpers.rs   # Tempfile creation helpers
├── simple_model_tests.rs          # Model unit tests
├── edge_case_tests.rs             # Boundary condition tests
├── e2e_inference_tests.rs         # End-to-end tests
├── attention_tests.rs             # Attention module tests
└── ...
```

**Module-level test structure:**
```rust
// In src/attention/mod.rs
#[cfg(test)]
#[cfg(feature = "rocm")]
mod kernel_tests;

#[cfg(test)]
#[cfg(feature = "rocm")]
mod rope_gpu_tests;
```

## Test Structure

**Suite Organization:**
```rust
#[cfg(test)]
mod phase3_flash_attention_tests {
    use crate::attention::cpu::CpuBackend;
    use crate::attention::kernels::flash_attention_gpu_kernel;
    use crate::backend::{DeviceTensor, HipBackend};
    use serial_test::serial;

    const TEST_TOLERANCE: f32 = 1e-3;

    /// Helper: Get GPU backend or skip test if not available
    fn get_backend_or_skip() -> Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\nWARNING: GPU not available: {}", e);
                panic!("GPU_SKIP");  // Caught by test harness
            }
        }
    }

    /// Test 1: FlashAttention matches CPU - small dimensions
    #[test]
    #[serial]
    fn test_flash_attention_matches_cpu_small_no_mask() {
        let batch_size = 1;
        let seq_len = 4;
        let head_dim = 4;

        let (q, k, v) = create_qkv_tensors(batch_size, seq_len, head_dim);

        // CPU reference
        let cpu_result = CpuBackend::forward(head_dim, &q, &k, &v, None, None)
            .expect("CPU attention failed");

        // GPU run
        let backend = get_backend_or_skip();
        // ... GPU test code ...

        // Compare
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(diff < TEST_TOLERANCE, "Mismatch at {}: CPU={}, GPU={}, diff={}",
                    i, cpu_val, gpu_val, diff);
        }
    }
}
```

**Patterns:**

**Setup:**
- Helper functions for common setup: `get_backend_or_skip()`, `create_qkv_tensors()`
- Global fixture via `once_cell::sync::Lazy` for GPU backend
- Configuration creation helpers

**Teardown:**
- Explicit drop where needed: `drop(tensor);`
- Memory leak detection: `fixture.assert_no_leak(5);`

**Assertion pattern:**
```rust
// Basic assertion
assert!(result.is_ok(), "Operation should succeed");

// Error matching
assert!(result.is_err(), "Should fail with invalid input");
match result {
    Err(KvCacheError::CapacityExceeded) => { /* expected */ }
    Err(e) => panic!("Expected CapacityExceeded, got: {:?}", e),
    Ok(_) => panic!("Should not succeed when cache is full"),
}

// Float comparison with tolerance
assert!((cpu_val - gpu_val).abs() < TEST_TOLERANCE,
        "Values differ at {}: {} vs {}", i, cpu_val, gpu_val);

// Vector element-wise comparison
for (i, (&cpu_val, &gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
    let diff = (cpu_val - gpu_val).abs();
    assert!(diff < 1e-4, "mismatch at {}: CPU={}, GPU={}", i, cpu_val, gpu_val);
}
```

## Mocking

**Framework:** `mockall` (in dev-dependencies)

**Patterns:**
```rust
// mockall not extensively used in current codebase
// Most tests use real implementations or simplified versions
```

**What to Mock:**
- External services (HTTP clients)
- File I/O (using temp files instead)
- Time sources (for deterministic testing)

**What NOT to Mock:**
- CPU reference implementations (used for verification)
- GPU kernels (tested directly via comparison)
- Data structures (use real implementations)

**GPU Backend Mocking:**
- Use `HipBackend::new_checked()` for availability check
- Graceful skip pattern when GPU unavailable:
```rust
fn get_backend_or_skip() -> Arc<HipBackend> {
    match HipBackend::new_checked() {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("WARNING: GPU not available - skipping");
            panic!("GPU_SKIP");
        }
    }
}
```

## Fixtures and Factories

**Test Data:**

**GGUF file creation** (from `tests/common/fixtures.rs`):
```rust
/// Create a minimal valid GGUF file for testing
pub fn create_test_gguf(path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write GGUF magic
    writer.write_all(b"GGUF")?;
    writer.write_all(&3u32.to_le_bytes())?;  // version
    writer.write_all(&0u64.to_le_bytes())?;  // tensor count
    // ... metadata ...
    Ok(())
}

/// Create a minimal GGUF with token embeddings
pub fn create_embedding_gguf(
    path: &Path,
    vocab_size: usize,
    hidden_size: usize,
) -> anyhow::Result<()>
```

**Tensor creation:**
```rust
/// Create test Q, K, V tensors
fn create_qkv_tensors(
    batch_size: usize,
    seq_len: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total_size = batch_size * seq_len * head_dim;
    let q: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let v: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1 + 2.0).collect();
    (q, k, v)
}
```

**GPU Backend fixture** (from `tests/common/mod.rs` and `src/backend/gpu_test_common.rs`):
```rust
pub static GPU_FIXTURE: Lazy<Option<GpuTestFixture>> = Lazy::new(|| {
    if !HipBackend::gpu_available() {
        eprintln!("WARNING: GPU not available - skipping GPU tests");
        return None;
    }
    match GpuTestFixture::new() {
        Ok(fixture) => Some(fixture),
        Err(e) => {
            eprintln!("ERROR: Failed to initialize GPU test fixture: {}", e);
            None
        }
    }
});

pub struct GpuTestFixture {
    backend: std::sync::Arc<HipBackend>,
    initial_free_mb: usize,
    initial_total_mb: usize,
    device_name: String,
}
```

**Location:**
- `tests/common/fixtures.rs` - Reusable fixtures
- `tests/common/mod.rs` - GPU test fixture
- `src/backend/gpu_test_common.rs` - Internal GPU fixture

## Coverage

**Requirements:** No enforced coverage target

**View Coverage:**
```bash
# No coverage tool configured (use cargo-tarpaulin or similar if needed)
cargo tarpaulin --features rocm --out Html
```

**Coverage areas:**
- GPU kernels: Tested via CPU comparison
- CPU implementations: Unit tests + property tests
- Error paths: Explicit error variant tests
- Edge cases: Boundary value tests in `tests/edge_case_tests.rs`

## Test Types

**Unit Tests:**
- Co-located in `#[cfg(test)]` modules
- Test individual functions and methods
- Fast execution (no GPU or minimal GPU usage)

**Integration Tests:**
- `tests/` directory
- Test module interactions
- File format tests (GGUF loading)
- End-to-end inference tests

**E2E Tests:**
- `tests/e2e_inference_tests.rs`
- `tests/e2e_suite.rs`
- Full model loading and inference
- Requires actual model files (configurable via env var)

**GPU Accuracy Tests:**
- Compare GPU kernel output to CPU reference
- Use `TEST_TOLERANCE` for float comparison
- Pattern from `src/attention/flash_attention_tests.rs`:
```rust
const TEST_TOLERANCE: f32 = 1e-3;

let cpu_result = CpuBackend::forward(...);
let gpu_result = run_gpu_kernel(...);

for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
    let diff = (cpu_val - gpu_val).abs();
    assert!(diff < TEST_TOLERANCE, "Mismatch at {}: CPU={}, GPU={}", ...);
}
```

**Property Tests:**
- Framework: `proptest` (in dev-dependencies)
- Files with property tests:
  - `src/sampler/sampler.rs` - Sampling properties
  - `src/kv_cache/kv_cache.rs` - KV cache invariants
  - `src/scheduler/scheduler.rs` - Scheduling properties
  - `tests/scheduler_tests.rs`, `tests/kv_cache_tests.rs`, `tests/loader_tests.rs`

**Property test pattern:**
```rust
proptest! {
    #[test]
    fn test_sampling_properties(
        logits in prop::collection::vec(-10.0f32..10.0f32, 1..100),
        temperature in 0.1f32..2.0f32,
        top_k in 1usize..20usize,
        top_p in 0.1f32..1.0f32
    ) {
        let config = SamplingConfig::new(temperature, top_k, top_p).unwrap();
        let mut sampler = Sampler::new(config);

        if !logits.is_empty() {
            let token_id = sampler.sample(&logits);
            prop_assert!(token_id.is_ok());
            let token_id = token_id.unwrap();
            prop_assert!(token_id < logits.len() as u32);
        }
    }
}
```

**Regression Tests:**
- Named with phase numbers: `phase*N_*`
- Track specific bugs with targeted tests
- Pattern: `phase5_paged_tests.rs`, `phase6_integration_*.rs`

**Smoke Tests:**
- Quick sanity checks
- `tests/inference_smoke_tests.rs`, `tests/hip_backend_smoke_tests.rs`
- Verify basic functionality works

## Common Patterns

**Async Testing:**
```rust
// tokio test macro for async tests
#[tokio::test]
async fn test_async_operation() {
    let result = async_function().await.unwrap();
    assert!(result.is_some());
}
```

**Error Testing:**
```rust
#[test]
fn test_invalid_input_returns_error() {
    let config = CacheConfig::new(0, 100, 32, 128, 24);
    assert!(config.is_err(), "Zero page size should be invalid");

    match config {
        Err(KvCacheError::InvalidConfiguration) => { /* expected */ }
        Err(e) => panic!("Expected InvalidConfiguration, got: {:?}", e),
        Ok(_) => panic!("Should not accept zero page size"),
    }
}
```

**Deterministic Testing:**
```rust
#[test]
fn test_model_deterministic_with_same_seed() {
    let model1 = SimpleModel::new(20, 4, 1, 3, ModelBackend::Cpu, 999);
    let model2 = SimpleModel::new(20, 4, 1, 3, ModelBackend::Cpu, 999);

    let input_tokens = vec![1, 2, 3];

    let output1 = model1.forward(&input_tokens).unwrap();
    let output2 = model2.forward(&input_tokens).unwrap();

    for (i, (&v1, &v2)) in output1.iter().zip(output2.iter()).enumerate() {
        assert!((v1 - v2).abs() < 1e-6, "Output at index {} differs", i);
    }
}
```

**GPU Memory Leak Detection:**
```rust
#[test]
#[serial]
fn test_gpu_operation_no_leak() {
    let fixture = GPU_FIXTURE.as_ref().expect("GPU not available");

    // ... GPU operations ...

    // Check for memory leaks (5% tolerance)
    fixture.assert_no_leak(5);
}
```

**Numerical Stability Testing:**
```rust
#[test]
fn test_softmax_gpu_numerical_stability() {
    // Large values that would overflow exp()
    let input: Vec<f32> = vec![1000.0, 1001.0, 1002.0, 1003.0];

    // GPU run
    // ... kernel launch ...

    // Verify all values in valid range
    for &val in &gpu_input {
        assert!(val > 0.0 && val <= 1.0, "Invalid softmax value {}", val);
    }
}
```

**Test constants:**
```rust
const TEST_TOLERANCE: f32 = 1e-3;      // GPU vs CPU comparison
const SMALL_TOLERANCE: f32 = 1e-6;     // CPU reference
const EPSILON: f32 = 1e-9;             // Near-zero checks
```

**Ignored tests:**
- Tagged with `#[ignore]` for:
  - E2E tests requiring specific hardware/models
  - Performance benchmarks
  - Experimental features

---

*Testing analysis: 2026-01-20*
