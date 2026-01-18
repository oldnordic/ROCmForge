# Testing Patterns

**Analysis Date:** 2026-01-18

## Test Framework

**Runner:**
- Rust built-in `cargo test`
- Config: `Cargo.toml`

**Assertion Library:**
- Built-in `assert!`, `assert_eq!`, `assert_matches!`

**Run Commands:**
```bash
cargo test                              # Run all tests
cargo test --test test_name             # Single test
cargo test --release                    # Release mode tests
cargo bench                             # Run Criterion benchmarks
```

## Test File Organization

**Location:**
- `tests/` directory for integration tests
- Inline `#[cfg(test)]` modules in source files
- `benches/` for Criterion benchmarks

**Naming:**
- `*_test.rs` - Standalone test files
- `tests/{name}_tests.rs` - Integration test files
- `benches/{name}.rs` - Benchmark files

**Structure:**
```
tests/
├── hip_blas_matmul_tests.rs
├── loader_tests.rs
├── e2e_suite.rs
└── common/
    └── mod.rs

src/
├── tensor/matmul_test.rs (inline)
├── kv_cache/kv_cache_test.rs (inline)
└── ...
```

## Test Structure

**Suite Organization:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_name() {
        // arrange
        let input = create_test_input();

        // act
        let result = function_call(input);

        // assert
        assert_eq!(result, expected);
    }

    #[test]
    fn test_error_case() {
        assert!(function_call(invalid).is_err());
    }
}
```

**Patterns:**
- `#[test]` attribute for test functions
- `serial_test` for GPU-dependent tests (prevent concurrent execution)
- Property-based testing with `proptest`

## Mocking

**Framework:**
- `mockall` 0.12 for mocking
- Manual mocks via trait implementations

**Patterns:**
```rust
// Mock trait
#[automock]
trait MyTrait {
    fn do_something(&self) -> Result<u32>;
}

// Use in test
let mock = MockMyTrait::new();
mock.expect_do_something()
    .returning(|| Ok(42));
```

**What to Mock:**
- External API calls (when implemented)
- File I/O (using temp files)
- GPU operations (via traits)

**What NOT to Mock:**
- Pure functions
- Simple utilities

## Fixtures and Factories

**Test Data:**
```rust
// Factory functions (inline)
fn create_test_tensor() -> Tensor {
    Tensor::zeros(&[2, 2])
}

// Shared fixtures in tests/common/mod.rs
pub mod fixtures {
    pub fn test_model_path() -> PathBuf {
        // ...
    }
}
```

**Location:**
- Factory functions: inline in test files
- Shared fixtures: `tests/common/mod.rs`
- Note: Some duplication due to integration test import limitations

## Coverage

**Requirements:**
- No enforced coverage target
- Coverage tracked for awareness

**Configuration:**
- No explicit coverage tool configured
- 96 files contain test functions

**View Coverage:**
```bash
cargo test
# Use external tools like tarpaulin or grcov for coverage reports
```

## Test Types

**Unit Tests:**
- Test single function/module in isolation
- Inline `#[cfg(test)]` modules
- Fast: typically <100ms per test

**Integration Tests:**
- Test multiple modules together
- Files in `tests/` directory
- May require GPU access

**GPU Tests:**
- Require AMD GPU with ROCm
- Use `serial_test` to prevent concurrent execution
- May be skipped in CI without GPU

**Benchmarks:**
- Criterion framework in `benches/`
- Performance regression detection
- HTML reports for analysis

## Common Patterns

**Async Testing:**
```rust
#[tokio::test]
async fn test_async_operation() {
    let result = async_function().await.unwrap();
    assert_eq!(result, expected);
}
```

**Error Testing:**
```rust
#[test]
fn test_error_case() {
    let result = function_call(invalid_input);
    assert!(result.is_err());
    assert_matches!(result, Err(MyError::SpecificVariant));
}
```

**Serial Testing (GPU):**
```rust
#[test]
#[serial]
fn test_gpu_operation() {
    // Only one GPU test runs at a time
}
```

## Known Issues

**Test Infrastructure:**
- 6 test files have compilation errors (as of Jan 2026)
- 20+ commented tests needing rewrite for new GGUF API
- `tests/loader_tests.rs` - Multiple TODO: Rewrite tests
- `tests/embedding_to_lmhead_tests.rs` - Entire file needs rewrite

**Debug Output:**
- 42 files still contain `eprintln!` statements (should use `tracing`)

**Coverage Gaps:**
- End-to-end inference flow not fully tested
- HTTP server integration incomplete

---

*Testing analysis: 2026-01-18*
*Update when test patterns change*
