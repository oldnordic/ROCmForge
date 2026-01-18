# Plan 02-04: Replace unwrap() with Proper Error Handling in Tests

**Phase**: 02 - Test Infrastructure
**Status**: Pending
**Complexity**: Medium
**Estimated Time**: 3-4 hours

---

## Problem Statement

The test suite contains 463+ instances of `unwrap()` calls. While `unwrap()` is acceptable in some test scenarios (e.g., creating test fixtures), excessive use hides error information and makes test failures harder to debug.

**Current State**:
- 463 `unwrap()` calls in `tests/` directory
- Some are appropriate (fixture creation)
- Many should use proper error handling

---

## Analysis

### Appropriate Uses of unwrap()

These cases **should keep** `unwrap()`:

1. **Test fixture creation** where failure means "test setup broken"
2. **Literal values** that can't fail (e.g., `Some(42).unwrap()`)
3. **After explicit checks** (e.g., `assert!(x.is_ok()); x.unwrap()`)

### Inappropriate Uses of unwrap()

These cases **should replace** `unwrap()`:

1. **File I/O operations** - use `?` with `anyhow::Result`
2. **Model loading** - use `?` for better error messages
3. **GPU operations** - propagate errors for debugging
4. **External dependencies** - report what failed

---

## Implementation Plan

### Task 1: Categorize unwrap() Usage

Analyze the 463 unwrap() calls by category:

```bash
# Find unwrap() in each test file
grep -rn "unwrap()" tests/ --include="*.rs" | sort
```

**Categories**:
1. **Keep**: Fixture creation (tempfile, test data)
2. **Replace**: File/model loading
3. **Replace**: GPU operations
4. **Replace**: External API calls

### Task 2: Add anyhow::Result to Test Functions

Convert test functions to return `anyhow::Result<()>`:

**Before:**
```rust
#[test]
fn test_something() {
    let file = File::open("test.txt").unwrap();
    // ...
}
```

**After:**
```rust
#[test]
fn test_something() -> anyhow::Result<()> {
    let file = File::open("test.txt")?;
    // ...
    Ok(())
}
```

### Task 3: Priority Files (High Impact)

Focus on these high-value files first:

| File | unwrap() count | Priority |
|------|----------------|----------|
| `tests/gguf_loader_tests.rs` | ~50 | High |
| `tests/hip_blas_matmul_tests.rs` | ~40 | High |
| `tests/loader_tests.rs` | ~30 | Medium |
| `tests/execution_plan_*.rs` | ~80 | Medium |
| `tests/transformer_integration_tests.rs` | ~60 | Medium |

### Task 4: GPU Test Pattern

For GPU tests, use the `?` operator with context:

**Before:**
```rust
#[test]
#[serial]
fn test_gpu_matmul() {
    let backend = HipBackend::new(0).unwrap();
    let result = backend.matmul(&a, &b).unwrap();
    assert_eq!(result.len(), 100);
}
```

**After:**
```rust
#[test]
#[serial]
fn test_gpu_matmul() -> anyhow::Result<()> {
    let backend = HipBackend::new(0)
        .context("Failed to create HIP backend")?;
    let result = backend.matmul(&a, &b)
        .context("Matmul operation failed")?;
    assert_eq!(result.len(), 100);
    Ok(())
}
```

### Task 5: Model Loading Pattern

For model/tensor loading, propagate errors:

**Before:**
```rust
#[test]
fn test_model_load() {
    let loader = GgufLoader::new("model.gguf").unwrap();
    let tensors = loader.load_tensors().unwrap();
    assert!(tensors.len() > 0);
}
```

**After:**
```rust
#[test]
fn test_model_load() -> anyhow::Result<()> {
    let loader = GgufLoader::new("model.gguf")?;
    let tensors = loader.load_tensors()?;
    assert!(tensors.len() > 0);
    Ok(())
}
```

### Task 6: Keep unwrap() Where Appropriate

Don't change these cases:

```rust
// OK: Test fixture creation with tempfile
let temp_dir = tempdir().unwrap();

// OK: Constant values
let value = Some(42).unwrap();

// OK: After explicit assertion
assert!(result.is_ok());
let unwrapped = result.unwrap();
```

---

## Strategy

1. **Batch by file**: Complete one test file at a time
2. **Run tests after each file**: Verify changes don't break tests
3. **Use `anyhow::Context`**: Add error context for debugging
4. **Keep simple tests simple**: Don't over-engineer trivial tests

---

## Dependencies

**No Dependencies**: Can run in parallel with 02-01, 02-02, 02-03

**Affects**: All test files (read-only changes, no production code)

---

## Definition of Done

- [ ] Reduce unwrap() count by 60%+ (target: <200 instances)
- [ ] All GPU operations use proper error handling
- [ ] All file/model loading uses `?` operator
- [ ] All tests still pass: `cargo test`
- [ ] Error messages are descriptive (use `.context()`)
- [ ] Appropriate unwrap() uses remain (fixture creation)

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| unwrap() count | 463 | <200 |
| Tests returning Result | ~10% | >80% |
| GPU tests with proper errors | ~20% | 100% |

---

## Notes

- Use `anyhow::Context` for error context: `.context("message")?`
- Don't change `unwrap_or()`, `unwrap_or_default()` - those are fine
- Focus on test functions, not helper functions (those may use unwrap intentionally)
- Consider this a "polish" task - tests should work the same, just with better error messages

---

## Example Conversion

**Full file example** (`tests/simple_test.rs`):

**Before:**
```rust
#[test]
fn test_example() {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::new(&data).unwrap();
    let result = tensor.compute().unwrap();
    assert_eq!(result.len(), 3);
}
```

**After:**
```rust
#[test]
fn test_example() -> anyhow::Result<()> {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::new(&data)
        .context("Failed to create tensor from data")?;
    let result = tensor.compute()
        .context("Tensor computation failed")?;
    assert_eq!(result.len(), 3);
    Ok(())
}
```

---

*Plan: 02-04*
*Created: 2026-01-18*
