# Plan 02-01: Rewrite Commented GGUF Loader Tests

**Phase**: 02 - Test Infrastructure
**Status**: Pending
**Complexity**: Medium
**Estimated Time**: 3-4 hours

---

## Problem Statement

The file `tests/loader_tests.rs` contains 5 commented-out tests that use the obsolete `GgufModel` API. These tests were commented out when the loader API was refactored, leaving gaps in test coverage for the GGUF loading functionality.

**Commented Tests:**
1. `test_gguf_model_loading` - Lines 77-87
2. `test_gguf_tensor_access` - Lines 90-99
3. `test_gguf_f32_conversion` - Lines 102-109
4. `test_gguf_invalid_magic` - Lines 112-122
5. `test_gguf_unsupported_version` - Lines 125-135

---

## Current State

### Old (Obsolete) API
```rust
// Old API - no longer exists
let loader = GgufLoader::new(file.path().to_str().unwrap());
let metadata = loader.metadata();
assert_eq!(metadata.architecture, "test");

let tensors = loader.load_tensors().unwrap();
let weight1 = tensors.get("weight1");
```

### New (Current) API
```rust
// Current API in src/loader/gguf.rs
let loader = GgufLoader::new(path)?;
let metadata = loader.metadata(); // Returns &GgufMetadata
let tensors = loader.load_tensors()?; // Returns HashMap<String, GgufTensor>
let gpu_tensors = loader.load_to_gpu(&backend)?; // For GPU upload
```

### Key API Changes
- `GgufLoader::new()` returns `Result<GgufLoader>` (not `Option`)
- `metadata()` returns `&GgufMetadata` (no need to clone)
- `load_tensors()` returns `HashMap<String, GgufTensor>` (not custom struct)
- Added `load_to_gpu()` for direct GPU upload

---

## Implementation Plan

### Task 1: Create GGUF Test Helper Functions
Create utility functions for generating test GGUF files:

**File**: `tests/loader_tests.rs` (add at top of file)

```rust
/// Create a minimal valid GGUF file for testing
fn create_test_gguf(path: &Path) -> anyhow::Result<()> {
    // Write GGUF magic, version, tensor count, KV count
    // Write minimal metadata (architecture: "test")
    // Write 1-2 test tensors with known shapes
    // Return Ok if successful
}
```

### Task 2: Rewrite test_gguf_model_loading
Test that `GgufLoader::new()` successfully loads a valid GGUF file:

```rust
#[test]
fn test_gguf_model_loading() -> anyhow::Result<()> {
    let temp_dir = tempdir()?;
    let gguf_path = temp_dir.path().join("test.gguf");
    create_test_gguf(&gguf_path)?;

    let loader = GgufLoader::new(gguf_path.to_str().unwrap())?;
    let metadata = loader.metadata();

    assert_eq!(metadata.architecture, "test");
    Ok(())
}
```

### Task 3: Rewrite test_gguf_tensor_access
Test that loaded tensors can be accessed by name:

```rust
#[test]
fn test_gguf_tensor_access() -> anyhow::Result<()> {
    let temp_dir = tempdir()?;
    let gguf_path = temp_dir.path().join("test.gguf");
    create_test_gguf(&gguf_path)?;

    let loader = GgufLoader::new(gguf_path.to_str().unwrap())?;
    let tensors = loader.load_tensors()?;

    let weight1 = tensors.get("weight1");
    assert!(weight1.is_some());
    assert_eq!(weight1.unwrap().shape.dims(), &[128, 256]);

    Ok(())
}
```

### Task 4: Rewrite test_gguf_f32_conversion
Test F32 tensor data loading and conversion:

```rust
#[test]
fn test_gguf_f32_conversion() -> anyhow::Result<()> {
    let temp_dir = tempdir()?;
    let gguf_path = temp_dir.path().join("test.gguf");
    create_test_gguf_with_f32(&gguf_path)?;

    let loader = GgufLoader::new(gguf_path.to_str().unwrap())?;
    let tensors = loader.load_tensors()?;
    let tensor = tensors.get("f32_tensor").unwrap();

    assert_eq!(tensor.tensor_type, GgufTensorType::F32);
    assert!(!tensor.data.is_empty());

    Ok(())
}
```

### Task 5: Rewrite test_gguf_invalid_magic
Test error handling for invalid GGUF files:

```rust
#[test]
fn test_gguf_invalid_magic() -> anyhow::Result<()> {
    let mut file = NamedTempFile::new()?;
    file.write_all(&0x12345678u32.to_le_bytes())?; // Invalid magic
    file.write_all(&3u32.to_le_bytes())?;
    file.write_all(&0u64.to_le_bytes())?;
    file.write_all(&0u64.to_le_bytes())?;

    let result = GgufLoader::new(file.path().to_str().unwrap());
    assert!(result.is_err());
    assert_matches!(result, Err(ref e) if e.to_string().contains("magic"));

    Ok(())
}
```

### Task 6: Rewrite test_gguf_unsupported_version
Test error handling for unsupported GGUF versions:

```rust
#[test]
fn test_gguf_unsupported_version() -> anyhow::Result<()> {
    let mut file = NamedTempFile::new()?;
    file.write_all(&0x46554747u32.to_le_bytes())?; // "GGUF" magic
    file.write_all(&999u32.to_le_bytes())?; // Unsupported version
    file.write_all(&0u64.to_le_bytes())?;
    file.write_all(&0u64.to_le_bytes())?;

    let result = GgufLoader::new(file.path().to_str().unwrap());
    assert!(result.is_err());

    Ok(())
}
```

---

## Testing Strategy

1. **Write tests first** (TDD): Each rewritten test should fail initially, then pass after implementation
2. **Use `anyhow::Result`** return type for test functions for clean error handling
3. **Verify GGUF file creation**: Ensure `create_test_gguf` produces valid files
4. **Test error paths**: Verify invalid inputs produce appropriate errors
5. **Run all tests**: `cargo test --test loader_tests`

---

## Dependencies

**No dependencies** - This plan can execute independently of other Phase 2 plans.

**Affects**: None (test-only changes)

---

## Definition of Done

- [ ] All 5 commented tests rewritten with current API
- [ ] `create_test_gguf()` helper function implemented
- [ ] All new tests pass: `cargo test --test loader_tests`
- [ ] No compiler warnings
- [ ] Code follows project conventions (RESULT types, proper error handling)
- [ ] COMMENTS added explaining any non-obvious test behavior

---

## Notes

- Reference `tests/gguf_loader_tests.rs` for examples of the new API usage
- Use `tempfile` crate for temporary file creation
- Use `anyhow::Result` for test function return types
- Keep tests simple and focused - one assertion per test ideally

---

*Plan: 02-01*
*Created: 2026-01-18*
