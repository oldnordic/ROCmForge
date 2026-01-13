# Vocab Size Inference Fix Plan

**Date**: 2026-01-12
**Phase**: 24 - GGUF Metadata Compatibility
**Status**: ✅ COMPLETE - Implemented

---

## Problem Statement

ROCmForge fails to load GGUF models when `vocab_size` metadata is missing or zero.
The embedding tensor exists but is rejected due to over-constrained validation.

### Root Cause Analysis

**File**: `src/model/execution_plan.rs:396` (OLD CODE)

```rust
// OLD (BROKEN):
if shape.len() == 2 && shape[0] == config.vocab_size {
    return Ok(Arc::new(lazy.clone()));
}
```

When `config.vocab_size == 0`, this condition can NEVER succeed.

---

## Implementation Status: ✅ COMPLETE

All 4 functions have been updated with llama.cpp-compatible vocab size inference:

| Function | Lines | Status |
|----------|-------|--------|
| `map_embedding_lazy()` | 400-468 | ✅ IMPLEMENTED |
| `map_lm_head_lazy()` | 475-508 | ✅ IMPLEMENTED |
| `map_embedding()` | 1214-1275 | ✅ IMPLEMENTED |
| `map_lm_head()` | 1282+ | ✅ IMPLEMENTED |

### What Was Implemented

```rust
// Infer vocab_size if unknown (vocab_size == 0)
let actual_vocab_size = if config.vocab_size == 0 {
    // Use hidden_size as anchor to disambiguate
    if d0 == hidden_size && d1 != hidden_size {
        d1  // [hidden, vocab] layout
    } else if d1 == hidden_size && d0 != hidden_size {
        d0  // [vocab, hidden] layout
    } else {
        d0.max(d1)  // fallback: larger dimension is vocab
    }
} else {
    config.vocab_size
};

// Accept: [vocab, hidden] OR [hidden, vocab]
if (d0 == actual_vocab_size && d1 == hidden_size) ||
   (d0 == hidden_size && d1 == actual_vocab_size) {
    tracing::info!("Found embedding '{}' with shape {:?}, vocab_size={}",
                  name, shape, actual_vocab_size);
    return Ok(Arc::new(lazy.clone()));
}
```

---

## Test Plan

### Test 1: Load qwen2.5-0.5b.gguf (missing vocab metadata)

```bash
cargo run --release --bin rocmforge_cli -- generate \
  --gguf models/qwen2.5-0.5b.gguf \
  --prompt "Hello" \
  --max-tokens 5
```

**Status**: ⏳ NOT YET TESTED

### Test 2: Verify vocab_size is inferred correctly

**Status**: ⏳ NOT YET TESTED

### Test 3: Run existing unit tests

```bash
cargo test --lib --features rocm
```

**Expected**: All existing tests still pass

---

## Success Criteria

| Criteria | Status |
|----------|--------|
| qwen2.5-0.5b.gguf loads without error | ⏳ NOT YET TESTED |
| Vocab size is inferred from tensor shape | ⏳ NOT YET TESTED |
| Both `[vocab, hidden]` and `[hidden, vocab]` layouts are accepted | ✅ CODE COMPLETE |
| Error messages provide evidence (list found tensors) | ✅ CODE COMPLETE |
| All existing unit tests pass | ⏳ NOT YET TESTED |
| First token generation succeeds | ⏳ NOT YET TESTED |

---

## Files Modified

| File | Function | Lines | Status |
|------|----------|-------|--------|
| `src/model/execution_plan.rs` | `map_embedding_lazy` | 400-468 | ✅ DONE |
| `src/model/execution_plan.rs` | `map_lm_head_lazy` | 475-508 | ✅ DONE |
| `src/model/execution_plan.rs` | `map_embedding` | 1214-1275 | ✅ DONE |
| `src/model/execution_plan.rs` | `map_lm_head` | 1282+ | ✅ DONE |

---

## References

- llama.cpp philosophy: "Metadata is advisory, not authoritative"
- llama.cpp GGUF loading: `gguf_load_tensor_meta()` in `gguf.c`
- Existing ROCmForge vocab inference: `infer_vocab_size_from_tensors()` in `src/loader/gguf.rs:1692`
