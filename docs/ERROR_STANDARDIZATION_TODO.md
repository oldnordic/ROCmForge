# Error Message Standardization TODO

**Date**: 2026-01-08
**Status**: IN PROGRESS
**Goal**: Standardize all GPU memory-related error messages across the codebase

---

## Standardized Terminology

| Concept | Standard Term |
|---------|---------------|
| GPU memory allocation failures | Describe the HIP function that failed |
| Memory pool allocation | "GPU memory pool" |
| Sub-buffer creation | "GPU memory sub-allocation" |
| Tensor creation from pool | "GPU memory pool tensor creation" |
| Buffer copy operations | "GPU memory copy" |
| Handle creation failures | "Failed to create [handle type]" |

**Note**: For HIP-specific operations, use the HIP function name (e.g., "hipMalloc failed", "hipMemcpyDtoH failed") as this provides the most diagnostic value.

---

## Complete Inventory

### Files Analyzed

| File | Status | Notes |
|------|--------|-------|
| `src/backend/hip_backend.rs` | ✅ Reviewed | Error messages use HIP function names - appropriate |
| `src/loader/gguf.rs` | ✅ Fixed | Pool error messages standardized |
| `src/attention/gpu.rs` | ⚠️ Issue | Uses DimensionError for all failures (enum lacks proper variants) |
| `src/model/simple_transformer.rs` | ✅ OK | Uses warnings for optional GPU fallback |
| `src/ops/attention_gpu.rs` | ✅ OK | Error messages are clear |
| `src/model/execution_plan.rs` | ✅ OK | Error messages are clear |
| `src/model/glm_position.rs` | ⚠️ Issue | Uses DimensionError for non-dimension errors |
| `src/attention/rope.rs` | ⚠️ Issue | Uses DimensionError for non-dimension errors |
| `src/attention/multi_query.rs` | ⚠️ Issue | Uses DimensionError for non-dimension errors |
| Test files | ✅ OK | Using `.expect()` in tests is acceptable |

### Known Issues

**Issue 1**: `AttentionError` enum lacks variants for:
- Memory allocation failures
- Memory copy failures
- Handle/creation failures
- Kernel launch failures

Current workarounds:
- Everything is mapped to `DimensionError` (semantically incorrect)

**Resolution**: This requires enum expansion, which is a larger refactoring.

---

## Tasks

### Phase 1: Inventory (COMPLETE)
- [x] Task 1.1: Audit `src/backend/hip_backend.rs`
- [x] Task 1.2: Audit `src/loader/gguf.rs`
- [x] Task 1.3: Audit `src/attention/gpu.rs`
- [x] Task 1.4: Audit `src/model/simple_transformer.rs`
- [x] Task 1.5: Audit remaining production files

### Phase 2: BUG-11 Fixes (COMPLETE)
- [x] Task 2.1: Fix "Sub-buffer out of bounds" → "GPU memory sub-allocation failed"
- [x] Task 2.2: Fix pool allocation error messages
- [x] Task 2.3: Fix tensor creation error messages

### Phase 3: AttentionError Enum (DEFERRED)
The `AttentionError` enum needs new variants for proper error categorization:
- `MemoryAllocationFailed(String)`
- `MemoryCopyFailed(String)`
- `HandleCreationFailed(String)`

This is tracked as a separate task (requires changing the error enum and updating all call sites).

---

## Progress

**Started**: 2026-01-08
**Last Updated**: 2026-01-08

**BUG-11 Status**: ✅ COMPLETE
The original BUG-11 scope was specifically about GPU memory pool terminology:
- "Sub-buffer out of bounds" → "GPU memory sub-allocation failed"
- "Failed to create tensor '{}' from pool #{}" → "GPU memory pool #{} tensor '{}' creation failed"
- Pool allocation messages → "GPU memory pool #{} allocation failed"

All 4 specific messages from BUG-11 are now standardized.

---

## Deferred Work

1. **AttentionError enum expansion** - Add proper variants for memory/HIP failures
2. **Update all AttentionError::DimensionError usages** - Use correct error types
