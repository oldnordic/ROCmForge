# Architectural Decision: Per-Request Runtime Duplication Fix

**Date**: 2026-01-12
**Status**: IMPLEMENTED
**Phase**: 24

---

## Decision

Separate **engine-scoped** resources (ModelRuntime with GPU buffers) from **request-scoped** state (only logical counters like `processed_tokens`).

**Per-request code must NEVER allocate GPU memory.**

---

## Problem

`RequestRuntimeState` owned a full `ModelRuntime` per request:

```rust
// BEFORE (WRONG)
struct RequestRuntimeState {
    runtime: ModelRuntime,  // ~560-976MB duplicated per request!
    processed_tokens: usize,
}
```

**Impact**: Each request created:
- ScratchBufferManager: ~225-640MB (attention buffer)
- KVCache: ~336MB-4GB (24 layers × 2 buffers)
- Result: Allocator hangs, timeouts, massive memory waste

---

## Solution

```rust
// AFTER (CORRECT)
/// Request-scoped state only - NO GPU resources owned here.
struct RequestRuntimeState {
    processed_tokens: usize,  // Only 8 bytes!
}
```

Per-request state now only tracks logical progress. The shared `model_runtime` at engine level handles all GPU operations.

---

## Files Modified

- `src/engine.rs:81-87` - RequestRuntimeState simplified
- `src/engine.rs:378-402` - ensure_request_state() no longer creates runtime
- `src/engine.rs:627-746` - run_forward_pass() uses shared model_runtime

---

## Alternatives Considered

1. **Add a "verifier"** to check if buffers exist before creating
   - **REJECTED**: Hides the bug, creates implicit coupling, partial initialization states

2. **Use Arc<Mutex<>>** for shared resources
   - **USED**: `Arc<RwLock<ModelRuntime>>` at engine level for async synchronization

---

## Trade-offs

| Trade-off | Impact | Mitigation |
|-----------|--------|------------|
| Single-request serialization | Only one request holds runtime write lock at a time | Acceptable for CLI use case |
| No concurrent multi-request | Cannot easily do parallel inference yet | Future work: paged KV cache |

---

## Validation

CLI inference test confirms:
- ✅ NO duplicate scratch buffer allocation
- ✅ NO duplicate KV cache allocation
- ✅ Only per-request state: `processed_tokens` counter

---

## References

This fix empirically validates why production systems (vLLM, TensorRT-LLM, llama.cpp) use **engine/request separation**:

```
┌─────────────────────────────────────────┐
│         Engine/Model Lifetime          │
│  (GPU-heavy, created once)              │
│  • ScratchBufferManager                 │
│  • KVCache                              │
│  • ExecutionPlan                        │
└─────────────────────────────────────────┘
                    │ shared reference
                    ▼
┌─────────────────────────────────────────┐
│         Request Lifetime               │
│  (lightweight, many)                    │
│  • processed_tokens                     │
│  • token positions                      │
└─────────────────────────────────────────┘
```
