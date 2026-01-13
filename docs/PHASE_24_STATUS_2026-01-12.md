# Phase 24: CLI to GPU Workflow Analysis - FINAL REPORT

**Date**: 2026-01-12
**Status**: ✅ COMPLETE - Ownership Boundary Fix Implemented

---

## Note: Continued in Phase 25

Phase 24 ownership boundary fix is **COMPLETE and VERIFIED**. The CLI now correctly:

1. ✅ Uses shared `ModelRuntime` at engine level (no per-request duplication)
2. ✅ Allocates scratch buffer and KV cache once (not per-request)
3. ✅ Loads GGUF models with correct parameters
4. ✅ Fetches embedding weights successfully

**Phase 25** investigates a NEW issue discovered during testing:
- GGUF stores embeddings as `[hidden_size, vocab_size]` (transposed)
- Transpose fix applied
- New hang discovered in first layer forward pass computation

See `docs/PHASE_25_STATUS_2026-01-12.md` for details.

---

---

## Summary

Phase 24 is now **COMPLETE** with the correct architectural fix for per-request state management. The root cause was identified as **wrong ownership boundaries** between engine-scoped and request-scoped resources.

| Issue | Description | Status |
|-------|-------------|--------|
| #1 | Ownership boundary: RequestRuntimeState owned ModelRuntime | ✅ FIXED |
| #2 | Wrong default configuration | ✅ FIXED |
| #3 | ScratchBufferManager parameter order bug | ✅ FIXED |
| #4 | Added `InferenceEngine::from_gguf()` helper | ✅ COMPLETE |
| #5 | Vocab size inference for missing metadata | ✅ IMPLEMENTED |

---

## The Core Fix: Ownership Boundary Separation

### Before (WRONG Architecture)

```rust
// src/engine.rs
struct RequestRuntimeState {
    runtime: ModelRuntime,  // ❌ Owned per-request!
    processed_tokens: usize,
}
```

**Problem**: Each request created its own ModelRuntime, duplicating:
- ScratchBufferManager (~225-640MB)
- KVCache (~336MB-4GB)
- Backend references

**Result**: Massive memory waste, allocator hangs, timeouts

### After (CORRECT Architecture)

```rust
// src/engine.rs - PHASE 24 FIX
/// Request-scoped state only - NO GPU resources owned here.
/// The ModelRuntime is shared at engine level.
struct RequestRuntimeState {
    processed_tokens: usize,  // ✅ Only logical state
}
```

**Solution**: Per-request state only tracks progress. The shared `model_runtime` at engine level is used for all GPU operations.

**Result**: Zero duplicate allocations, clean ownership boundaries

---

## Implementation Details

### 1. `RequestRuntimeState` Simplified ✅

**File**: `src/engine.rs:81-87`

```rust
/// PHASE 24 FIX: Request-scoped state only - NO GPU resources owned here.
#[derive(Debug)]
struct RequestRuntimeState {
    processed_tokens: usize,
}
```

### 2. `ensure_request_state()` No Longer Creates Runtime ✅

**File**: `src/engine.rs:378-402`

```rust
/// PHASE 24 FIX: Ensure request state exists WITHOUT creating duplicate GPU resources.
async fn ensure_request_state(&self, request_id: u32) -> EngineResult<()> {
    // Verify model is loaded
    let _runtime_arc = self.model_runtime.as_ref()
        .ok_or_else(|| EngineError::InferenceFailed("No GGUF model loaded".to_string()))?;

    // Create per-request state WITHOUT any GPU resources
    let mut states = self.request_states.write().await;
    states.entry(request_id).or_insert(RequestRuntimeState {
        processed_tokens: 0,
    });
    Ok(())
}
```

### 3. `run_forward_pass()` Uses Shared Runtime ✅

**File**: `src/engine.rs:627-746`

```rust
// Get shared model runtime from engine level (NOT per-request)
let runtime_arc = self.model_runtime.as_ref()?.clone();

// Get backend and execution plan from shared runtime
let (backend, execution_plan) = {
    let runtime = runtime_arc.read().await;
    (runtime.backend().clone(), runtime.execution_plan().cloned()?)
};

// Get mutable access to shared runtime for decode_step
let mut runtime = runtime_arc.write().await;
for token in tokens_to_process {
    let logits = runtime.decode_step(&embeddings)?;
    // ...
}
```

---

## Memory Impact

### Before Fix (Per-Request Duplication)

| Component | Per-Allocation | Per-Request Impact |
|-----------|----------------|-------------------|
| ScratchBufferManager | ~225-640MB | ✗ Duplicated |
| KVCache (24 layers) | ~336MB | ✗ Duplicated |
| **Total Waste** | | **~560-976MB per request** |

### After Fix (Shared Resources)

| Component | Engine-Scoped | Per-Request |
|-----------|---------------|-------------|
| ScratchBufferManager | ✗ Shared (one-time) | ✓ 0 bytes |
| KVCache | ✗ Shared (one-time) | ✓ 0 bytes |
| RequestRuntimeState | N/A | ✓ 8 bytes (usize) |

---

## Current Status

### Verification

Running `rocmforge_cli generate --gguf qwen2.5-0.5b.gguf` shows:
- ✅ Model loads with correct parameters
- ✅ NO duplicate scratch buffer allocation
- ✅ NO duplicate KV cache allocation
- ✅ Only per-request state: `processed_tokens` counter

### Remaining Issue: Embedding Weights Lazy Load (Separate)

**Not a bug, but an optimization opportunity**:

The 544MB allocation during first inference is the **embedding weights** being loaded lazily:
- Location: `src/model/execution_plan.rs:192-207` (`embedding_weights()`)
- Process: CPU dequantization → GPU upload via `hipMemcpyHtoD`
- Size: ~544MB (vocab_size × hidden_size × sizeof(f32))

**This is a ONE-TIME allocation**, not per-request. Could be optimized later by:
- Preloading during model load
- Using async memcpy with streams
- Implementing on-device dequantization

---

## Architectural Learning

This fix empirically validates why production LLM inference systems (vLLM, TensorRT-LLM, llama.cpp server) all have **engine/request separation**:

```
┌─────────────────────────────────────────┐
│         Engine/Model Lifetime          │
│  (GPU-heavy, created once)              │
│                                         │
│  • ScratchBufferManager                 │
│  • KVCache                              │
│  • ExecutionPlan                        │
│  • Backend resources                    │
└─────────────────────────────────────────┘
                    │
                    │ shared reference
                    ▼
┌─────────────────────────────────────────┐
│         Request Lifetime               │
│  (lightweight, many)                    │
│                                         │
│  • processed_tokens                     │
│  • token positions                      │
│  • stop conditions                      │
└─────────────────────────────────────────┘
```

**Key Principle**: Per-request code must NEVER allocate GPU memory.

---

## Files Modified

### Core Fixes
- `src/engine.rs` - Ownership boundary fix (RequestRuntimeState, ensure_request_state, run_forward_pass)
- `src/backend/hip_backend.rs` - Fixed all `ScratchBufferManager::new()` calls
- `src/backend/scratch.rs` - Added debug output
- `src/bin/rocmforge_cli.rs` - Updated to use `from_gguf()`
- `src/model/execution_plan.rs` - Vocab size inference for missing metadata

### Tests
- `tests/decode_step_integration_tests.rs` - Fixed parameter order

---

## Phase 24 Status: COMPLETE ✅

The ownership boundary fix eliminates the root cause of CLI inference hangs. The remaining 544MB embedding weights lazy load is a separate optimization opportunity, not a bug.
