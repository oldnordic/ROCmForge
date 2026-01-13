# ROCmForge Quick Restart - 2026-01-12

**SUPERSEDED**: See `QUICK_RESTART_2026-01-13.md` for latest status.

**Phase 25 Status**: ✅ COMPLETE - See new file for details.

---

## Archive (Historical)

### What Was Accomplished

1. **Phase 24 COMPLETE** ✅
   - Fixed ownership boundary bug (per-request runtime duplication)
   - No more duplicate scratch buffer/KV cache allocation
   - Model loads successfully

2. **Phase 25 Started** 🔄
   - Discovered GGUF stores embeddings TRANSPOSED: `[hidden_size, vocab_size]` instead of `[vocab_size, hidden_size]`
   - Implemented transpose fix for `token_embd.weight`, `lm_head.weight`, `output.weight`
   - Embedding lookup now passes validation

3. **New Issue Discovered** ❌
   - First layer `forward_layer()` HANGS after weights load
   - NOT a loading issue - computation hangs
   - Added timing logs to `decode_step()` to track progress

---

## Current Bug Symptom

```
>>> decode_step: Layer 1/24 starting...
>>> [all 7 weights for layer 0 load successfully]
>>> [HANG - never see "Layer 1/24 complete"]
```

The hang is in `ExecutionPlan::forward_layer()` computation, NOT in weight loading.

---

## ROOT CAUSE IDENTIFIED (2026-01-13)

**QKV weight shape corruption bug**:

| Stage | Shape | Status |
|-------|-------|--------|
| GGUF file | `[2688, 896]` | ✅ Correct |
| `load_tensor_to_gpu()` | `[2688, 896]` | ✅ Correct |
| `matmul()` receives | `[896, 896]` | ❌ **CORRUPTED** |

**Why it hangs**: Matmul with wrong shape produces 896 elements instead of 2688. QKV extraction tries to read K from offset 896, but buffer only has 896 elements → `hipMemcpy` blocks indefinitely.

---

## Files Modified

| File | Change |
|------|--------|
| `src/loader/gguf.rs:901-921` | Embedding transpose logic |
| `src/backend/hip_backend.rs:2510-2571` | Layer timing logs |
| `src/model/execution_plan.rs:664-723` | Enhanced embedding_lookup logging |
| `docs/TODO.md` | Updated with Phase 25 status |
| `docs/PHASE_25_STATUS_2026-01-12.md` | Detailed investigation notes |
| `docs/ARCH_DECISION_2026-01-12_GGUF_SHAPE.md` | Architectural decision record |

---

## Next Steps (When You Return)

1. **Find where QKV weight shape gets corrupted**
   - Trace shape from `load_tensor_to_gpu()` → `LazyTensor` → `matmul()`
   - Read `DeviceTensor::from_host_vec()` implementation
   - Check for `.0` vs `.shape()` bugs
   - Add shape logging at each transition

2. **Fix the shape corruption**
   - Once found, fix the shape handling
   - Add assertion to catch this earlier
   - Test with full model

3. **Verify other weights not affected**
   - MLP weights (gate, up, down)
   - Output projection
   - Layer norm weights

---

## Key Code Locations

- `forward_layer()`: `src/model/execution_plan.rs` - search for `pub fn forward_layer`
- `decode_step()`: `src/backend/hip_backend.rs:2510`
- `forward_layer` calls: attention, MLP, layer norm

---

## Test Command

```bash
RUST_LOG=warn timeout 60 ./target/release/rocmforge_cli generate \
  --gguf /home/feanor/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "Hi" --max-tokens 1
```

Expected: Should see layer timing logs. Currently hangs at "Layer 1/24 starting...".

---

## Documentation to Read

1. `docs/PHASE_25_STATUS_2026-01-12.md` - Detailed investigation
2. `docs/ARCH_DECISION_2026-01-12_GGUF_SHAPE.md` - Transpose decision
3. `docs/TODO.md` - Overall project status
