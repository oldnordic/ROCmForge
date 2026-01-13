# ROCmForge Quick Restart - 2026-01-13

## Current State

**Phase**: 25 COMPLETE (GQA Architecture Support) → Phase 26 (LM Head Matmul Hang)

---

## Phase 25 COMPLETE ✅

### What Was Accomplished

**Root Cause Discovered** (using CodeMCP tools):
- Code expected: Fused QKV attention `attn_qkv.weight` [2688, 896] (LLaMA-style)
- Model had: Separate Q,K,V with GQA:
  - `attn_q.weight` [896, 896] - 14 query heads
  - `attn_k.weight` [128, 896] - 2 KV heads
  - `attn_v.weight` [128, 896] - 2 KV heads

**Fixes Applied**:
1. ✅ Tensor format detection - `create_layer_plan_lazy` detects both fused and separate formats
2. ✅ Separate QKV attention path - New `self_attention_separate()` function
3. ✅ RoPE for GQA - CPU-side with separate head counts (14 for Q, 2 for K/V)
4. ✅ KV cache skip - Temporary workaround for incompatible cache
5. ✅ Attention kernel KV expansion - CPU-side 2→14 head expansion

**Result**: All 24 transformer layers complete successfully (~60-80ms per layer)

---

## Phase 26: LM Head Matmul Hang 🔍

### Current Bug Symptom

After all 24 layers complete:
```
>>> decode_step: Layer 24/24 complete (62.8ms)
>>> lm_head(): Getting LM head tensor...
>>> lm_head(): Not cached, loading...
>>> lm_head(): Tensor loaded successfully (519 MB)
>>> apply_lm_head(): Got LM head tensor, calling matmul...
[HANG - matmul ENTRY log never appears]
```

### Analysis

The matmul function is called but the first log statement inside doesn't execute. Possible causes:
1. Binary not updated with new logging (need clean rebuild)
2. Function prologue issue
3. Borrowing/lifetime problem preventing entry

---

## Files Modified (Phase 25)

| File | Lines | Change |
|------|-------|--------|
| `src/model/execution_plan.rs` | 126-172 | `LayerPlan` struct - separate Q,K,V fields |
| `src/model/execution_plan.rs` | 526-637 | `create_layer_plan_lazy` - format detection |
| `src/model/execution_plan.rs` | 1127-1264 | `self_attention_separate()` function |
| `src/model/execution_plan.rs` | 1569-1660 | GQA KV expansion in SDPA |
| `src/model/execution_plan.rs` | 225-259 | LM head diagnostic logging |

---

## Test Command

```bash
RUST_LOG=warn timeout 180 ./target/release/rocmforge_cli generate \
  --gguf /home/feanor/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "Hi" --max-tokens 1
```

Expected: All 24 layers complete (~1.5-2 seconds total), hang at LM head matmul.

---

## Next Steps (Phase 26)

1. **Clean rebuild** - Ensure new logging code is included
2. **Add pre-matmul logging** - Confirm function is reachable
3. **Investigate matmul entry** - Check for prologue/lifetime issues
4. **Verify hipBLAS state** - Check if hipBLAS handle is valid for 519MB tensor

---

## Documentation Updated

- `docs/PHASE_25_STATUS_2026-01-13.md` - Complete Phase 25 report
- `docs/CHANGELOG.md` - Phase 25 entry added
- `docs/TODO.md` - Updated with Phase 25/26 status
- `docs/QUICK_RESTART_2026-01-13.md` - This file

---

## CodeMCP Tools Used

During Phase 25 investigation:
- `find_symbols` - Located exact byte spans for QKV loading code
- `discover_summary` - Understood function purposes before editing
- `magellan_init` - Built code graph for impact analysis
- `label_symbols` - Found all attention-related symbols

These tools enabled **evidence-based debugging** instead of trial-and-error.
