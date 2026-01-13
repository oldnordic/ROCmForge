# Phase 25: GQA Architecture Support - COMPLETE

**Date**: 2026-01-13
**Status**: ✅ COMPLETE - GQA (Grouped Query Attention) support implemented, all 24 layers working
**Issue**: Code expected fused QKV attention, but Qwen2 uses separate Q,K,V with GQA (fewer KV heads than query heads)

---

## Executive Summary

Phase 25 discovered and fixed a fundamental architecture mismatch: the code was designed for **fused QKV** attention weights (common in LLaMA models), but Qwen2 uses **separate Q,K,V** weights with **Grouped Query Attention (GQA)** where K/V have fewer heads than Q.

All 24 transformer layers now complete successfully. The remaining hang at LM head matmul is a separate issue.

---

## Root Cause Analysis

### Discovery Process

Using CodeMCP tools (Magellan, find_symbols, discover_summary), the investigation traced the exact data flow:

1. **Initial symptom**: First layer hung in `forward_layer()`
2. **Shape diagnostics**: QKV weight was `[896, 896]` instead of expected `[2688, 896]`
3. **Tensor inspection**: Model had `attn_q.weight`, `attn_k.weight`, `attn_v.weight` (separate), not `attn_qkv.weight` (fused)
4. **Architecture detection**: Qwen2.5-0.5B uses GQA with 14 query heads, 2 KV heads

### The Architecture Mismatch

| Component | Expected (Fused QKV) | Actual (Qwen2 GQA) |
|-----------|---------------------|-------------------|
| Attention weights | `attn_qkv.weight` [2688, 896] | `attn_q.weight` [896, 896] |
| | | `attn_k.weight` [128, 896] |
| | | `attn_v.weight` [128, 896] |
| Num heads | 14 (same for Q,K,V) | Q: 14, K: 2, V: 2 |
| Attention type | Multi-Head Attention (MHA) | Grouped Query Attention (GQA) |

---

## Fixes Implemented

### Fix #1: Tensor Format Detection

**File**: `src/model/execution_plan.rs:526-637`

```rust
// Detect attention tensor format:
let qkv_key = &format!("{}.attn_qkv.weight", prefix);
let q_key = &format!("{}.attn_q.weight", prefix);
let k_key = &format!("{}.attn_k.weight", prefix);
let v_key = &format!("{}.attn_v.weight", prefix);

let has_fused_qkv = lazy_tensors.contains_key(qkv_key);
let has_separate_qkv = has_q && has_k && has_v;

// Store both formats in LayerPlan
pub struct LayerPlan {
    pub qkv_weight: Arc<LazyTensor>,  // Fused format
    pub q_weight: Option<Arc<LazyTensor>>,  // Separate format
    pub k_weight: Option<Arc<LazyTensor>>,
    pub v_weight: Option<Arc<LazyTensor>>,
    // ...
}
```

### Fix #2: Separate QKV Attention Path

**File**: `src/model/execution_plan.rs:1127-1264`

Implemented `self_attention_separate()` function:
1. Projects Q, K, V separately using GPU matmul
2. Reshapes for multi-head format (14 for Q, 2 for K/V)
3. Applies RoPE with different head counts (CPU path for GQA)
4. Performs scaled dot-product attention with KV expansion
5. Projects output

### Fix #3: RoPE for GQA

**Problem**: RoPE function assumed Q and K have same `num_heads`
**Solution**: CPU-side RoPE with separate head dimensions

```rust
// GQA: Q has 14 heads, K has 2 heads
let q_rope = apply_rope_cpu_separate(
    &q_reshaped, positions, num_heads, head_dim, rope_dim
)?;
let k_rope = apply_rope_cpu_separate(
    &k_reshaped, positions, num_kv_heads, head_dim, rope_dim
)?;
```

### Fix #4: KV Cache Skip for GQA

**Problem**: KV cache expected `num_heads=14`, but K/V tensors have `num_kv_heads=2`
**Solution**: Skip KV cache for GQA models (temporary workaround)

```rust
let use_kv_cache = self.cache.is_some()
    && num_kv_heads == num_heads  // Only if MHA, not GQA
    && seq_len > 1;
```

### Fix #5: Attention Kernel KV Expansion

**Problem**: Attention kernels expect same head count for Q,K,V
**Solution**: CPU-side expansion of K/V to match Q's head count

```rust
// For GQA: expand K and V from 2 heads to 14 heads
let heads_per_kv = num_heads / num_kv_heads;  // 14 / 2 = 7
for kv_head in 0..num_kv_heads {
    for q_head_offset in 0..heads_per_kv {
        let target_head = kv_head * heads_per_kv + q_head_offset;
        // Copy KV data to multiple Q heads
    }
}
```

---

## Current Status

### Working (All 24 Layers Complete)

```
>>> decode_step: Layer 24/24 starting...
>>> Layer 24: attn_norm done (1.2ms)
>>> Layer 24: self_attention done (45.3ms)
>>> Layer 24: ffn_norm done (0.8ms)
>>> Layer 24: mlp done (12.1ms)
>>> Layer 24: output_projection done (3.4ms)
>>> decode_step: Layer 24/24 complete (62.8ms)
```

### Remaining Issue: LM Head Hang

After all 24 layers:
```
>>> lm_head(): Getting LM head tensor...
>>> lm_head(): Not cached, loading...
>>> lm_head(): Tensor loaded successfully (519 MB)
>>> apply_lm_head(): Got LM head tensor, calling matmul...
[HANG - matmul ENTRY log never appears]
```

The matmul function is called but the first log statement doesn't execute, suggesting either:
1. Compilation issue (binary not updated)
2. Function prologue issue
3. Borrowing/lifetime problem

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `src/model/execution_plan.rs` | 126-172 | `LayerPlan` struct - added separate Q,K,V fields |
| `src/model/execution_plan.rs` | 526-637 | `create_layer_plan_lazy` - format detection |
| `src/model/execution_plan.rs` | 1127-1264 | `self_attention_separate` - new function |
| `src/model/execution_plan.rs` | 1569-1660 | GQA KV expansion in SDPA |
| `src/model/execution_plan.rs` | 225-259 | LM head diagnostic logging |

---

## Technical Notes

### GQA vs MHA

| Aspect | MHA (Multi-Head) | GQA (Grouped-Query) |
|--------|------------------|---------------------|
| Q heads | 14 | 14 |
| K heads | 14 | 2 |
| V heads | 14 | 2 |
| KV cache size | 14 × seq_len × dim | 2 × seq_len × dim (smaller!) |
| Memory bandwidth | Higher | Lower (more efficient) |
| Quality | Baseline | Slight quality trade-off |

### Why Qwen Uses GQA

1. **Inference efficiency**: Fewer KV heads = less memory for KV cache
2. **Batch processing**: Smaller KV tensors = better cache utilization
3. **Modern architecture**: GQA is standard in newer models (Llama-3, Qwen2, Mistral)

---

## Next Steps

### Option A: Fix LM Head Hang (PRIORITY)
1. Clean rebuild to ensure new logging is included
2. Add logging before matmul call to confirm reachability
3. Investigate if issue is in matmul function or calling code
4. Check for borrow checker / lifetime issues

### Option B: Proper GQA KV Cache (DEFERRED)
Implement GQA-aware KV cache that:
1. Stores K,V at their native head count (2 heads)
2. Expands to Q's head count (14 heads) during attention
3. Reduces KV cache memory by 7× for Qwen2

---

## Test Command

```bash
RUST_LOG=warn timeout 180 ./target/release/rocmforge_cli generate \
  --gguf /home/feanor/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "Hi" --max-tokens 1
```

Expected: All 24 layers complete (~60-80ms per layer), hang at LM head.

---

## References

- **Phase 24**: Vocab size inference fix
- **Phase 23**: hipDeviceSynchronize fix
- **Qwen2 Architecture**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- **GQA Paper**: "GQA: Training Generalized Multi-Query Transformer Models for Fast Inference"
