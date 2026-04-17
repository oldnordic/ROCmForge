# GPU Sequential Token Corruption - Partial Fix Applied

**Date:** 2026-04-17
**Status:** PARTIALLY FIXED - Non-graph path improved, graph path still broken
**Root Cause:** Multiple bugs in RoPE and attention kernels using baked-in parameters instead of GPU state pointers

## Bugs Fixed

### 1. GQA RoPE in ops.rs (Line 887-888)
**Problem:** `gpu_dispatch_fused_qkv_gqa_on_stream` used `rope_heads_on_stream` with CPU-side `pos` parameter
**Fix:** Changed to `rope_heads_from_state_on_stream` with GPU state pointer
**Impact:** GQA graph replay and non-graph paths

### 2. MHA/GQA RoPE in non-graph path (Lines 1371-1392)
**Problem:** `gpu_layer_forward_hybrid` used `rope_heads_on_stream` with CPU-side `pos` parameter
**Fix:** Changed to `rope_heads_from_state_on_stream` with GPU state pointer
**Impact:** Non-graph fallback path for both MHA and GQA

### 3. Attention in non-graph path (Line 1410)
**Problem:** `gpu_layer_forward_hybrid` used `gpu_attention_decode` with baked-in seq_len parameter
**Fix:** Changed to `gpu_attention_decode_from_state` with GPU state pointer
**Impact:** Non-graph fallback path attention computation

### 4. Missing decode state upload in non-graph path
**Problem:** Non-graph path didn't upload decode state before token generation
**Fix:** Added `scratch.upload_decode_state(pos, pos + 1, device.stream())?` before layer loop
**Impact:** Ensures GPU memory has correct position/seq_len values

## Current Status

### Non-Graph Path (ROCMFORGE_DISABLE_DECODE_GRAPH=1)
**Token 1:** Correct ✅
**Token 2+:** Still corrupted ❌ (but improved from before)

**Example:**
- CPU: "ertha"
- GPU (non-graph): "er " (space instead of "tha")

### Graph Path (default)
**Token 1:** Correct ✅
**Token 2+:** Still corrupted ❌

**Example:**
- CPU: "ertha"
- GPU (graph): "er失" (Chinese character instead of "tha")

## Root Cause Analysis Summary

The bug was caused by HIP graph capture "baking in" kernel parameters during graph capture with token 1 (pos=0, seq_len=1). When replaying for tokens 2+, the kernels continued using the baked-in values instead of reading updated GPU state.

**Affected Operations:**
1. RoPE application - position parameter baked in
2. Attention computation - seq_len parameter baked in
3. Decode state - not uploaded in non-graph path

**Why Token 1 Works:**
- Position 0 is correct for first token
- Baked-in values match actual values
- All kernels operate correctly

**Why Token 2+ Fails:**
- Should use position 1, 2, 3, etc.
- But kernels still use position 0 (baked in)
- Incorrect RoPE rotation → wrong attention scores → wrong tokens

## Next Steps

### Immediate Actions
1. Investigate remaining corruption in non-graph path
   - Verify all GPU state pointers are correct
   - Check for other kernels using baked-in parameters
   - Examine KV-cache read/write operations

2. Debug graph path corruption
   - Verify graph capture uses correct "from_state" kernels
   - Check if graph replay is using captured kernels correctly
   - Investigate synchronization issues in graph replay

### Debug Commands
```bash
# Test non-graph path
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /path/to/model.gguf --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# Test graph path
./target/release/rocmforge --gpu \
  --model /path/to/model.gguf --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# Compare with CPU baseline
./target/release/rocmforge \
  --model /path/to/model.gguf --prompt "The" --no-template --top-p 1.0 --max-tokens 5
```

## Files Modified

1. `src/gpu/ops.rs` - Fixed GQA RoPE to use GPU state pointer
2. `src/gpu/forward.rs` - Fixed MHA/GQA RoPE and attention to use GPU state pointers
3. `src/gpu/forward.rs` - Added decode state upload to non-graph path
4. `src/gpu/cache.rs` - Added debug output for decode state upload

## Verification

Decode state upload is working correctly:
```
[DEBUG] upload_decode_state: pos=0, seq_len=1, state[0]=0, state[1]=1
[DEBUG] upload_decode_state: pos=1, seq_len=2, state[0]=1, state[1]=2
[DEBUG] upload_decode_state: pos=2, seq_len=3, state[0]=2, state[1]=3
```

This confirms GPU memory is receiving correct values, but corruption persists.

## Hypothesis for Remaining Issues

### Non-Graph Path
- Possible issue with KV-cache indexing or memory layout
- Potential numerical precision differences in attention computation
- Race condition or synchronization issue between kernels

### Graph Path
- Graph may not be capturing the correct "from_state" kernel variants
- Graph replay may have issues with state pointer handling
- Possible graph invalidation or cache coherency issue

## Conclusion

Significant progress made - identified and fixed 4 major bugs related to GPU state management. However, corruption persists in both graph and non-graph paths for token 2+ generation. Further investigation needed to identify remaining root cause(s).
