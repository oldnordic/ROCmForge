# GPU Sequential Token Corruption - LM Head Bug Fixed

**Date:** 2026-04-17  
**Status:** CRITICAL BUG FIXED - LM head projection was being skipped  
**Remaining:** Corruption still present in both graph and non-graph paths

## Critical Bug Found and Fixed

### Bug #5: LM Head Projection Skipped in Non-Graph Path
**Location:** `src/gpu/forward.rs` line 1626

**Problem:** 
- Non-graph path was calling `gpu_try_greedy_decode_graph()` which has type signature `-> GpuResult<u32>`
- This function returns early when graphs are disabled (line 413-414)
- The non-graph path expected `-> GpuResult<Option<u32>>` and used `.map(Some)`
- Result: LM head projection was COMPLETELY SKIPPED in non-graph GreedyArgmax mode

**Root Cause:**
When `ROCMFORGE_DISABLE_DECODE_GRAPH=1` with `GreedyArgmax`:
1. `gpu_try_full_greedy_decode_graph()` returns `Ok(None)` (line 1224)
2. Code falls through to LM head projection section (line 1595-1629)
3. `gpu_try_greedy_decode_graph()` was called instead of `gpu_greedy_logits_tail_token()`
4. `gpu_try_greedy_decode_graph()` returns early without doing GPU computation
5. **LM HEAD PROJECTION WAS NEVER EXECUTED**

**Fix Applied:**
```rust
// Before (BROKEN):
GpuLogitsMode::GreedyArgmax => {
    gpu_try_greedy_decode_graph(device, gpu_weights, scratch, config).map(Some)
}

// After (FIXED):
GpuLogitsMode::GreedyArgmax => {
    gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config).map(Some)
}
```

**Impact:** 
- Non-graph path now properly computes LM head projection on GPU
- Token generation should use correct logits instead of stale/uninitialized data

## Current Status (After LM Head Fix)

### Non-Graph Path (ROCMFORGE_DISABLE_DECODE_GRAPH=1)
**Token 1:** Correct ✅  
**Token 2+:** Still corrupted ❌

**Example:**
- CPU: "ertha"
- GPU (non-graph): "er " (space instead of "tha")

### Graph Path (default)
**Token 1:** Correct ✅  
**Token 2+:** Still corrupted ❌

**Example:**
- CPU: "ertha"
- GPU (graph): "er失" (Chinese character instead of "tha")

**Model Architecture Confirmed:**
- Using GQA path: q_size=896, kv_size=128, head_dim=64
- Confirms Qwen2.5-0.5B uses GQA (14 query heads, 2 KV heads)

## All Bugs Fixed So Far

1. **GQA RoPE in ops.rs** - Changed to use GPU state pointer
2. **MHA/GQA RoPE in non-graph path** - Changed to use GPU state pointer  
3. **Attention in non-graph path** - Changed to use GPU state pointer
4. **Missing decode state upload** - Added upload to non-graph path
5. **LM head projection skipped** - Fixed to use correct GPU function

## Remaining Issues

Despite fixing 5 major bugs, corruption persists in both paths for token 2+. This suggests:

### Possible Remaining Root Causes:

1. **KV-cache corruption**
   - KV values may be written incorrectly for token 2+
   - Cache indexing may be wrong for sequential positions
   - Memory layout issues in strided cache access

2. **RoPE still incorrect**
   - Despite using state pointers, RoPE may still be computing wrong rotations
   - RoPE kernel itself may have bugs
   - Position values may not be propagating correctly

3. **Hidden state corruption**
   - Layer outputs may be corrupted between layers
   - Residual connections may have issues
   - Normalization may be incorrect

4. **Attention computation bugs**
   - Attention scores may be computed incorrectly
   - Softmax may have numerical issues
   - Value aggregation may be wrong

5. **Synchronization issues**
   - Kernels may be executing in wrong order
   - Memory barriers may be missing
   - Stream synchronization may be insufficient

## Next Investigation Steps

1. **Verify KV-cache write/read operations**
   - Check if KV values are written correctly for position 1, 2, 3, etc.
   - Verify cache indexing matches expected layout
   - Examine memory strides in cache access patterns

2. **Debug RoPE computation**
   - Add debug output to verify RoPE is using correct position
   - Check RoPE kernel for numerical precision issues
   - Verify RoPE is applied to both Q and K correctly

3. **Compare intermediate values**
   - Dump Q, K, V values for token 1 vs token 2
   - Compare attention scores between CPU and GPU
   - Check softmax output for correctness

4. **Verify attention computation**
   - Check if attention weights sum to 1.0
   - Verify context aggregation is correct
   - Examine value cache reading

## Test Commands

```bash
# Test non-graph path
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# Test graph path  
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# CPU baseline
./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 5
```

## Conclusion

Fixed 5 major bugs related to GPU state management and LM head projection. However, corruption persists for token 2+ generation, indicating additional issues in the attention pipeline, KV-cache operations, or RoPE computation. Further systematic debugging of intermediate values is needed to isolate the remaining root cause(s).
