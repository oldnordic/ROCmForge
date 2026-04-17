# GQA Corruption - seq_len Fix (Incomplete)

**Date:** 2026-04-17
**Status:** seq_len bug fixed, but corruption persists

## Bug Fixed: Incorrect seq_len

**Root Cause:** `upload_decode_state(pos, pos + 1, ...)` was uploading wrong seq_len value for causal attention.

**Problem:** When processing token at position `pos`, attention should only attend to previous tokens at positions `[0, pos)`, NOT include position `pos` itself.

**Fix Applied:**
```rust
// Before (WRONG):
scratch.upload_decode_state(pos, pos + 1, device.stream())?;

// After (CORRECT):
scratch.upload_decode_state(pos, pos, device.stream())?;
```

**Locations Fixed:**
1. `/home/feanor/Projects/rocmforge/src/gpu/forward.rs:1773` - Non-graph decode path
2. `/home/feanor/Projects/rocmforge/src/gpu/forward.rs:1258` - Graph decode path

## Verification

**seq_len values now correct:**
- pos=1: seq_len=1 ✓ (attend to position 0 only)
- pos=2: seq_len=2 ✓ (attend to positions 0 and 1)
- pos=3: seq_len=3 ✓ (attend to positions 0, 1, and 2)

**Attention outputs now differ:**
- pos=1: `[0.008067593, 0.02008115, ...]`
- pos=2: `[0.029841192, -0.017958924, ...]` (DIFFERENT from pos=1 ✓)

**Q values now normal:**
- pos=1: `[-0.10698192, -0.12904091, ...]` (max magnitude ~0.48 ✓)
- pos=2: `[-0.24421328, -0.2358655, ...]` (max magnitude ~2.05 ✓)
- Before fix: Q values had extreme magnitudes (-34.1, -15.9)

## Remaining Issue

**Output still corrupted:**
- CPU (correct): "ertha is"
- GPU non-graph: "er,Hëlen" (corrupted)
- GPU graph: "er,\nup-" (corrupted, different pattern)

**Analysis:**
- First two characters "er" are correct (first generated token)
- Subsequent characters are wrong
- Both graph and non-graph paths produce corruption (different patterns)
- Q values are now normal (seq_len fix resolved the abnormal Q magnitudes)
- Attention outputs differ between positions (seq_len fix resolved the identical output issue)

## Hypothesis

The seq_len bug was masking the real issue, not causing it directly. Now that seq_len is fixed, we can see that:

1. **QKV projection or RoPE may have issues** - Q values are now reasonable but may still be incorrect
2. **Attention computation may have bugs** - Attention outputs differ but may be computing wrong values
3. **GQA-specific logic may be wrong** - The 7:1 grouping ratio may have implementation bugs

## Next Investigation Steps

1. **Compare attention weights** - Download and compare attention weights between CPU and GPU for same input
2. **Verify K/V values in cache** - Ensure cached K/V values are correct for all positions
3. **Check GQA grouping logic** - Verify query head → KV head mapping is correct
4. **Test with MHA model** - If available, test with a model using MHA instead of GQA to isolate GQA-specific issues

## Test Commands

```bash
# Non-graph path
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# Graph path
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# CPU baseline
./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 5
```
