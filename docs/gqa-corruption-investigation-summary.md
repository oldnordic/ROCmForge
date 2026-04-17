# GQA Corruption Investigation Summary

**Date:** 2026-04-17
**Status:** Root cause NOT found - seq_len was a red herring

## What We Know

**Symptoms:**
- CPU produces correct output: "ertha is"
- GPU non-graph path: "er管辖 lash lash lash"
- GPU graph path: "er,\nup-" (different corruption pattern)

**Investigated:**
1. ✅ KV-cache read/write - working correctly
2. ✅ seq_len values - using pos + 1 (matches CPU)
3. ✅ Code flow - kernel calls in correct order with correct parameters
4. ❌ Root cause - NOT FOUND

## seq_len Investigation (Inconclusive)

**Hypothesis:** seq_len should be pos (causal) instead of pos + 1

**Test Results:**
- seq_len = pos + 1: "er管辖 lash lash lash" (original corruption)
- seq_len = pos: "er,Hëlen" (different corruption)
- CPU uses seq_len = pos + 1: "ertha is" (correct ✓)

**Conclusion:** seq_len value affects output but neither value is correct on GPU. Since CPU uses pos + 1 and works correctly, the bug is NOT in seq_len.

**Reverted:** seq_len back to pos + 1 to match CPU

## Code Flow Analysis

**GQA path (non-graph):**
```
1. RMSNorm
2. QKV GEMV (separate kernels)
3. RoPE on Q
4. RoPE on K
5. KV-write to cache
6. Flash attention from state
7. Output projection GEMV
```

All kernel calls appear correct with proper parameters.

## Remaining Hypotheses

The bug is likely in one of the HIP kernels:

1. **QKV GEMV kernels** - Might compute incorrect Q, K, or V values
2. **RoPE kernels** - Might apply incorrect rotation
3. **Flash attention kernel** - Might compute incorrect attention weights or aggregation
4. **Output projection GEMV** - Might compute incorrect hidden state

**Evidence:**
- Graph and non-graph paths produce DIFFERENT corruption
- This suggests multiple bugs OR bug manifests differently based on timing/synchronization
- Both paths share the same HIP kernels, so bug is likely in kernel implementation

## Systematic Debugging Status

**Principle:** NO FIXES WITHOUT ROOT CAUSE

**Fixes attempted:** 1 (seq_len change - reverted)
**Threshold:** 3+ fixes = question architecture

**Current approach:** Continue investigation to find root cause before attempting more fixes.

## Next Investigation Steps

1. **Compare intermediate values** - Add diagnostics to download and compare Q, K, V, attention weights at each stage
2. **Test with simpler model** - If available, test with non-GQA model to isolate GQA-specific issues
3. **Inspect HIP kernel assembly** - Check for obvious bugs in kernel implementation
4. **Numerical precision analysis** - Check for precision/accuracy issues in GPU computation

## Test Commands

```bash
# CPU baseline
./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 3

# GPU non-graph
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 3

# GPU graph
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 3
```
