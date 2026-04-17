# GPU Sequential Token Corruption - KV-Cache Hypothesis

**Date:** 2026-04-17
**Status:** NEW HYPOTHESIS - Abnormal Q values affect both tokens equally

## Critical Discovery: Q Values Abnormal for BOTH Tokens

### Diagnostic Data Comparison

**Token pos=1 (produces "er" - CORRECT):**
```
Q[0..10] BEFORE RoPE: [..., -14.440233, ..., -15.445072, -33.985577]
Max abs Q before RoPE: 33.985577
Output: "er" ✓ CORRECT
```

**Token pos=2 (produces garbage - CORRUPTED):**
```
Q[0..10] BEFORE RoPE: [..., -14.4222765, ..., -15.40466, -33.897907]
Max abs Q before RoPE: 33.897907
Output: " " (space) ❌ CORRUPTED
```

**KEY FINDING:** Both tokens have IDENTICAL abnormal Q values (~34.0 magnitude), but only pos=2 produces corrupted output.

## Conclusion: Q Values Are NOT the Root Cause

Since both tokens have the same Q value magnitudes but different outputs:
- ❌ Abnormal Q values are NOT causing the corruption
- ✅ The model can handle abnormal Q values correctly for pos=1
- ❌ Something ELSE is going wrong for pos=2

## New Hypothesis: KV-Cache Corruption

### Why KV-Cache is the Likely Culprit:

**Token pos=1 (first decode step):**
- KV-cache is EMPTY
- Attention computes Q·K^T where K is from CURRENT token only
- Even with abnormal Q, if current K is also abnormal in the same way, the ratio Q·K might still work
- Output: CORRECT ✓

**Token pos=2 (second decode step):**
- KV-cache contains K/V from pos=1
- Attention computes Q·K^T where K includes CACHED values from pos=1
- If cached K values are wrong (quantization error, write bug, memory corruption), the attention scores are wrong
- Output: CORRUPTED ❌

### Specific KV-Cache Issues to Investigate:

1. **KV-Cache Write Error (pos=1):**
   - K/V values written to cache for pos=1 might be incorrect
   - Quantization error during cache write
   - Memory layout issue (wrong stride/indexing)

2. **KV-Cache Read Error (pos=2):**
   - K/V values read from cache for pos=1 might be incorrect
   - Reading from wrong memory location
   - Incorrect cache indexing for GQA (grouped query attention)

3. **Quantization Mismatch:**
   - K/V are stored in Q4_0 format in cache
   - Dequantization during attention read might have bugs
   - Scale factors might be wrong

4. **GQA-Specific Issues:**
   - Qwen2.5-0.5B uses GQA (14 query heads, 2 KV heads)
   - KV heads are shared across multiple query heads
   - Cache indexing for GQA might be wrong

## Next Investigation Steps

### Priority 1: Verify KV-Cache Write (pos=1)
1. Add diagnostic to download K/V values AFTER they're written to cache
2. Compare cached K/V values with local K/V values (before cache write)
3. Check if cache write is corrupting the values

### Priority 2: Verify KV-Cache Read (pos=2)
1. Add diagnostic to download K/V values read from cache for pos=1
2. Compare cached values with what was written in step 1
3. Check if cache read is returning wrong values

### Priority 3: Verify Attention Computation
1. Add diagnostic to check attention scores (Q·K^T / sqrt(d))
2. Compare attention weights between CPU and GPU
3. Check if softmax output is correct

## Test Commands

```bash
# Test with KV-cache diagnostics
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 2

# CPU baseline for comparison
./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 2
```

## Conclusion

**PARADIGM SHIFT:** The root cause is NOT in Q projection (both tokens affected equally). The issue is likely in KV-cache operations (affects only pos=2 which relies on cached values).

**NEXT STEP:** Add diagnostics to verify KV-cache write/read operations to identify where corruption occurs.
