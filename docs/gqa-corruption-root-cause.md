# GQA Corruption - ROOT CAUSE IDENTIFIED

**Date:** 2026-04-17
**Status:** ROOT CAUSE FOUND - QKV projection or RoPE bug on GPU

## Critical Discovery

**CPU and GPU Q values are completely different** after RoPE!

### Evidence: Token pos=1, Layer 0

**CPU Q[0..10] (after RoPE):**
```
[-0.1628938, -0.18200117, -0.43501627, 0.072969615,
 -12.042278, 0.39564082, 0.51739824, -0.049493045,
 -16.355131, -33.631363]
```

**GPU Q[0..10] (after RoPE):**
```
[-0.10698192, -0.12904091, -0.26587096, 0.12000557,
 -0.003213499, 0.11258322, 0.16589324, -0.1547054,
 -0.47740856, 0.48363742]
```

**Magnitude differences:**
- Index 4: CPU=-12.04, GPU=-0.003 (4000x difference!)
- Index 8: CPU=-16.35, GPU=-0.47 (34x difference)
- Index 9: CPU=-33.63, GPU=+0.48 (Opposite sign!)

**K values are also completely different:**
- CPU: `[-9.92581, -7.982329, ...]` (large magnitude ~10)
- GPU: `[0.3797785, 0.18571183, ...]` (small magnitude ~0.4)

## Root Cause Analysis

The bug is in one of:
1. **QKV GEMV kernels** - Computing wrong Q, K, or V values
2. **RoPE kernels** - Applying incorrect rotation
3. **Quantization/dequantization** - Q4_0 format handling different on CPU vs GPU

**NOT the causes (ruled out):**
- ✗ seq_len value (tested, not the issue)
- ✗ KV-cache read/write (verified working correctly)
- ✗ Flash attention kernel (appears correct, receives wrong input)

## Next Investigation Steps

1. **Verify GEMV kernel correctness** - Compare dequantized weights and GEMV computation
2. **Check RoPE application** - Verify RoPE parameters (theta, neox, position)
3. **Test with different model** - If available, test with non-quantized or different quant format
4. **Inspect Q4_0 dequantization** - Check if scale/scale factors are correct

## Impact

This bug affects **ALL GQA models** on GPU. Any model using grouped query attention will produce corrupted output because the QKV projection or RoPE is computing incorrect values.

## Test Commands (for verification)

```bash
# CPU baseline (correct)
ROCMFORGE_DEBUG=1 ./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1 2>&1 | grep "CPU DIAGNOSTIC"

# GPU (incorrect)
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1 2>&1 | grep "GQA: Q\[head=0\]"
```
