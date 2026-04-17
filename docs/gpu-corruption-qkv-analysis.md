# GPU Sequential Token Corruption - QKV Analysis

**Date:** 2026-04-17
**Status:** CRITICAL FINDING - Abnormal Q values causing numerical instability

## Critical Discovery: Extreme Q Value Magnitudes

### Diagnostic Data from Token 2 Generation (Non-Graph Path)

**Input Hidden State (from token 1):**
```
[0.0, 0.00026154518, 0.013861895, 0.011507988, 0.0052309036]
Sum: 0.030862331
```
✅ **Normal** - Small magnitude values as expected

**Q Values after QKV projection + RoPE:**
```
[-0.065522365, -0.0589669, -0.1711932, -0.057182737,
 -12.044628, 0.2620814, 0.35846516, 0.12859204,
 -15.881239, -34.104332]
```
❌ **ABNORMAL** - Values up to -34.1 magnitude are EXTREMELY large
❌ **Inconsistent** - Some values normal (around 0.1), others catastrophically large (-34.1)

**K Values after QKV projection + RoPE:**
```
[-8.804068, -3.4432926, -6.238553, 0.6017397,
 -0.14521708, 9.352216, 8.023085, -1.3291036,
 -0.12585078, -0.24078582]
```
⚠️ **Large but plausible** - K values have larger magnitude than typical (up to 9.35)

**V Values:**
```
[0.0012399361, 0.009329853, -0.0040042875, 0.0032822215,
 0.008135275, -0.0065581333, -0.019629223, 0.0019433857,
 -0.014660343, 0.015207353]
```
✅ **Normal** - Small magnitude values as expected

**Final Hidden State (before LM head):**
```
[-1.35694, -0.30211258, 0.4461969, -2.671262, 0.73523724]
Sum: 5.5117483
```
⚠️ **Enlarged** - Magnitude has grown from 0.03 to 5.5 (183x increase)

## Root Cause Hypothesis

The extreme Q values (-34.1, -15.9, -12.0) are causing:
1. **Numerical instability** in attention score computation (Q·K^T)
2. **Incorrect softmax** due to extreme values dominating
3. **Wrong attention weights** → wrong context aggregation → corrupted output

## Possible Sources of Abnormal Q Values

### 1. Incorrect QKV Projection Computation
- **Hypothesis:** GEMV kernel computing Q projection has bugs
- **Evidence:** Only Q is affected, K and V are normal
- **Test:** Compare Q values between CPU and GPU for same input

### 2. RoPE Application Error
- **Hypothesis:** RoPE rotation is corrupting Q values
- **Evidence:** Abnormal values appear AFTER RoPE (not before)
- **Test:** Check Q values before and after RoPE

### 3. Quantization Issues
- **Hypothesis:** Q4_0 quantization/dequantization errors
- **Evidence:** Using Q4_0 quantized model
- **Test:** Compare with unquantized model or different quant format

### 4. Memory Corruption
- **Hypothesis:** Buffer overlap or incorrect memory access
- **Evidence:** Inconsistent pattern (some normal, some extreme)
- **Test:** Check memory layout and buffer sizes

## Current Status

### Bugs Fixed So Far (6 total):
1. ✅ GQA RoPE in ops.rs - Use GPU state pointer
2. ✅ MHA/GQA RoPE in non-graph path - Use GPU state pointer
3. ✅ Attention in non-graph path - Use GPU state pointer
4. ✅ Missing decode state upload - Added to non-graph path
5. ✅ LM head projection skipped - Fixed function call
6. ✅ MHA path RoPE - Use GPU state pointer (just fixed)

### Remaining Issues:
- ❌ **Q values have extreme magnitudes (-34.1, -15.9, -12.0)**
- ❌ **Corruption persists in both graph and non-graph paths**
- ❌ **Token 2+ generation produces garbage output**

## Next Investigation Steps

### Immediate Priority: Verify QKV Projection Correctness
1. Add diagnostic to check Q values **BEFORE RoPE** (after GEMV only)
2. Compare with CPU Q values for same input
3. If pre-RoPE Q is abnormal → GEMV bug
4. If pre-RoPE Q is normal → RoPE bug

### Secondary Investigation:
1. Check if K and V values are correct (compare with CPU)
2. Verify attention score computation (Q·K^T)
3. Check softmax output for numerical issues
4. Verify final attention aggregation

## Test Commands

```bash
# Non-graph path with diagnostics
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 3

# Graph path with diagnostics
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 3

# CPU baseline
./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 3
```

## Conclusion

**KEY FINDING:** Q values have extreme magnitudes (-34.1, -15.9, -12.0) which is causing numerical instability in attention computation. This is likely the root cause of corruption in token 2+ generation.

**NEXT STEP:** Add diagnostic to check Q values BEFORE RoPE to determine if the issue is in GEMV computation or RoPE application.
