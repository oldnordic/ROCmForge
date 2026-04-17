# GQA GEMV Bug Investigation Summary

**Date:** 2026-04-17
**Status:** Bug NOT fixed - requires further investigation

## Problem Statement

GPU Q4_0 GEMV kernel produces values 10-100x too small, causing GQA corruption.

## Investigation Completed

### 1. Structure Fix Attempt ✅ Applied but ❌ Did NOT Fix

**Bug Found:** `Q4_0_block::qs` declared as `int8_t` instead of `uint8_t`

**Files Fixed:**
- `hip_kernels/quant/q4_0_gemv.hip`
- `hip_kernels/quant/q4_0_gemm.hip`
- `hip_kernels/quant/q4_0_fused_norm_gate_up.hip`

**Result:** After rebuild, GPU still produces incorrect values

### 2. Memory Layout Verification ✅ Correct

- CPU uses row-major format (`gemm_q4_0`, not transposed)
- GPU uses column-based access with matching offsets
- Block indexing formulas are consistent (both use 28 blocks × 18 bytes)

### 3. CPU Dequantization Verification ✅ Working

```
Scale: -0.0037059784
Dequantized weights: [-0.0037, -0.0037, 0.0185, 0.0111, ...]
Dot product (8 elements): -0.0000317575
Final Q values after accumulation: ~12-33 magnitude
```

CPU correctly accumulates all 896 values.

### 4. GPU Values ❌ Wrong

```
Q[0..10] BEFORE RoPE: [-0.032, -0.038, -0.222, 0.058, -0.005, ...]
Max: 0.490 (should be ~33)
```

## Key Evidence

### Inconsistent Ratios Indicate Wrong Data Access

| Index | GPU | CPU | Ratio | Sign |
|-------|-----|-----|-------|------|
| 4 | -0.005 | -12.04 | **2400x** | ✓ |
| 8 | -0.49 | -16.36 | **33x** | ✓ |
| 9 | **+0.48** | **-33.63** | 70x | **✗ WRONG!** |

**Critical Insight:** The wildly inconsistent ratios (33x to 2400x) and wrong sign prove the GPU is reading from wrong memory locations, not just scaling incorrectly.

## Hypotheses

### Most Likely: Block Weight Access Issue

1. **Block pointer arithmetic error** - GPU computing wrong block addresses
2. **Scale factor read error** - `__half2float(b->d)` not working correctly
3. **Endianness mismatch** - f16 bytes in wrong order
4. **Alignment issue** - Memory alignment causing misreads

### Less Likely: Accumulation Error

Warp reduction using `__shfl_down` should work correctly for summation.

### Ruled Out

- ✗ Structure signedness (fixed but didn't help)
- ✗ Memory layout (verified matches CPU)
- ✗ Block indexing formulas (consistent CPU/GPU)
- ✗ Dequantization formula (identical CPU/GPU)

## Recommended Next Steps

### 1. Add GPU Kernel Printf Diagnostics

Add `printf` to the GPU kernel to print:
- First scale factor read: `__half2float(b->d)`
- First few dequantized values
- First partial sum computed

This requires modifying the HIP kernel and recompiling.

### 2. Verify Raw Weight Data

Add host-side code to download and compare:
- First 100 bytes of Q weights (CPU vs GPU)
- Verify identical byte patterns
- Check f16 scale factor bytes

### 3. Inspect HIP Assembly

Compile with `--save` flag and check:
- How `__half2float` is being compiled
- Memory load instructions
- Block address calculation

### 4. Create Minimal Test Case

Write a simple test that:
- Allocates known Q4_0 weight data
- Runs GPU kernel
- Downloads result
- Compares with CPU reference

### 5. Check for ROCm/HIP Bugs

Search for known issues with:
- `__half2float` conversion
- Q4_0 block alignment
- Memory access patterns

## Files Created During Investigation

- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-structure-fix.md` - Structure fix details
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-current-status.md` - Current investigation status
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-investigation-summary.md` - This file

## Test Commands

```bash
# CPU baseline (correct)
ROCMFORGE_DEBUG=1 ./target/release/rocmforge \
  --model /path/to/model.gguf --prompt "The" --no-template --top-p 1.0 --max-tokens 1

# GPU (incorrect)
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /path/to/model.gguf --prompt "The" --no-template --top-p 1.0 --max-tokens 1
```

## Impact

Affects **ALL Q4_0 quantized models** on GPU. Critical bug preventing correct inference.
