# GQA GEMV Investigation - Current Status

**Date:** 2026-04-17
**Status:** Structure fix applied, but bug persists. Suspecting incorrect weight data access.

## Summary

Fixed the `int8_t` → `uint8_t` signedness bug in Q4_0_block structure, but GPU GEMV still produces values 10-100x too small.

## Evidence of Incorrect Data Access

The issue is NOT just scaling - it's that the GPU is reading wrong values entirely:

| Index | GPU Q (before RoPE) | CPU Q (after RoPE) | Ratio | Note |
|-------|---------------------|-------------------|-------|------|
| 4 | -0.005 | -12.04 | **2400x** | Wrong magnitude |
| 8 | -0.49 | -16.36 | **33x** | Wrong magnitude |
| 9 | **+0.48** | **-33.63** | 70x | **WRONG SIGN!** |

**Critical insight:** The inconsistent ratios and wrong sign at index 9 suggest the GPU is reading from the wrong memory locations, not just scaling incorrectly.

## Verified Correct: CPU Dequantization

```
[CPU DIAGNOSTIC] First Q weight block scale: -0.0037059784
[CPU DIAGNOSTIC] First 8 dequantized Q weights (low nibbles):
  [-0.0037, -0.0037, 0.0185, 0.0111, 0.0037, -0.0074, 0.0074, -0.0]
[CPU DIAGNOSTIC] Dot product (first 8): -0.0000317575
```

- Dequantized weights are small (correct for Q4_0)
- After accumulating 896 values → final Q values are ~12-33 magnitude (correct)
- CPU is working correctly

## GPU: Values 10-100x Too Small

```
[GPU DIAGNOSTIC] Q[0..10] BEFORE RoPE:
  [-0.032, -0.038, -0.222, 0.058, -0.005, 0.121, 0.172, -0.137, -0.490, 0.482]
Max: 0.490
```

These should be in the ~12-33 range after accumulation, not ~0.5.

## Possible Root Causes

### 1. **Block indexing error** (Most Likely)
The GPU might be reading weights from wrong blocks:
```cpp
w_cols[c] = reinterpret_cast<const Q4_0_block*>(
    static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
) : nullptr;
```

- CPU: `block = &w[col_offset + b * Q4_BLOCK_BYTES..]`
- GPU: `w_cols[c] + block_idx` where `w_cols[c] = base + col * n_blocks_total * 18`
- These might be computing different locations

### 2. **Half-float conversion issue**
```cpp
const float d = __half2float(b->d);
```
- `__half2float` might not be working correctly
- Endianness issue with f16 bytes

### 3. **Missing accumulation**
- Warp reduction using `__shfl_down` might not be working correctly
- Some threads contributing zero incorrectly

### 4. **Weight loading mismatch**
- GPU weights might be loaded differently than CPU
- Transformation during upload

## Next Steps

1. **Verify weight data access** - Check if GPU is reading from correct memory locations
2. **Add GPU kernel printf** - Print scale factor and first few dequantized values from kernel
3. **Compare raw bytes** - Verify CPU and GPU have identical weight data
4. **Check block offset formula** - Verify `col * n_blocks_total * 18` calculation
5. **Inspect half conversion** - Verify `__half2float` is working correctly

## Files Modified

1. ✅ `hip_kernels/quant/q4_0_gemv.hip` - Fixed int8_t → uint8_t
2. ✅ `hip_kernels/quant/q4_0_gemm.hip` - Fixed int8_t → uint8_t
3. ✅ `hip_kernels/quant/q4_0_fused_norm_gate_up.hip` - Fixed int8_t → uint8_t
4. ✅ `src/cpu/forward.rs` - Added dequantization diagnostics

## Related

- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-structure-fix.md` - Structure fix details
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-bug.md` - Original bug discovery
