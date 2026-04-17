# GQA Corruption Investigation - Q4_0 Structure Fix

**Date:** 2026-04-17
**Status:** Structure fix applied but did NOT resolve the bug

## Issue Found: Q4_0 Block Structure Incorrect Signedness

**Location:** Multiple HIP kernel files

**Problem:** The `Q4_0_block` structure used `int8_t` instead of `uint8_t` for the quantized values array.

### Files Fixed

1. **hip_kernels/quant/q4_0_gemv.hip (line 16)**
   ```cpp
   // BEFORE (BROKEN):
   struct Q4_0_block {
       half d;
       int8_t qs[16];  // ← WRONG: signed
   };

   // AFTER (FIXED):
   struct Q4_0_block {
       half d;
       uint8_t qs[16];  // ← CORRECT: unsigned
   };
   ```

2. **hip_kernels/quant/q4_0_gemm.hip (line 12)**
   ```cpp
   // BEFORE:
   int8_t qs[16];   // quantized values packed as 2x4-bit per byte (16 bytes)

   // AFTER:
   uint8_t qs[16];   // quantized values packed as 2x4-bit per byte (16 bytes)
   ```

3. **hip_kernels/quant/q4_0_fused_norm_gate_up.hip (line 9)**
   ```cpp
   // BEFORE:
   struct Q4_0_block_ngu {
       half d;
       int8_t qs[16];
   };

   // AFTER:
   struct Q4_0_block_ngu {
       half d;
       uint8_t qs[16];
   };
   ```

### Reference: llama.cpp

The correct definition from llama.cpp's ggml-common.h:
```c
typedef struct {
    ggml_half d;           // delta (scale factor)
    uint8_t qs[QK4_0 / 2]; // nibbles / quants (16 bytes)
} block_q4_0;
```

## Test Results After Fix

**Expected:** Q values should match CPU (magnitude ~12-33)
**Actual:** Q values still 10-100x too small (magnitude ~0.005-0.5)

```
GPU Q values (pos=1, before RoPE):
[-0.032, -0.038, -0.222, 0.058, -0.005, 0.121, 0.172, -0.137, -0.490, 0.482]
Max: 0.490

CPU Q values (pos=1, after RoPE):
[-0.163, -0.182, -0.435, 0.073, -12.042, 0.396, 0.517, -0.050, -16.355, -33.631]
Max: 33.631
```

## Conclusion

**Fixing the signedness bug did NOT resolve the GEMV value issue.**

The GPU GEMV kernel is still producing values 10-100x too small even with the correct `uint8_t` type.

### Possible Remaining Issues

1. **Scale factor loading** - `__half2float(b->d)` might not be converting correctly
2. **Block indexing** - Might be reading blocks from wrong location
3. **Missing multiplication** - Some normalization/division might be missing
4. **Compiler optimization** - HIP compiler might be optimizing away critical operations
5. **Weight loading** - GPU weights might be loaded differently than CPU

### CPU Scale Factor Verified

```
[CPU DIAGNOSTIC] First Q weight block scale: -0.0037059784
```

This scale factor is applied to both CPU and GPU (same weights from GGUF file).

### Next Steps

1. Add diagnostic to GPU kernel to print actual scale factor being read
2. Compare first few dequantized weight values between CPU and GPU
3. Check if there's a division by QK4_0 (32) that shouldn't be there
4. Verify block offset calculation: `col * n_blocks_total * 18`
5. Inspect HIP assembly to verify __half2float conversion

## Related Documentation

- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-bug.md` - Original bug discovery
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-investigation-complete.md` - Complete investigation summary
