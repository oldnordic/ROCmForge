# Q4_K Coherence Bug - Root Cause Analysis and Fix

**Date:** 2026-04-15
**Status:** ✅ Resolved
**Severity:** Critical (complete output corruption)

## Problem Description

Q4_K quantized models were producing completely incoherent output with mixed Chinese/English characters and repetitive patterns.

**Example Symptoms:**
```
"?strappsc郢stinianThinitraitraitra人格该院pscainless email enf"
"collisioningly ├拯_FALL_FALL告诉她udasudasudas"
"? 1: 1: 1: 1: 1"
```

**Expected:** Coherent English text like Q4_0 produces.

## Investigation Process

### Initial Hypotheses (All Wrong)

1. ❌ **Memory layout corruption** - Verified GGUF loading was correct
2. ❌ **Half-float byte order** - Verified f16 conversion was correct  
3. ❌ **GPU kernel bugs** - Not a GPU issue, also affected CPU mode
4. ❌ **Block structure mismatch** - Q4_K block structure was correct
5. ❌ **Mathematical algorithm** - Verified llama.cpp formula was correctly implemented

### What We Fixed (That Didn't Solve It)

1. ✅ Fixed `BlockQ4K::dequantize` to use llama.cpp formula `d * sc * q - dmin * m`
2. ✅ Added `get_scale_min_k4` helper function
3. ✅ Fixed `dispatch_gemv` to call Q4_K × float instead of Q4_K × Q8_K
4. ✅ Rewrote GEMV version of `gemm_q4_k_transposed_fallback` with correct algorithm
5. ✅ Created unit test verifying `BlockQ4K::dequantize` works correctly

**All these fixes were correct but didn't solve the problem.**

### The Breakthrough

**User Insight:** "check which hotpath is being called for q4_k"

This led to discovering there were **TWO different functions** with the same name:
- `gemv_q4_k_transposed_fallback(w, x, y, out_dim, in_dim)` - GEMV version (line 1556)
- `gemm_q4_k_transposed_fallback(w, x, y, m, n, k)` - GEMM version (line 1639)

Rust allows this overloading since they have different signatures.

### Root Cause

The **GEMM version** (called during prefill for QKV projections and FFN layers) was using the **wrong Q4_0 formula**:

```rust
// WRONG (was in GEMM version):
let d = half::f16::from_le_bytes(block.d).to_f32();
let weight = d * (q4_value as f32 - 8.0);  // Q4_0 formula!
```

Instead of the correct llama.cpp Q4_K formula:

```rust
// CORRECT (llama.cpp Q4_K):
let d = half::f16::from_le_bytes(block.d).to_f32();
let dmin = half::f16::from_le_bytes(block.dmin).to_f32();
let (sc1, m1) = get_scale_min_k4(is, &block.scales);
let d1 = d * sc1 as f32;
let m1_val = dmin * m1 as f32;
let weight = d1 * q_val - m1_val;  // d * sc * q - dmin * m
```

### Why This Was Missed

1. The GEMV version was fixed first (used during decode)
2. The GEMM version had the same function name but different signature
3. Both `dispatch_gemm` and `dispatch_gemv` needed fixing
4. The symptom manifested during prefill (which uses GEMM) but we initially tested decode

## The Fix

**File:** `src/cpu/ops.rs`

**Line 1639-1684:** Updated `gemm_q4_k_transposed_fallback(w, x, y, m, n, k)` to use correct llama.cpp Q4_K dequantization:

1. Load `d` and `dmin` from f16
2. Extract scales/mins using `get_scale_min_k4` 
3. Apply formula: `output = d * sc * q - dmin * m`
4. Process in groups of 64 values (matching llama.cpp pattern)

**Also fixed:**
- `dispatch_gemm` line 843-847: Always use dequantize-on-the-fly for Q4_K
- `dispatch_gemv` line 1763-1767: Always use dequantize-on-the-fly for Q4_K

## Verification

**Before Fix:**
```
"?strappsc郢stinianThinitraitraitra人格该院pscainless email enf"
```

**After Fix:**
```
"I'm doing well, thank you for asking! How can I assist you today? If you have"
"The capital of France is Paris. Paris is a famous city known for its art, culture,"
```

✅ Q4_K now produces coherent English text matching Q4_0 quality.

## Lessons Learned

### 1. Trace Execution Paths, Don't Just Fix Code

**Mistake:** Fixed algorithms in isolation without verifying which code paths were actually being executed.

**Correct approach:** 
- Add debug logging to trace actual function calls
- Check both prefill and decode code paths  
- Verify all overloaded functions
- Use profiling to see what's running

### 2. Function Overloading Can Hide Bugs

**Problem:** Two functions with the same name but different signatures:
- `gemv_q4_k_transposed_fallback(w, x, y, out_dim, in_dim)`  
- `gemm_q4_k_transposed_fallback(w, x, y, m, n, k)`

Fixing one doesn't fix the other!

**Solution:** Search for all definitions/uses when fixing bugs in overloaded functions.

### 3. Q4_K × Q8_K Kernels Are Different From Q4_K × Float

**Critical distinction:**
- `gemv_q4_k_q8_k_*` kernels: For Q4_K weights × Q8_K activations (both quantized)
- `gemv_q4_k_transposed_fallback`: For Q4_K weights × float activations (dequantize-on-the-fly)

Using the wrong kernel produces garbage because it expects quantized input but receives float data.

### 4. CPU and GPU Can Have Different Bugs

**Don't assume:** "Works on CPU = works on GPU" or vice versa.

In this case:
- Both CPU and GPU had Q4_K bugs
- Same root cause (wrong kernel selection)
- Different manifestations (GPU: HIP graph errors, CPU: garbage output)

## Technical Details

### Q4_K Block Structure

```
struct Q4_K_block {
    half d;                          // 2 bytes - super-block scale
    half dmin;                       // 2 bytes - super-block scale for mins
    uint8_t scales[12];              // 12 bytes - packed scales and mins
    uint8_t qs[128];                 // 128 bytes - 4-bit quantized values
};  // Total: 144 bytes
```

### Correct Dequantization Formula

**llama.cpp formula:**
```c
output = d * sc * q - dmin * m
```

Where:
- `d`, `dmin` are f16 super-block scales
- `sc`, `m` are 6-bit scale/min values extracted via `get_scale_min_k4`
- `q` is the 4-bit quantized value (0-15)

### Scale/Min Extraction (`get_scale_min_k4`)

```rust
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}
```

### Processing Pattern

**llama.cpp processes 256 values in 4 iterations:**
```
for j in (0..256).step_by(64):
    get_scale_min_k4(is + 0, scales) -> (sc1, m1)  // for first 32 values (low nibbles)
    get_scale_min_k4(is + 1, scales) -> (sc2, m2)  // for second 32 values (high nibbles)
    
    // Process 32 bytes containing 64 4-bit values
    for l in 0..32:
        output[j + l] = d * sc1 * (qs[l] & 0xF) - dmin * m1        // low nibbles
        output[j + 32 + l] = d * sc2 * (qs[l] >> 4) - dmin * m2   // high nibbles
    
    q += 32  // advance by 32 bytes
    is += 2
```

## Files Modified

1. **`src/cpu/kernels/q4.rs`**
   - Added `get_scale_min_k4` helper function
   - Fixed `BlockQ4K::dequantize` to use llama.cpp formula

2. **`src/cpu/ops.rs`**
   - Fixed `gemv_q4_k_transposed_fallback` (line 1556) with llama.cpp pattern
   - Fixed `gemm_q4_k_transposed_fallback` (line 1639) with llama.cpp pattern
   - Fixed `dispatch_gemm` to always use Q4_K × float fallback
   - Fixed `dispatch_gemv` to always use Q4_K × float fallback

## Related Issues

- Task #121: Fix Q4_K dequantize method to use llama.cpp formula
- Task #122: Fix Q4_K kernel selection - use Q4_K × float, not Q4_K × Q8_K

## References

- llama.cpp Q4_K implementation: `/home/feanor/Projects/llama.cpp/ggml/src/ggml-quants.c:1274`
- llama.cpp get_scale_min_k4: ggml-common.h
- Previous investigation: `docs/investigation/Q4_K_INVESTIGATION.md`

---

**Key Takeaway:** Always trace the actual execution path. Fixing code in isolation without verifying it's being called is ineffective. Function overloading can hide duplicate implementations that both need fixing.
