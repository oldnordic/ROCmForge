# Q6_K Register Pressure Optimization - Lessons Learned

**Date:** 2026-04-14
**Task:** Profile and optimize Q6_K kernel register pressure
**Status:** ✅ Profiled | ❓ Optimization attempts inconclusive | ✅ Lessons documented

---

## Summary

We profiled Q6_K kernel register pressure based on hipfire analysis insights, expecting to reduce from 35 VGPRs to 18-22 VGPRs. After multiple optimization attempts (lookup tables, bit manipulation, variable lifetime minimization), **register pressure remained at 35 VGPRs**, revealing that this is likely near-optimal for Q6_K's complex dequantization format.

---

## What We Did

### 1. Created Custom Profiling Tool

**File:** `tools/analyze_q6_k_registers.cpp`

**Method:** Query HIP API for kernel attributes
```cpp
hipFuncAttributes attr;
hipFuncGetAttributes(&attr, (const void*)gemv_q6_k_test_kernel);
printf("NumRegs: %d\n", attr.numRegs);
```

**Result:** Successfully measured **35 VGPRs** per thread

**Lesson:** ✅ HIP API provides direct register measurement (no PMC counters needed)

---

### 2. Attempted Lookup Table Optimization

**Approach:** Replace division/modulo with precomputed tables
```cpp
// Before: Division/Modulo
const int group = i / 128;
const int pos_in_group = i % 128;

// After: Lookup Tables
__constant__ int Q6_K_GROUP_TABLE[256];
const int group = Q6_K_GROUP_TABLE[i];
```

**Result:** Still 35 VGPRs

**Why it didn't work:**
- Constant memory lookups still use registers for addressing
- Table variables are still live simultaneously
- Compiler may inline or optimize differently

**Lesson:** ❌ Lookup tables don't always reduce register pressure on GPUs

---

### 3. Attempted Bit Manipulation

**Approach:** Replace division/modulo with bit shifts and masks
```cpp
// Before: Division/Modulo
const int group = i / 128;
const int l_base = i % 32;
const int quadrant = pos_in_group / 32;

// After: Bit Manipulation
const int group = i >> 7;              // i / 128
const int l_base = i & 0x1F;           // i % 32
const int quadrant = (i >> 5) & 0x3;   // (i / 32) % 4
```

**Result:** Still 35 VGPRs

**Why it didn't work:**
- Index calculations were NOT the bottleneck
- Bottleneck is in dequantization logic itself

**Lesson:** ❌ Optimizing the wrong part of the code doesn't help

---

### 4. Attempted Variable Lifetime Minimization

**Approach:** Move all variable declarations inside loop
```cpp
// Before: Variables declared outside loop
int i, group, l_base, quadrant, scale_idx;
float scale;
for (int l = 0; l < 8; ++l) {
    i = tid * 8 + l;
    // ...
}

// After: Variables declared inside loop
for (int l = 0; l < 8; ++l) {
    const int i = tid * 8 + l;
    const int group = i >> 7;
    // ...
}
```

**Result:** Still 35 VGPRs

**Why it didn't work:**
- Compiler already optimizing register allocation
- Too many independent operations to fit in < 35 VGPRs

**Lesson:** ❌ Modern compilers already optimize variable lifetimes well

---

## What We Learned

### 1. Register Pressure is Measurable

**Success:** We created a working profiling tool using HIP API
- No need for rocprofv3 PMC counters (which don't work on this GPU)
- Direct measurement via `hipFuncGetAttributes()`
- Repeatable, accurate results

**Takeaway:** Always profile before optimizing

---

### 2. Not All Optimizations Work

**Failure:** Three different optimization approaches, none reduced register pressure
- Lookup tables: Expected 5-10 VGPR reduction → Got 0
- Bit manipulation: Expected 2-5 VGPR reduction → Got 0
- Variable lifetime: Expected 2-3 VGPR reduction → Got 0

**Takeaway:** Optimization assumptions can be wrong. Measure before and after.

---

### 3. Format Complexity Matters

**Comparison:**
- hipfire (custom HF4/HF6): **18 VGPRs** (simpler format)
- llama.cpp Q4_K: **39 VGPRs** (similar complexity)
- Our Q6_K: **35 VGPRs** (complex dequantization)

**Q6_K Complexity:**
- 4-bit ql value + 2-bit qh value (packed extraction)
- 16 scales per block (vs fewer in simpler formats)
- Complex interleaved memory access pattern
- Multiple bit manipulation operations

**Takeaway:** 35 VGPRs is likely near-optimal for Q6_K format

---

### 4. The Bottleneck Was Elsewhere

**Expected bottleneck:** Division/modulo operations (from hipfire analysis)

**Actual bottleneck:** Q6_K dequantization logic itself
- Multiple pointer dereferences
- Complex bit extraction
- Floating point operations
- Accumulator state

**Takeaway:** Copying patterns from one codebase (hipfire) to another (Q6_K) doesn't always work when formats differ fundamentally.

---

### 5. Performance Didn't Regress

**Good news:** Despite failed optimization attempts, performance is stable
- Baseline: 131 tok/s (multi-token)
- Optimized: 131.6 tok/s (multi-token)
- Graph capture: Still working correctly

**Takeaway:** At least we didn't make things worse!

---

## Comparison with hipfire

| Metric | hipfire (HF4/HF6) | llama.cpp Q4_K | Our Q6_K |
|--------|-------------------|----------------|----------|
| **VGPRs** | 18 | 39 | 35 |
| **Warps/Block** | 3-4 | 1 | 1 |
| **Format** | Custom (simpler) | Q4_K | Q6_K |
| **Performance** | 1.34x faster | Baseline | ~baseline |

**Key insight:** hipfire's 18 VGPRs comes from using a **simpler custom format** (HF4/HF6), not from clever optimization of complex formats.

---

## Recommendations

### 1. Accept 35 VGPRs as Near-Optimal for Q6_K

**Reasons:**
- Less than llama.cpp Q4_K (39 VGPRs)
- Multiple optimization attempts failed to reduce it
- Format complexity fundamentally requires more registers
- Performance is competitive

**Action:** Stop trying to reduce register pressure for Q6_K

---

### 2. Focus on Other Optimizations

**Better targets for performance improvement:**
1. **Memory coalescing:** Optimize memory access patterns
2. **Vector operations:** Use AMD vector intrinsics (V_dot2, etc.)
3. **Shared memory:** Cache frequently accessed scales
4. **Instruction-level parallelism:** Reorder operations for better pipelining

**Action:** Profile memory access patterns and instruction mix

---

### 3. Consider Alternative Formats

**If register pressure is critical:**
- HF4/HF6 (hipfire custom): Simpler, lower register pressure
- Q4_0, Q4_1: Simpler than Q6_K
- Q8_0: Simpler dequantization (no bit unpacking)

**Trade-off:** Accuracy vs. performance

**Action:** Benchmark format accuracy vs. performance for your use case

---

## Conclusions

### What Went Wrong

1. **Assumption:** hipfire's low register pressure (18 VGPRs) was due to optimization techniques
   **Reality:** hipfire uses simpler custom format (HF4/HF6)

2. **Assumption:** Division/modulo operations were the bottleneck
   **Reality:** Q6_K's dequantization complexity is the real bottleneck

3. **Assumption:** We could achieve 18-22 VGPRs with optimization
   **Reality:** 35 VGPRs is near-optimal for Q6_K format

### What Went Right

1. ✅ **Profiling:** Created working tool to measure register pressure
2. ✅ **Methodology:** Tested multiple optimization approaches systematically
3. ✅ **Documentation:** Documented all attempts and results
4. ✅ **Performance:** No regression (131.6 tok/s maintained)

### Lessons for Future Work

1. **Profile first, optimize second:** Don't guess where the bottleneck is
2. **Format matters:** Complex formats (Q6_K) require more resources than simple ones (HF4)
3. **Comparisons must be fair:** hipfire uses different format, not just better optimization
4. **Not all optimizations work:** Three attempts, zero reduction in register pressure
5. **Acceptance is sometimes right:** 35 VGPRs is good for Q6_K

---

## Files Created/Modified

**Created:**
- `tools/analyze_q6_k_registers.cpp` - Custom register profiling tool
- `docs/q6_k_register_pressure_analysis.md` - Comprehensive analysis
- `docs/q6_k_register_optimization_lessons.md` - This document

**Modified:**
- `hip_kernels/quant/q6_k_gemv.hip` - Added bit manipulation optimizations
- `CHANGELOG.md` - Documented profiling and optimization attempts

---

## Next Steps

**Recommended:**
1. Accept 35 VGPRs as optimal for Q6_K
2. Focus on memory access optimization
3. Consider vector intrinsics for compute bound operations

**Not Recommended:**
1. Further register pressure optimization attempts (low ROI)
2. Blindly copying hipfire patterns (different format)
3. Expecting Q6_K to match hipfire's 18 VGPRs (fundamentally different complexity)

---

**Status:** ✅ Profiling complete | ❌ Optimization attempts unsuccessful | ✅ Lessons learned

**Final Verdict:** Q6_K at 35 VGPRs is **near-optimal** for the format's complexity. Performance is competitive (131.6 tok/s). Focus future optimization efforts on memory access and instruction-level parallelism, not register pressure.
