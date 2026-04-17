# Colleague Fork Integration Plan

**Date:** 2026-04-15
**Goal:** Achieve 600 tok/s while maintaining correctness-first philosophy
**Baseline:** ~146 tok/s
**Target:** 600 tok/s (~4x improvement)

---

## Phase 1: Foundation & Correctness (Week 1)

### Task 1.1: Verify Q4_K Formula ✅ ALREADY DONE
**Status:** Complete - Our Q4_K uses identical dequantization formula
**Finding:** Line 25 of q4_k_gemv.hip uses `(q4 / d + dmin)` - same as colleague
**Action:** No changes needed

### Task 1.2: Create Multi-Row Correctness Tests
**Status:** Pending
**Files to Create:**
- `tests/q8_0_multirow_correctness_test.rs` - Verify 4-column and 8-column outputs match single-column
- `tests/q4_0_multirow_correctness_test.rs` - Verify multi-row residual correctness

**Test Pattern:**
```rust
#[test]
#[ignore = "Requires GPU"]
fn test_q8_0_multirow_matches_single() {
    // Generate random input
    // Run single-column kernel
    // Run multi-row kernel (4-col and 8-col)
    // Assert outputs match bitwise
}
```

**Success Criteria:** Multi-row outputs match single-column within numerical precision

### Task 1.3: Audit Shared Memory Usage
**Status:** Pending
**Action:** Review all kernels with hardcoded `__shared__` arrays
**Priority Files:**
- `hip_kernels/quant/q4_1_gemv.hip:59` - Hardcoded `[256]`
- `hip_kernels/elementwise/*.hip` - Multiple hardcoded `[BLOCK_SIZE]`

**Refactor Pattern:**
```cpp
// Before:
__shared__ float partial_sums[256];

// After:
extern __shared__ float partial_sums[];
// Launch with:<<<blocks, threads, threads * sizeof(float)>>>
```

---

## Phase 2: High-Impact Performance (Week 2-3) 🚀

### Task 2.1: Implement Q8_0 LM Head Multi-Row
**Status:** Pending
**Impact:** 🚀 HIGH - Primary contributor to 600 tok/s
**File:** `hip_kernels/quant/q8_0_gemv.hip`

**Implementation Steps:**
1. **Create template kernel** (4-column and 8-column variants)
   ```cpp
   template<int N_COLS>
   __global__ void gemv_q8_0_f32_lm_head_multi_row_kernel(...)
   ```

2. **Add shared memory input staging**
   ```cpp
   extern __shared__ float s_input[];
   for (int i = tid; i < n_rows; i += blockDim.x) {
       s_input[i] = input[i];
   }
   __syncthreads();
   ```

3. **Process multiple columns per subwave**
   ```cpp
   const Q8_0_block* w_cols[N_COLS];
   float sums[N_COLS] = {0.0f};
   // ... compute dot products for all columns using staged input
   ```

4. **Float4 vectorized loads**
   ```cpp
   const float4* input_vec = reinterpret_cast<const float4*>(&s_input[row_offset]);
   #pragma unroll
   for (int i = 0; i < 8; ++i) {
       const float4 in = input_vec[i];
       dot += block->qs[q_offset + 0] * in.x;
       dot += block->qs[q_offset + 1] * in.y;
       // ...
   }
   ```

5. **Update Rust FFI**
   - Add variant parameter to launch function
   - Select 4-col vs 8-col based on ncols_dst alignment

**Verification:**
- Run `test_q8_0_multirow_matches_single`
- Benchmark vs baseline: `cargo bench --bench gpu_decode --features gpu`
- Expected speedup: 2-3x on LM head

### Task 2.2: Implement Q4_0 Multi-Row for Residual
**Status:** Pending
**Impact:** 🚀 HIGH - Reduces residual connection overhead
**File:** `hip_kernels/quant/q4_0_gemv.hip`

**Implementation:** Similar pattern to Q8_0, but with Q4_0_block structure

**Verification:**
- Run `test_q4_0_multirow_correctness_test`
- Benchmark FFN-down hotspot

### Task 2.3: Benchmark 600 tok/s Claim
**Status:** Pending
**Test Model:** Qwen 2.5 0.5B Q4_0
**Command:** `cargo test --release --features gpu --test gpu_decode_real test_gpu_greedy_decode_profile_real_model -- --ignored --nocapture --test-threads=1`

**Metrics to Track:**
- Tokens per second (target: 600 tok/s)
- Kernel launch count (should decrease with multi-row)
- Memory bandwidth utilization (should increase)

**Success Criteria:**
- Achieve ≥80% of 600 tok/s (≥480 tok/s) with correctness

---

## Phase 3: Attention & Norm Optimization (Week 4)

### Task 3.1: Implement Flash Attention
**Status:** Pending
**Impact:** 🚀 HIGH for attention-heavy workloads (prefill, long sequences)
**File:** `hip_kernels/attention.hip`

**Implementation:**
1. **Two-pass algorithm**
   - Pass 1: Compute all Q*K scores, store in shared memory
   - Pass 2: Compute softmax weights (with max for numerical stability)
   - Pass 3: Accumulate V * weights

2. **Numerically stable softmax**
   ```cpp
   // Find max before exp to avoid overflow
   float m = block_reduce_max(local_max, s_reduce);
   float weight = expf(score - m);
   ```

3. **Shared memory score buffer**
   ```cpp
   extern __shared__ float s_scores[];
   // Store scores for reuse in V accumulation
   ```

**Verification:**
- Compare with current attention kernel
- Check numerical accuracy (should match within 1e-5)
- Profile on prefill workload (long sequences)

### Task 3.2: Add Float4 Vectorization to Norm Kernels
**Status:** Pending
**Impact:** 📈 MEDIUM - Better memory bandwidth utilization
**File:** `hip_kernels/norm.hip`

**Implementation:**
```cpp
// Process 4 elements per load/store
int i = tid * 4;
for (; i + 3 < n; i += blockDim.x * 4) {
    float4 val4 = reinterpret_cast<const float4*>(&x[i])[0];
    sum_sq += val4.x * val4.x;
    sum_sq += val4.y * val4.y;
    sum_sq += val4.z * val4.z;
    sum_sq += val4.w * val4.w;
}

// Apply norm with float4
float4 out4;
out4.x = x4.x * inv_rms * w4.x;
out4.y = x4.y * inv_rms * w4.y;
out4.z = x4.z * inv_rms * w4.z;
out4.w = x4.w * inv_rms * w4.w;
reinterpret_cast<float4*>(&out[i])[0] = out4;
```

**Verification:**
- Compare outputs with scalar version
- Benchmark RMS norm kernel

---

## Phase 4: Code Quality & Documentation (Week 5)

### Task 4.1: Refactor Q6_K to Pointer Arithmetic
**Status:** Pending
**Impact:** Neutral - Code clarity only
**File:** `hip_kernels/quant/q6_k_gemv.hip`

**Implementation:** Replace hardcoded group offsets with pointer advancement
```cpp
for (int n = 0; n < 2; ++n) {
    // Process 128 elements
    // Advance pointers
    ql += 64;
    qh += 32;
    sc += 8;
    offset += 128;
}
```

**Verification:**
- Run Q6_K correctness test
- Benchmark (should be same performance)

### Task 4.2: Add Template-Based Kernel Dispatch
**Status:** Pending
**Impact:** 📈 MEDIUM - Compile-time optimization
**Pattern:**
```cpp
template<int N_SUBWAVES>
__global__ void multi_row_kernel(...) { ... }

extern "C" hipError_t launch(...) {
    const int n_subwaves = (ncols_dst <= 4096) ? 4 : 8;
    if (n_subwaves == 4) {
        multi_row_kernel<4><<<...>>>(...);
    } else {
        multi_row_kernel<8><<<...>>>(...);
    }
}
```

**Verification:**
- Check compiler unrolling with `#pragma unroll`
- Benchmark template vs runtime dispatch

### Task 4.3: Update GPU_SAFETY.md
**Status:** Pending
**Sections to Add:**
1. Multi-row kernel design patterns
2. Shared memory staging best practices
3. Template-based optimization guidelines
4. Flash attention correctness protocol

---

## Correctness-First Validation Checklist

Before marking ANY task complete:

- [ ] Failing test created before implementation
- [ ] Implementation compared with llama.cpp reference
- [ ] All quant formats tested (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0)
- [ ] Multiple models tested (0.5B, 1.6B, 3B if available)
- [ ] Numerical accuracy verified (bitwise or within tolerance)
- [ ] Performance benchmarked (improvement documented)
- [ ] GPU_SAFETY.md updated with new patterns
- [ ] No regressions in existing tests

**Zero tolerance for correctness violations.**

---

## Risk Mitigation

### Risk 1: Multi-Row Shared Memory Limits
**Mitigation:** Add runtime check, fall back to single-row if input too large
```cpp
const size_t input_shared_mem = n_rows * sizeof(float);
if (input_shared_mem > SHARED_MEM_LIMIT) {
    // Use single-row kernel
}
```

### Risk 2: Hardware-Specific Tuning
**Mitigation:** Profile on target GPUs (RDNA3, RDNA4), add tuning heuristics
```cpp
static inline int select_multi_row_cols(int gpu_arch, int ncols_dst) {
    // RDNA4: Prefer 8 columns
    // RDNA3: Use 4 or 8 based on ncols_dst
}
```

### Risk 3: Code Complexity
**Mitigation:**
- Keep simple kernels as fallback
- Add clear comments for complex optimizations
- Use templates to keep code DRY

---

## Integration Timeline

| Week | Phase | Tasks | Expected Speedup |
|------|-------|-------|------------------|
| 1 | Foundation | Tests + audit | 1.0x (baseline) |
| 2-3 | Performance | Q8_0/Q4_0 multi-row | 2.5-3.0x |
| 4 | Attention | Flash attention + float4 | 3.0-3.5x |
| 5 | Polish | Code quality + docs | 3.5-4.0x |

**Final Target:** 600 tok/s (4x baseline)

---

## Success Metrics

### Performance
- [ ] ≥480 tok/s on Qwen 2.5 0.5B Q4_0 (80% of target)
- [ ] ≤80% baseline kernel launch count (multi-row fusion)
- [ ] ≥1.5x memory bandwidth utilization

### Correctness
- [ ] All tests pass (no regressions)
- [ ] Multi-row outputs match single-column (bitwise)
- [ ] No GPU crashes or hangs

### Code Quality
- [ ] GPU_SAFETY.md updated
- [ ] No hardcoded shared memory arrays
- [ ] All kernels documented

---

## Next Steps

1. ✅ **Start Task 1.2** - Create multi-row correctness tests
2. ✅ **Start Task 1.3** - Audit shared memory usage
3. ✅ **Prepare Task 2.1** - Design Q8_0 multi-row kernel

**Remember:** 600 tok/s is meaningless if the output is garbage. Correctness first, performance second.
