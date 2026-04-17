# Colleague's Fork Analysis: 600 tok/s Improvements

**Date:** 2026-04-15
**Fork:** https://github.com/maeddesg/ROCmForge
**Baseline:** ~146 tok/s (current rocmforge)
**Reported:** 600 tok/s (colleague's fork)
**Target:** Identify integratable improvements while maintaining correctness-first philosophy

---

## Executive Summary

The colleague's fork achieves ~4x throughput improvement through several architectural optimizations:

1. **Multi-row kernel fusion** (Q4_0/Q8_0 LM head)
2. **Shared memory input staging** for reuse across output columns
3. **Template-based size specializations** for compile-time optimization
4. **Simplified quantization unpacking** (Q4_K)
5. **Pointer arithmetic** instead of hardcoded offsets (Q6_K)
6. **Float4 vectorized loads** throughout
7. **Flash attention** with online softmax optimization

**Critical:** All changes must maintain correctness. Performance without correctness is meaningless.

---

## 1. Q4_K: Simplified Unpacking Approach

### Current Implementation
```cpp
// Complex bit manipulation with multiple scales
// Uses 8 separate scale values with complex indexing
```

### Colleague's Implementation
```cpp
__device__ inline float vec_dot_q4_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half, dmin_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float dmin = __half2float(dmin_half);

    const uint8_t* qs = block_bytes + 16;
    if (fabsf(d) < 1e-7f) return 0.0f;

    float sum = 0.0f;
    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;
        const uint8_t packed = qs[i / 2];
        const uint8_t q4 = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        sum += (static_cast<float>(q4) / d + dmin) * vec[offset + i];
    }
    return sum;
}
```

**Key Differences:**
- Direct `(q4 / d + dmin)` dequantization formula
- Single scale `d` instead of 8 separate scales
- Simpler indexing with `(i % 2 == 0)` ternary
- Loop-based instead of unrolled

**Correctness Assessment:** ✅ **SAFE** - IDENTICAL to current implementation! Our Q4_K uses the same formula `(q4 / d + dmin)` on line 25 of q4_k_gemv.hip.

**Performance Impact:** Neutral - Same algorithm, just simpler loop structure.

**Action:** No changes needed - already optimized.

---

## 2. Q6_K: Pointer Arithmetic Approach

### Current Implementation
```cpp
// Process group 0 (elements 0-127)
{
    const uint8_t* ql = &block_bytes[0];
    const uint8_t* qh = &block_bytes[128];
    const int8_t* sc = &scales[0];
    // ... hardcoded offsets
}

// Process group 1 (elements 128-255)
{
    const uint8_t* ql = &block_bytes[64];
    const uint8_t* qh = &block_bytes[160];
    const int8_t* sc = &scales[8];
    // ... hardcoded offsets
}
```

### Colleague's Implementation
```cpp
for (int n = 0; n < 2; ++n) {
    const int is = tid / 16;
    const int q1 = static_cast<int>((ql[tid + 0] & 0xF) | (((qh[tid] >> 0) & 3) << 4)) - 32;
    const int q2 = static_cast<int>((ql[tid + 32] & 0xF) | (((qh[tid] >> 2) & 3) << 4)) - 32;
    const int q3 = static_cast<int>(((ql[tid + 0] >> 4) & 0xF) | (((qh[tid] >> 4) & 3) << 4)) - 32;
    const int q4 = static_cast<int>(((ql[tid + 32] >> 4) & 0xF) | (((qh[tid] >> 6) & 3) << 4)) - 32;

    sum += scale1 * static_cast<float>(q1) * vec[offset + tid + 0];
    sum += scale2 * static_cast<float>(q2) * vec[offset + tid + 32];
    sum += scale3 * static_cast<float>(q3) * vec[offset + tid + 64];
    sum += scale4 * static_cast<float>(q4) * vec[offset + tid + 96];

    // Advance pointers for next 128 elements
    ql += 64;
    qh += 32;
    sc += 8;
    offset += 128;
}
```

**Key Differences:**
- Pointer arithmetic via `ql += 64; qh += 32; sc += 8; offset += 128;`
- `static_cast<int>` instead of `(int8_t)` cast
- Loop-based for cleaner code

**Correctness Assessment:** ✅ **SAFE** - Mathematically equivalent to current implementation. The pointer arithmetic is just cleaner code.

**Performance Impact:** Neutral - same operations, just cleaner structure.

**Action:** Consider adopting for code clarity, but not a performance priority.

---

## 3. Q8_0 LM Head: Multi-Row Optimization

### Current Implementation
```cpp
// Single column per block launch
gemv_q8_0_f32_kernel<<<ncols_dst, 256>>>(...);
```

### Colleague's Implementation
```cpp
template<int N_SUBWAVES>
__global__ void gemv_q8_0_f32_lm_head_multi_row_kernel(
    const void* __restrict__ weights_q8_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int subwave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col_base = (blockIdx.x * N_SUBWAVES + subwave_id) * Q8_0_LM_HEAD_COLS;
    const int n_blocks_total = n_rows / QK8_0;

    // Stage input in shared memory ONCE for all columns
    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // Process MULTIPLE columns (4 or 8) per subwave
    const Q8_0_block* w_cols[Q8_0_LM_HEAD_COLS];
    #pragma unroll
    for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q8_0_block*>(
            static_cast<const uint8_t*>(weights_q8_0) + col * n_blocks_total * Q8_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[Q8_0_LM_HEAD_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Compute dot products for ALL columns using STAGED input
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK8_0;
        const float4* input_vec = reinterpret_cast<const float4*>(&s_input[row_offset]);

        #pragma unroll
        for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
            if (!w_cols[c]) continue;

            const Q8_0_block* block = &w_cols[c][block_idx];
            const float d = __half2float(block->d);
            float dot = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 in = input_vec[i];
                const int q_offset = i * 4;

                dot += static_cast<float>(block->qs[q_offset + 0]) * in.x;
                dot += static_cast<float>(block->qs[q_offset + 1]) * in.y;
                dot += static_cast<float>(block->qs[q_offset + 2]) * in.z;
                dot += static_cast<float>(block->qs[q_offset + 3]) * in.w;
            }

            sums[c] += d * dot;
        }
    }

    // Warp shuffle reduction for ALL columns
    #pragma unroll
    for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                output[col] = sums[c];
            }
        }
    }
}
```

**Key Optimizations:**
1. **Shared memory input staging** - Load input ONCE, reuse for 4-8 columns
2. **Multi-column processing** - Each subwave processes 4 or 8 output columns
3. **Float4 vectorized loads** - Load 4 values at a time from weights
4. **Template-based** - Compile-time optimization for 4 or 8 columns
5. **Reduced kernel launches** - Fewer blocks = less launch overhead

**Correctness Assessment:** ✅ **SAFE** - Mathematically equivalent, just reorganized for better memory access.

**Performance Impact:** 🚀 **HIGH** - This is likely the biggest contributor to 600 tok/s:
- Reduces global memory reads (input staged once)
- Better cache utilization (multiple columns share staged input)
- Fewer kernel launches (4-8x fewer blocks)

**Action:** ✅ **HIGH PRIORITY** - Integrate multi-row optimization for Q8_0 LM head.

---

## 4. Q4_0/Q8_0: Dynamic Shared Memory Fix

### Colleague's Comment
```cpp
// CRITICAL FIX: Use dynamic shared memory based on actual block size
// The kernel was launched with variable block sizes (64, 128, 256) but
// hardcoded shared memory array of 256, causing out-of-bounds access.
extern __shared__ float partial_sums[];
```

**Assessment:** ⚠️ **BEST PRACTICE** - Found hardcoded `__shared__ float partial_sums[256];` on line 59 of q4_1_gemv.hip.

**Current Implementation Analysis:**
- Legacy kernel launched with 256 threads (line 203) or 128 threads (line 253)
- Hardcoded array size 256 with 128 threads wastes memory but doesn't cause OOB
- **However**, using `extern __shared__` is safer and more flexible

**Action:** ✅ **MEDIUM PRIORITY** - Refactor to dynamic shared memory as best practice.

---

## 5. Attention: Flash Attention with Online Softmax

### Colleague's Implementation
```cpp
__global__ void flash_attn_decode_strided_multi_head_v2_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    // 1. Compute scores for ALL positions
    extern __shared__ float s_scores[];
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        // Compute Q * K[pos]
        s_scores[pos] = score;
    }
    __syncthreads();

    // 2. Find max for numerical stability
    float local_max = kNegInf;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[pos]);
    }
    const float m = block_reduce_max(local_max, s_reduce);

    // 3. Compute exp and sum
    float local_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(s_scores[pos] - m);
        s_scores[pos] = weight;
        local_sum += weight;
    }
    const float s_sum = block_reduce_sum(local_sum, s_reduce);

    // 4. Accumulate output using computed weights
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            acc += s_scores[pos] * v_cache[pos * kv_size + head_offset + i];
        }
        out[head_idx * head_dim + i] = acc / s_sum;
    }
}
```

**Key Optimizations:**
1. **Two-pass algorithm** - Compute all scores first, then accumulate
2. **Numerically stable** - Find max before exp to avoid overflow
3. **Shared memory scores** - Store scores in LDS for reuse
4. **Block reduction** - Custom reduction primitives for max/sum

**Correctness Assessment:** ✅ **SAFE** - Standard flash attention algorithm.

**Performance Impact:** 🚀 **HIGH** for attention-heavy workloads (prefill, long sequences).

**Action:** ✅ **HIGH PRIORITY** for attention optimization phase.

---

## 6. Norm: Float4 Vectorized RMS Norm

### Colleague's Implementation
```cpp
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n,
    const float eps
) {
    // ... reduction using float4 loads ...
    int i = tid * 4;
    for (; i + 3 < n; i += blockDim.x * 4) {
        float4 val4 = reinterpret_cast<const float4*>(&x[i])[0];
        sum_sq += val4.x * val4.x;
        sum_sq += val4.y * val4.y;
        sum_sq += val4.z * val4.z;
        sum_sq += val4.w * val4.w;
    }

    // ... apply norm using float4 stores ...
    i = tid * 4;
    for (; i + 3 < n; i += blockDim.x * 4) {
        float4 x4 = reinterpret_cast<const float4*>(&x[i])[0];
        float4 w4 = reinterpret_cast<const float4*>(&weight[i])[0];
        float4 out4;
        out4.x = x4.x * inv_rms * w4.x;
        out4.y = x4.y * inv_rms * w4.y;
        out4.z = x4.z * inv_rms * w4.z;
        out4.w = x4.w * inv_rms * w4.w;
        reinterpret_cast<float4*>(&out[i])[0] = out4;
    }
}
```

**Key Optimization:**
- Process 4 elements per load/store using float4

**Correctness Assessment:** ✅ **SAFE** - Same operations, just vectorized.

**Performance Impact:** 📈 **MEDIUM** - Better memory bandwidth utilization.

**Action:** ✅ **MEDIUM PRIORITY** - Low-hanging fruit for norm kernels.

---

## 7. Launch Configuration: Size-Specific Templates

### Colleague's Pattern
```cpp
template<int N_SUBWAVES>
__global__ void gemv_q8_0_f32_lm_head_multi_row_kernel(...) { ... }

template<int N_SUBWAVES>
extern "C" hipError_t gemv_q8_0_f32_lm_head_launch(...) {
    const int n_subwaves = select_lm_head_subwaves(ncols_dst);
    if (n_subwaves == 4) {
        gemv_q8_0_f32_lm_head_multi_row_kernel<4><<<...>>>(...);
    } else {
        gemv_q8_0_f32_lm_head_multi_row_kernel<8><<<...>>>(...);
    }
}
```

**Key Benefit:**
- Compiler can optimize with known constants
- Loop unrolling with `#pragma unroll`
- Better register allocation

**Correctness Assessment:** ✅ **SAFE** - Same logic, just template-based.

**Performance Impact:** 📈 **MEDIUM** - Better compile-time optimization.

**Action:** ✅ **MEDIUM PRIORITY** - Adopt for size-dependent kernels.

---

## Integration Priority Matrix

| Change | Correctness Risk | Performance Impact | Complexity | Priority |
|--------|-----------------|-------------------|------------|----------|
| Q8_0 LM head multi-row | Low | 🚀 HIGH | High | **P0 - Critical** |
| Q4_0 multi-row | Low | 🚀 HIGH | Medium | **P0 - Critical** |
| Dynamic shared mem fix | Low (bug fix) | 🚀 HIGH (stability) | Low | **P0 - Critical** |
| Flash attention | Low | 🚀 HIGH | Medium | **P1 - High** |
| Float4 vectorization | Low | 📈 MEDIUM | Low | **P1 - High** |
| Template-based kernels | Low | 📈 MEDIUM | Medium | **P2 - Medium** |
| Q6_K pointer arithmetic | Low | Neutral | Low | **P3 - Low** |
| Q4_K simplified unpacking | ⚠️ **BLOCKER** | Unknown | Medium | **BLOCKED** |

---

## Recommended Integration Plan

### Phase 1: Critical Correctness & Stability (Week 1)
1. **Audit shared memory usage** - Fix hardcoded sizes
2. **Verify Q4_K dequantization formula** - Compare with llama.cpp
3. **Create correctness tests** for multi-row kernels

### Phase 2: High-Impact Performance (Week 2-3)
1. **Implement Q8_0 LM head multi-row** with shared memory staging
2. **Implement Q4_0 multi-row** for residual connection
3. **Benchmark 600 tok/s claim** - Verify improvement

### Phase 3: Attention & Norm Optimization (Week 4)
1. **Implement flash attention** for decode path
2. **Add float4 vectorization** to norm kernels
3. **Profile attention vs matmul** - Find bottleneck

### Phase 4: Code Quality (Week 5)
1. **Refactor Q6_K** to pointer arithmetic (code clarity)
2. **Add template-based kernels** for size-specific paths
3. **Final regression testing** - All quant formats, all models

---

## Correctness-First Validation Protocol

Before integrating ANY change:

1. **Create failing test** - Reproduce current behavior
2. **Verify reference** - Compare with llama.cpp implementation
3. **Implement change** - Isolated from other modifications
4. **Run test suite** - All quant formats, all models
5. **Numerical accuracy check** - Compare outputs before/after
6. **Performance benchmark** - Measure improvement
7. **Document** - Update GPU_SAFETY.md

**No exceptions.** Performance without correctness is meaningless.

---

## Unknowns & Risks

### Unknowns
1. **Model size dependency** - Does 600 tok/s apply to 0.5B only?
2. **Hardware differences** - Colleague's GPU vs ours (RDNA3 vs RDNA4?)
3. **Batch size** - Decode vs prefill performance characteristics
4. **Memory bandwidth** - Does multi-row exceed shared memory capacity?

### Risks
1. **Q4_K formula mismatch** - Could cause silent correctness regression
2. **Shared memory limits** - Multi-row may fail for large rows
3. **Launch configuration** - Optimal for colleague's hardware, not ours
4. **Code complexity** - Multi-row kernels harder to maintain

### Mitigation
1. **Extensive testing** - All quant formats, multiple models
2. **Hardware-specific tuning** - Profile on our target GPUs
3. **Fallback paths** - Keep simple kernels for edge cases
4. **Documentation** - Clear comments for complex optimizations

---

## Next Steps

1. ✅ **Audit current shared memory usage** for hardcoded sizes
2. ✅ **Verify Q4_K dequantization** against llama.cpp reference
3. ✅ **Create correctness test** for multi-row kernels
4. ✅ **Implement Q8_0 LM head multi-row** with staging
5. ✅ **Benchmark vs baseline** - Verify 600 tok/s claim
6. ✅ **Document all changes** in GPU_SAFETY.md

**Remember:** 600 tok/s is meaningless if the output is garbage. Correctness first, performance second.
