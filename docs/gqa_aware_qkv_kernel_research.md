# GQA-Aware QKV Fusion Kernel Research

**Date:** 2026-04-15
**Status:** Research Phase
**Performance Impact:** Potential +111 tok/s (535 → 646 tok/s)

---

## Executive Summary

This document investigates the feasibility of modifying the QKV+RoPE+KV-Write fusion kernel to support Grouped Query Attention (GQA). Current implementation achieves 535 tok/s but requires disabling QKV fusion for GQA models, forfeiting ~111 tok/s of potential performance.

**Key Finding:** GQA support is technically feasible but requires significant kernel redesign with non-trivial performance implications.

---

## Background

### GQA Architecture

Grouped Query Attention (GQA) reduces memory bandwidth and compute requirements by sharing K/V projections across multiple query heads:

```
Standard MHA:  n_q_heads = n_kv_heads (e.g., 14 Q = 14 K = 14 V)
GQA:           n_q_heads > n_kv_heads (e.g., 14 Q = 2 K = 2 V)

For Qwen2.5-0.5B:
- n_q_heads = 14, n_kv_heads = 2
- q_size = 14 * 64 = 896
- kv_size = 2 * 64 = 128
- kv_group = 14 / 2 = 7 (7 query heads share each KV pair)
```

### Current Kernel Architecture

The `gemv_norm_qkv_rope_kvwrite_q4_0_f32_kernel` processes Q, K, V in a single fused pass:

```cpp
// Current assumption: n_q == n_kv (no GQA)
const int total_cols = n_q + 2 * n_kv;  // 896 + 2*128 = 1152

// Single loop processes all outputs uniformly
for (int col_base = 0; col_base < total_cols; col_base += 4) {
    if (col_base < n_q) {
        // Process Q columns [0-895]
    } else if (col_base < n_q + n_kv) {
        // Process K columns [896-1023]
    } else {
        // Process V columns [1024-1151]
    }
}
```

**Why This Breaks with GQA:**

1. **RoPE Head Calculation:**
   ```cpp
   const int head = col_in_q / head_dim;  // Works for Q [0-13]
   // But for K/V, we only have 2 KV heads, not 14!
   ```

2. **KV Cache Writing:**
   ```cpp
   const size_t cache_base = (size_t)pos * kv_size;  // 128 floats per position
   // We write 128 K values and 128 V values to cache
   // But kernel assumes 896 K/V values (wrong!)
   ```

3. **Attention Stage Mismatch:**
   - Attention kernel expects: `q_size=896, kv_size=128`
   - QKV fusion produces: `q_size=896, kv_size=896` (wrong!)
   - Result: Memory access corruption in flash attention

---

## Root Cause Analysis

### The Fundamental Assumption Violation

The QKV fusion kernel assumes:
```cpp
total_cols = n_q + 2 * n_kv;  // Linear concatenation

// Q  : [0, n_q)
// K  : [n_q, n_q + n_kv)
// V  : [n_q + n_kv, n_q + 2*n_kv)
```

This works for MHA (n_q == n_kv) because:
- All heads have same dimension
- RoPE calculation is uniform
- KV cache layout matches kernel output

**For GQA (n_q > n_kv):**
- K/V are computed once but **replicated** during attention
- Kernel should only produce `kv_size=128` values
- But attention needs to access replicated K/V across 14 heads

### Why the Kernel Produces Corrupted Output

When QKV fusion runs with GQA parameters (n_q=896, n_kv=128):

1. ✅ **Q computation correct:** Produces 896 values (14 heads × 64 dim)
2. ❌ **K computation wrong:** Produces 128 values but uses RoPE calculation for 14 heads
3. ❌ **V computation wrong:** Produces 128 values but wrong cache addressing
4. ❌ **KV cache corrupt:** Writes to wrong offsets, confusing attention kernel

The corruption manifests as nonsense output (repeated "奘" characters) because attention reads garbage from misaligned KV cache.

---

## Design Options for GQA-Aware Kernel

### Option 1: Memory Layout Transformation (Rejected)

**Approach:** Replicate K/V weights to match Q dimension

```rust
// Transform GQA weights to MHA layout
// Before: w_k.shape = [128, hidden], w_v.shape = [128, hidden]
// After:  w_k_rep.shape = [896, hidden], w_v_rep.shape = [896, hidden]

for group in 0..7 {
    w_k_rep[group*128..(group+1)*128] = w_k[0..128];
    w_v_rep[group*128..(group+1)*128] = w_v[0..128];
}
```

**Pros:**
- No kernel changes
- Enables QKV fusion
- Simple implementation

**Cons:**
- ❌ **7× memory overhead** for K/V weights (128 → 896)
- ❌ **Memory bandwidth bottleneck:** Loading 896 K/V values vs 128
- ❌ **Likely negates performance gains** due to memory pressure
- ❌ **Increases model VRAM** significantly

**Verdict:** Rejected - memory cost dominates any compute benefit

---

### Option 2: Dynamic Checking (Current Approach) ✅

**Approach:** Disable QKV fusion for GQA models

```rust
let use_fused_kernel = gpu_layer.attn_q_meta.wtype == GgmlType::Q4_0
    && gpu_layer.attn_k_meta.wtype == GgmlType::Q4_0
    && gpu_layer.attn_v_meta.wtype == GgmlType::Q4_0
    && q_size % 4 == 0 && kv_size % 4 == 0
    && q_size == kv_size; // ⚠️ GQA check: only use when n_q == n_kv
```

**Pros:**
- ✅ **Zero memory overhead**
- ✅ **Guaranteed correctness**
- ✅ **Works for all model architectures**
- ✅ **Simple and maintainable**

**Cons:**
- ⚠️ **Forfeits 111 tok/s** for GQA models
- ⚠️ **Incomplete optimization** for modern models

**Verdict:** ✅ **Current production approach** - prioritizes correctness

---

### Option 3: GQA-Aware Kernel Rewrite (Research Candidate)

**Approach:** Modify kernel to handle different Q and KV dimensions

#### 3.1 Architecture Design

```cpp
template<int N_WAVES>
__global__ void gemv_norm_qkv_rope_kvwrite_q4_0_f32_gqa_kernel(
    // ... same parameters ...
    int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
    int kv_group  // n_heads / n_kv_heads
) {
    // Phase 1: RMS norm (unchanged)
    // ...

    // Phase 2: GEMV with GQA-aware dispatch
    const int total_cols = n_q + 2 * n_kv;  // Still 1152 columns

    if (col_base < n_q) {
        // Q path: 896 columns, 14 heads
        process_q_columns(n_q, n_heads, head_dim);
    } else if (col_base < n_q + n_kv) {
        // K path: 128 columns, 2 KV heads (replicated to 14 during attention)
        process_kv_columns(/*kind=*/0, n_kv, n_kv_heads, head_dim);
    } else {
        // V path: 128 columns, 2 KV heads
        process_kv_columns(/*kind=*/1, n_kv, n_kv_heads, head_dim);
    }
}
```

#### 3.2 Key Challenges

**Challenge 1: Asymmetric Output Processing**

```cpp
// Current: Uniform processing for all columns
for (int c = 0; c < 4; ++c) {
    int col = col_base + c;
    if (col < n_q) { /* Q */ }
    else if (col < n_q + n_kv) { /* K */ }
    else { /* V */ }
}

// GQA: Different head calculations for Q vs K/V
if (output_kind == 0) {  // Q
    const int head = col_in_q / head_dim;  // 0-13
} else {  // K/V
    const int kv_head = col_in_kv / head_dim;  // 0-1 only!
}
```

**Challenge 2: KV Cache Addressing with Replication**

```cpp
// Current: Direct write (wrong for GQA)
k_cache[cache_base + col_in_k] = value;

// GQA: Must replicate during attention, not during KV write!
// Still write only 128 values to cache:
k_cache[cache_base + col_in_kv] = value;  // col_in_kv: 0-127

// Attention kernel handles replication:
// Q[head=0] attends to KV[kv_head=0]
// Q[head=1] attends to KV[kv_head=0]
// Q[head=7] attends to KV[kv_head=1]
// ...
```

**Challenge 3: Wave Scheduling Imbalance**

```cpp
// Current: Balanced waves (total_cols = 3 × n_q = 2688)
const int n_waves_q = n_q / 4;       // 224 waves
const int n_waves_k = n_kv / 4;      // 32 waves
const int n_waves_v = n_kv / 4;      // 32 waves
// Total: 288 waves, balanced

// GQA: Imbalanced workload (n_q=896, n_kv=128)
const int n_waves_q = 896 / 4;       // 224 waves
const int n_waves_k = 128 / 4;       // 32 waves
const int n_waves_v = 128 / 4;       // 32 waves
// Total: 288 waves, but Q takes 78% of time!
```

**Impact:** Occupancy imbalance - Q waves finish early, K/V waves become bottleneck.

#### 3.3 Performance Implications

**Theoretical Best Case:**
- Launch reduction: 4 → 1 kernels (-75% overhead)
- Shared memory reuse: Same as current
- Expected gain: ~67 tok/s (+17%)

**Realistic Estimate:**
- Occupancy penalty: -10-15% (wave imbalance)
- Complex branching: -5-10% (register pressure)
- Net gain: ~40-50 tok/s (+8-10% vs current 535 tok/s)

**Upper Bound:**
- Best case: 585 tok/s (still 61 tok/s short of 646 target)
- Worst case: 520 tok/s (slower than current FFN-only!)

---

### Option 4: Hybrid Approach (Recommended) ⭐

**Approach:** Split Q and KV processing into separate kernels

```rust
// Kernel 1: Q-only fusion (RMSNorm + Q projection + RoPE)
gemv_norm_q_rope_q4_0_f32_on_stream(
    raw_hidden, norm_weight, eps,
    w_q, bias_q,
    out_q,  // 896 values
    n_rows, n_q, pos_ptr, head_dim, theta_base, neox, stream
);

// Kernel 2: KV fusion (RMSNorm + KV projection + RoPE + KV write)
gemv_norm_kv_rope_kvwrite_q4_0_f32_on_stream(
    raw_hidden, norm_weight, eps,
    w_k, w_v, bias_k, bias_v,
    k_cache, v_cache,  // 128 values each
    n_rows, n_kv, pos_ptr, head_dim, theta_base, neox, stream
);
```

**Pros:**
- ✅ **No RMS norm recomputation** (only norm weight sharing, not norm itself)
- ✅ **Balanced wave scheduling** (separate launches for Q and KV)
- ✅ **Simpler RoPE logic** (each kernel handles its own heads)
- ✅ **No memory overhead**
- ✅ **Maintains most fusion benefits** (2 kernels vs 4)

**Cons:**
- ⚠️ **2 kernel launches** (vs 1 in full fusion)
- ⚠️ **Recomputes RMSNorm** twice (can optimize with caching)
- ⚠️ **More complex FFI** (2 function calls vs 1)

**Performance Estimate:**
- Launch overhead: 2 vs 4 kernels (-50% overhead, not -75%)
- Shared memory: Same as current (per kernel)
- Expected gain: ~50-60 tok/s (+10-12% vs 535 tok/s)
- Realistic target: 585-595 tok/s

---

## llama.cpp Analysis

### Finding: No QKV Fusion in llama.cpp

After thorough investigation, llama.cpp **does not implement** QKV+RoPE+KV-Write fusion kernels. Their approach:

1. **Separate kernels for each operation:**
   - `ggml_mul_mat_q` for matrix multiplication
   - Custom RoPE kernels
   - KV cache write kernels

2. **GQA handling in attention kernels:**
   ```cpp
   const int gqa_ratio = Q->ne[2] / K->ne[2];  // 7 for Qwen2.5-0.5B
   const half2 *K_h2 = (const half2 *)(K + nb12*(blockIdx.z / gqa_ratio));
   const half2 *V_h2 = (const half2 *)(V + nb12*(blockIdx.z / gqa_ratio));

   // blockIdx.z is the Q head index (0-13)
   // blockIdx.z / gqa_ratio maps to KV head index (0-1)
   ```

3. **Optimization strategy:** Focus on fast attention kernels (flash attention, MMA), not fusion

**Key Insight:** llama.cpp prioritizes **fast attention kernels** over projection fusion. This suggests our colleague's 646 tok/s claim may come from different optimizations, not QKV fusion.

---

## Mathematical Foundation

### GQA Projection Mathematics

For standard attention (MHA):
```
Q = X @ W_q  # [batch, seq, n_q_heads * head_dim]
K = X @ W_k  # [batch, seq, n_kv_heads * head_dim]
V = X @ W_v  # [batch, seq, n_kv_heads * head_dim]

where n_q_heads = n_kv_heads
```

For GQA:
```
Q = X @ W_q  # [batch, seq, n_q_heads * head_dim]
K = X @ W_k  # [batch, seq, n_kv_heads * head_dim]
V = X @ W_v  # [batch, seq, n_kv_heads * head_dim]

where n_q_heads > n_kv_heads

# During attention, K/V are replicated:
for head in 0..n_q_heads:
    kv_head = head // (n_q_heads // n_kv_heads)
    scores[head] = Q[head] @ K[kv_head].T / sqrt(d)
    attn[head] = softmax(scores[head])
    out[head] = attn[head] @ V[kv_head]
```

### Implications for Kernel Design

1. **Projection stage:** K/V computed once (n_kv_heads dimensions)
2. **Attention stage:** K/V replicated virtually (no memory copy)
3. **Our fusion kernel:** Operates at projection stage, should only compute n_kv_heads

**Critical Insight:** KV cache should **only store n_kv_heads dimensions**, not n_q_heads. Replication happens during attention read, not during projection write.

---

## Implementation Complexity Analysis

### Option 3 (GQA-Aware Kernel) Complexity

**Kernel Changes Required:**
1. ✅ Add `n_heads`, `n_kv_heads`, `kv_group` parameters
2. ✅ Split RoPE calculation (Q heads vs KV heads)
3. ✅ Fix KV cache addressing (write only 128, not 896)
4. ✅ Rebalance wave scheduling (account for Q vs KV workload)
5. ⚠️ Add complex branching (Q vs KV paths)
6. ⚠️ Handle edge cases (kv_group not integer, etc.)

**Estimated Development Time:**
- Design: 4 hours
- Implementation: 8 hours
- Testing: 6 hours
- Performance tuning: 4 hours
- **Total: 22 hours**

**Risk Assessment:**
- High complexity (register pressure, branching)
- Uncertain performance (occupancy imbalance)
- May not achieve target 646 tok/s

### Option 4 (Hybrid Approach) Complexity

**Kernel Changes Required:**
1. ✅ Split into 2 separate kernels (Q-only, KV-only)
2. ✅ Remove complex branching (each kernel has simple path)
3. ✅ Reuse RMSNorm logic (or compute once and cache)
4. ✅ Simplify RoPE (single head count per kernel)
5. ⚠️ Manage 2 kernel launches (synchronization)

**Estimated Development Time:**
- Design: 2 hours
- Implementation: 4 hours
- Testing: 4 hours
- Performance tuning: 2 hours
- **Total: 12 hours**

**Risk Assessment:**
- Lower complexity (separate concerns)
- More predictable performance
- Likely achieves 585-595 tok/s (close to optimal for GQA)

---

## Recommendation

### Short-Term (Production)

✅ **Stick with Option 2 (Dynamic Checking)**

**Justification:**
- Current 535 tok/s performance is solid (+19% vs baseline)
- Zero technical debt or maintenance burden
- Works correctly for all model architectures
- FFN fusion provides significant benefit

### Medium-Term (Performance)

⭐ **Implement Option 4 (Hybrid Approach)**

**Justification:**
- Lower complexity than full GQA rewrite
- Predictable performance gains (~50-60 tok/s)
- Maintains code maintainability
- Achieves 585-595 tok/s (87-92% of theoretical max)

### Long-Term (Research)

🔬 **Investigate Alternative Optimizations**

**Potential Avenues:**
1. **Attention kernel optimization** (follow llama.cpp lead)
2. **Multi-row GEMV optimizations** for Q projection
3. **KV cache compression** for GQA models
4. **Quantization improvements** (Q4_K, Q5_K)

**Colleague's 646 tok/s Mystery:**

Hypothesis: Colleague's performance may come from:
1. Non-GQA model (e.g., LLaMA with n_q == n_kv)
2. Different optimizations (attention kernels, not fusion)
3. Compiler/hardware differences
4. Measurement methodology differences

**Action:** Reproduce exact benchmark setup before committing to complex implementation.

---

## Conclusion

The QKV fusion kernel is fundamentally incompatible with GQA due to architectural assumptions. While a GQA-aware rewrite is technically possible, the complexity and performance uncertainty make it questionable.

**Recommended Path:**
1. ✅ **Accept 535 tok/s as production baseline** (Option 2)
2. ⭐ **Implement hybrid approach** for ~60 tok/s gain (Option 4)
3. 🔬 **Investigate colleague's setup** to understand 646 tok/s claim
4. 📊 **Focus on attention kernels** (llama.cpp approach) for further gains

**Final Verdict:** The 111 tok/s gap to 646 tok/s is **not recoverable through QKV fusion alone** for GQA models. Alternative optimization paths are more promising.

---

## References

- **Current Kernel:** `hip_kernels/quant/q4_0_fused_norm_qkv_rope.hip`
- **Forward Pass:** `src/gpu/forward.rs`
- **GQA Paper:** "Grouped Query Attention" (Ainslie et al., 2023)
- **llama.cpp:** `/home/feanor/Projects/llama.cpp`
- **Performance Baseline:** Criterion benchmark `benches/gpu_decode.rs`

---

**Next Steps:**

1. [ ] Decide between Option 2 (status quo) vs Option 4 (hybrid)
2. [ ] If Option 4: Implement hybrid Q+KV kernels
3. [ ] Investigate colleague's benchmark methodology
4. [ ] Explore attention kernel optimizations

**Document Status:** Ready for review and decision
