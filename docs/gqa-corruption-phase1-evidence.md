# GQA Corruption Investigation - Phase 1 Evidence

**Date:** 2026-04-17
**Status:** KV-cache working, corruption persists

## Evidence Collected

### 1. KV-Cache Operations ✅ WORKING

**GQA KV-cache write verification (pos=1):**
```
K BEFORE cache: [0.3797785, 0.18571183, 0.17014426, 0.4550724, ...]
K FROM cache:  [0.3797785, 0.18571183, 0.17014426, 0.4550724, ...] ✅ MATCH
V BEFORE cache: [0.029841192, -0.017958924, -0.016362073, ...]
V FROM cache:  [0.029841192, -0.017958924, -0.016362073, ...] ✅ MATCH
```

**Conclusion:** KV-cache read/write is working correctly in GQA path.

### 2. Flash Attention Kernel Comparison

**CPU Implementation (working):**
```rust
for h in 0..num_heads {
    let kv_h = h / kv_group;  // maps query head to KV head
    for t in 0..seq_len {
        let k_start = t * kv_size + kv_h * head_dim;
        let v_start = t * kv_size + kv_h * head_dim;
        // dot product and attention
    }
}
```

**GPU Implementation (kernel):**
```cpp
const int head_idx = blockIdx.x;  // 0-13 for 14 query heads
const int kv_head_idx = head_idx / (num_heads / num_kv_heads);  // 0 or 1
const int head_offset = kv_head_idx * head_dim;  // 0 or 64

for (int pos = tid; pos < seq_len; pos += blockDim.x) {
    const size_t cache_base = (size_t)pos * kv_size + head_offset;
    // compute attention scores
}
```

**Analysis:** The indexing logic appears equivalent. For Qwen:
- CPU: kv_h = h / 7 (h=0→0, h=7→1)
- GPU: kv_head_idx = head_idx / 7 (head_idx=0→0, head_idx=7→1)

Both map heads 0-6 to KV head 0, heads 7-13 to KV head 1.

### 3. Hidden State Values

**Token pos=1 (first generation):**
- Input: [-0.007846355, 0.00026154518, 0.013861895, 0.011507988, 0.0052309036]
- Output: [-1.4447902, -1.263134, 0.47485244, 1.2352886, -0.8342049]
- Sum abs: 5.25

**Token pos=2 (second generation):**
- Input: [-1.4447902, -1.263134, 0.47485244, 1.2352886, -0.8342049]
- Output: [0.3141359, 1.2872293, 0.9557446, 3.8782415, -1.6634861]
- Sum abs: 8.10 (significantly different)

**Analysis:** Hidden states are very different between pos=1 and pos=2, suggesting the layer computation produces different results.

### 4. Output Corruption Pattern

**CPU (correct):** "ertha is a "
**GPU (GQA):** "er管辖 lash lash lash"

**Pattern:**
- First 2 chars correct ("er")
- Then Chinese characters (corruption)
- Then repetitive "lash" (4x repetition)

This suggests the model is getting stuck in a loop, generating the same token repeatedly.

## Hypothesis

**Hypothesis:** The flash attention kernel is computing incorrect attention weights for GQA, leading to wrong context aggregation and corrupted hidden states.

**Possible causes:**
1. **Race condition in shared memory:** Multiple threads writing to same cache_base location
2. **Incorrect cache indexing:** Subtle bug in how cache is indexed for GQA
3. **Numerical precision:** Different computation order causing accumulation errors
4. **Missing synchronization:** Threads not synchronized properly

## Next Investigation Steps

### Priority 1: Download Attention Output
Add diagnostic to download `attn_out` (attention output) for first query head:
1. Compare with CPU attention output for same input
2. Check if attention weights are correct
3. Verify context aggregation

### Priority 2: Check Flash Attention Kernel
Review the HIP kernel code for potential issues:
1. Shared memory access patterns
2. Thread synchronization
3. Cache indexing calculations

### Priority 3: Compare CPU vs GPU Computation
Create minimal test case:
1. Same Q, K, V values
2. Compare attention output byte-by-byte
3. Identify exact divergence point
