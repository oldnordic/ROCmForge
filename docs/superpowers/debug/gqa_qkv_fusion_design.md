# GQA-Aware QKV Fusion Kernel Design

## Kernel Signature

```cpp
__launch_bounds__(256, 1)
__global__ void fused_qkv_rope_q4_0_gqa_kernel(
    // Quantized weights
    const void* __restrict__ w_q,  // [n_heads * hidden_size * sizeof_q4_0]
    const void* __restrict__ w_k,  // [n_kv_heads * hidden_size * sizeof_q4_0]
    const void* __restrict__ w_v,  // [n_kv_heads * hidden_size * sizeof_q4_0]

    // Bias (optional)
    const float* __restrict__ bias_q,  // [n_heads * head_dim] or nullptr
    const float* __restrict__ bias_k,  // [n_kv_heads * head_dim] or nullptr
    const float* __restrict__ bias_v,  // [n_kv_heads * head_dim] or nullptr

    // Input
    const float* __restrict__ input,   // [hidden_size] = [n_heads * head_dim]

    // Output (RoPE'd)
    float* __restrict__ out_q,  // [n_heads * head_dim]
    float* __restrict__ out_k,  // [n_kv_heads * head_dim]
    float* __restrict__ out_v,  // [n_kv_heads * head_dim]

    // RoPE parameters
    const int pos,              // Current position
    const int n_heads,          // Number of query heads (14)
    const int n_kv_heads,       // Number of KV heads (2)
    const int head_dim,         // Head dimension (128)
    const float rope_theta,     // RoPE theta (10000.0)
    const bool rope_neox        // RoPE mode (true for GQA)
);
```

## Thread Block Organization

- **Threads per block:** 256 (warp size 32 × 8 warps)
- **Blocks per grid:** n_kv_heads (2 blocks for Qwen2.5-0.5B)
- **Each block processes one KV head and its associated query heads**

**GQA Grouping Strategy:**
- Block 0: Processes KV head 0 + query heads 0-6 (7 query heads)
- Block 1: Processes KV head 1 + query heads 7-13 (7 query heads)

## Work Distribution per Block

For block processing KV head `kv_head`:
1. Load quantized K and V weights for this KV head
2. Load quantized Q weights for 7 associated query heads
3. For each of 7 query heads:
   - Compute Q projection from input
   - Apply RoPE to Q
   - Compute K projection from input
   - Apply RoPE to K
   - Compute V projection from input
4. Write outputs to global memory

## Memory Access Pattern

**Coalesced reads from input:**
- All threads read input [hidden_size] sequentially
- Reuse input for all projections (broadcast via shared memory)

**Quantized weight access:**
- Each thread loads different quantized blocks
- Q weights: 7 query heads × multiple blocks
- K/V weights: 1 KV head × multiple blocks

**RoPE computation:**
- Thread-local: each thread computes RoPE for one element
- Use cos/sin precomputed or computed on-fly

## AMD HIP Standards Compliance

1. **Explicit warpSize in __shfl_down**
   ```cpp
   for (int offset = 16; offset > 0; offset >>= 1) {
       sum += __shfl_down(sum, offset, 32);  // Explicit warpSize
   }
   ```

2. **Thread block limits**
   - Max threads per block: 256 (safe for RX 7900 XT)
   - Max shared memory: ~64KB (leaves room for other kernels)

3. **Memory alignment**
   - Use alignas(16) for shared memory
   - Ensure coalesced global memory access

4. **Launch bounds**
   - `__launch_bounds__(max_threads_per_block, min_blocks_per_sm)`
   - Helps compiler optimize register usage

## Implementation Strategy

**Phase 1: Basic fusion (QKV + RoPE)**
- Load quantized weights
- Compute Q, K, V projections
- Apply RoPE to Q and K
- Write outputs
- NO bias support initially

**Phase 2: Add bias support**
- Add optional bias loading
- Add to projection before RoPE

**Phase 3: Optimize**
- Use shared memory for input caching
- Optimize quantized weight loading
- Vectorize where possible
