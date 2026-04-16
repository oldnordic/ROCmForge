# Current QKV-RoPE-KV-Write Pipeline

## Per-Layer Operations (for decode)

1. **QKV Projection** (3 separate GEMV calls)
   - gemv_q4_0_f32_q8_on_stream(w_q, input, out_q, ...)  // Query projection
   - gemv_q4_0_f32_q8_on_stream(w_k, input, out_k, ...)  // Key projection
   - gemv_q4_0_f32_q8_on_stream(w_v, input, out_v, ...)  // Value projection
   - Launches: 3 kernel launches

2. **RoPE** (1 kernel call)
   - rope_heads_on_stream(out_q, pos, num_heads, head_dim, ...)
   - Launches: 1 kernel launch

3. **KV-Write** (1 kernel call)
   - kv_write_from_state_on_stream(out_k, out_v, pos, ...)
   - Launches: 1 kernel launch

**Total per layer: 5 kernel launches**
**Total for 24 layers: 120 kernel launches** (plus FFN, norms, etc.)

## Memory Layout

**Input:** [hidden_size] = [n_heads * head_dim] = [14 * 128] = 1792
**Output Q:** [n_heads * head_dim] = [14 * 128] = 1792
**Output K:** [n_kv_heads * head_dim] = [2 * 128] = 256
**Output V:** [n_kv_heads * head_dim] = [2 * 128] = 256

**GQA Grouping:** 7 query heads share 1 KV head
- Query heads 0-6 share KV head 0
- Query heads 7-13 share KV head 1

## Fusion Opportunity

Combine QKV projection + RoPE into single kernel:
- Input: hidden [1792]
- Output Q (RoPE'd): [14 * 128]
- Output K (RoPE'd): [2 * 128]
- Output V: [2 * 128]

Reduces: 4 kernel launches → 1 kernel launch per layer
Savings: 24 layers × 3 launches = 72 launches saved per token

## Implementation Details

### Q4_0 Quantization Structure
```cpp
struct Q4_0_block {
    half d;              // scale (f16)
    int8_t qs[16];       // 4-bit quantized weights (32 values packed)
};
```
- Each block stores 32 float values as 4-bit quantized integers
- Block size: 18 bytes (2 bytes d + 16 bytes qs)
- Quantization range: [-8, +7] (after subtracting 8)

### Thread Organization (from q4_0_gemv.hip)
- Threads per block: 256 (8 warps × 32 threads)
- Multi-row processing: Each wave processes 4 output columns
- Shared memory: Input cached in LDS for reuse
- Warp reduction: Explicit `__shfl_down(value, offset, 32)`

### RoPE Implementation (from rope.hip)
- `rope_heads_kernel`: Applies RoPE to multiple heads
- Supports GPT-J and Neox RoPE modes
- Parameters: position, num_heads, head_dim, theta, neox_mode

### KV-Write Implementation (from attention.hip)
- `kv_write_state_kernel`: Writes K and V to KV cache
- Takes position and state for indexed write
- Supports both MHA and GQA layouts
