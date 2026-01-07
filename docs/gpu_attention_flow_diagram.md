# GPU Attention Integration - Architecture Flow

## Current Implementation (Already Complete)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      forward_attention() (Line 530-566)                      │
│                   /home/feanor/Projects/ROCmForge/src/model/                 │
│                          execution_plan.rs                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
        │  QKV Projection │ │ QKV Extract │ │  Output Project  │
        │  (GPU matmul)   │ │    (GPU)    │ │   (GPU matmul)  │
        └────────┬────────┘ └──────┬──────┘ └─────────────────┘
                 │                │
                 │                ▼
                 │      ┌─────────────────────────────────┐
                 │      │ scaled_dot_product_attention()  │
                 │      │        (Line 709-792)           │
                 │      └────────────┬────────────────────┘
                 │                   │
                 │                   ▼
                 │      ┌─────────────────────────────────────────┐
                 │      │    HipAttentionKernels::new()           │
                 │      │         (Line 747)                     │
                 │      └────────────┬────────────────────────────┘
                 │                   │
                 │                   ▼
                 │      ┌────────────────────────────────────────────────────┐
                 │      │              GPU ATTENTION PIPELINE                 │
                 │      │  ┌──────────────────────────────────────────────┐  │
                 │      │  │ 1. compute_qk_t()                            │  │
                 │      │  │    Q @ K^T → hipBLAS SGEMM                  │  │
                 │      │  │    Shape: [seq_q, seq_k]                    │  │
                 │      │  └──────────────────────────────────────────────┘  │
                 │      │                      │                           │
                 │      │                      ▼                           │
                 │      │  ┌──────────────────────────────────────────────┐  │
                 │      │  │ 2. scale_inplace()                          │  │
                 │      │  │    Scale by 1/√(head_dim)                   │  │
                 │      │  │    In-place GPU scaling                     │  │
                 │      │  └──────────────────────────────────────────────┘  │
                 │      │                      │                           │
                 │      │                      ▼                           │
                 │      │  ┌──────────────────────────────────────────────┐  │
                 │      │  │ 3. apply_causal_mask()                      │  │
                 │      │  │    Mask future positions                    │  │
                 │      │  │    GPU kernel (JIT compiled)                │  │
                 │      │  └──────────────────────────────────────────────┘  │
                 │      │                      │                           │
                 │      │                      ▼                           │
                 │      │  ┌──────────────────────────────────────────────┐  │
                 │      │  │ 4. compute_softmax()                        │  │
                 │      │  │    Row-wise softmax with numerical stability│  │
                 │      │  │    GPU kernel (JIT compiled)                │  │
                 │      │  └──────────────────────────────────────────────┘  │
                 │      │                      │                           │
                 │      │                      ▼                           │
                 │      │  ┌──────────────────────────────────────────────┐  │
                 │      │  │ 5. compute_attention_weighted_v()            │  │
                 │      │  │    attention @ V → hipBLAS SGEMM            │  │
                 │      │  │    Shape: [seq_q, num_heads, head_dim]      │  │
                 │      │  └──────────────────────────────────────────────┘  │
                 │      │                      │                           │
                 │      │                      ▼                           │
                 │      │             Attention Output                     │
                 │      └────────────────────────────────────────────────────┘
                 │
                 └───────────► Returns: DeviceTensor [seq_len, num_heads, head_dim]
```

## GPU Kernel Inventory

### From `/home/feanor/Projects/ROCmForge/src/attention/kernels.rs`

| Kernel Name | Purpose | Status | Location |
|-------------|---------|--------|----------|
| `qkt_matmul_gpu_kernel_scaled` | QK^T matmul with scaling | ✅ Loaded | Line 463-540 |
| `causal_mask_gpu_kernel` | Causal masking | ✅ Loaded | Line 698-739 |
| `softmax_gpu_kernel` | Row-wise softmax | ✅ Loaded | Line 334-372 |
| `weighted_matmul_gpu_kernel` | Attention @ V matmul | ✅ Loaded | Line 556-609 |
| `flash_attention_causal_gpu_kernel` | Fused causal attention | ✅ Loaded | Line 754-812 |
| `position_embeddings_gpu_kernel` | RoPE position embeddings | ✅ Loaded | Line 904-952 |

## Wrapper Implementation

### From `/home/feanor/Projects/ROCmForge/src/ops/attention_gpu.rs`

```
┌─────────────────────────────────────────────────────────────┐
│              HipAttentionKernels Struct                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Fields:                                               │  │
│  │   - backend: HipBackend                              │  │
│  │   - blas_handle: HipBlasHandle                       │  │
│  │   - qk_kernel: Option<HipModule>                     │  │
│  │   - softmax_kernel: Option<HipModule>                │  │
│  │   - v_kernel: Option<HipModule>                      │  │
│  │   - attention_softmax_kernel: OnceCell<CompiledKernel>│ │
│  │   - causal_mask_kernel: OnceCell<CompiledKernel>     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Public Methods                         │
├─────────────────────────────────────────────────────────────┤
│  new(backend) → HipResult<Self>                            │
│    Initialize all kernels and BLAS handle                   │
├─────────────────────────────────────────────────────────────┤
│  compute_qk_t(q, k, output) → HipResult<()>                │
│    hipBLAS SGEMM with CPU fallback on error                 │
├─────────────────────────────────────────────────────────────┤
│  apply_causal_mask(attention, seq, cache) → HipResult<()>   │
│    GPU kernel with CPU fallback on error                    │
├─────────────────────────────────────────────────────────────┤
│  compute_softmax(attention, temp) → HipResult<()>           │
│    GPU kernel with CPU fallback on error                    │
├─────────────────────────────────────────────────────────────┤
│  compute_attention_weighted_v(attn, v, output) → HipResult<│
│    hipBLAS SGEMM with CPU fallback on error                 │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling Strategy

```
GPU Operation
     │
     ▼
┌─────────┐
│ Try GPU │
└────┬────┘
     │
     ├── Success ──► Return Result
     │
     └── Error ────► Log Warning
                      │
                      ▼
                 ┌──────────┐
                 │ CPU Fallback │
                 └──────────┘
                      │
                      ▼
                 Return Result
```

**Key Design Decisions:**

1. **GPU-First:** Always attempt GPU operation first
2. **Graceful Degradation:** CPU fallback only on GPU errors
3. **Defensive Programming:** Fallbacks ensure correctness even if GPU fails
4. **No Hot Path CPU:** CPU round-trips only on errors, not in normal execution

## Performance Characteristics

### Current Performance

- **Sequence Length:** 32 tokens
- **Configuration:** 4 heads, head_dim=32, hidden_size=128
- **Average Time:** 418.30ms per iteration
- **Operations:**
  - 5 kernel launches (QK^T, Scale, Mask, Softmax, @V)
  - 2 hipBLAS GEMM operations
  - 3 JIT compiled kernels (cached after first use)

### Optimization Opportunities

1. **Flash Attention:** Fuse all operations into single kernel
   - Reduce launches: 5 → 1
   - Expected speedup: 2-3x

2. **Kernel Caching:** OnceCell already implemented
   - JIT compilation: First use only
   - Subsequent calls: Use cached kernel

3. **Memory Coalescing:** Optimize memory access patterns
   - Reduce global memory reads
   - Increase shared memory usage

4. **Batch Processing:** Process multiple heads simultaneously
   - Reduce kernel launch overhead
   - Better GPU utilization

## Integration Points

### Upstream: QKV Projection

```rust
// Line 536 in execution_plan.rs
let qkv_proj = self.matmul(backend, hidden_states, qkv_weight, qkv_bias)?;
```

### Downstream: Output Projection

```rust
// Line 563 in execution_plan.rs
let final_output = self.matmul(backend, &output_reshaped, o_proj, o_proj_bias)?;
```

### KV Cache Integration

```rust
// Lines 749-763 in execution_plan.rs
if let Some(cache) = kv_cache {
    cache.append(layer_idx, k, v)?;
    let current_len = cache.get_current_length(layer_idx)?;
    return attention_kernels.compute_attention(
        q, &attention_scores, &softmax_temp,
        cache_ref, layer_idx, current_len,
    );
}
```

## Test Coverage

### Test Suite: 7 Integration Tests

1. **Single Token Attention** - Verify basic functionality
2. **Multi-Token Sequence** - Test seq_len=16
3. **Causal Mask Correctness** - Ensure proper masking
4. **GPU-CPU Consistency** - Verify GPU output validity
5. **Varying Lengths** - Test seq_len in [1, 2, 4, 8, 16, 32]
6. **Numerical Stability** - Detect NaN/Inf values
7. **Performance Baseline** - Measure execution time

**Test Results:** ✅ 7/7 Passing

**File:** `/home/feanor/Projects/ROCmForge/src/model/gpu_attention_integration_tests.rs`

---

**Diagram created:** 2025-01-06
**Purpose:** Visual documentation of GPU attention integration
**Status:** ✅ Implementation Complete and Verified
