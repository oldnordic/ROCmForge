# Flash Attention Research - ROCmForge

**Date:** 2026-01-18
**Phase:** 06 - Attention Optimization
**Purpose:** Research existing flash attention implementation and document integration strategy

---

## 1. Existing Flash Attention Kernels

### 1.1 Overview

Three flash attention kernels exist in the codebase:

| Kernel | File | Purpose |
|--------|------|---------|
| `flash_attention_kernel` | `kernels/flash_attention.hip` | Generic fused attention with optional mask |
| `flash_attention_causal_kernel` | `kernels/flash_attention_causal.hip` | Fused causal attention |
| `flash_attention_nocausal_kernel` | `kernels/flash_attention_nocausal.hip` | Fused non-causal attention |

All three kernels are registered in `build.rs` and have Rust wrapper functions in `src/attention/kernels.rs`.

### 1.2 Kernel Specifications

#### 1.2.1 flash_attention.hip (Generic)

**File:** `kernels/flash_attention.hip` (252 lines)

**Algorithm:**
- Fused operation: QK^T + scale + mask + softmax + softmax*V in single kernel
- Each thread block processes one (batch, head, query_pos) triple
- Block size: 256 threads (8 waves of 32 for RDNA3)

**Tensor Layouts (row-major):**
```
Q: [batch_size, seq_len, num_heads, head_dim]
K: [batch_size, seq_len, num_heads, head_dim]
V: [batch_size, seq_len, num_heads, head_dim]
output: [batch_size, seq_len, num_heads, head_dim]
mask: [batch_size, seq_len, seq_len] (optional)
```

**Key Implementation Details:**
- Head dimension limit: 128 (due to register storage `float q_row[128]`)
- Shared memory: `s_partial[BLOCK_SIZE]`, `s_scores[256]`, `s_max`, `s_sum`
- Each thread loads multiple Q elements if head_dim > BLOCK_SIZE
- Wave32 optimized reduction for dot products
- No automatic causal mask (uses mask tensor if provided)

**Grid/Block Configuration:**
```cpp
// Grid: (seq_len, num_heads, batch_size)
// Block: 256 threads (BLOCK_SIZE)
gridDim = (seq_len, num_heads, batch_size);
blockDim = (BLOCK_SIZE, 1, 1);  // 256 threads
```

**Memory Access Pattern:**
1. Load Q row for query position into registers
2. For each key position:
   - Compute QK^T dot product with wave reduction
   - Store score in s_scores
3. Two-pass softmax (max finding, then exp-sum-normalize)
4. For each head_dim: compute weighted sum of V

**Limitations:**
- Head dim <= 128 (hardcoded register limit)
- Seq_len must fit in s_scores array (hardcoded 256)
- No automatic causal masking (must pass mask tensor)

---

#### 1.2.2 flash_attention_causal.hip (Causal)

**File:** `kernels/flash_attention_causal.hip` (172 lines)

**Algorithm:**
- Fused causal attention: QK^T + scale + causal mask + softmax + softmax*V
- Causal masking: query at position i can only attend to keys at positions j <= i
- Block size: 32 threads (1 wave of 32) - exact wavefront size

**Tensor Layouts (4D explicit):**
```
Q: [batch, heads, seq_q, dim]
K: [batch, heads, seq_k, dim]
V: [batch, heads, seq_k, dim]
output: [batch, heads, seq_q, dim]
```

**Key Implementation Details:**
- Each thread computes full dot product directly from global memory
- Shared memory: `s_scores[32]`, `s_partial[32]` for wave32 reduction
- Simpler than generic kernel: no mask parameter (causal is built-in)
- Uses `fmaxf` and loop unrolling for softmax

**Grid/Block Configuration:**
```cpp
// Grid: (seq_q, num_heads, batch_size) blocks
// Block: WARP_SIZE threads (exactly one wavefront)
gridDim = (seq_q, num_heads, batch_size);
blockDim = (WARP_SIZE, 1, 1);  // 32 threads
```

**Causal Mask Implementation:**
```cpp
// Apply causal mask: key_pos > query_pos -> -inf
for (int key_pos = tid; key_pos < seq_k; key_pos += WARP_SIZE) {
    if (key_pos > query_pos) {
        s_scores[key_pos] = -(__builtin_inff());  // -inf
    }
}
```

**Optimizations:**
- Loop unrolling with `#pragma unroll` for small seq_k
- Wave32 reduction pattern minimizes shared memory usage
- No mask tensor needed (causal is intrinsic)

---

#### 1.2.3 flash_attention_nocausal.hip (Non-Causal)

**File:** `kernels/flash_attention_nocausal.hip` (156 lines)

**Algorithm:**
- Fused non-causal attention: QK^T + scale + softmax + softmax*V
- Same as causal kernel but without the mask application step

**Key Difference from Causal:**
- NO mask parameter at all
- Simpler softmax pass (no causal masking step)
- Otherwise identical to causal variant

**When to Use:**
- Bi-directional attention (e.g., encoder-only models like BERT)
- Cross-attention where causal masking not needed

---

### 1.3 Kernel Comparison

| Aspect | flash_attention.hip | flash_attention_causal.hip | flash_attention_nocausal.hip |
|--------|---------------------|---------------------------|------------------------------|
| **Block size** | 256 threads | 32 threads | 32 threads |
| **Causal mask** | Via mask tensor | Built-in | None |
| **Mask param** | Optional | No | No |
| **Head dim limit** | 128 | Unbounded* | Unbounded* |
| **Seq_len limit** | 256 | 32* | 32* |
| **Shared memory** | ~512 floats + overhead | 64 floats | 64 floats |
| **Complexity** | High | Medium | Low |
| **Use case** | Generic/masked attention | Decoder autoregressive | Encoder/bidirectional |

*Limited by s_scores array size, could be increased

---

## 2. Backend Registry Analysis

### 2.1 BackendImplementation Trait

**File:** `src/attention/backend_registry.rs`

**Trait Definition:**
```rust
pub trait BackendImplementation: Send + Sync {
    fn name(&self) -> &str;
    fn supports(&self, config: &AttentionConfig) -> bool;
    fn required_kv_layout(&self) -> Option<KvCacheLayout>;
    fn forward(
        &self,
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
    ) -> AttentionBackendResult<Vec<f32>>;
}
```

**Key Observations:**
1. Trait operates on host slices (`&[f32]`), not device tensors
2. Returns owned `Vec<f32>` (output allocated on host)
3. Configuration via `AttentionConfig` struct
4. Error handling via `AttentionBackendResult<T>`

### 2.2 Existing Backends

#### CPU Backend (CpuAttentionBackend)

**Location:** `src/attention/backend_registry.rs:244-287`

**Capabilities:**
- Supports all configurations (fallback)
- Calls `super::super::cpu::CpuBackend::forward()`
- No specific KV cache layout requirement

**Selection Logic:**
```rust
fn supports(&self, _config: &AttentionConfig) -> bool {
    true  // CPU always supports everything (fallback)
}
```

#### GPU Backend (GpuAttentionBackend)

**Location:** `src/attention/backend_registry.rs:295-352`

**Capabilities:**
- Requires `#[cfg(feature = "rocm")]`
- Validates config (dim divisible by num_heads)
- Has `use_flash_attention` flag (currently always false)
- Calls `super::super::gpu::GpuBackend::forward()`

**Selection Logic:**
```rust
fn supports(&self, config: &AttentionConfig) -> bool {
    config.validate().is_ok()
}
```

**KV Cache Layout:**
```rust
fn required_kv_layout(&self) -> Option<KvCacheLayout> {
    if self.use_flash_attention {
        Some(KvCacheLayout::FlashAttention)
    } else {
        Some(KvCacheLayout::BlockSparse)
    }
}
```

### 2.3 Backend Registry

**Location:** `src/attention/backend_registry.rs:139-237`

**Initialization:**
```rust
pub fn new() -> Self {
    let mut backends: Vec<Box<dyn BackendImplementation>> =
        vec![Box::new(cpu_backend::CpuAttentionBackend::new())];

    #[cfg(feature = "rocm")]
    {
        backends.push(Box::new(gpu_backend::GpuAttentionBackend::new()));
    }

    AttentionBackendRegistry {
        backends,
        default_backend: None,
    }
}
```

**Selection Logic:**
1. If `default_backend` is set, use it (if supports config)
2. Otherwise, auto-select first backend that supports config
3. Return error if no suitable backend found

---

## 3. Current GPU Backend Implementation

### 3.1 GpuBackend::forward() Analysis

**File:** `src/attention/gpu.rs`

**Current Implementation (Multi-Kernel Approach):**
```rust
pub fn forward(dim, q, k, v, mask, dropout) -> Result<Vec<f32>> {
    // 1. Compute QK^T on GPU (via hipBLAS matmul)
    // 2. Copy to CPU, scale on GPU
    // 3. Apply mask on GPU (if provided)
    // 4. Apply softmax on GPU
    // 5. Compute final output: scores * V on GPU (via hipBLAS matmul)
}
```

**Memory Flow:**
```
Host -> GPU (Q, K, V)
GPU: QK^T matmul
GPU: scale kernel
GPU -> Host (scores)
Host -> GPU (scores, mask)
GPU: mask kernel
GPU: softmax kernel
GPU -> Host (scores)
Host -> GPU (scores, V)
GPU: scores * V matmul
GPU -> Host (output)
```

**Issues with Current Approach:**
1. Multiple CPU-GPU round trips (3 sync points)
2. Multiple kernel launches (5+ separate operations)
3. Batch dimension handled via loops (not fully GPU)
4. Creates intermediate buffers (scores matrix)

### 3.2 Flash Attention Kernel Wrappers

**File:** `src/attention/kernels.rs`

**Available Functions:**
```rust
// Non-causal flash attention
pub unsafe fn flash_attention_nocausal_gpu_kernel(
    q: *const f32, k: *const f32, v: *const f32, output: *mut f32,
    scale: f32, batch_size: u32, seq_len: u32,
    num_heads: u32, head_dim: u32,
) -> Result<(), String>

// Causal flash attention
pub unsafe fn flash_attention_causal_gpu_kernel(
    q: *const f32, k: *const f32, v: *const f32, output: *mut f32,
    scale: f32, batch_size: u32, seq_len: u32,
    num_heads: u32, head_dim: u32,
) -> Result<(), String>

// Generic flash attention (with mask)
pub unsafe fn flash_attention_gpu_kernel(
    Q: *const f32, K: *const f32, V: *const f32,
    output: *mut f32, mask: *const f32,
    scale: f32, batch_size: u32, seq_len: u32,
    num_heads: u32, head_dim: u32,
) -> i32
```

**Kernel Loading:**
- All kernels loaded via `get_or_init_cache()`
- Uses global `Mutex<Option<KernelCache>>` for lazy initialization
- HSACO paths from build.rs environment variables

---

## 4. Flash Attention Algorithm

### 4.1 What is Flash Attention?

Flash Attention is an IO-aware exact attention algorithm that:
1. **Fuses operations** - QK^T, scale, mask, softmax, weighted sum in one kernel
2. **Reduces memory bandwidth** - Minimizes HBM reads/writes
3. **Uses tiling** - Processes attention in blocks to fit in SRAM/registers
4. **Eliminates materialization** - Never stores full attention matrix

**Key Innovation:** Instead of computing the full attention matrix (O(seq_len^2)) and then applying softmax, Flash Attention computes the output incrementally using online algorithms.

### 4.2 Speedup Mechanisms

#### Memory Bandwidth Reduction

**Traditional Attention:**
```
Read Q, K, V: 3 * batch * seq * heads * head_dim * 4 bytes
Write scores: batch * seq * seq * 4 bytes  (expensive!)
Read scores: batch * seq * seq * 4 bytes   (expensive!)
Write output: batch * seq * heads * head_dim * 4 bytes

Total: ~O(seq^2) memory for scores matrix
```

**Flash Attention:**
```
Read Q, K, V: 3 * batch * seq * heads * head_dim * 4 bytes
Write output: batch * seq * heads * head_dim * 4 bytes

No scores matrix materialization!
```

**Bandwidth savings:** For seq_len=2048, scores matrix is 16MB per batch/head, while Q/K/V are ~32KB each.

#### Kernel Fusion Benefits

1. **Fewer kernel launches** - 1 vs 5+
2. **No CPU-GPU sync** - Computation stays on GPU
3. **Better cache utilization** - Intermediate values in registers/shared memory
4. **Reduced latency** - No round trips to host

### 4.3 Limitations and Tradeoffs

#### Shared Memory Constraints

Flash Attention requires storing per-block intermediate values:
- `s_scores[seq_len]` - Attention scores for softmax
- For seq_len > 256, requires multiple passes or larger shared memory

**RDNA3 Shared Memory:** 64KB per compute unit
- Max s_scores: ~16K floats (64KB / 4 bytes)
- Practical limit: seq_len <= 2048 (fits with other buffers)

#### Register Pressure

Each thread needs registers for:
- Q row (head_dim elements)
- Partial dot products
- Reduction accumulators

**Practical head_dim limit:** ~128-256 depending on compiler

#### Causal vs Non-Causal

- **Causal:** Can use incremental computation (only need to attend to past)
- **Non-causal:** Need full softmax across all positions
- **Masked:** Requires checking mask tensor (more branches)

---

## 5. Integration Strategy Options

### Option 1: Separate FlashAttention Backend

**Description:** Create a new `FlashAttentionBackend` struct implementing `BackendImplementation`.

**Architecture:**
```rust
pub struct FlashAttentionBackend {
    use_causal: bool,
    max_seq_len: usize,
}

impl BackendImplementation for FlashAttentionBackend {
    fn name(&self) -> &str { "flash_attention" }

    fn supports(&self, config: &AttentionConfig) -> bool {
        // Check constraints:
        // - seq_len <= max_seq_len (shared memory limit)
        // - head_dim <= 128 (register limit)
        // - CUDA/ROCm feature enabled
        config.is_causal == self.use_causal
            && config.max_sequence_length <= self.max_seq_len
            && config.head_dim <= 128
    }

    fn forward(&self, config, q, k, v, mask) -> Result<Vec<f32>> {
        // Copy to GPU
        // Launch flash_attention kernel
        // Copy result back
    }
}
```

**Pros:**
- Clean separation of concerns
- Easy to enable/disable via registry
- Can have multiple variants (causal, non-causal)
- Follows existing pattern

**Cons:**
- Code duplication with GPU backend (buffer management, sync)
- Requires explicit registration
- User must know about FlashAttention backend

**Implementation Complexity:** Medium

---

### Option 2: Optimization Variant of GPU Backend

**Description:** Modify `GpuAttentionBackend` to automatically use FlashAttention when conditions are met.

**Architecture:**
```rust
pub struct GpuAttentionBackend {
    use_flash_attention: bool,  // Existing field
    max_flash_seq_len: usize,
}

impl GpuAttentionBackend {
    pub fn forward(&self, config, q, k, v, mask) -> Result<Vec<f32>> {
        let can_use_flash = self.can_use_flash_attention(config, mask);

        if can_use_flash {
            self.forward_flash(config, q, k, v, mask)
        } else {
            self.forward_traditional(config, q, k, v, mask)
        }
    }

    fn can_use_flash_attention(&self, config, mask) -> bool {
        self.use_flash_attention
            && config.head_dim <= 128
            && config.max_sequence_length <= 2048
            && mask.is_none() || config.is_causal
    }
}
```

**Pros:**
- Transparent to user (automatic optimization)
- Single backend interface
- Can fallback gracefully
- Leverages existing infrastructure

**Cons:**
- More complex forward() logic
- Harder to test separate paths
- May hide performance issues

**Implementation Complexity:** Medium

---

### Option 3: Automatic Detection with Fallback

**Description:** Runtime detection of FlashAttention applicability with automatic fallback to traditional path.

**Architecture:**
```rust
impl GpuAttentionBackend {
    pub fn forward(&self, config, q, k, v, mask) -> Result<Vec<f32>> {
        // Detect optimal path
        match self.select_implementation(config, mask) {
            Implementation::FlashCausal => self.flash_causal(config, q, k, v),
            Implementation::FlashNonCausal => self.flash_nocausal(config, q, k, v),
            Implementation::FlashGeneric => self.flash_generic(config, q, k, v, mask),
            Implementation::Traditional => self.forward_traditional(config, q, k, v, mask),
        }
    }

    fn select_implementation(&self, config, mask) -> Implementation {
        // Decision tree based on:
        // - Causal vs non-causal
        // - Sequence length
        // - Head dimension
        // - Mask presence
    }
}
```

**Pros:**
- Most flexible
- Optimal for all workloads
- Future-proof (can add more variants)

**Cons:**
- Most complex implementation
- Harder to predict behavior
- More code paths to test

**Implementation Complexity:** High

---

## 6. Recommended Strategy

### Recommendation: Option 2 (Optimization Variant)

**Rationale:**

1. **User Experience:** Flash attention is an optimization, not a fundamentally different backend. Users shouldn't need to know about it.

2. **Code Simplicity:** Extends existing `GpuAttentionBackend` without adding new types.

3. **Gradual Rollout:** Can enable/disable via `use_flash_attention` flag.

4. **Fallback Safety:** Always has traditional path if FlashAttention conditions not met.

5. **Testing:** Can compare outputs between paths for correctness.

### Implementation Plan

#### Phase 1: Add Flash Support Detection (06-02)

```rust
impl GpuAttentionBackend {
    fn can_use_flash_attention(&self, config: &AttentionConfig, mask: Option<&[f32]>) -> bool {
        if !self.use_flash_attention {
            return false;
        }

        // Check head_dim limit (from kernel docs)
        if config.head_dim > 128 {
            return false;
        }

        // Check sequence length limit
        if config.max_sequence_length > 2048 {
            return false;
        }

        // Check if mask is compatible
        // - Causal: use flash_attention_causal
        // - No mask: use flash_attention_nocausal
        // - Custom mask: must use generic kernel
        if mask.is_some() && !config.is_causal {
            return false;  // Custom masks not supported yet
        }

        true
    }

    fn select_flash_kernel(&self, config: &AttentionConfig, mask: Option<&[f32]>) -> FlashKernelType {
        if config.is_causal {
            FlashKernelType::Causal
        } else if mask.is_none() {
            FlashKernelType::NonCausal
        } else {
            FlashKernelType::Generic
        }
    }
}
```

#### Phase 2: Implement Flash Forward Path (06-03)

```rust
impl GpuAttentionBackend {
    fn forward_flash(
        &self,
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
    ) -> AttentionBackendResult<Vec<f32>> {
        // 1. Allocate GPU buffers for Q, K, V, output
        // 2. Copy Q, K, V to GPU
        // 3. Launch appropriate flash attention kernel
        // 4. Sync and copy output back
    }

    fn launch_flash_causal(&self, ...) -> Result<(), AttentionError> {
        unsafe {
            flash_attention_causal_gpu_kernel(
                q_gpu, k_gpu, v_gpu, output_gpu,
                scale, batch_size, seq_len, num_heads, head_dim,
            )
        }
    }

    fn launch_flash_nocausal(&self, ...) -> Result<(), AttentionError> {
        unsafe {
            flash_attention_nocausal_gpu_kernel(
                q_gpu, k_gpu, v_gpu, output_gpu,
                scale, batch_size, seq_len, num_heads, head_dim,
            )
        }
    }
}
```

#### Phase 3: Integrate with Registry (06-03)

Modify `GpuAttentionBackend::forward()`:

```rust
pub fn forward(
    &self,
    config: &AttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
) -> AttentionBackendResult<Vec<f32>> {
    if self.can_use_flash_attention(config, mask) {
        return self.forward_flash(config, q, k, v, mask);
    }

    // Fall back to traditional implementation
    self.forward_traditional(config, q, k, v, mask)
}
```

#### Phase 4: Benchmark and Validate (06-04)

- Correctness: Compare outputs with traditional path
- Performance: Measure speedup for different seq_len, head_dim
- Limits: Verify behavior at boundaries

---

## 7. ROCm/AMDGPU Specific Considerations

### 7.1 RDNA3 Architecture

**Target GPU:** AMD Radeon RX 7900 XT (gfx1100)

**Key Characteristics:**
- Wave32: 32 threads per wavefront (vs CUDA's 32)
- Wave64: Also available for some workloads
- Shared memory: 64KB per compute unit
- L2 cache: Large (helps with memory access patterns)

### 7.2 Kernel Optimization Patterns

**Wave32 Reduction:**
```cpp
// Flash attention kernels use this pattern
for (int stride = 16; stride > 0; stride >>= 1) {
    if (tid < stride) {
        s_partial[tid] += s_partial[tid + stride];
    }
    __syncthreads();
}
```

**Register Blocking:**
- Q row stored in registers: `float q_row[128]`
- Each thread loads multiple elements
- Reduces global memory access

### 7.3 Build System Integration

**Already configured in build.rs:**
```rust
(
    "kernels/flash_attention.hip",
    "FLASH_ATTENTION_HSACO",
    "flash_attention_kernel",
),
(
    "kernels/flash_attention_causal.hip",
    "FLASH_ATTENTION_CAUSAL_HSACO",
    "flash_attention_causal_kernel",
),
(
    "kernels/flash_attention_nocausal.hip",
    "FLASH_ATTENTION_NCAUSAL_HSACO",
    "flash_attention_nocausal_kernel",
),
```

**No changes needed** - kernels already compiled and available via environment variables.

---

## 8. Next Steps

### 06-02: Flash Attention Backend Registration

1. Add `can_use_flash_attention()` detection to `GpuAttentionBackend`
2. Add `select_flash_kernel()` for kernel type selection
3. Add `forward_flash()` stub (to be implemented in 06-03)
4. Update `forward()` to call flash path when conditions met
5. Add tests for detection logic

### 06-03: Flash Attention Kernel Integration

1. Implement buffer allocation and GPU memory management
2. Implement kernel launch functions (causal, non-causal, generic)
3. Add synchronization and error handling
4. Integrate with existing `GpuAttentionBackend::forward()`
5. Add correctness tests vs traditional path

### 06-04: Benchmark and Optimize

1. Benchmark suite for different seq_len, head_dim, batch sizes
2. Profile memory bandwidth usage
3. Identify and fix bottlenecks
4. Document performance characteristics

---

## Summary

**Existing Implementation:**
- Three flash attention kernels exist and are built
- Rust wrapper functions available in `src/attention/kernels.rs`
- Not currently used in GPU backend path

**Integration Path:**
1. Extend `GpuAttentionBackend` to detect when FlashAttention is applicable
2. Add flash-aware forward path that calls existing kernels
3. Automatic fallback to traditional implementation when conditions not met
4. Validate correctness and benchmark performance

**Key Constraints:**
- Head dimension: <= 128 (register limit)
- Sequence length: <= 2048 (shared memory)
- Causal masking: Supported via dedicated kernel
- Custom masks: Requires generic flash_attention kernel

**Expected Performance Improvement:**
- 2-4x speedup for typical workloads (seq_len=512-2048, head_dim=64-128)
- Primary benefit from reduced memory bandwidth (no attention matrix materialization)
- Secondary benefit from kernel fusion (fewer launches, no sync)

---

*End of Research Document*
