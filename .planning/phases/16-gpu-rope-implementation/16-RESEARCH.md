# Phase 16: GPU RoPE Implementation - Research

**Researched:** 2026-01-19
**Domain:** Rotary Positional Embeddings (RoPE) on AMD ROCm/HIP
**Confidence:** HIGH

## Summary

Phase 16 focuses on ensuring Rotary Positional Embeddings (RoPE) are applied on GPU for GPU tensors, eliminating CPU-GPU transfer overhead per transformer layer. The codebase **already has** RoPE GPU infrastructure in place:

1. **Existing HIP kernel**: `kernels/rope.hip` - implements RoPE rotation on GPU
2. **Rust wrappers**: `src/attention/rope.rs` - `apply_q_device()` and `apply_k_device()` methods
3. **Fused kernels**: `kernels/fused_rope_kvappend.hip` - combines RoPE with KV cache append
4. **CPU fallback**: `src/model/glm_position.rs` - implements GPU kernel with CPU fallback

The primary gap is that **the RoPE kernel source files exist but are not being compiled to HSACO** by `build.rs`. The `ROPE_HSACO` environment variable is never set because the kernel compilation loop in `build.rs` references source files that may have compilation issues or the HSACO output is not being properly tracked.

**Key findings:**
- RoPE HIP kernel (`kernels/rope.hip`) exists and follows correct patterns
- `rope_gpu_kernel()` wrapper in `kernels.rs` is implemented
- `apply_q_device()` and `apply_k_device()` in `rope.rs` call the GPU kernel
- **Root issue**: The compiled HSACO files are not present in `kernels/` directory
- **Secondary issue**: `src/ggml/hip_backend/ops/rope.rs` has the GPU kernel call commented out (TODO marker)

**Primary recommendation:** Ensure RoPE kernels compile successfully and verify the GPU code path is actually used for GPU tensors.

## Standard Stack

The established libraries/tools for GPU RoPE on AMD ROCm:

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| ROCm HIP | 5.7+ | AMD GPU programming model | Required for AMD GPU |
| hipcc | 5.7+ | HIP compiler | Compiles `.hip` to HSACO |
| HSACO | - | Compiled HIP binary | Loaded at runtime via env vars |

### Internal Codebase Components
| Module | Location | Purpose | Status |
|--------|----------|---------|--------|
| `rope.hip` | `kernels/rope.hip` | RoPE GPU kernel | Exists, needs verification |
| `position_embeddings.hip` | `kernels/position_embeddings.hip` | Q+K RoPE in single kernel | Exists |
| `fused_rope_kvappend.hip` | `kernels/fused_rope_kvappend.hip` | RoPE + KV append fusion | Exists |
| `rope.rs` | `src/attention/rope.rs` | RoPE CPU + GPU API | GPU path implemented |
| `kernels.rs` | `src/attention/kernels.rs` | Kernel cache and launchers | `rope_gpu_kernel` defined |
| `fused_ops.rs` | `src/ggml/hip_backend/ops/fused_ops.rs` | Fused kernel wrappers | Implements fused RoPE+KV |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Standalone RoPE kernel | Fused RoPE+KV append | Fused reduces memory bandwidth but less flexible |
| Precompute cos/sin on CPU | Compute on GPU | CPU precomputation is standard, GPU compute wastes resources |

**Installation:**
```bash
# ROCm environment
export ROCM_PATH=/opt/rocm
export HIPCC=${ROCM_PATH}/bin/hipcc
export ROCm_ARCH=gfx1100  # RX 7900 XT (RDNA3)
```

## Architecture Patterns

### Existing RoPE Implementation Structure

```
src/attention/
├── rope.rs              # Rope, RopeConfig, apply_q/apply_k (CPU + GPU)
├── rope_gpu_tests.rs    # CPU vs GPU comparison tests
└── kernels.rs           # rope_gpu_kernel() wrapper

kernels/
├── rope.hip                     # Basic RoPE kernel
├── position_embeddings.hip      # Q+K RoPE in single kernel
└── fused_rope_kvappend.hip      # RoPE + KV cache append fusion

src/ggml/hip_backend/ops/
├── rope.rs              # TODO: GPU kernel commented out
└── fused_ops.rs         # Fused RoPE+KV append wrappers
```

### Pattern 1: RoPE Frequency Precomputation

**What:** Cosine and sine values are precomputed on CPU for all positions up to `max_seq_len`

**When to use:** Standard approach for models with fixed context length

**Example (from `src/attention/rope.rs:86-103`):**
```rust
// Precompute frequencies
for pos in 0..max_seq_len {
    for i in 0..head_dim / 2 {
        let idx = pos * (head_dim / 2) + i;

        // Apply scaling if enabled (for long context)
        let effective_pos = if config.scaled {
            (pos as f32 / config.scale).floor() as usize
        } else {
            pos
        };

        let freq = effective_pos as f32 / config.base.powf(2.0 * i as f32 / head_dim as f32);
        cos[idx] = freq.cos();
        sin[idx] = freq.sin();
    }
}
```

### Pattern 2: GPU Kernel Launch Pattern

**What:** Grid/block layout and kernel launch follows the established pattern

**Grid:** `(seq_len, num_heads, 1)` - one block per token per head
**Block:** `(BLOCK_SIZE, 1, 1)` where `BLOCK_SIZE = 256` threads

**Example (from `src/attention/kernels.rs:491-540`):**
```rust
pub unsafe fn rope_gpu_kernel(
    mut input: *mut f32,
    cos: *const f32,
    sin: *const f32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> i32 {
    // ... (cache initialization)

    let grid_dim = (seq_len, num_heads, 1);
    let block_dim = (BLOCK_SIZE, 1, 1);

    let args: &[*mut c_void] = &[
        &mut input as *mut _ as *mut c_void,
        &mut (cos as *mut f32) as *const _ as *mut c_void,
        &mut (sin as *mut f32) as *const _ as *mut c_void,
        &mut seq_len_arg as *mut _ as *mut c_void,
        &mut num_heads_arg as *mut _ as *mut c_void,
        &mut head_dim_arg as *mut _ as *mut c_void,
    ];

    match backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}
```

### Pattern 3: Position Embedding with GPU Fallback

**What:** Try GPU kernel first, fall back to CPU if it fails

**When to use:** When GPU availability is uncertain or kernel may fail

**Example (from `src/model/glm_position.rs:277-385`):**
```rust
// Apply RoPE if configured
if let Some(rope) = &self.rope {
    // Try GPU kernel first
    let gpu_success = if let Ok(backend) = HipBackend::new() {
        // Upload cos/sin to GPU
        // ...

        let result = unsafe {
            position_embeddings_gpu_kernel(
                q_ptr, k_ptr, cos_ptr, sin_ptr,
                seq_len as u32, num_heads as u32, head_dim as u32,
            )
        };

        result == 0 && backend.synchronize().is_ok()
    } else {
        false
    };

    // If GPU kernel failed, fall back to CPU
    if !gpu_success {
        // CPU fallback: download, apply RoPE, upload
        let q_host = q.to_host_vec()?;
        let mut q_with_pos = q_host.clone();
        rope.apply_q(&mut q_with_pos, position_ids, num_heads)?;
        // ...
    }
}
```

### Anti-Patterns to Avoid

- **Re-uploading cos/sin per layer**: Cache cos/sin in GPU memory for the entire forward pass
- **Synchronizing after each RoPE call**: Use streams or batch operations
- **Computing cos/sin on GPU**: Precompute on CPU once and upload

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RoPE rotation formula | Custom rotation implementation | `kernels/rope.hip` pattern | The 2D rotation is non-trivial to get correct |
| Position scaling for long context | Custom scaling logic | YaRN/NTK-aware scaling formulas | These are well-researched approaches |
| Cos/sin precomputation | Runtime computation | Precompute at model init | Saves GPU compute cycles |

**Key insight:** RoPE looks like a simple rotation but requires:
- Correct pairing of dimensions (x0, x1) with cos/sin
- Proper handling of head_dim parity (must be even)
- Position scaling for sequences longer than training context

## Common Pitfalls

### Pitfall 1: Kernel Not Actually Being Used

**What goes wrong:** Code has GPU kernel calls but they're disabled, commented out, or the kernel never loads

**Why it happens:** TODO markers, failed compilation silently skipped, environment variables not set

**How to avoid:**
1. Verify `ROPE_HSACO` environment variable is set at runtime
2. Check that `rope_gpu_kernel()` returns 0 (success)
3. Add tracing/logging to confirm GPU path is taken

**Warning signs:**
- TODO comments near GPU kernel calls
- Compilation warnings about missing `.hip` files
- CPU fallback always being taken

### Pitfall 2: Head Dimension Not Even

**What goes wrong:** RoPE requires pairs of dimensions, fails on odd `head_dim`

**Why it happens:** Some models have non-standard head dimensions

**How to avoid:**
```rust
assert_eq!(
    head_dim % 2,
    0,
    "Head dimension must be even for RoPE, got {}",
    head_dim
);
```

**Warning signs:**
- Panic in `Rope::new()` during model load
- Incorrect rotation results

### Pitfall 3: Position ID Overflow

**What goes wrong:** Position IDs exceed `max_seq_len`, causing out-of-bounds access

**Why it happens:** Generation continues beyond preallocated cos/sin array

**How to avoid:**
```rust
for &pos in position_ids {
    if pos >= self.config.max_seq_len {
        return Err(AttentionError::DimensionError(format!(
            "Position ID {} exceeds maximum sequence length {}",
            pos, self.config.max_seq_len
        )));
    }
}
```

**Warning signs:**
- Panic during long generation runs
- Garbage values in output

### Pitfall 4: CPU-GPU Transfer Overhead

**What goes wrong:** RoPE is applied on CPU, causing round-trip per layer

**Why it happens:** GPU kernel path not taken, or tensors unnecessarily copied

**How to avoid:**
- Keep tensors on GPU throughout the forward pass
- Only transfer to CPU for final output
- Use DeviceTensor API consistently

**Warning signs:**
- `to_host_vec()` calls in hot loop
- Slow inference despite GPU availability

### Pitfall 5: Cos/Sin Re-upload Per Layer

**What goes wrong:** Uploading cos/sin to GPU for each transformer layer

**Why it happens:** Not caching GPU cos/sin buffers

**How to avoid:**
- Allocate cos/sin GPU buffers once at model initialization
- Pass cached buffers to each layer
- Reuse across all layers

**Warning signs:**
- Many `DeviceTensor::from_host_vec()` calls for cos/sin
- Slow RoPE application

## Code Examples

Verified patterns from the codebase:

### RoPE Kernel Source (from `kernels/rope.hip`)

```cpp
extern "C" __global__ void rope_kernel(
    float* __restrict__ input,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_pair = threadIdx.x;

    const int half_dim = head_dim / 2;

    if (token_idx >= seq_len || head_idx >= num_heads || dim_pair >= half_dim) {
        return;
    }

    const int base = (token_idx * num_heads + head_idx) * head_dim;
    const int i0 = base + dim_pair;
    const int i1 = base + dim_pair + half_dim;

    const int cos_idx = token_idx * half_dim + dim_pair;

    const float c = cos[cos_idx];
    const float s = sin[cos_idx];

    const float x0 = input[i0];
    const float x1 = input[i1];

    // RoPE rotation formula
    input[i0] = x0 * c - x1 * s;
    input[i1] = x0 * s + x1 * c;
}
```

### RoPE DeviceTensor API (from `src/attention/rope.rs:204-327`)

```rust
#[cfg(feature = "rocm")]
pub fn apply_q_device(
    &self,
    x: &mut DeviceTensor,
    position_ids: &[usize],
    num_heads: usize,
) -> AttentionResult<()> {
    use crate::attention::kernels::rope_gpu_kernel;
    use crate::backend::hip_backend::HipBackend;
    use crate::loader::mmap_loader::TensorShape;

    let head_dim = self.config.head_dim;
    let seq_len = position_ids.len();

    // Validate input
    let expected_elements = seq_len * num_heads * head_dim;
    if x.len() != expected_elements {
        return Err(AttentionError::ShapeMismatch(...));
    }

    // Create backend
    let backend = HipBackend::new().map_err(|e| ...)?;

    // Upload cos/sin for the positions we need
    let half_dim = head_dim / 2;
    let mut cos_gpu = Vec::with_capacity(seq_len * half_dim);
    let mut sin_gpu = Vec::with_capacity(seq_len * half_dim);

    for &pos in position_ids {
        if pos >= self.config.max_seq_len {
            return Err(AttentionError::DimensionError(...));
        }
        let cos_offset = pos * half_dim;
        let sin_offset = pos * half_dim;
        cos_gpu.extend_from_slice(&self.cos[cos_offset..cos_offset + half_dim]);
        sin_gpu.extend_from_slice(&self.sin[sin_offset..sin_offset + half_dim]);
    }

    // Create device tensors for cos/sin
    let cos_device = DeviceTensor::from_host_vec(&backend, cos_gpu, cos_shape)?;
    let sin_device = DeviceTensor::from_host_vec(&backend, sin_gpu, sin_shape)?;

    // Get pointers and launch kernel
    let input_ptr = x.buffer().as_mut_ptr() as *mut f32;
    let cos_ptr = cos_device.as_ptr() as *const f32;
    let sin_ptr = sin_device.as_ptr() as *const f32;

    let result = unsafe {
        rope_gpu_kernel(
            input_ptr, cos_ptr, sin_ptr,
            seq_len as u32, num_heads as u32, head_dim as u32,
        )
    };

    if result != 0 {
        return Err(AttentionError::GpuOperation("GPU kernel execution failed".to_string()));
    }

    backend.synchronize()?;

    Ok(())
}
```

## State of the Art

### Long Context RoPE Scaling (for ROPE-03)

For handling position IDs beyond the original training context length (>2048 tokens), several approaches exist:

| Approach | Description | Used By | When to Use |
|----------|-------------|---------|-------------|
| **Position Interpolation (PI)** | Scale positions to fit original context | Original LLaMA | Simple extension, 2x-4x context |
| **NTK-Aware Scaling** | Adjust frequency base for longer contexts | Many open models | Better extrapolation |
| **YaRN** | NTK-by-parts + scaling ramp | Qwen2.5, DeepSeek-R1 | Best for ultra-long context |

**YaRN** (as of 2025) has become the preferred approach:
- Partitions RoPE dimensions and applies different scaling factors
- Uses NTK-by-parts interpolation with ramp function
- Models like Qwen2.5 and DeepSeek-R1 use YaRN for context extension

**Implementation in existing codebase:**
```rust
// From RopeConfig (src/attention/rope.rs:20-22)
pub scaled: bool,      // Whether to use scaled RoPE
pub scale: f32,        // Scaling factor (default 8.0)

// From RoPE::new (src/attention/rope.rs:91-96)
let effective_pos = if config.scaled {
    (pos as f32 / config.scale).floor() as usize
} else {
    pos
};
```

This simple scaling provides basic long-context support but is less sophisticated than YaRN.

### Fused RoPE + KV Cache Append

**Optimization:** Combine RoPE application with KV cache write to eliminate intermediate buffer

**Memory bandwidth comparison:**
- Unfused: Read K/V + Write rotated K/V + Read K/V for append + Write to cache = 8 × hidden_size
- Fused: Read K/V + Read cos/sin + Write to cache = 5 × hidden_size
- **Bandwidth reduction: ~1.6x**

**Kernel exists:** `kernels/fused_rope_kvappend.hip`

**Build.rs entry:** Line 152-155 (compiled but HSACO not present in output)

## Open Questions

### 1. Why is ROPE_HSACO not being generated?

**What we know:**
- `build.rs` has `rope.hip` in the kernels list (line 48)
- `ROPE_HSACO` env var should be set during compilation
- No `rope.hsaco` file exists in `kernels/` directory
- Other kernels (topk_sampling, softmax, etc.) compiled successfully

**What's unclear:**
- Is `rope.hip` failing to compile?
- Is there a hipcc error being silently swallowed?
- Are the source files in the correct location?

**Recommendation:** Run `cargo build -vv` to see hipcc compilation output, check for rope compilation errors.

### 2. Is the GPU RoPE code path actually used?

**What we know:**
- `apply_q_device()` and `apply_k_device()` exist
- `src/ggml/hip_backend/ops/rope.rs` has GPU call commented out with TODO
- `position_embeddings_gpu_kernel()` exists and is called from `glm_position.rs`

**What's unclear:**
- Which RoPE code path does the actual model execution use?
- Is the ggml path using the commented-out version or the attention module version?

**Recommendation:** Trace the actual execution path for a model forward pass to confirm which RoPE implementation is used.

## Sources

### Primary (HIGH confidence)

| Source | Topic | URL/Path |
|--------|-------|----------|
| Codebase analysis | Existing RoPE implementation | `/home/feanor/Projects/ROCmForge/src/attention/rope.rs` |
| Codebase analysis | GPU kernel wrappers | `/home/feanor/Projects/ROCmForge/src/attention/kernels.rs` |
| Codebase analysis | RoPE HIP kernel | `/home/feanor/Projects/ROCmForge/kernels/rope.hip` |
| Codebase analysis | Fused RoPE kernel | `/home/feanor/Projects/ROCmForge/kernels/fused_rope_kvappend.hip` |
| Codebase analysis | Build system | `/home/feanor/Projects/ROCmForge/build.rs` |

### Secondary (MEDIUM confidence)

| Source | Topic | URL |
|--------|-------|-----|
| Dao-AILab/flash-attention | FlashAttention RoPE patterns | [GitHub](https://github.com/Dao-AILab/flash-attention) |
| AMD ROCm Blog | Flash Attention on AMD GPUs | [Blog](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html) |

### Tertiary (LOW confidence - requires verification)

| Source | Topic | URL |
|--------|-------|-----|
| Aman Arora | RoPE context extension guide | [Blog](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html) |
| LongRoPE2 paper | Near-lossless context scaling | [arXiv](https://arxiv.org/pdf/2502.20082) |
| YaRN explanation | YaRN scaling approach | [Medium](https://xiaolishen.medium.com/hands-on-transformer-deep-dive-part-4-yarn-yet-another-rope-extension-8b7f769eff9e) |

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Based on codebase analysis and existing patterns
- Architecture: HIGH - Verified against actual source code
- Pitfalls: HIGH - Based on code review and common GPU programming issues
- Long context scaling: MEDIUM - WebSearch verified, needs actual implementation testing

**Research date:** 2026-01-19
**Valid until:** 30 days (codebase patterns stable, but verification of compilation issues needed)

---

## Phase 16 Implementation Notes

### Current Status Assessment

Based on research findings, Phase 16 has two sub-tasks:

1. **Verify RoPE kernel compilation** - Ensure `rope.hip` compiles and HSACO is generated
2. **Verify GPU code path is used** - Ensure model execution calls GPU RoPE, not CPU fallback

### Key Files to Modify/Verify

| File | Action |
|------|--------|
| `build.rs` | Verify rope.hip compilation succeeds |
| `src/ggml/hip_backend/ops/rope.rs` | Uncomment GPU kernel call (remove TODO) |
| `src/attention/rope.rs` | Ensure apply_q_device/apply_k_device are called |
| `src/attention/rope_gpu_tests.rs` | Use existing tests for verification |

### Test Strategy

The existing `rope_gpu_tests.rs` module already provides CPU vs GPU comparison tests. These tests should pass once:
1. The kernel compiles successfully
2. The GPU code path is actually executed
