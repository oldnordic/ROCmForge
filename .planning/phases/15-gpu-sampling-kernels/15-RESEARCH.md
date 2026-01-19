# Phase 15: GPU Sampling Kernels - Research

**Researched:** 2026-01-19
**Domain:** GPU token sampling (top-k, top-p, temperature) for LLM inference
**Confidence:** HIGH

## Summary

This research phase investigates implementing GPU-accelerated token sampling kernels for ROCmForge. The project already has `.hip` kernel source files for sampling operations, but they are **not compiled** (missing from build.rs) and partially implemented as stubs.

The current state:
- **4 sampling kernel `.hip` files exist**: `topk_sampling.hip`, `topp_sampling.hip`, `topk_topp_sampling.hip`, `sampling_utils.hip`
- **Kernels are NOT in build.rs**: Must be added to compilation list
- **topp_sampling.hip contains stub kernels**: Deliberately disabled due to GPU watchdog timeout issues (documented in file)
- **GPU sampler structs exist**: `GpuTopKSampler`, `GpuTopPSampler`, `GpuFusedSampler` with CPU fallbacks

Key findings from industry research:
- **FlashInfer (2025)**: Uses sorting-free "dual-pivot rejection sampling" algorithm for O(log K) complexity
- **ROCm rocRAND**: Provides `rocrand_generate_uniform()` for GPU random number generation
- **Main challenge**: Large vocabulary sizes (151936 tokens) cause single-threaded loops to trigger GPU watchdog timeout

**Primary recommendation**: Add sampling kernels to build.rs, implement proper multi-kernel pipeline following FlashInfer's approach to avoid watchdog timeouts, and create comprehensive correctness tests.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ROCm HIP | 6.x | GPU kernel programming | AMD GPU platform |
| rocRAND | 4.2.0 | GPU random number generation | Official AMD RNG library |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rand (Rust) | latest | CPU fallback RNG | When GPU unavailable |
| FlashInfer patterns | 2025 | Algorithm reference | Rejection sampling approach |

### Existing Codebase Components
| Component | Purpose | Status |
|-----------|---------|--------|
| `src/sampler/gpu.rs` | GPU sampler structs | Implemented, uses CPU fallback |
| `src/sampler/sampler.rs` | CPU sampler | Fully functional |
| `kernels/*.hip` | GPU kernel sources | Exist but not compiled |
| `src/backend/hip_backend/` | HIP FFI bindings | Provides `load_module`, `get_kernel_function`, `launch_kernel_with_module_shared` |

**Installation:**
Sampling kernels use ROCm which is already a dependency. rocRAND linkage needs to be verified in build.rs.

## Architecture Patterns

### Current Kernel Infrastructure

The project uses a consistent pattern for kernel loading and execution:

```rust
// Pattern from src/attention/kernels.rs and src/mlp/kernels.rs

// 1. Load kernel module from HSACO file
let module = backend.load_module(&hsaco_path)?;

// 2. Get kernel function by name
let kernel = backend.get_kernel_function(&module, "kernel_name")?;

// 3. Prepare kernel arguments (all as mut locals)
let mut arg1 = &value1 as *const _ as *mut c_void;
let args: &[*mut c_void] = &[arg1, arg2, ...];

// 4. Launch kernel
backend.launch_kernel_with_module_shared(
    kernel,
    grid_dim,
    block_dim,
    args,
    shared_mem_bytes,
)?;
```

### Recommended Multi-Kernel Pipeline

Based on FlashInfer's approach and the documented watchdog timeout issues:

```
Input: logits [batch, vocab_size]
  |
  v
[Kernel 1] temperature_scale_kernel (optional)
  | Divides logits by temperature
  v
[Kernel 2] softmax_kernel
  | Converts logits to probabilities (row-wise)
  v
[Kernel 3] topp_threshold_kernel OR topk_threshold_kernel
  | Computes threshold for filtering
  v
[Kernel 4] sampling_kernel (with rejection sampling loop)
  | Dual-pivot rejection sampling for convergence
  v
Output: sampled_tokens [batch]
```

**Why multi-kernel**: Single-kernel approaches with large nested loops over vocab_size=151936 trigger GPU watchdog timeout (~1-2 seconds). Breaking into multiple kernels allows each to complete quickly.

### Kernel File Organization

```
kernels/
├── sampling_utils.hip      # Softmax, temperature scale, prefix sum
├── topk_sampling.hip       # Top-k only sampling (partially implemented)
├── topp_sampling.hip       # Top-p only sampling (STUB - needs rewrite)
├── topk_topp_sampling.hip  # Fused top-k + top-p (partially implemented)
```

### Pattern: Global Kernel Cache

The project uses a global cache pattern for kernel modules (from `src/sampler/gpu.rs`):

```rust
static GLOBAL_SAMPLING_CACHE: Mutex<Option<SamplingKernelCache>> = Mutex::new(None);

fn get_or_init_sampling_cache() -> Result<&'static Mutex<...>, HipError> {
    // Double-checked locking pattern
    // Load modules once, reuse for all sampler instances
}
```

**Key insight**: Do NOT store `HipBackend` in the cache (causes separate stream issues). Pass backend as parameter to kernel launch functions.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU Random Numbers | `rand::thread_rng()` on CPU | rocRAND `rocrand_generate_uniform` | GPU-generated avoids CPU-GPU sync |
| Prefix Sum / CDF | Manual loop implementations | Parallel scan (CUB-style) | O(n) serial is too slow for 151k vocab |
| Sorting | Naive O(v*k) selection | FlashInfer's sorting-free rejection | Avoids O(v log v) bottleneck |
| Numerical Stability | Ad-hoc max subtraction | Block-level reduce with shared memory | Prevents NaN propagation |

**Key insight**: The existing kernels use naive O(v*k) selection which causes watchdog timeouts. FlashInfer's rejection sampling avoids sorting entirely.

## Common Pitfalls

### Pitfall 1: GPU Watchdog Timeout
**What goes wrong**: Single-threaded loops over large vocabulary (151936 tokens) exceed GPU watchdog timeout (~1-2 seconds), causing system hang.

**Why it happens**: The current `topk_sampling.hip` and `topp_sampling.hip` use thread-0-only loops:
```cpp
if (tid == 0) {
    // Single thread processes 151936 elements - TOO SLOW
    for (int i = 0; i < vocab_size; i++) { ... }
}
```

**How to avoid**:
1. Use multi-kernel approach (each completes in <1ms)
2. Parallelize with wave-level primitives (RDNA3 wave32)
3. Use rocRAND for random values on GPU

**Warning signs**: Kernel launch succeeds but system hangs, GPU resets, or timeout errors.

### Pitfall 2: Stream Ownership Issues
**What goes wrong**: Storing `HipBackend` in sampler cache causes separate HIP streams, leading to hangs.

**Why it happens**: Each `HipBackend` has its own stream. Using different streams for dependent operations breaks synchronization.

**How to avoid**: Pass `HipBackend` as parameter to sampling functions, don't store in cache:
```rust
// WRONG
struct GpuTopPSampler {
    backend: Arc<HipBackend>,  // Separate stream!
}

// BETTER
fn sample(backend: &HipBackend, ...)  // Use caller's stream
```

**Documented in**: `src/sampler/gpu.rs:24-27`

### Pitfall 3: Prefix Sum Numerical Instability
**What goes wrong**: Parallel prefix sum (CDF) produces non-monotonic output due to floating-point non-associativity.

**Why it happens**: Floating-point addition is not associative: `(a + b) + c != a + (b + c)` in parallel execution.

**How to avoid**: Use verified parallel scan implementation or add tolerance checks.

**Documented in**: FlashInfer blog: "parallel prefix-sum cannot guarantee monotonic outputs even with non-negative inputs"

### Pitfall 4: Missing from build.rs
**What goes wrong**: Kernel source files exist but aren't compiled, causing runtime fallback to CPU.

**Why it happens**: New kernel files not added to build.rs compilation list.

**How to avoid**: Always add kernels to build.rs kernel list:
```rust
let kernels = [
    // ... existing kernels ...
    ("kernels/sampling_utils.hip", "SAMPLING_UTILS_HSACO", "softmax_kernel"),
    ("kernels/topk_sampling.hip", "TOPK_SAMPLING_HSACO", "topk_sampling_kernel"),
    ("kernels/topp_sampling.hip", "TOPP_SAMPLING_HSACO", "topp_sampling_kernel"),
    ("kernels/topk_topp_sampling.hip", "TOPK_TOPP_SAMPLING_HSACO", "topk_topp_sampling_kernel"),
];
```

## Code Examples

### Existing Kernel Loading Pattern

Verified from `src/sampler/gpu.rs:74-85`:
```rust
let softmax_path = std::env::var("SOFTMAX_HSACO")
    .unwrap_or_else(|_| "kernels/softmax.hsaco".to_string());

let (softmax_module, softmax_kernel) = if Path::new(&softmax_path).exists() {
    let module = load_backend.load_module(&softmax_path)?;
    let kernel = load_backend.get_kernel_function(&module, "softmax_kernel")?;
    (Some(module), Some(kernel))
} else {
    tracing::warn!("Softmax kernel not found at {}, using CPU fallback", softmax_path);
    (None, None)
};
```

### Kernel Launch Pattern

Verified from `src/sampler/gpu.rs:197-220`:
```rust
// Prepare kernel arguments - ALL args must be copied to mut locals first
let mut probabilities_arg = probabilities;
let mut random_values_arg = random_values;
let mut output_arg = output;
let mut top_p_arg = top_p;
let mut batch_size_arg = batch_size;
let mut vocab_size_arg = vocab_size;

let args: &[*mut c_void] = &[
    &mut probabilities_arg as *mut _ as *mut c_void,
    &mut random_values_arg as *mut _ as *mut c_void,
    &mut output_arg as *mut _ as *mut c_void,
    &mut top_p_arg as *mut _ as *mut c_void,
    &mut batch_size_arg as *mut _ as *mut c_void,
    &mut vocab_size_arg as *mut _ as *mut c_void,
];

backend.launch_kernel_with_module_shared(
    kernel,
    grid_dim,
    block_dim,
    args,
    shared_mem_bytes,
).map_err(|e| format!("Failed to launch top-p kernel: {:?}", e))?;
```

### Softmax Kernel Example (from sampling_utils.hip)

Source: `kernels/sampling_utils.hip:39-124`
```cpp
extern "C" __global__ void softmax_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probabilities,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int row_offset = batch_idx * vocab_size;

    // Shared memory for reduction
    __shared__ float s_max;
    __shared__ float s_sum;

    // Step 1: Find max value in row (for numerical stability)
    // [parallel reduction implementation]

    // Step 2: Compute exp(x - max) and sum
    // [parallel reduction implementation]

    // Step 3: Normalize by sum
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        probabilities[row_offset + i] *= sum_inv;
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sorting-based top-k/p | Sorting-free rejection sampling | FlashInfer 2025 | 50%+ speedup in sampling |
| Single-kernel sampling | Multi-kernel pipeline | FlashInfer v0.2.3 | O(log K) complexity |
| CPU random generation | GPU random (rocRAND) | Industry standard | Eliminates CPU-GPU sync |

**Deprecated/outdated:**
- **Single-threaded kernel loops**: Cause watchdog timeout on large vocab
- **Sorting-based selection**: O(v log v) bottleneck for v=151936
- **CPU fallback as default**: Should be temporary, not primary path

## Open Questions

1. **rocRAND Integration**
   - What we know: rocRAND 4.2.0 provides `rocrand_generate_uniform()` for GPU random values
   - What's unclear: Whether rocRAND is currently linked in build.rs (only hipblas, amdhip64, hiprtc are linked)
   - Recommendation: Verify rocRAND linkage, add if missing

2. **Watchdog Timeout Timeout Threshold**
   - What we know: GPU watchdog triggers at ~1-2 seconds
   - What's unclear: Exact timeout value for target hardware
   - Recommendation: Measure with timing kernel, target <500ms per kernel

3. **Vocabulary Size Handling**
   - What we know: Current models use vocab_size=151936
   - What's unclear: Whether block-based prefix sum can handle this efficiently
   - Recommendation: Test with actual model vocab sizes

## Sources

### Primary (HIGH confidence)
- [FlashInfer: Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html) - Algorithm reference for dual-pivot rejection sampling
- [ROCm rocRAND Host API Documentation](https://rocm.docs.amd.com/projects/rocRAND/en/develop/doxygen/html/group__rocrandhost.html) - GPU random number generation
- Source code analysis: `kernels/*.hip` - Existing kernel implementations
- Source code analysis: `src/sampler/gpu.rs` - Current GPU sampler implementation
- Source code analysis: `src/sampler/sampler.rs` - CPU reference implementation
- Source code analysis: `src/backend/hip_backend/backend.rs` - HIP backend FFI

### Secondary (MEDIUM confidence)
- [ROCm/rocRAND GitHub Repository](https://github.com/ROCm/rocRAND) - Library source and examples

### Tertiary (LOW confidence)
- None - all findings verified against primary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Verified from ROCm documentation
- Architecture: HIGH - Verified from existing codebase patterns
- Pitfalls: HIGH - Documented in existing code (watchdog timeout comments)
- Algorithms: HIGH - Verified from FlashInfer blog with mathematical proofs

**Research date:** 2026-01-19
**Valid until:** 30 days (FlashInfer patterns are stable, ROCm API evolves slowly)
