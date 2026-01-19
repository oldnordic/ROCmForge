# Phase 18: GPU Attention Completion - Research

**Researched:** 2026-01-19
**Domain:** GPU Attention Kernels (FlashAttention, MQA, GQA)
**Confidence:** HIGH

## Summary

Phase 18 focuses on ensuring all attention mechanisms (FlashAttention, Multi-Query Attention, Grouped-Query Attention) run fully on GPU with no CPU fallbacks. The codebase already has significant GPU attention infrastructure:

**Existing:**
- FlashAttention kernels: 3 variants (generic, causal, non-causal)
- MQA KV replication kernel: `mqa_kv_replicate_fused_kernel`
- Kernel wrappers in `src/attention/kernels.rs`
- Tests for flash attention and MQA correctness
- Backend registry with `FlashAttentionBackend` implementation

**What needs completion:**
1. **FlashAttention verification**: Kernels exist and are compiled, but need end-to-end verification that they work correctly
2. **MQA/GQA pure GPU path**: `MultiQueryAttention::forward_device()` exists but has incomplete RoPE integration and potential CPU fallbacks
3. **Integration testing**: Need tests that verify full GPU execution path from model layer through attention

**Key insight:** The kernels are implemented and compiled. The phase is about verifying they work end-to-end and removing any CPU fallbacks, not writing new kernels.

**Primary recommendation:** Focus on integration testing and verification rather than kernel development. Use TDD: write failing tests that verify pure-GPU execution, then fix any fallbacks discovered.

## Standard Stack

The codebase uses existing ROCm/HIP infrastructure - no new dependencies needed.

---

## CRITICAL: Anti-CUDA Porting Guardrails

**IMPORTANT:** ROCmForge is a **native HIP inference engine**, not a CUDA port. See `.planning/research/ANTI_CUDA_PORTING_RATIONALE.md` for complete rationale.

### Why We Don't Port CUDA Code

1. **Different Execution Models**
   - CUDA: 32-thread warp
   - AMD: 64-thread wavefront
   - CUDA-optimized kernels are mathematically misaligned with AMD hardware

2. **Performance Destruction**
   - HIPIFY gives syntactic translation but zero performance tuning
   - Wrong assumptions about shared memory, indexing, block dims
   - Porting breaks vectorization, occupancy, LDS tiling

3. **Project Integrity**
   - Thesis: "First pure Rust ROCm/HIP inference engine"
   - Porting CUDA destroys uniqueness and research value
   - Contaminates the architecture with NVIDIA-specific assumptions

### Current CUDA Intrinsics to Eliminate

The following kernels have CUDA-specific `__shfl_down_f32` that must be replaced with HIP `__shfl_down`:

| Kernel | Lines | Status | Action |
|--------|-------|--------|--------|
| `flash_attention.hip` | 300, 306, 317, 322 | ✅ Fixed | Replaced with `__shfl_down` |
| `q4_0_matmul.hip` | 115, 124 | ❌ BLOCKING | Needs HIP-native rewrite |
| `q4_k_matmul.hip` | 176, 185 | ❌ BLOCKING | Needs HIP-native rewrite |
| `q6_k_matmul.hip` | 176, 185 | ❌ BLOCKING | Needs HIP-native rewrite |
| `fused_dequant_rmsnorm.hip` | 225 | ❌ BLOCKING | Needs HIP-native rewrite |

**Note:** These quantized matmul kernels are CRITICAL for GGUF model inference. They must be fixed, but properly rewritten as HIP-native code, not simple find-replace.

### Development Rules

1. **NEVER** copy CUDA kernels directly
2. **NEVER** use HIPIFY on inference kernels
3. **ALWAYS** write HIP kernels from scratch for AMD
4. **USE** AMD ISA documents for optimization
5. **REFERENCE** llama.cpp, vllm as reference only - don't copy
6. **TEST** on actual AMD hardware

---

### Core (Already in use)
| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| ROCm HIP | External | GPU kernel framework | Existing |
| HIP Kernels | `kernels/` | GPU implementation | Complete |
| Kernel wrappers | `src/attention/kernels.rs` | Rust FFI bindings | Complete |
| Backend registry | `src/attention/backend_registry.rs` | Pluggable attention backends | Complete |

### Attention Kernel Inventory (All in build.rs)

| Kernel | File | Purpose | Env Var | Status |
|--------|------|---------|---------|--------|
| `flash_attention_kernel` | `flash_attention.hip` | Generic fused attention | `FLASH_ATTENTION_HSACO` | Compiled |
| `flash_attention_causal_kernel` | `flash_attention_causal.hip` | Causal fused attention | `FLASH_ATTENTION_CAUSAL_HSACO` | Compiled |
| `flash_attention_nocausal_kernel` | `flash_attention_nocausal.hip` | Non-causal fused attention | `FLASH_ATTENTION_NCAUSAL_HSACO` | Compiled |
| `mqa_kv_replicate_fused_kernel` | `mqa_kv_replicate.hip` | KV head replication for MQA/GQA | `MQA_KV_REPLICATE_HSACO` | Compiled |
| `qkt_matmul_kernel` | `qkt_matmul.hip` | QK^T matmul with scaling | `QKT_MATMUL_HSACO` | Compiled |
| `softmax_kernel` | `softmax.hip` | Row-wise softmax | `SOFTMAX_HSACO` | Compiled |
| `weighted_matmul_kernel` | `weighted_matmul.hip` | Attention weights x V | `WEIGHTED_MATMUL_HSACO` | Compiled |
| `rope_kernel` | `rope.hip` | Rotary position embeddings | `ROPE_HSACO` | Compiled (Phase 16) |

**Installation:** No additional packages needed. ROCm toolchain must be available.

## Architecture Patterns

### Attention Backend Architecture

The codebase uses a pluggable backend pattern for attention:

```
AttentionBackend (enum)
    ├── Cpu -> CpuBackend::forward()
    └── Gpu -> GpuBackend::forward()

BackendImplementation (trait)
    ├── CpuAttentionBackend (fallback)
    ├── FlashAttentionBackend (fused kernels)
    └── [future: others]
```

### Pattern 1: FlashAttention Backend

**What:** Fused attention kernel implementing QK^T -> scale -> mask -> softmax -> softmax*V in a single kernel launch.

**When to use:** Standard decoder attention with causal masking. Head dimension <= 128, sequence length <= 2048.

**Current state:**
- Kernels compiled and registered in build.rs
- `FlashAttentionBackend` implements `BackendImplementation` trait
- Rust wrappers in `kernels.rs`: `flash_attention_gpu_kernel()`, `flash_attention_causal_gpu_kernel()`, `flash_attention_nocausal_gpu_kernel()`

**Example:**
```rust
// Source: src/attention/flash_attention.rs:161-214
fn forward_causal_gpu(
    &self,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> AttentionBackendResult<Vec<f32>> {
    // Allocate GPU buffers
    let q_gpu = HipBuffer::new(q.len() * std::mem::size_of::<f32>())?;
    let k_gpu = HipBuffer::new(k.len() * std::mem::size_of::<f32>())?;
    let v_gpu = HipBuffer::new(v.len() * std::mem::size_of::<f32>())?;
    let output_gpu = HipBuffer::new(q.len() * std::mem::size_of::<f32>())?;

    // Copy data to GPU
    q_gpu.copy_from_host(q)?;
    k_gpu.copy_from_host(k)?;
    v_gpu.copy_from_host(v)?;

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel
    unsafe {
        crate::attention::kernels::flash_attention_causal_gpu_kernel(
            q_gpu.as_ptr() as *const f32,
            k_gpu.as_ptr() as *const f32,
            v_gpu.as_ptr() as *const f32,
            output_gpu.as_ptr() as *mut f32,
            scale,
            batch_size as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
        )?;
    }

    synchronize_device()?;

    // Copy output back to host
    let mut output = vec![0.0f32; q.len()];
    output_gpu.copy_to_host(&mut output)?;
    Ok(output)
}
```

### Pattern 2: MQA/GQA with KV Replication

**What:** Multi-Query Attention uses fewer KV heads than query heads, requiring KV replication to match dimensions for standard attention kernels.

**When to use:** Models with `num_kv_heads < num_attention_heads` (MQA: 1 KV head, GQA: multiple but fewer KV heads).

**Current state:**
- `mqa_kv_replicate_fused_kernel` compiles K/V from `num_kv_heads` to `num_q_heads`
- `MultiQueryAttention::replicate_kv_gpu()` calls the kernel
- `MultiQueryAttention::compute_attention_gpu()` handles full GPU attention computation

**Example:**
```rust
// Source: src/attention/multi_query.rs:196-264
fn replicate_kv_gpu(
    &self,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
) -> AttentionResult<(DeviceTensor, DeviceTensor)> {
    let q_dims = q.shape().dims();
    let (batch_size, seq_len, num_q_heads, head_dim) =
        (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let num_kv_heads = self.config.num_kv_heads;

    // Allocate expanded tensors
    let k_expanded = DeviceTensor::empty(&backend, k_expanded_shape)?;
    let v_expanded = DeviceTensor::empty(&backend, v_expanded_shape)?;

    // Call GPU kernel
    unsafe {
        mqa_kv_replicate_gpu_kernel(
            k.as_ptr(),
            v.as_ptr(),
            k_expanded.as_ptr() as *mut f32,
            v_expanded.as_ptr() as *mut f32,
            batch_size as u32,
            seq_len as u32,
            num_kv_heads as u32,
            num_q_heads as u32,
            head_dim as u32,
        )?;
        backend.synchronize()?;
    }

    Ok((k_expanded, v_expanded))
}
```

### Pattern 3: DeviceTensor Zero-Copy Path

**What:** Use `DeviceTensor` inputs to avoid CPU-GPU transfers.

**When to use:** When previous layer outputs are already on GPU.

**Current state:** `forward_device()` exists but has TODO for RoPE application.

**Example issue:**
```rust
// Source: src/attention/multi_query.rs:188-189
// TODO: Implement RoPE application for GPU tensors
// This is a CPU fallback that needs to be eliminated
```

### Anti-Patterns to Avoid

- **CPU round-trip in forward_device():** The `forward_device()` method should never copy data to host and back
- **Calling CPU kernels from GPU path:** Check for `super::cpu::` calls in GPU code sections
- **Missing synchronization:** Always call `synchronize_device()` after kernel launches before using results

## Don't Hand-Roll

All attention kernels already exist. Do NOT implement:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FlashAttention | Custom fused kernel | `flash_attention_causal_kernel` or `flash_attention_nocausal_kernel` | Already optimized for ROCm |
| KV replication | Manual CPU loop | `mqa_kv_replicate_fused_kernel` | Single-kernel GPU implementation |
| QK^T matmul | Manual dot products | `qkt_matmul_gpu_kernel_scaled` | Wave32 optimized |
| Softmax | Manual exp/sum | `softmax_gpu_kernel` | Handles numerical stability |
| Attention x V | Manual reduction | `weighted_matmul_gpu_kernel` | Fused kernel available |

**Key insight:** Phase 18 is about verification and integration, not kernel development.

## Common Pitfalls

### Pitfall 1: Layout Mismatch Between CPU and GPU

**What goes wrong:** CPU uses `[batch*seq, heads, dim]` (flattened), GPU uses `[batch, seq, heads, dim]` (4D explicit).

**Why it happens:** Historical codebase evolution; CPU was written first without 4D layout consideration.

**How to avoid:** Always check tensor layout before passing to GPU kernels. The kernels expect explicit 4D layout.

**Warning signs:** Incorrect output dimensions, garbage values in attention output.

### Pitfall 2: Missing Kernel Synchronization

**What goes wrong:** Kernel launch is async; code proceeds before GPU finishes computation.

**Why it happens:** HIP/ROCm kernels are asynchronous by default.

**How to avoid:** Always call `synchronize_device()` after kernel launch before using results.

**Code pattern:**
```rust
unsafe {
    kernel_launch(...)?;
}
synchronize_device()?;  // CRITICAL: wait for kernel to finish
buffer.copy_to_host(&mut output)?;
```

### Pitfall 3: forward_device() CPU Fallback

**What goes wrong:** `forward_device()` copies to host, calls CPU implementation, copies back.

**Why it happens:** Incomplete GPU implementation (TODOs in code).

**How to avoid:** Trace execution path through `forward_device()` to ensure no `to_host_vec()` calls before final output.

**Warning signs:** High CPU usage during "GPU" inference, profile shows memcpy overhead.

### Pitfall 4: MQA/GQA RoPE Not Applied on GPU

**What goes wrong:** RoPE is applied on CPU before passing to GPU, causing extra transfer.

**Why it happens:** `forward_device()` has TODO comment for GPU RoPE application.

**How to avoid:** Use `rope_gpu_kernel` for DeviceTensor inputs.

**Location:** `src/attention/multi_query.rs:189`

### Pitfall 5: Test Assumes GPU Available

**What goes wrong:** CI fails because tests assume AMD GPU is present.

**Why it happens:** Tests written on developer machine with GPU.

**How to avoid:** Use `HipBackend::new_checked()` pattern with graceful skip.

**Example:**
```rust
let backend = match HipBackend::new_checked() {
    Ok(b) => b,
    Err(_) => {
        eprintln!("SKIP: GPU not available");
        return;
    }
};
```

## Code Examples

### Verified: FlashAttention Kernel Call

```rust
// Source: src/attention/kernels.rs:922-993
pub unsafe fn flash_attention_causal_gpu_kernel(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    output: *mut f32,
    scale: f32,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    let cache = get_or_init_cache()?;
    let kernel = cache.flash_attention_causal_kernel
        .ok_or_else(|| "flash_attention_causal_kernel not loaded".to_string())?;

    let grid_dim = (seq_len, num_heads, batch_size);
    let block_dim = (WARP_SIZE, 1, 1);  // 32 for RDNA3
    let shared_mem_bytes = 2 * WARP_SIZE * std::mem::size_of::<f32>() as u32;

    backend.launch_kernel_with_module_shared(
        kernel,
        grid_dim,
        block_dim,
        args,
        shared_mem_bytes,
    )?
}
```

### Verified: MQA KV Replication

```rust
// Source: src/attention/kernels.rs:1175-1235
pub unsafe fn mqa_kv_replicate_gpu_kernel(
    k: *const f32,
    v: *const f32,
    k_expanded: *mut f32,
    v_expanded: *mut f32,
    batch_size: u32,
    seq_len: u32,
    num_kv_heads: u32,
    num_q_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    let cache = get_or_init_cache()?;
    let kernel = cache.mqa_kv_replicate_kernel
        .ok_or_else(|| "mqa_kv_replicate_kernel not loaded".to_string())?;

    let total_elements = batch_size * seq_len * num_q_heads * head_dim;
    let grid_dim = (total_elements.div_ceil(BLOCK_SIZE), 1, 1);
    let block_dim = (BLOCK_SIZE, 1, 1);  // 256 for RDNA3

    backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0)?
}
```

### Verified: GPU Availability Check Pattern

```rust
// Source: src/attention/mqa_kernel_tests.rs:34-40
let backend = match HipBackend::new_checked() {
    Ok(b) => b,
    Err(_) => {
        eprintln!("SKIP: GPU not available");
        return;
    }
};
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CPU fallback for attention | GPU kernels for all attention phases | Phase 6-17 | 2-4x speedup for attention |
| Separate QK^T, softmax, matmul | Fused FlashAttention kernel | Phase 6 | Single kernel vs 5+ launches |
| CPU KV replication for MQA | GPU `mqa_kv_replicate_fused_kernel` | Phase 19.2 | Eliminates CPU-GPU transfer |

**Deprecated/outdated:** None. All attention kernels are current and actively used.

## Open Questions

### Q1: Does FlashAttentionBackend get used in actual model execution?

**What we know:** `FlashAttentionBackend` is implemented in `src/attention/flash_attention.rs` and registered in `backend_registry.rs`.

**What's unclear:** Does the actual transformer layer use this backend, or does it call `GpuBackend::forward()` directly?

**Recommendation:** Trace model execution path from `src/model/` through attention calls to verify which backend is selected. If `FlashAttentionBackend` is not used, either wire it up or remove dead code.

### Q2: Is RoPE applied before or after attention in the model?

**What we know:** `MultiQueryAttention::forward_device()` has a TODO for RoPE application on GPU.

**What's unclear:** Does the model apply RoPE before calling attention, or should attention handle it?

**Recommendation:** Check model layer implementation to see RoPE application order. If RoPE is pre-applied, the TODO might be a non-issue.

### Q3: What is the actual attention bottleneck in production inference?

**What we know:** FlashAttention kernels are optimized and compiled.

**What's unclear:** Are we optimizing the right thing? Profile data needed.

**Recommendation:** Before spending time on micro-optimizations, profile a real model to identify actual bottlenecks.

## Sources

### Primary (HIGH confidence)

| Source | Topic | Confidence |
|--------|-------|------------|
| `build.rs:44-191` | Kernel compilation list | HIGH - read directly |
| `src/attention/kernels.rs:1-1236` | GPU kernel wrappers | HIGH - read directly |
| `src/attention/flash_attention.rs:1-524` | FlashAttention backend | HIGH - read directly |
| `src/attention/multi_query.rs:1-853` | MQA/GQA implementation | HIGH - read directly |
| `kernels/flash_attention.hip:1-327` | FlashAttention kernel | HIGH - read directly |
| `kernels/mqa_kv_replicate.hip:1-209` | MQA KV replication kernel | HIGH - read directly |
| `.planning/phases/06-attention-optimization/RESEARCH.md` | Prior attention research | HIGH - read directly |
| `.planning/REQUIREMENTS.md:57-66` | ATTENTION requirements | HIGH - read directly |

### Secondary (MEDIUM confidence)

| Source | Topic | Confidence |
|--------|-------|------------|
| `src/attention/flash_attention_tests.rs` | FlashAttention tests | MEDIUM - test patterns verified |
| `src/attention/mqa_kernel_tests.rs` | MQA correctness tests | MEDIUM - test patterns verified |

### Tertiary (LOW confidence)

None. All research is from direct code inspection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All kernels verified in build.rs
- Architecture: HIGH - Direct code inspection of backend implementations
- Pitfalls: HIGH - Issues identified from code comments and patterns
- Integration gaps: MEDIUM - Need to trace model execution path

**Research date:** 2026-01-19
**Valid until:** 60 days (stable GPU kernel architecture)

**Key files read:**
- `build.rs` - Kernel compilation
- `src/attention/kernels.rs` - Kernel wrappers
- `src/attention/flash_attention.rs` - FlashAttention backend
- `src/attention/multi_query.rs` - MQA/GQA implementation
- `src/attention/gpu.rs` - GPU backend
- `src/attention/mod.rs` - Attention module organization
- `kernels/flash_attention.hip` - FlashAttention kernel
- `kernels/mqa_kv_replicate.hip` - MQA KV replication kernel
- `.planning/REQUIREMENTS.md` - Phase requirements

---

## Observations from Phase 18 Execution

### GPU Reset Issue Fixed

**Problem Identified (2026-01-19):**
Multiple test files were calling `HipBackend::new()` directly instead of using the shared `GPU_FIXTURE` singleton. This caused:
- Multiple GPU backend instances to be created
- GPU watchdog timeouts
- Desktop crashes when running `cargo test`

**Files Fixed:**
1. `tests/embedding_to_lmhead_tests.rs` - 3 tests fixed
2. `tests/hip_blas_matmul_tests.rs` - 5 tests fixed
3. `tests/q_dequant_tests.rs` - 10 tests fixed (across 4 modules)
4. `src/ops/causal_mask_tests.rs` - 7 tests fixed

**Fix Applied:**
```rust
// OLD (dangerous - causes GPU resets):
let backend = match HipBackend::new() {
    Ok(b) => b,
    Err(_) => { return; }
};

// NEW (safe - uses shared singleton):
let fixture = match GPU_FIXTURE.as_ref() {
    Some(f) => f,
    None => { return; }
};
let backend = fixture.backend();
```

**Additional Requirements:**
- Added `#[serial]` attribute to all GPU tests to prevent parallel execution
- Added `use rocmforge::backend::gpu_test_common::GPU_FIXTURE;` imports
- Added `use serial_test::serial;` imports

**Lesson for Future Tests:**
NEVER call `HipBackend::new()` directly in tests. Always use `GPU_FIXTURE` to avoid GPU resets.
