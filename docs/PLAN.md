# ROCmForge Implementation Plan

> GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
> Last Updated: 2026-01-03
> Rule: **Make it correct → make it measurable → then make it fast.**

---

## Current Status

| Phase | Description | Status | Tests | Date |
|-------|-------------|--------|-------|------|
| Phase 1 | Basic kernels (scale, mask, softmax) | ✅ Complete | 3/3 | 2025-01-03 |
| Phase 2 | RoPE + KV Append | ✅ Complete | 5/5 | 2025-01-03 |
| Phase 3a | Non-Causal FlashAttention | ✅ Complete | 17/17 | 2025-01-03 |
| Phase 3b | Causal Masking | ✅ Complete | 8/8 | 2025-01-03 |
| Phase 4 | MLP Ops (SwiGLU, RMSNorm) | ✅ Complete | 8/8 | 2026-01-03 |
| Phase 5 | Optional Optimizations | Pending | - | - |

**Progress**: 4/5 phases complete (41/41 tests passing)

---

## Phase 4: MLP Ops - Summary ✅

### What Was Done

1. **SwiGLU Activation Kernel** (`kernels/swiglu.hip`)
   - Formula: `SwiGLU(x) = gate(x) * swish(up(x))` where `swish(x) = x * sigmoid(x)`
   - Element-wise operation (no reduction)
   - 5 tests passing

2. **RMSNorm Kernel** (`kernels/rms_norm.hip`)
   - Formula: `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight`
   - Row-wise reduction with shared memory
   - 3 tests passing

3. **GPU-Only Integration**
   - Replaced CPU fallback in `src/backend/hip_backend.rs`
   - Verified `HipBuffer::copy_from_buffer` uses `hipMemcpyDeviceToDevice`
   - No `to_host_vec` in MLP forward pass

### Key Technical Discoveries

1. **Kernel Argument Pattern**: ALL arguments (including pointers) must go through intermediate mutable variables
2. **Parallel Reduction**: Starting stride must be `BLOCK_SIZE / 2` to process all elements
3. **GPU-to-GPU Copy**: Direct device-to-device memory copy confirmed

---

## Phase 5: Optional Optimizations (Next)

> These are performance optimizations, not required for correctness.
> Profile first, then optimize what actually matters.

---

## Phase 4.5: GGUF Loader Vocab Size Inference ✅ COMPLETE

**Priority:** High - Required for model compatibility
**Status:** Complete
**Completed:** 2026-01-04

### Problem

Some GGUF models do not include `{architecture}.vocab_size` metadata, causing:
- `vocab_size = 0` in GgufMetadata
- Potential crashes or incorrect model initialization
- Models fail to load even though tensor data is valid

### Solution Implemented

Infer vocab_size from tensor shapes when metadata is missing:
1. Check for `token_embd.weight`, `output.weight`, `lm_head.weight`, or `embed_tokens.weight` tensors
2. Use tensor shape `[vocab_size, hidden_size]` or `[hidden_size, vocab_size]`
3. Compare against known `hidden_size` to determine which dimension is vocab_size
4. Fallback to architecture-specific defaults if inference fails

### Files Modified
- `src/loader/gguf.rs` - Added inference logic (lines 671-712, modified 229-278)

### Summary
- Helper method `infer_vocab_size_from_tensors()` implemented ✅
- `to_model_config()` uses inferred vocab_size as fallback ✅
- Code compiles successfully ✅
- Documentation updated ✅

### Technical Details

**Method Added:**
- `infer_vocab_size_from_tensors()` - Infers vocab_size from tensor shapes
- Searches 4 common tensor names
- Compares against known hidden_size
- Uses heuristic when hidden_size unknown
- Debug logging for transparency

**Integration:**
- `to_model_config()` now has 3-tier fallback:
  1. Metadata vocab_size (if > 0)
  2. Inferred from tensor shapes
  3. Architecture-specific defaults (qwen2: 151936, llama: 32000, glm: 151552)

### 5.1: GPU Sampler (top-k/top-p)

**Goal**: Move token sampling to GPU to overlap with next token computation.

**Current**: CPU-side sampling after decoding
**Target**: GPU-side sampling with device-to-host transfer only for final token

**Tasks**:
- [ ] Profile current sampling latency
- [ ] Implement `top_k_kernel` (select top-k values per row)
- [ ] Implement `top_p_kernel` (nucleus sampling)
- [ ] Implement `sample_kernel` (weighted random selection)
- [ ] Integrate into decode loop

**Expected Impact**: ~10-20% latency reduction for decode steps

---

### 5.2: Custom GEMM Kernels

**Goal**: Evaluate if rocBLAS GEMM is a bottleneck.

**Prerequisite**: Profile with rocprofiler to identify hot paths

**Tasks**:
- [ ] Profile current matmul performance
- [ ] Compare rocBLAS vs custom implementation
- [ ] If rocBLAS is fast enough: skip
- [ ] If not: implement tiled GEMM with LDS

**Expected Impact**: Unknown (profiling required)

**Note**: RDNA3 does not have MFMA instructions, so matrix multiply acceleration
is limited compared to CDNA3 datacenter GPUs.

---

### 5.3: FP16 Support

**Goal**: Use half precision for weights and activations to reduce memory bandwidth.

**Tasks**:
- [ ] Convert weights to FP16 (loader changes)
- [ ] Implement FP16 kernels (or use __half in HIP)
- [ ] Careful numerical stability testing
- [ ] Benchmark FP16 vs FP32 latency

**Expected Impact**: ~2x memory bandwidth reduction, ~1.5-2x speedup

**Risks**:
- Numerical underflow/overflow
- Gradient scaling issues (if training)

---

### 5.4: Wave64 Tuning (CDNA3)

**Goal**: Optimize for CDNA3 GPUs (wave64 instead of wave32).

**Current**: Tuned for RDNA3 (wave32, RX 7900 XT)
**Target**: CDNA3 (MI300X, wave64)

**Tasks**:
- [ ] Detect GPU architecture at runtime
- [ ] Implement wave64 variants of kernels
- [ ] Tune block sizes (512 threads instead of 256)
- [ ] Benchmark both configurations

**Expected Impact**: Better performance on CDNA3 datacenter GPUs

---

## Future Work (Beyond Phase 5)

### Quantization
- INT8/INT4 weight quantization
- Quantization-aware training

### Multi-GPU
- Tensor parallelism for large models
- Pipeline parallelism for layer distribution

### Compilation
- AOT compilation for specific model architectures
- Kernel fusion for entire layers

---

## Hardware Reference

| Component | Value |
|-----------|-------|
| **GPU** | AMD Radeon RX 7900 XT (Navi 31) |
| **Architecture** | gfx1100 (RDNA3) |
| **Wavefront Size** | 32 (not 64!) |
| **ROCm** | 7.1.52802 |
| **Target Flag** | --offload-arch=gfx1100 |
| **Block Size** | 256 threads (8 waves of 32) |
| **Reduction** | stride starts at 16 for wave32 |

### Block Size Formula

```cpp
// For RDNA3 (wave32)
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;

// Wave32 reduction
for (int stride = 16; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
    __syncthreads();
}
```

### For CDNA3 (future)

```cpp
// For CDNA3 (wave64)
constexpr int BLOCK_SIZE = 512;  // 8 waves of 64 threads
constexpr int WARP_SIZE = 64;

// Wave64 reduction
for (int stride = 32; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
    __syncthreads();
}
```

---

## Test Coverage

### Current Tests (41 total)

| Category | Tests | File |
|----------|-------|------|
| Basic kernels | 3 | `kernel_tests.rs` |
| RoPE | 5 | `rope_gpu_tests.rs` |
| QK^T matmul | 4 | `qkt_matmul_tests.rs` |
| Softmax explicit | 4 | `softmax_explicit_tests.rs` |
| Weighted matmul | 4 | `weighted_matmul_tests.rs` |
| Flash non-causal | 5 | `flash_nocausal_tests.rs` |
| Causal mask | 4 | `causal_mask_tests.rs` |
| Flash causal | 4 | `flash_causal_tests.rs` |
| SwiGLU | 5 | `swiglu_tests.rs` |
| RMSNorm | 3 | `rms_norm_tests.rs` |

### Running Tests

```bash
# All tests
cargo test --features rocm

# Specific phase
cargo test --features rocm --lib mlp

# Specific test
cargo test --features rocm --lib test_swiglu_matches_cpu_small

# With output
cargo test --features rocm --lib -- --nocapture
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `CHANGELOG.md` | Chronological history of all changes |
| `docs/TODO.md` | Detailed task tracking with progress |
| `docs/PLAN.md` | This file - roadmap and future work |
| `docs/implementation_roadmap.md` | Original plan with contract specs |
| `docs/implementation_principles.md` | Engineering methodology |
| `docs/kernel_research.md` | Kernel implementation research |
| `docs/codebase_audit.md` | Current state audit |

---

## Quick Command Reference

```bash
# Build
cargo build --features rocm

# Clean build
cargo clean && cargo build --features rocm

# Release build
cargo build --features rocm --release

# Run tests
cargo test --features rocm

# GPU monitoring
watch -n 1 rocm-smi

# Check GPU info
rocm-smi --showproductname
rocm-smi --showmem
rocm-smi --showuse
```
