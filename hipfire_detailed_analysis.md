# hipfire Detailed Analysis - Features Applicable to GGUF Q4_0

**Date:** 2026-04-18
**Goal:** Identify features that can be implemented WITHOUT changing quantization format
**Focus:** Techniques applicable to current GGUF Q4_0 setup

---

## Part 1: Architecture-Aware Kernel Dispatch

### What hipfire does:

From `crates/rdna-compute/src/dispatch.rs`:

```rust
fn gemv_rows_default(arch: &str) -> u32 {
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => 1,  // RDNA3: single-row optimal
        "gfx1030" | "gfx1031" => 1,              // RDNA2: has specialized kernels
        _ => 2,                                   // RDNA1/APU: multi-row helps
    }
}
```

They detect GPU architecture at runtime and select optimized kernels:

- **gfx1010** (RX 5700 XT): No v_dot2_f32_f16, uses multi-row GEMV
- **gfx1013** (BC-250 APU): No v_dot2_f32_f16, needs workarounds
- **gfx1030** (RX 6900 XT): Has v_dot2_f32_f16, dp4a support
- **gfx1100** (RX 7900 XTX): Has WMMA, single-row optimal

### Feature detection:

```rust
fn has_dot2_f32_f16(arch: &str) -> bool {
    matches!(arch,
        "gfx1011" | "gfx1012"
        | "gfx1030" | "gfx1031" | "gfx1032"
        | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
        | "gfx1150" | "gfx1151"
        | "gfx1200" | "gfx1201")
}
```

**Apply to ROCmForge:**
- Current: Single kernel for all architectures
- Improvement: Architecture detection + kernel variants
- Implementation: Detect gfx arch, select optimal kernel path

---

## Part 2: Instruction-Level Optimizations (No Format Change)

### 2.1 DP4A (Dot Product Accumulate) for RDNA2+

**From:** `kernels/src/gemv_hfq4g256.gfx1030.v4.hip`

**Technique:** Use `v_dot4_i32_i8` (4-way int8 multiply-accumulate)

```cpp
// Pack nibbles into int32
int nib_even = (int)(b0 & 0xF)
             | ((int)(b1 & 0xF) << 8)
             | ((int)(b2 & 0xF) << 16)
             | ((int)(b3 & 0xF) << 24);

int nib_odd  = (int)(b0 >> 4)
             | ((int)(b1 >> 4) << 8)
             | ((int)(b2 >> 4) << 16)
             | ((int)(b3 >> 4) << 24);

// Quantize x to int8 on-the-fly
int xq_even = (xq0 & 0xFF) | ((xq2 & 0xFF) << 8)
            | ((xq4 & 0xFF) << 16) | ((xq6 & 0xFF) << 24);

// 2 dp4a instructions = 8 multiply-accumulates
int dot_sum = __builtin_amdgcn_sdot4(nib_even, xq_even, 0, false);
dot_sum = __builtin_amdgcn_sdot4(nib_odd, xq_odd, dot_sum, false);
```

**Why faster:**
- 2 instructions vs 8 FP32 multiplies
- Native hardware instruction on RDNA2+
- ~2-3× throughput for integer ops

**Adapt to Q4_0:**
- Q4_0 stores values as 0-15 (fits in int8)
- On-the-fly quantization of x values
- Same dp4a pattern works
- **Trade-off:** 0.4% noise from x quantization vs 2× speed

### 2.2 Packed 32-bit Loads

**Current ROCmForge (broken pattern):**
```cpp
const uint32_t q = reinterpret_cast<const uint32_t*>(b->qs)[i];
// Wrong bit extraction...
```

**hipfire pattern:**
```cpp
// Packed 32-bit load for 4 consecutive bytes
unsigned int pk = *(const unsigned int*)(nib + boff);
unsigned char b0 = pk & 0xFF;
unsigned char b1 = (pk >> 8) & 0xFF;
unsigned char b2 = (pk >> 16) & 0xFF;
unsigned char b3 = (pk >> 24) & 0xFF;
```

**Why better:**
- Single 32-bit load vs 4 × 8-bit loads
- Better memory coalescing
- Cleaner bit extraction

**Apply to Q4_0:**
- Q4_0 `qs[16]` array = 16 bytes
- Load as 4 × uint32_t
- Extract bytes, then extract nibbles from bytes

### 2.3 Factored Dequantization

**hipfire formulation:**
```cpp
// Instead of: sum(scale[i] * (nib[i] - zero[i]) * x[i])
// Use: scale * sum(nib*x) + zero * sum(x)

float sum_x = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
float nib_dot_x = (float)dot_sum * x_scale;
acc += scale * nib_dot_x + zero * sum_x;
```

**Why faster:**
- Reduces multiply count: 8 → 2 multiplies
- Precompute sum_x once
- Vectorizable

**Apply to Q4_0:**
- Q4_0 has single scale per 32 values (not per-value like HFQ)
- Still benefit from reduced multiplication count
- Formulate as: `scale * (sum(nib * x) - 8 * sum(x))`

---

## Part 3: Memory Access Patterns

### 3.1 Multi-Row GEMV

**hipfire:** Processes multiple output rows per kernel launch

```cpp
// R=2: Each thread block computes 2 rows instead of 1
// Better wave scheduler utilization on some architectures
```

**Architecture-specific defaults:**
- RDNA3 (gfx1100): R=1 (single-row already optimal)
- RDNA2 (gfx1030): R=1 (specialized kernels exist)
- RDNA1/APU: R=2 (multi-row improves throughput)

**Apply to Q4_0:**
- Add multi-row variant for fusion kernel
- Process 2-4 QKV outputs per block
- Better occupancy on smaller GPUs

### 3.2 Vector Loads for Input

**hipfire:**
```cpp
// Load 8 consecutive x values
float x0 = x[base];
float x1 = x[base + 1];
// ... etc (compiler vectorizes)
```

**Optimization:**
- Use float4 for 4-way loads
- Better cache line utilization
- Reduces load instructions

**Apply to Q4_0:**
- Our fusion kernel uses float4 for s_input loads
- Already doing this correctly ✓

---

## Part 4: Kernel Organization

### 4.1 Per-Architecture Kernel Variants

**hipfire structure:**
```
kernels/src/
  gemv_hfq4g256.gfx1030.v1.hip  # RDNA2 variant 1
  gemv_hfq4g256.gfx1030.v2.hip  # RDNA2 variant 2 (dp4a)
  gemv_hfq4g256.gfx1030.v3.hip  # RDNA2 variant 3
  gemv_hfq4g256.gfx1030.v4.hip  # RDNA2 variant 4 (latest)
```

**Benefits:**
- Clear optimization history
- A/B testing in production
- Rollback if regression
- Architecture-specific tuning

**Apply to ROCmForge:**
- Rename: `q4_0_fused_norm_qkv_rope.hip` → `q4_0_fused_norm_qkv_rope.gfx1030.v1.hip`
- Add variants: `.v2.hip` with optimizations
- Keep old versions as fallback
- Clear versioning in dispatch

### 4.2 Descriptive Naming

**hipfire pattern:**
```
attention_flash_asym3_tile_batched.hip
gemv_hfq4g256_residual_wmma.hip
```

**Encodes:**
- Operation (attention/gemv)
- Quantization (hfq4g256)
- Optimization (asym3, tile, wmma)
- Features (residual, batched)

**Apply to ROCmForge:**
- Current: `q4_0_fused_norm_qkv_rope.hip`
- Better: `q4_0_fused_norm_qkv_rope_gqa.gfx1030.v1.hip`
- Encodes: format, features, architecture, version

---

## Part 5: WMMA (Wave Matrix Multiply) for RDNA3

### 5.1 WMMA Intrinsics

**From:** `kernels/src/gemm_hfq4g256_residual_wmma.hip`

```cpp
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// 16×16×16 matrix multiply
acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
```

**Requirements:**
- gfx1100+ (RDNA3)
- Wave32 mode
- FP16 inputs

**Performance:**
- 16 × 16 × 16 = 4096 operations per instruction
- ~10× faster than scalar FP32
- 654 GiB/s effective bandwidth on 7900 XTX

### 5.2 Adapting WMMA to Q4_0

**Challenge:** WMMA needs FP16 inputs, Q4_0 is 4-bit

**Solution: Two-stage approach**

Stage 1: Dequantize Q4_0 → FP16 in shared memory
```cpp
// Each thread block dequantizes 16×16 chunk
__shared__ half smem_chunk[16][16];

// Load Q4_0 weights
const Q4_0_block* b = &w[col];
float d = b->d;
uint8_t qs[16] = b->qs;

// Dequant to FP16
for (int i = 0; i < 16; i++) {
    float val = d * ((qs[i] & 0xF) - 8.0f);
    smem_chunk[row][col] = __float2half(val);
}
__syncthreads();
```

Stage 2: WMMA GEMM with FP16
```cpp
half16_t a = load_from_shared(smem_chunk, ...);
half16_t b = load_from_shared(x_fp16, ...);
float8_t acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
```

**Where to apply:**
- **Prefill:** Large batch sizes, benefits from WMMA
- **Decode:** Small batches (1 token), stick with scalar
- hipfire uses WMMA for prefill only

**Implementation for ROCmForge:**
1. Detect gfx1100+ at runtime
2. For prefill (batch_size > 1): Use WMMA kernel
3. For decode (batch_size = 1): Use current scalar kernel
4. Separate code path for WMMA vs non-WMMA

---

## Part 6: Attention Optimizations

### 6.1 Tiled Attention

**From:** `kernels/src/attention_flash_asym3_tile.hip`

**Concept:** Process sequence in tiles (chunks) instead of all-at-once

```cpp
int tile_size = 32; // Configurable
int tile_start = tile_id * tile_size;
int tile_end = min(tile_start + tile_size, seq_len);

// Process only this tile
for (int t_local = 0; t_local < tile_len; t_local++) {
    int t = tile_start + t_local;
    // Compute attention for position t in tile
}
```

**Benefits:**
- Better cache locality
- Reduced memory pressure
- Shared memory reuse
- Extensible to variable sequence lengths

**Apply to ROCmForge:**
- Current attention processes full sequence
- Add tiled variant for long sequences
- Configurable tile size (32, 64, 128)
- Reduces register pressure

### 6.2 In-Register RoPE with Givens Rotations

**hipfire pattern:**
```cpp
// Precompute cos/sin for all positions
const float* cos_theta = precomputed_cos;
const float* sin_theta = precomputed_sin;

// Apply in-register (no memory access)
givens_rot_fwd(mq0, mq1, cos_theta[b0 + 0], sin_theta[b0 + 0]);
```

**Optimization:**
- Precompute cos/sin once (not per-token)
- Store in constant cache or read-only cache
- In-register application avoids memory latency

**Apply to ROCmForge:**
- Current: Compute cos/sin per kernel launch
- Improvement: Precompute for all positions
- Cache in GPU constant memory
- RoPE becomes just table lookup + multiply

---

## Part 7: Software-Level Optimizations

### 7.1 Environment Variable Overrides

**hipfire:**
```bash
HIPFIRE_GEMV_ROWS=2  # Override multi-row setting
```

**Benefits:**
- Per-model tuning without recompilation
- A/B testing in production
- User-level performance tuning

**Apply to ROCmForge:**
```bash
ROCMFORGE_TILE_SIZE=64
ROCMFORGE_USE_WMMA=1
ROCMFORGE_ATTENTION_TILES=1
```

### 7.2 Performance Profiling Infrastructure

**hipfire has:** `redline` crate for monitoring

**Features:**
- Per-kernel timing
- Memory bandwidth tracking
- Bottleneck identification
- Regression detection

**Apply to ROCmForge:**
- Add timing instrumentation
- Track effective bandwidth
- Alert on performance regressions
- Help users identify bottlenecks

---

## Part 8: Testing Infrastructure

### 8.1 Kernel-Level Unit Tests

**hipfire has:**
```
tests/
  batch_attn.rs     # Test attention correctness
  quantization.rs   # Test dequantization accuracy
  wmma.rs           # Test WMMA kernels
  spec_decode.rs    # Test speculative decoding
```

**Apply to ROCmForge:**
- Add dequantization correctness test
- Compare CPU vs GPU outputs
- Test across architectures
- Catch regressions early

### 8.2 Benchmarks

**hipfire examples:**
```
bench_hfq6_vs_hfq4.rs  # Compare quantization formats
bench_wide.rs          # Test wide activation variants
debug_dequant.rs       # Validate dequantization
```

**Apply to ROCmForge:**
- Benchmark fusion vs separate kernels
- Measure bandwidth utilization
- Test with different models
- Track performance over time

---

## Part 9: Concrete Implementation Plan

### Phase 1: Correctness & Foundation (Week 1)

**Task 1.1: Fix current fusion kernel**
- ✅ DONE: Fixed Q4_0 dequantization pattern
- ✅ DONE: Kernel produces coherent output

**Task 1.2: Add architecture detection**
```rust
// src/gpu/device.rs
pub fn detect_architecture(device: &GpuDevice) -> String {
    // Query HIP for architecture string
    hip::get_device_architecture(device)
}
```

**Task 1.3: Add feature detection**
```rust
pub struct GpuFeatures {
    pub arch: String,
    pub has_wmma: bool,         // gfx1100+
    pub has_dp4a: bool,         // gfx1030+
    pub has_dot2_f32_f16: bool, // gfx1011/1012, gfx1030+
}
```

**Deliverable:** Can detect GPU and features at runtime

---

### Phase 2: DP4A Optimization for RDNA2+ (Week 2)

**Task 2.1: Create DP4A variant of fusion kernel**

New file: `hip_kernels/quant/q4_0_fused_norm_qkv_rope_dp4a.hip`

Key changes:
1. Pack Q4_0 nibbles into int32
2. Quantize x to int8 on-the-fly
3. Use `__builtin_amdgcn_sdot4`
4. Factored dequantization formulation

**Task 2.2: Add dispatch logic**
```rust
// src/gpu/ops.rs
pub fn gpu_dispatch_fused_norm_qkv_rope_kvwrite_on_stream(...) {
    if features.has_dp4a && config.use_dp4a {
        launch_dp4a_kernel();
    } else {
        launch_scalar_kernel(); // Current fixed kernel
    }
}
```

**Task 2.3: Benchmark**
- Compare DP4A vs scalar on RDNA2
- Measure accuracy impact (0.4% noise)
- Verify speedup (target: 1.5-2×)

**Deliverable:** DP4A-optimized kernel for RDNA2+

---

### Phase 3: Multi-Row GEMV for Better Occupancy (Week 2)

**Task 3.1: Implement multi-row variant**

Current: 1 output row per block
New: 2-4 rows per block

```cpp
// Process 4 QKV outputs instead of 1
int row_base = blockIdx.x * 4;
if (row_base >= M) return;
for (int r = 0; r < 4 && row_base + r < M; r++) {
    // Process row_base + r
}
```

**Task 3.2: Add tuning parameter**
```rust
gemv_rows_default(arch) -> usize {
    match arch {
        "gfx1100" => 1,  // RDNA3: single-row optimal
        "gfx1010" => 2,  // RDNA1: multi-row helps
        _ => 1,
    }
}
```

**Task 3.3: Benchmark**
- Test on RDNA1, RDNA2, RDNA3
- Measure occupancy impact
- Verify correctness

**Deliverable:** Multi-row kernel with architecture-aware defaults

---

### Phase 4: WMMA for RDNA3 Prefill (Week 3)

**Task 4.1: Create WMMA variant**

New file: `hip_kernels/quant/q4_0_fused_norm_qkv_rope_wmma.hip`

Two-stage approach:
1. Dequant Q4_0 → FP16 in shared memory
2. WMMA GEMM with FP16 tensors

**Task 4.2: Add prefill/decode split**
```rust
pub fn dispatch_fused_kernel(...) {
    if batch_size > 1 && features.has_wmma {
        // Prefill: Use WMMA
        launch_wmma_kernel();
    } else {
        // Decode: Use scalar/DP4A
        launch_optimized_scalar_kernel();
    }
}
```

**Task 4.3: Benchmark**
- Test on RX 7900 XT (gfx1100)
- Measure prefill speedup (target: 2-3×)
- Verify decode not affected

**Deliverable:** WMMA kernel for RDNA3 prefill

---

### Phase 5: Tiled Attention (Week 4)

**Task 5.1: Implement tiled attention**

New file: `hip_kernels/attention_tiled.hip`

Changes:
1. Add tile_size parameter
2. Process sequence in chunks
3. Shared memory for tiles
4. Better cache locality

**Task 5.2: Integrate with forward pass**
```rust
// src/gpu/forward.rs
let tile_size = match seq_len {
    0..=512 => 32,
    513..=2048 => 64,
    _ => 128,
};
```

**Task 5.3: Benchmark**
- Test with different sequence lengths
- Measure memory bandwidth reduction
- Verify correctness

**Deliverable:** Tiled attention kernel

---

### Phase 6: Precomputed RoPE (Week 4)

**Task 6.1: Precompute cos/sin tables**

```rust
// src/gpu/kernels/rope.rs
pub struct PrecomputedRoPE {
    cos_table: GpuBuffer,  // [max_seq_len * head_dim / 2]
    sin_table: GpuBuffer,
}

impl PrecomputedRoPE {
    pub fn new(max_seq_len: usize, head_dim: usize) -> Self {
        // Precompute all cos/sin values
        // Upload to GPU as read-only buffers
    }
}
```

**Task 6.2: Update RoPE kernel**
```cpp
// Load from precomputed table instead of computing
float cos_val = cos_theta[pos * pair_idx];
float sin_val = sin_theta[pos * pair_idx];
```

**Task 6.3: Benchmark**
- Measure reduction in cosine computation
- Verify accuracy maintained

**Deliverable:** Precomputed RoPE tables

---

### Phase 7: Testing & Profiling (Week 5)

**Task 7.1: Add kernel-level tests**

New file: `tests/kernel_correctness.rs`

Tests:
- Dequantization accuracy (GPU vs CPU)
- Attention correctness
- RoPE application
- End-to-end forward pass

**Task 7.2: Add performance benchmarks**

New file: `benches/kernel_performance.rs`

Metrics:
- Tokens/second
- Memory bandwidth (GiB/s)
- % of peak bandwidth
- Kernel timing breakdown

**Task 7.3: Add profiling infrastructure**

```rust
// src/gpu/profile.rs
pub struct KernelProfile {
    pub name: String,
    pub avg_time_ns: u64,
    pub calls: u64,
    pub bandwidth_gbs: f64,
}
```

**Deliverable:** Comprehensive test + profiling suite

---

### Phase 8: Documentation & Integration (Week 5)

**Task 8.1: Update documentation**

Files to update:
- `CLAUDE.md`: Add architecture-specific notes
- `README.md`: Document new features
- `ARCHITECTURE.md`: Explain kernel variants

**Task 8.2: Add environment variables**

Document:
- `ROCMFORGE_USE_DP4A=1`
- `ROCMFORGE_USE_WMMA=1`
- `ROCMFORGE_TILE_SIZE=N`
- `ROCMFORGE_GEMV_ROWS=N`

**Task 8.3: Add CLI flags**

```bash
rocmforge --gpu --arch-optimized
rocmforge --gpu --profile-kernels
rocmforge --gpu --tile-size 64
```

**Deliverable:** User-facing features documented

---

## Expected Performance Improvements

### Baseline (Current)
- **0.5B model:** ~154 tok/s
- **Effective bandwidth:** ~150 GiB/s (poor)
- **Kernel:** Fixed Q4_0 fusion (correct but slow)

### After Phase 2 (DP4A)
- **RDNA2 (6900 XT):** 230-280 tok/s (**1.5-1.8×**)
- **RDNA3 (7900 XT):** 200-240 tok/s (**1.3-1.6×**)
- **Trade-off:** 0.4% quantization noise

### After Phase 3 (Multi-row)
- **RDNA1 (5700 XT):** 180-200 tok/s (**1.2-1.3×**)
- **APU (BC-250):** 200-220 tok/s (**1.3-1.4×**)
- **RDNA3:** Minimal impact (already optimal)

### After Phase 4 (WMMA Prefill)
- **RDNA3 prefill (7900 XTX):** 400-600 tok/s (**2.5-4×** prefill only)
- **Decode:** Unchanged (WMMA not beneficial for batch=1)

### After Phase 5 (Tiled Attention)
- **Long sequences (2K+ tokens):** 10-20% faster
- **Memory bandwidth:** 15-25% reduction
- **Best case:** 512-2048 token range

### Combined (All Phases)
- **RDNA1:** 180-200 tok/s (baseline: 150 tok/s) = **1.2-1.3×**
- **RDNA2:** 250-300 tok/s (baseline: 150 tok/s) = **1.7-2.0×**
- **RDNA3:** 250-350 tok/s (baseline: 150 tok/s) = **1.7-2.3×**
- **Prefill (RDNA3):** 400-600 tok/s = **2.5-4.0×**

**Comparison to hipfire:**
- hipfire 0.8B: 353 tok/s
- Our target 0.5B: 250-350 tok/s
- **Gap narrows from 2.3× to ~1.0×**

---

## Risks & Mitigations

### Risk 1: DP4A Accuracy Impact
- **Risk:** 0.4% noise from x quantization
- **Mitigation:** Make DP4A opt-in via env var
- **Fallback:** Scalar kernel still available
- **Validation:** Test with perplexity benchmarks

### Risk 2: WMMA Only Works on RDNA3
- **Risk:** RDNA1/2 users don't benefit
- **Mitigation:** Keep non-WMMA kernels
- **Architecture:** Clear dispatch logic based on feature detection
- **Testing:** Verify all code paths

### Risk 3: Multi-Row May Reduce Occupancy
- **Risk:** Larger blocks = fewer concurrent blocks
- **Mitigation:** Per-architecture tuning (gemv_rows_default)
- **Fallback:** Single-row still available
- **Validation:** Test on multiple GPU models

### Risk 4: Implementation Complexity
- **Risk:** Multiple kernel variants = more maintenance
- **Mitigation:** Clear naming, version control
- **Testing:** Comprehensive test suite
- **Documentation:** Explain each variant

---

## Non-Goals (Explicitly NOT Doing)

These require quantization format changes - explicitly excluded:

1. ❌ **Switching to HFQ format** - Stay with GGUF Q4_0
2. ❌ **Q8_0 KV cache** - Requires format change
3. ❌ **Asym3/Asym4 quantization** - Different format
4. ❌ **MagnumQuant rotations** - Format-dependent
5. ❌ **3-bit quantization** - Format change

We're optimizing **within** the GGUF Q4_0 constraint.

---

## Success Criteria

### Correctness
- ✅ All kernels pass correctness tests
- ✅ CPU vs GPU output matches (within tolerance)
- ✅ No regression from baseline

### Performance
- ✅ RDNA2: 1.7-2.0× speedup over baseline
- ✅ RDNA3: 1.7-2.3× speedup over baseline
- ✅ Prefill (RDNA3): 2.5-4.0× speedup

### Compatibility
- ✅ Still works with GGUF Q4_0 models
- ✅ No breaking changes to API
- ✅ Graceful fallback for unsupported features

### Maintainability
- ✅ Clear kernel naming and versioning
- ✅ Comprehensive test coverage
- ✅ Documentation for all optimizations

---

## Files to Create/Modify

### New Files
```
hip_kernels/quant/
  q4_0_fused_norm_qkv_rope_dp4a.hip           # DP4A variant
  q4_0_fused_norm_qkv_rope_wmma.hip           # WMMA variant
  q4_0_fused_norm_qkv_rope_multirow.hip       # Multi-row variant

src/gpu/
  features.rs                                  # Feature detection
  profile.rs                                   # Performance profiling

tests/
  kernel_correctness.rs                        # Kernel tests

benches/
  kernel_performance.rs                        # Performance benchmarks
```

### Modified Files
```
src/gpu/
  device.rs                                    # Add arch detection
  ops.rs                                       # Add dispatch logic
  forward.rs                                   # Use optimized kernels
  kernels/mod.rs                              # Export new kernels

Cargo.toml                                    # Add test/bench dependencies
CLAUDE.md                                      # Document new features
```

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Correctness & Foundation | Arch detection, feature detection |
| 2 | DP4A + Multi-row | RDNA2/1 optimizations |
| 3 | WMMA Prefill | RDNA3 prefill speedup |
| 4 | Tiled Attention + RoPE | Memory optimizations |
| 5 | Testing & Docs | Test suite, profiling, docs |

**Total: 5 weeks** for full implementation

---

## Conclusion

This plan focuses on **techniques that work with GGUF Q4_0**, no format changes required.

**Key insights:**
1. DP4a gives 1.5-2× on RDNA2 (minor accuracy trade-off)
2. WMMA critical for RDNA3 prefill (2-4× speedup)
3. Architecture-aware dispatch maximizes each GPU
4. Software optimizations (tiling, precompute) add 10-20%

**Expected result:** Close the gap with hipfire while staying GGUF-compatible.
