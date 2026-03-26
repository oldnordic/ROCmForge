# AVX2 Q4_K × Q8_K GEMM/GEMV Kernel Implementation

**Date:** 2026-03-26
**Status:** Design Approved
**Priority:** High (Performance Critical)

---

## Overview

Implement AVX2-optimized Q4_K × Q8_K matrix multiplication kernels for rocmforge, following llama.cpp's proven algorithm. Target: 3-6x speedup over baseline for 7B model inference (target: 22.5 tok/s).

**Reference:** `/home/feanor/Projects/llama.cpp`
- `ggml/src/ggml-common.h` - Block structure definitions
- `ggml/src/ggml-cpu/arch/x86/quants.c` - AVX2 SIMD kernels

---

## Goals

1. **Performance:** Achieve 3-6x speedup over Q4_0 × f32 baseline
2. **Correctness:** Match llama.cpp numerical results within floating-point tolerance
3. **Maintainability:** Keep files under 1000 LOC with clear separation of concerns
4. **Extensibility:** Enable future AVX-512 and NEON ports

---

## Non-Goals

1. AVX-512 implementation (future work)
2. ARM NEON implementation (future work)
3. Transposed GEMM variants (future work)
4. GQA/MQA optimizations (future work)

---

## Architecture

### Data Structures

#### BlockQ4K (Existing)
```rust
// src/cpu/kernels/q4.rs
#[repr(C, align(16))]
pub struct BlockQ4K {
    pub d: [u8; 2],      // f16 scale (2 bytes)
    pub dmin: [u8; 2],   // f16 min scale (2 bytes)
    pub scales: [u8; 12], // 6-bit quantized scales/mins
    pub qs: [u8; 128],    // 4-bit quants (2 per byte)
} // Total: 144 bytes
```

#### BlockQ8K (New)
```rust
// src/cpu/kernels/q8.rs
#[repr(C, align(16))]
pub struct BlockQ8K {
    pub d: f32,           // Delta scale
    pub qs: [i8; 256],    // 8-bit quants (QK_K elements)
    pub bsums: [i16; 16], // Sum of quants in groups of 16
} // Total: 292 bytes
```

### Module Structure

```
src/cpu/kernels/
├── mod.rs              # Module exports, ~50 LOC
├── q4.rs               # Existing: BlockQ4K (470 LOC)
├── q8.rs               # NEW: BlockQ8K + quantization (~300 LOC)
├── q8_scalar.rs        # NEW: Scalar Q8_K operations (~200 LOC)
├── gemm_q4k_q8.rs      # NEW: AVX2 GEMM/GEMV (~400 LOC)
└── gemm_q4k_q8_scalar.rs # NEW: Scalar fallback (~300 LOC)
```

---

## Component Specifications

### 1. Q8_K Block (src/cpu/kernels/q8.rs, ~300 LOC)

**Purpose:** Define Q8_K block structure and quantization function.

**Functions:**
- `quantize_q8_k(values: &[f32]) -> BlockQ8K` - Quantize 256 f32 values
- `BlockQ8K::zero() -> Self` - Create zero block
- `BlockQ8K::dequantize(&self, output: &mut [f32])` - Dequantize to f32 (testing)

**Algorithm:**
1. Find max absolute value
2. Compute scale: d = max_abs / 127.0
3. Quantize each value: qs[i] = clamp(round(f32[i] / d), -127, 127)
4. Compute block sums: bsums[j] = sum(qs[j*16 .. (j+1)*16])

**Testing:**
- Verify block size matches 292 bytes
- Test quantize/dequantize roundtrip
- Verify bsums match actual sums

---

### 2. AVX2 Dot Product (src/cpu/kernels/gemm_q4k_q8.rs, ~400 LOC)

**Purpose:** Compute dot product of one Q4_K block and one Q8_K block using AVX2.

**Function:**
```rust
unsafe fn dot_q4_k_q8_k_block_avx2(
    q4_block: &BlockQ4K,
    q8_block: &BlockQ8K,
) -> f32
```

**Algorithm (from llama.cpp):**
1. Load and combine scales: d = q8.d × q4.d, dmin = -q8.d × q4.dmin
2. Unpack Q4_K 6-bit scales using bit manipulation
3. Load Q8_K block sums for min contribution
4. Process 4 groups of 64 values:
   - Shuffle scales for each sub-block
   - Load Q4_K nibbles, split into low/high
   - Load Q8_K int8 values
   - `_mm256_maddubs_epi16` for multiply-accumulate
   - Apply scales with `_mm256_madd_epi16`
5. Horizontal sum and add min contribution

**Intrinsics:**
- `_mm256_loadu_si256` - Load 256 bits unaligned
- `_mm256_shuffle_epi8` - Byte-wise shuffle
- `_mm256_maddubs_epi16` - Multiply unsigned 8-bit, add adjacent pairs
- `_mm256_madd_epi16` - Multiply signed 16-bit, add adjacent pairs
- `_mm256_fmadd_ps` - Fused multiply-add

---

### 3. Scalar Fallback (src/cpu/kernels/gemm_q4k_q8_scalar.rs, ~300 LOC)

**Purpose:** Provide reference implementation for systems without AVX2.

**Function:**
```rust
fn dot_q4_k_q8_k_block_scalar(
    q4_block: &BlockQ4K,
    q8_block: &BlockQ8K,
) -> f32
```

**Algorithm:**
1. Same scale unpacking as AVX2 version
2. Process 8 sub-blocks of 32 elements
3. For each sub-block: compute dot product, apply scale, add min contribution
4. Sum all sub-blocks

**Testing:**
- Verify scalar matches AVX2 within floating-point tolerance

---

### 4. GEMV Wrapper (src/cpu/kernels/gemm_q4k_q8.rs, ~100 LOC)

**Purpose:** Single-token decode path: y = W × x where W is Q4_K.

**Function:**
```rust
pub fn gemv_q4_k_q8_k(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
)
```

**Algorithm:**
1. Assert in_dim is multiple of 256
2. Quantize input x to Q8_K blocks (once per call)
3. For each output row:
   - For each block: compute dot product
   - Accumulate result
4. Store to y

---

### 5. GEMM Wrapper (src/cpu/kernels/gemm_q4k_q8.rs, ~100 LOC)

**Purpose:** Batched prefill path: Y = W × X where W is Q4_K.

**Function:**
```rust
pub fn gemm_q4_k_q8_k(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
)
```

**Algorithm:**
1. Assert k is multiple of 256
2. For each batch row:
   - Quantize row to Q8_K blocks
   - For each output column:
     - Compute dot products across blocks
     - Accumulate result

---

### 6. Dispatch Integration (src/cpu/ops.rs, ~50 LOC modification)

**Purpose:** Wire Q4_K × Q8_K kernels into existing dispatch.

**Changes:**
- Add `GgmlType::Q4_K` case to `dispatch_gemm`
- Add `GgmlType::Q4_K` case to `dispatch_gemv`
- Export new functions from `cpu/kernels/mod.rs`

---

## Dependencies

### External
- `std::arch::x86_64` - AVX2 intrinsics (Rust stdlib)
- `rayon` - Parallel iteration (existing dependency)
- `half` - f16 conversion (existing dependency)

### Internal
- `crate::cpu::kernels::q4::BlockQ4K` - Existing Q4_K block
- `crate::cpu::ops::load_f16_scale` - f16 loading helper
- `crate::loader::GgmlType` - Quantization type enum

---

## Testing Strategy

### Unit Tests (~200 LOC)

1. **BlockQ8K tests:**
   - Size verification (292 bytes)
   - Quantize/dequantize roundtrip
   - bsums correctness

2. **Dot product tests:**
   - AVX2 vs scalar comparison
   - Known input/output verification

### Integration Tests (~100 LOC)

1. **GEMV tests:**
   - Compare against reference implementation
   - Verify output shape and values

2. **GEMM tests:**
   - Small matrix (2×2 blocks)
   - Random matrix comparison

### Regression Tests

1. **llama.cpp comparison:**
   - Export known Q4_K blocks from llama.cpp
   - Verify dot products match

---

## Performance Targets

### Benchmarks (benches/gemm_q4k_q8.rs, ~150 LOC)

| Metric | Target | Measurement |
|--------|--------|-------------|
| GEMV throughput | > 1000 calls/sec | `cargo bench --bench gemm_q4k_q8` |
| GEMV latency | < 1 ms per call | Timestamp measurements |
| Speedup vs Q4_0×f32 | 3-6x | Comparative benchmark |
| Prefill tokens/sec | > 20 tok/s | End-to-end test |

### Profiling

- Use `perf` to identify hotspots
- Measure cache hit rates
- Verify AVX2 instruction utilization

---

## Implementation Plan

### Phase 1: Foundation (Day 1)
1. Create `q8.rs` with BlockQ8K and quantization
2. Add unit tests for Q8_K block
3. Verify BlockQ4K compatibility

### Phase 2: Scalar Kernel (Day 1-2)
1. Implement `gemm_q4k_q8_scalar.rs`
2. Add scalar dot product
3. Add GEMV/GEMM wrappers
4. Test with random matrices

### Phase 3: AVX2 Kernel (Day 2-3)
1. Implement `gemm_q4k_q8.rs` with AVX2
2. Port scale shuffle masks from llama.cpp
3. Add AVX2 dot product
4. Verify AVX2 matches scalar

### Phase 4: Integration (Day 3-4)
1. Wire into dispatch_gemm/gemv
2. Add comprehensive tests
3. Benchmark vs baseline
4. Performance tuning

### Phase 5: Cleanup (Day 4)
1. Remove dead code (Q4_0 × Q8_0 if unused)
2. Update documentation
3. Final verification

---

## Risk Mitigation

### Risk 1: Floating-Point Divergence
- **Mitigation:** Use same rounding mode as llama.cpp
- **Verification:** Compare intermediate results

### Risk 2: Memory Alignment
- **Mitigation:** Use `#[repr(C, align(16))]` on all blocks
- **Verification:** Assert alignment in debug builds

### Risk 3: AVX2 Availability
- **Mitigation:** Provide scalar fallback
- **Verification:** Test on non-AVX2 hardware

### Risk 4: Scale Unpacking Bugs
- **Mitigation:** Unit test each scale mask
- **Verification:** Compare against llama.cpp reference

---

## Success Criteria

1. ✅ All tests pass (unit + integration)
2. ✅ AVX2 implementation matches scalar within 1e-5 tolerance
3. ✅ GEMV achieves > 3x speedup over Q4_0×f32 baseline
4. ✅ No files exceed 1000 LOC
5. ✅ Code documented with examples
6. ✅ Benchmarks demonstrate target performance

---

## Open Questions

1. Should we support transposed GEMM from day one?
   - **Decision:** No, defer to future work (non-blocking)

2. Should Q8_K quantization be cached?
   - **Decision:** No, recompute each call (simpler, cache-friendly)

3. Should we implement NEON variant now?
   - **Decision:** No, focus on AVX2 first (x86_64 priority)

---

## References

1. **llama.cpp source:** `/home/feanor/Projects/llama.cpp`
   - `ggml/src/ggml-common.h` - Block structures
   - `ggml/src/ggml-cpu/arch/x86/quants.c` - AVX2 kernels

2. **Intel Intrinsics Guide:** https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

3. **Existing code:**
   - `src/cpu/kernels/q4.rs` - BlockQ4K implementation
   - `src/cpu/ops.rs` - Dispatch patterns
   - `src/cpu/prefill.rs` - Batched prefill usage

---

## Appendix: Scale Mask Explanation

The Q4_K format uses 12 bytes to store 16 scale values (8 scales + 8 mins), each packed as 6-bit values. The unpacking algorithm rearranges these to enable efficient SIMD shuffle operations.

**Example:**
- Input: `scales[12] = [s0, s1, s2, ...]` packed 6-bit values
- Output: `utmp[4]` = 32-bit values ready for `vpshufb`

The masks `kmask1`, `kmask2`, `kmask3` extract specific bit fields to reconstruct the 16 scale values.
