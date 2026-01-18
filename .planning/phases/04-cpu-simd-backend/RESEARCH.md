# Rust SIMD Ecosystem Research

**Date:** 2026-01-18
**Phase:** 04-cpu-simd-backend
**Plan:** 04-01

---

## Executive Summary

**Critical Finding:** Rust's `std::simd` (portable SIMD) was stabilized in Rust 1.82.0 (November 2024). This eliminates the need for external crates like `packed_simd` or nightly-only features.

**Recommendation:** Use `std::simd` for CPU SIMD operations.

---

## Option 1: std::simd (STABLE) - RECOMMENDED

### Status
- **Stabilized:** Rust 1.82.0 (November 2024)
- **Location:** `std::simd` / `core::simd`
- **Requires:** Stable Rust compiler (1.82+)

### Pros
- Standard library - no external dependencies
- Stable API - won't break between compiler versions
- Portable across x86_64 (SSE/AVX), ARM64 (NEON), and other architectures
- Active maintenance by Rust team
- Zero-cost abstraction - compiler generates optimal SIMD instructions

### Cons
- Requires Rust 1.82+ (newer than some projects may use)
- API may be less mature than external crates (but improving rapidly)

### Platform Support
- x86/x86_64: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2
- ARM64: NEON
- WASM: SIMD128
- PowerPC: Altivec/VSX

### Example Usage

```rust
use std::simd::{f32x4, Simd, SimdFloat};

// SIMD-accelerated dot product
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x4::splat(0.0);

    let chunks = a.chunks_exact(4)
        .zip(b.chunks_exact(4));

    for (a_chunk, b_chunk) in chunks {
        let a_vec = f32x4::from_array(*a_chunk.try_into().unwrap());
        let b_vec = f32x4::from_array(*b_chunk.try_into().unwrap());
        sum += a_vec * b_vec;
    }

    // Horizontal sum of the vector
    sum.reduce_sum()
}
```

---

## Option 2: packed_simd crate - DEPRECATED

### Status
- **Deprecated:** Superseded by `std::simd`
- **Last update:** 2021-2022 era
- **Repository:** rust-lang/packed_simd

### Pros
- Works on older Rust versions (< 1.82)
- Similar API to what became std::simd

### Cons
- **Deprecated and unmaintained**
- External dependency
- API may not match stabilized std::simd
- No future development

### Verdict
**DO NOT USE.** This crate has been superseded by the standard library.

---

## Option 3: wide crate

### Status
- **Active:** Under active development
- **Repository:** mttqbot/wide
- **Version:** 0.7.x

### Pros
- Active development
- Focus on ergonomic API
- Good documentation

### Cons
- External dependency
- Different API from std::simd
- Less platform coverage than std::simd
- Another dependency to maintain

### Verdict
**Not recommended** - std::simd provides equivalent functionality without external dependency.

---

## Option 4: std::arch (intrinsics)

### Status
- **Stable:** Direct platform intrinsics
- **Location:** `std::arch::x86_64::*`, `std::arch::aarch64::*`

### Pros
- Maximum control over SIMD instructions
- Stable API

### Cons
- Platform-specific code required
- Must maintain multiple code paths (SSE, AVX, AVX2, NEON)
- More complex to use
- No portable abstraction

### Verdict
Use only for platform-specific optimizations after portable SIMD is implemented.

---

## Performance Expectations

### Matrix Multiplication (f32)
- **Scalar baseline:** 1x
- **std::simd f32x4 (SSE):** ~3-4x speedup
- **std::simd f32x8 (AVX2):** ~6-8x speedup
- **std::simd f32x16 (AVX-512):** ~12-16x speedup (where available)

### Attention Operations
- **Softmax:** ~4-6x speedup (horizontal operations benefit from SIMD)
- **QK^T matmul:** ~6-8x speedup (standard matrix multiplication)
- **Weighted sum (scores * V):** ~6-8x speedup

### Real-world Impact
For a typical transformer layer:
- CPU matmul: ~100-500ms (scalar)
- CPU SIMD matmul: ~20-80ms (with AVX2)

This makes CPU fallback viable for inference when GPU is unavailable.

---

## Feature Detection Strategy

### Runtime Detection with std::simd

The `std::simd` module uses target-feature detection at compile time by default. For runtime detection:

```rust
// Use the `target_feature` crate for runtime detection
// or detect CPU capabilities via `std::is_x86_feature_detected!`

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn get_optimal_path() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            SimdLevel::AVX2
        } else if is_x86_feature_detected!("sse4.1") {
            SimdLevel::SSE41
        } else {
            SimdLevel::Scalar
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        SimdLevel::NEON  // ARM64 always has NEON
    }
}

enum SimdLevel {
    Scalar,
    SSE41,
    AVX2,
    NEON,
}
```

### Compile-time Detection (Simpler)

For initial implementation, use compile-time detection:

```rust
#[cfg(target_arch = "x86_64")]
type SimdF32 = std::simd::f32x8;  // AVX2

#[cfg(target_arch = "aarch64")]
type SimdF32 = std::simd::f32x4;  // NEON

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64"))]
type SimdF32 = std::simd::f32x4;  // Safe fallback
```

---

## Dependencies Impact

### Using std::simd
- **New dependencies:** None (stdlib)
- **MSRV increase:** Rust 1.82+ (from current unspecified)

### Cargo.toml Changes Required
None. `std::simd` is in the standard library.

### Optional Feature Flag
For users who want to disable SIMD (e.g., for testing):

```toml
[features]
default = ["cpu-simd"]
cpu-simd = []
```

---

## Implementation Approach

### Phase 1: Core SIMD Primitives (Plan 04-02)
- Add SIMD-enabled matmul_f32 using f32x4/f32x8
- Implement tiled matmul for cache efficiency
- Add feature detection

### Phase 2: Attention Operations (Plan 04-03)
- SIMD-accelerated softmax
- SIMD QK^T matmul
- SIMD weighted sum

### Phase 3: Integration and Testing (Plan 04-04)
- Integrate SIMD backend with existing CPU backend
- Benchmark vs scalar
- Correctness validation

---

## References

- [Rust std::simd documentation](https://doc.rust-lang.org/stable/std/simd/)
- [Portable SIMD RFC 2366](https://rust-lang.github.io/rfcs/2366-portable-simd.html)
- [Rust 1.82.0 Release Notes](https://blog.rust-lang.org/2024/11/28/Rust-1.82.0.html)
