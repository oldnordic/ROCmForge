# Phase 1: Wave32 Specialization - File-by-File Implementation Plan

**Date:** 2026-04-13
**Author:** Claude Sonnet 4.6 + Luiz Spies
**Status:** Design Approved - Ready for Implementation
**Scope:** Optimize prefill performance on RDNA3 (RX 7900 XT) via wave32 kernel specialization

## Overview

This document provides a complete file-by-file implementation plan for Phase 1 of the rocmforge optimization project. The goal is to specialize HIP kernels for wave32 execution on RDNA3 architecture to improve prefill performance by 20-30%.

**Success Criteria:**
- ✅ Prefill performance improves by ≥20% (target: 78→94+ tok/s)
- ✅ Decode performance stays at ~529 tok/s (no regression)
- ✅ All tests pass with `--test-threads=1` (sequential execution)
- ✅ No VRAM leaks detected (assert_vram_cleanup! passes)
- ✅ No GPU resets during testing

**Constraints:**
- Use existing VRAM safety infrastructure (do not rewrite `GpuLock`, `serial_test`)
- Preserve backward compatibility with CDNA (wave64) GPUs
- Minimal changes to Rust FFI layer (already works correctly)
- HIP kernels only - no changes to quantization algorithms

## Architecture Summary

```
build.rs → hip_kernels/common.hip → hip_kernels/{norm,quant,attention,elementwise}.hip
   ↓              ↓                           ↓
Wave32      WARP_SIZE template          Wave32-specialized
compilation   parameter                    kernels
```

**Key Insight:** The Rust layer already creates separate graph keys for wave32 vs wave64. The HIP kernels hardcode wave32 optimizations (`/ 32`, `% 32`, `__shfl_down(..., ..., 32)`) but incorrectly define `WARP_SIZE = 64`. We fix this by making wave size a compile-time template parameter.

---

## File-by-File Changes

### 1. `build.rs` (Lines 11-98)

**Current State:**
- Compiles kernels with `hipcc` targeting `gfx1100` (line 63)
- No wave size specialization
- Single kernel variant for all architectures

**Changes:**

```rust
// After line 28 (kernels array), add wave size detection:

fn detect_wave_size_for_arch(arch: &str) -> usize {
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => 32, // RDNA3
        "gfx1030" | "gfx1031" | "gfx1032" => 32, // RDNA2
        "gfx1010" | "gfx1011" | "gfx1012" => 32, // RDNA1
        "gfx900" | "gfx906" | "gfx908" | "gfx90a" | "gfx90c" | "gfx942" => 64, // CDNA
        _ => 32, // Conservative default for RDNA
    }
}

// In compile_hip_kernels(), update compile command (line 56):

let target_arch = "gfx1100"; // TODO: Make this configurable
let wave_size = detect_wave_size_for_arch(target_arch);

let compile_status = Command::new(&hipcc)
    .arg(source_file)
    .arg("-o")
    .arg(&obj_file)
    .arg("-c")
    .arg("-fPIC")
    .arg("-O3")
    .arg(format!("--offload-arch={}", target_arch))
    .arg(format!("-DWARP_SIZE={}", wave_size))  // NEW: Wave size specialization
    .arg(format!("-I{}", hip_include.display()))
    .status();
```

**Similarly for compile_quant_kernels() (line 100+):**
- Add `-DWARP_SIZE=32` to CMake flags
- Pass via `CXXFLAGS` or `hip_compile_options`

---

### 2. `hip_kernels/common.hip` (Lines 46-48)

**Current State:**
```cpp
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 64;  // ❌ WRONG for RDNA3
```

**Changes:**

```cpp
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCK_SIZE = 1024;

// Wave size is now a compile-time template parameter
// Default to 64 for backward compatibility, but build.rs overrides this
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

constexpr int WARP_SIZE_COMPTIME = WARP_SIZE;
```

**Rationale:**
- Use preprocessor macro `WARP_SIZE` to allow build.rs to override
- Keep `constexpr` version for compile-time constants in kernels
- Backward compatible: defaults to 64 if not specified

---

### 3. `hip_kernels/norm_vulkan_style.hip` (Lines 44-60)

**Current State:**
```cpp
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down(sum_sq, offset);  // ❌ Hardcoded wave32
}
```

**Changes:**

```cpp
#pragma unroll
for (int offset = WARP_SIZE_COMPTIME / 2; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down(sum_sq, offset, WARP_SIZE_COMPTIME);
}
```

**Pattern:**
- Replace `16` with `WARP_SIZE_COMPTIME / 2`
- Add third parameter to `__shfl_down`: `WARP_SIZE_COMPTIME`
- Apply to all shuffle operations in this file

**Locations:**
- Line 46 (first shuffle loop)
- Line 59-60 (second shuffle loop)
- Any other `__shfl_down` calls

---

### 4. `hip_kernels/quant/q4_0_fused_q8.hip` (Lines 29-135)

**Current State:**
```cpp
template<int N_WAVES>
__global__ void gemv_gate_up_q4_0_q8_0_vulkan_style_kernel(...) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;  // ❌ Hardcoded
    const int lane_id = tid % 32;  // ❌ Hardcoded
    // ...
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {  // ❌ Hardcoded
        // ...
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate0 += __shfl_down(sum_gate0, offset, 32);  // ❌ Hardcoded
        sum_up0 += __shfl_down(sum_up0, offset, 32);
        // ...
    }
}
```

**Changes:**

```cpp
template<int N_WAVES, int WARP_SIZE = WARP_SIZE_COMPTIME>
__global__ void gemv_gate_up_q4_0_q8_0_vulkan_style_kernel(...) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    // ...
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        // ...
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_gate0 += __shfl_down(sum_gate0, offset, WARP_SIZE);
        sum_up0 += __shfl_down(sum_up0, offset, WARP_SIZE);
        // ...
    }
}
```

**Pattern:**
- Add `int WARP_SIZE = WARP_SIZE_COMPTIME` to template parameters
- Replace `/ 32` with `/ WARP_SIZE`
- Replace `% 32` with `% WARP_SIZE`
- Replace `block_idx += 32` with `block_idx += WARP_SIZE`
- Replace `offset = 16` with `offset = WARP_SIZE / 2`
- Add `WARP_SIZE` parameter to all `__shfl_down` calls

**Files to Update (All in `hip_kernels/quant/`):**
- `q4_0_fused_q8.hip` (lines 29-135, 138+)
- `q4_0_gemv.hip` (similar patterns)
- `q4_0_fused.hip` (similar patterns)
- `q8_0_gemv.hip` (line ~40-150)
- Any other kernels with `__shfl_down` or wave operations

---

### 5. `hip_kernels/attention.hip`

**Current State:**
- Likely contains wave32 hardcoded patterns in attention kernels
- Needs similar `/ 32`, `% 32`, `__shfl_down` replacements

**Changes:**
- Apply same pattern as quantization kernels
- Search for: `/ 32`, `% 32`, `__shfl`, `waveSize`, `WARP_SIZE`
- Replace with `WARP_SIZE_COMPTIME` template parameter

---

### 6. `hip_kernels/elementwise.hip`

**Current State:**
- Elementwise operations may have wave-specific optimizations
- Check for reduction patterns using `__shfl_down`

**Changes:**
- Apply same pattern if wave operations found
- Otherwise, no changes needed (elementwise ops typically don't use shuffle)

---

### 7. `tests/common/mod.rs` (NEW - After Line 173)

**Current State:**
- Has `require_gpu!()` and `require_vram!()` macros
- Has `GpuLock` for cross-process GPU locking
- Uses `rocm-smi` for VRAM checks (external tool)

**Changes (Add at end of file, before `#[cfg(test)]`):**

```rust
/// HIP-based VRAM leak detection macro.
///
/// Uses device.vram_stats() for accurate HIP API measurements instead of rocm-smi.
/// Panics if VRAM leak exceeds tolerance (default 10 MB).
///
/// Usage:
/// ```rust
/// #[test]
/// #[serial]
/// fn test_something() {
///     require_gpu!();
///     let device = GpuDevice::init(0).unwrap();
///
///     let before = device.vram_stats().unwrap();
///     // ... test code ...
///     drop(device);  // Explicit cleanup
///     assert_vram_cleanup!(before, 10);  // Allow 10 MB tolerance
/// }
/// ```
#[macro_export]
macro_rules! assert_vram_cleanup {
    ($device:expr, $tolerance_mb:expr) => {
        let after = $device.vram_stats().expect("Failed to get VRAM stats");
        let before = $device.vram_stats().expect("Failed to get VRAM stats");

        let leaked_mb = (before.used_vram as i64 - after.used_vram as i64).abs() / (1024 * 1024);

        if leaked_mb > $tolerance_mb {
            panic!(
                "VRAM leak detected: {} MB leaked (tolerance: {} MB)\n\
                 Before: {} MB used, After: {} MB used\n\
                 Total: {} MB, Free: {} MB",
                leaked_mb,
                $tolerance_mb,
                before.used_vram_mb(),
                after.used_vram_mb(),
                after.total_vram_mb(),
                after.free_vram_mb()
            );
        }
    };
}
```

**Rationale:**
- More accurate than rocm-smi (uses HIP API directly)
- Integrates with existing `vram_stats()` method from `src/gpu/device.rs:229`
- Optional: tests can choose to use this or not
- Works alongside existing `require_vram!()` macro

---

### 8. `tests/gpu_wave32_integration.rs` (NEW FILE)

**Purpose:**
- Verify wave32 kernel specialization works correctly
- Test that VRAM is cleaned up properly
- Ensure no regression in correctness

**Content:**

```rust
#![cfg(feature = "gpu")]

mod common;

use rocmforge::gpu::GpuDevice;
use serial_test::serial;

#[test]
#[serial]
fn test_wave32_device_properties() {
    require_gpu!();

    let device = GpuDevice::init(0).expect("Failed to init GPU");

    // Verify RDNA3 wave32 detection
    assert_eq!(device.warp_size(), 32, "RX 7900 XT should have wave32");

    // Verify VRAM stats work
    let stats = device.vram_stats().expect("Failed to get VRAM stats");
    assert!(stats.total_vram_gb() > 16.0, "Should have >16 GB VRAM");
    assert!(stats.safely_allocatable_gb() > 8.0, "Should have >8 GB safe VRAM");

    // Explicit cleanup
    drop(device);
}

#[test]
#[serial]
fn test_wave32_kernel_launch_no_vram_leak() {
    require_gpu!();

    let device = GpuDevice::init(0).expect("Failed to init GPU");
    let before = device.vram_stats().expect("Failed to get VRAM before");

    // TODO: Launch a simple wave32 kernel here
    // For now, just verify VRAM tracking works

    drop(device);
    let after = GpuDevice::init(0).expect("Failed to re-init GPU");
    let after_stats = after.vram_stats().expect("Failed to get VRAM after");

    // Allow 10 MB tolerance for driver overhead
    let leaked_mb = (before.used_vram as i64 - after_stats.used_vram as i64).abs() / (1024 * 1024);
    assert!(leaked_mb <= 10, "VRAM leak detected: {} MB", leaked_mb);
}
```

---

## Build and Test Sequence

### Step 1: Baseline Measurement (Before Changes)

```bash
# Build current version
cargo build --release --features gpu

# Benchmark prefill (current performance)
cargo bench --bench gpu_decode --features gpu -- --noplot

# Run existing tests
cargo test --release --features gpu -- --test-threads=1
```

**Record These Metrics:**
- Prefill throughput: `XX.X tok/s` (current: ~78 tok/s)
- Decode throughput: `XXX.X tok/s` (current: ~529 tok/s)
- Test pass/fail: all pass
- VRAM usage: `XXX MB` (via rocm-smi)

---

### Step 2: Implement Changes (File-by-File)

**Order:**
1. `build.rs` - Add wave size detection
2. `hip_kernels/common.hip` - Add WARP_SIZE macro
3. `hip_kernels/norm_vulkan_style.hip` - Update shuffle operations
4. `hip_kernels/quant/q4_0_fused_q8.hip` - Update all wave operations
5. `hip_kernels/quant/q8_0_gemv.hip` - Update wave operations
6. `hip_kernels/attention.hip` - Update wave operations
7. `hip_kernels/elementwise.hip` - Update if needed
8. `tests/common/mod.rs` - Add `assert_vram_cleanup!` macro
9. `tests/gpu_wave32_integration.rs` - Create integration test

**After Each File:**
```bash
cargo build --release --features gpu
# If build fails, fix before continuing
```

---

### Step 3: Verification (After All Changes)

```bash
# Clean build
cargo clean && cargo build --release --features gpu

# Run all GPU tests sequentially
cargo test --release --features gpu -- --test-threads=1

# Run new wave32 integration test
cargo test --release --features gpu --test gpu_wave32_integration -- --test-threads=1

# Benchmark prefill (expect 20-30% improvement)
cargo bench --bench gpu_decode --features gpu -- --noplot

# Verify VRAM usage with rocm-smi
rocm-smi --showmeminfo vram
```

**Expected Results:**
- ✅ All tests pass
- ✅ Prefill: `94+ tok/s` (20-30% improvement from 78 tok/s)
- ✅ Decode: `529±10 tok/s` (no regression)
- ✅ VRAM leak detection passes (≤10 MB variance)

---

### Step 4: Performance Regression Testing

```bash
# Compare with llama-cli (baseline)
time /home/feanor/Projects/llama.cpp/build/bin/llama-cli \
  -m /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -p "Hello world" -n 64 -ngl 24

# Compare with rocmforge (after optimization)
time ./target/release/rocmforge \
  --gpu --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "Hello world" --max-tokens 64 --no-template --top-p 1.0
```

**Target:**
- Prefill: `rocmforge ≥ llama-cli` (currently: rocmforge 78 vs llama-cli 99)
- Decode: `rocmforge >> llama-cli` (currently: rocmforge 529 vs llama-cli 99)

---

## Rollback Plan

If anything goes wrong:

```bash
# Revert all changes
git checkout hip_kernels/*.hip
git checkout build.rs
git checkout tests/common/mod.rs
rm tests/gpu_wave32_integration.rs

# Rebuild
cargo clean && cargo build --release --features gpu
```

**Known Risks:**
- Template parameter changes might break compile-time optimizations
- Wave size detection might be incorrect for some GPU architectures
- HIP compiler might reject template specialization in some cases

**Mitigation:**
- Incremental changes (one file at a time)
- Test after each change
- Keep baseline build artifacts for comparison

---

## Success Criteria

### Phase 1 Complete When:
- ✅ All HIP kernels use `WARP_SIZE_COMPTIME` template parameter
- ✅ `build.rs` sets `-DWARP_SIZE=32` for `gfx1100`
- ✅ All tests pass with `--test-threads=1`
- ✅ `assert_vram_cleanup!` macro passes in integration tests
- ✅ Prefill performance improves by ≥20%
- ✅ Decode performance stays at ~529 tok/s
- ✅ No GPU resets during testing

### Phase 1 Deliverables:
- Wave32-specialized HIP kernels for RDNA3
- HIP-based VRAM leak detection macro
- Wave32 integration test
- Performance benchmark results (before/after)

---

## Next Steps (Phase 2 - Multi-User Foundation)

**After Phase 1 is complete and verified:**
1. Implement PagedAttention for multi-user KV cache management
2. Add continuous batching support
3. Integrate vLLM-style request scheduling

**NOT in Phase 1:**
- MFMA matrix operations (deferred to Phase 3)
- Multi-user/batching (deferred to Phase 2)
- Architecture auto-detection at runtime (deferred to Phase 3)

---

## References

- ROCm 7.2 Docs: https://rocm.docs.amd.com/en/latest
- RDNA3 ISA: https://www.amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
- Wave32 vs Wave64: https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/reduction
- Existing VRAM Safety: `tests/common/mod.rs`, `tests/gpu_test_utils.rs`

---

**End of File-by-File Implementation Plan**
