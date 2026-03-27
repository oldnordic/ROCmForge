# GPU Quantization - Phase 1 Foundation

**Status:** ✅ Complete (2026-03-27)

## Overview

Phase 1 establishes the infrastructure foundation for GPU-accelerated quantization kernels in ROCm-Forge. This phase focuses on build systems, safety infrastructure, and testing framework rather than implementing actual quantization algorithms.

## What Was Built

### 1. CMake Build System

**Location:** `hip_kernels/quant/CMakeLists.txt`

- Professional CMake-based build system parallel to `build.rs`
- Compiles HIP kernels to static libraries
- Outputs to `hip_kernels/quant/build/lib/`
- Integrated into Cargo build via `build.rs::compile_quant_kernels()`

**Libraries Built:**
- `libquant_common.a` - Common utilities and safety macros
- `libtest_quant.a` - Test kernels for build verification

### 2. GpuArchitecture Enum

**Location:** `src/gpu/arch.rs`

AMD GPU architecture identifiers with architecture-specific parameters:

```rust
pub enum GpuArchitecture {
    Gfx1100,  // RDNA3 (RX 7000 series) - 32-wide wavefront
    Gfx1030,  // RDNA2 (RX 6000 series) - 32-wide wavefront
    Gfx90a,   // CDNA2 (MI210) - 64-wide wavefront
    Gfx908,   // CDNA1 (MI100) - 64-wide wavefront
    Gfx900,   // Vega - 64-wide wavefront
    Unknown(u32),
}
```

**Key Methods:**
- `max_threads_per_block()` - Returns 1024 for known archs, 256 for unknown
- `warp_size()` - Returns 32 for RDNA, 64 for CDNA/Vega
- `shared_mem_per_block()` - Returns 64KB for known, 32KB for unknown
- `from_name(name: &str)` - Parses "gfx1100" format strings

### 3. Safety-First HIP Utilities

**Location:** `hip_kernels/quant/common.hip`

C++/HIP header with safety macros and quantization constants:

**Safety Macros:**
```cpp
CHECK_HIP(cmd)        // Wrap HIP API calls, return error on failure
CHECK_BOUNDS(idx, max) // Kernel-safe bounds check
CHECK_LAST()          // Check for kernel launch errors
```

**Quantization Constants:**
```cpp
constexpr int QK_K = 256;          // K-format block size (llama.cpp compatible)
constexpr int BLOCK_SIZE = 256;    // Default HIP block size
constexpr int WARP_SIZE = 32;      // Wavefront size
```

**Device Utilities:**
```cpp
get_global_id()    // Flattened thread ID
get_lane_id()      // Lane within warp (0-31)
get_warp_id()      // Warp within block
```

### 4. Three-Tier Test Framework

**Sanity Tests** (`tests/quant_sanity.rs`):
- Verify CMakeLists.txt exists and is valid
- Verify common.hip contains required macros
- Verify test_kernel.hip exists
- Verify CMake can configure and build
- Verify library files are produced

**Unit Tests** (`tests/quant_unit.rs`):
- Test GpuArchitecture enum properties
- Test architecture name parsing
- Test VRAM checking logic
- Test quantization constant values
- Test block size calculations

**Integration Tests** (`tests/quant_integration.rs`):
- Placeholder tests for Phase 2 (Q4_K, Q8_0 quantization)
- Placeholder tests for Phase 3/4 (GEMM, concurrent ops)
- Example test structure for future implementation

### 5. GPU Architecture Detection

**Changes:**
- Added `arch_name: String` field to `ffi::DeviceInfo`
- Added `architecture: GpuArchitecture` field to `detect::GpuCapabilities`
- Parse architecture name in `detect()` method
- Export `GpuArchitecture` publicly from `gpu` module

### 6. VRAM Management Utilities

**Location:** `tests/gpu_test_utils.rs`

Added VRAM checking functions:
```rust
check_vram_available(required_gb: f64) -> Result<(), String>
rocm_smi_verify() -> Result<(), String>
```

## Test Results

**Total Tests:** 32
- **Sanity:** 8 passed
- **Unit:** 16 passed
- **Integration:** 1 passed, 7 ignored (Phase 2/3/4)

```
cargo test --test quant_sanity --test quant_unit --test quant_integration --features gpu
```

## File Structure

```
rocmforge/
├── hip_kernels/quant/
│   ├── CMakeLists.txt          # CMake build configuration
│   ├── common.hip              # Safety macros and utilities
│   ├── test_kernel.hip         # Test kernel for build verification
│   └── build/
│       ├── CMakeCache.txt
│       ├── Makefile
│       └── lib/
│           ├── libquant_common.a
│           └── libtest_quant.a
├── src/gpu/
│   ├── arch.rs                 # GpuArchitecture enum (NEW)
│   ├── mod.rs                  # Export GpuArchitecture (UPDATED)
│   ├── detect.rs               # Add architecture field (UPDATED)
│   └── ffi.rs                  # Add arch_name field (UPDATED)
├── tests/
│   ├── quant_sanity.rs         # Sanity tier tests (NEW)
│   ├── quant_unit.rs           # Unit tier tests (NEW)
│   ├── quant_integration.rs    # Integration tier tests (NEW)
│   └── gpu_test_utils.rs       # VRAM checking (UPDATED)
└── build.rs                    # Add CMake invocation (UPDATED)
```

## Design Decisions

### 1. Parallel Build Systems (CMake + build.rs)

**Decision:** Use CMake for HIP kernels alongside existing `build.rs` approach.

**Rationale:**
- CMake is the standard build system for ROCm/HIP projects
- Easier integration with existing ROCm tooling
- `build.rs` handles CMake invocation and Cargo linking
- Keeps Rust and native code build concerns separate

### 2. Safety-First Error Handling

**Decision:** All HIP API calls wrapped with `CHECK_HIP` macro.

**Rationale:**
- Never continue past errors
- Early return prevents undefined behavior
- Consistent error handling across all kernels
- Matches project's "safety-first" philosophy

### 3. Three-Tier Testing

**Decision:** Separate tests into sanity → unit → integration tiers.

**Rationale:**
- Sanity tests catch build system issues immediately
- Unit tests verify individual components in isolation
- Integration tests (Phase 2) verify end-to-end functionality
- Clear progression from "does it build?" to "does it work?"

### 4. GpuArchitecture Enum with Properties

**Decision:** Store architecture as enum with methods for properties.

**Rationale:**
- Type-safe vs raw integers
- Central location for architecture-specific parameters
- Easy to add new architectures
- Parse from device name string

## What's Next: Phase 2

Phase 2 will implement actual quantization kernels:

1. **Q4_K Quantization** - 4-bit K-format (256 elements/block)
2. **Q8_0 Quantization** - 8-bit format (32 elements/block)
3. **Dequantization** - Convert back to f32 for verification
4. **Integration Tests** - Replace placeholders with real tests
5. **FFI Layer** - Add Rust wrappers for HIP kernels

## References

- **llama.cpp source:** `/home/feanor/Projects/llama.cpp`
  - `ggml/src/ggml-common.h` - Block structures
  - `ggml/src/ggml-quants.c` - Quantization algorithms
- **ROCm Documentation:** https://rocm.docs.amd.com/
- **HIP Programming Guide:** https://github.com/ROCm/HIP

## Commit History

```
64ab8cd feat(gpu): add CMake build system for quantization kernels
0b0d92c feat(gpu): implement CMake invocation for quantization kernels
3b73e7d test(gpu): add sanity tier tests for quantization infrastructure
b92939e test(gpu): add unit tier tests for quantization components
2f9603c test(gpu): add integration tier test placeholders for Phase 2
b7df556 feat(gpu): add GpuArchitecture enum module
... (plus earlier commits for FFI, detect, cache, weights fixes)
```
