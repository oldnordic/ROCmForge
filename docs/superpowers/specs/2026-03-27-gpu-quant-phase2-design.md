# GPU Quantization Phase 2 Design - Q4_K Implementation

**Date:** 2026-03-27
**Status:** Design Approved
**Phase:** 2 (Q4_K Quantization)

## Overview

Implement GPU-accelerated Q4_K quantization with complete end-to-end pipeline:
- Quantization: f32 weights → Q4_K blocks (4-bit K-format, 256 elements/block)
- Dequantization: Q4_K blocks → f32 weights
- Self-verification: GPU-side accuracy check (no CPU round-trip)

**Scope:** Focus on Q4_K first. Q8_0 will be added in a later phase after Q4_K is validated.

## Q4_K Format Overview

Q4_K is a 4-bit quantization format that stores 256 f32 values in 144 bytes (effective 4.5 bits per weight).

**Block Structure (from llama.cpp):**
```
Total: 144 bytes for 256 f32 values
├── d (f16): 2 bytes        - Global scale for all 256 elements
├── dmin (f16): 2 bytes     - Global minimum scale
├── scales[12]: 12 bytes    - Quantized scales for 8 sub-blocks (packed 6-bit)
└── qs[128]: 128 bytes      - 4-bit quantized values (256 elements / 2 per byte)
```

**Super-block Organization:**
- 256 elements divided into 8 sub-blocks of 32 elements each
- `d` and `dmin` provide global scaling
- `scales[12]` contains per-subblock scales and mins (quantized to 6 bits)
- Each value in `qs` stores 2 elements (4 bits each)

**Quantization Formula:**
```
For each element x in sub-block i:
    scale_i = dequantize_scale(scales[i])  // 6-bit → float
    q = round(x / (d * scale_i))
    clamp q to [-8, 7]  // 4-bit signed range
```

**Dequantization Formula:**
```
For each quantized value q:
    scale_i = dequantize_scale(scales[i])
    x = q * d * scale_i
```

## Architecture

### High-Level Data Flow

```
CPU f32 weights → GPU memory → Q4_K quantization kernel → GPU quantized blocks
                                                          ↓
                                                    Dequantize kernel
                                                          ↓
                              GPU verification (compare dequantized vs original)
                                                          ↓
                                                    Result: pass/fail
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Q4_K Quantization Kernel | `hip_kernels/quant/q4_k_quantize.hip` | 256 f32 → quantized block |
| Q4_K Dequantization Kernel | `hip_kernels/quant/q4_k_dequantize.hip` | Block → 256 f32 |
| Q4_K Verification Kernel | `hip_kernels/quant/q4_k_verify.hip` | GPU-side accuracy check |
| FFI Declarations | `src/gpu/ffi.rs` | extern "C" function declarations |
| GpuQuant Struct | `src/gpu/quant.rs` | Safe Rust wrapper |
| Q4_K Types | `src/gpu/quant/types.rs` | Rust representations of Q4_K blocks |

### Design Principles

- **Wavefront-aware:** Optimize for 32-wide (RDNA) and 64-wide (CDNA/Vega) wavefronts
- **Shared memory:** Use for scale computation optimization
- **GPU self-verification:** No CPU round-trip during accuracy checks
- **Safety-first:** Follow Phase 1's CHECK_HIP, CHECK_BOUNDS, CHECK_LAST patterns
- **Project patterns:** Consistent with existing `GpuKvCache`, `GpuForwardScratch` structs

## Components

### 1. HIP Kernels

**File:** `hip_kernels/quant/q4_k_quantize.hip`

```cpp
#include "common.hip"

/// Quantize 256 f32 values to one Q4_K block
/// Each thread block processes one 256-element block
__global__ void quantize_q4_k_kernel(
    const float* input,        // Input: 256 f32 values per block
    GpuQ4KBlock* output,       // Output: 1 quantized block
    int num_blocks
);

/// C wrapper for Rust FFI
extern "C" hipError_t hip_quantize_q4_k(
    const float* d_input,
    GpuQ4KBlock* d_output,
    int num_blocks,
    hipStream_t stream
);
```

**File:** `hip_kernels/quant/q4_k_dequantize.hip`

```cpp
#include "common.hip"

/// Dequantize one Q4_K block to 256 f32 values
/// Each thread processes 2 values (4-bit packing)
__global__ void dequantize_q4_k_kernel(
    const GpuQ4KBlock* input,
    float* output,
    int num_blocks
);

/// C wrapper for Rust FFI
extern "C" hipError_t hip_dequantize_q4_k(
    const GpuQ4KBlock* d_input,
    float* d_output,
    int num_blocks,
    hipStream_t stream
);
```

**File:** `hip_kernels/quant/q4_k_verify.hip`

```cpp
#include "common.hip"

/// Verify dequantized values match original within tolerance
/// Computes: max(abs(original - dequantized)) < tolerance
/// Returns single bool via device memory
__global__ void verify_q4_k_kernel(
    const float* original,
    const float* dequantized,
    int n,
    float tolerance,
    bool* result  // Device memory for result
);

/// C wrapper for Rust FFI
extern "C" hipError_t hip_verify_q4_k(
    const float* d_original,
    const float* d_dequantized,
    int n,
    float tolerance,
    bool* d_result,
    hipStream_t stream
);
```

**Kernel implementation notes:**
- Follow `test_kernel.hip` pattern: CHECK_BOUNDS, CHECK_LAST
- Use shared memory for scale computation (wavefront reduction)
- Launch with BLOCK_SIZE=256 threads per block
- Grid size = ceil(num_blocks / BLOCK_SIZE)

### 2. FFI Layer

**File:** `src/gpu/ffi.rs` (add to existing)

```rust
// Q4_K block structure (matches HIP layout)
#[repr(C)]
pub struct GpuQ4KBlock {
    pub d: f16,              // delta/scale (2 bytes)
    pub dmin: f16,           // minimum scale (2 bytes)
    pub scales: [u8; 12],    // quantized scales (K_SCALE_SIZE = 12 bytes)
    pub qs: [u8; 128],       // quants, 4-bit values (QK_K/2 = 128 bytes)
}

// FFI declarations
extern "C" {
    pub fn hip_quantize_q4_k(
        d_input: *const f32,
        d_output: *mut GpuQ4KBlock,
        num_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hip_dequantize_q4_k(
        d_input: *const GpuQ4KBlock,
        d_output: *mut f32,
        num_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hip_verify_q4_k(
        d_original: *const f32,
        d_dequantized: *const f32,
        n: i32,
        tolerance: f32,
        d_result: *mut bool,
        stream: hipStream_t,
    ) -> hipError_t;
}
```

### 3. GpuQuant Safe Wrapper

**File:** `src/gpu/quant.rs` (new module)

```rust
//! GPU quantization wrapper - safe interface to HIP kernels
//!
//! Provides Q4_K quantization with GPU self-verification.

use super::ffi::{hipError_t, hipStream_t, hipDevice_t, GpuQ4KBlock};
use super::{GpuDevice, GpuError, GpuResult};
use super::arch::GpuArchitecture;

/// Safe wrapper for GPU quantization operations
pub struct GpuQuant {
    device: hipDevice_t,
    stream: hipStream_t,
    architecture: GpuArchitecture,
}

impl GpuQuant {
    /// Create new GpuQuant from existing GpuDevice
    pub fn new(device: &GpuDevice) -> GpuResult<Self>;

    /// Quantize f32 weights to Q4_K format
    /// Returns GPU buffer with quantized blocks
    pub fn quantize_q4_k(&self, weights: &[f32]) -> GpuResult<Vec<GpuQ4KBlock>>;

    /// Dequantize Q4_K blocks back to f32
    pub fn dequantize_q4_k(&self, blocks: &[GpuQ4KBlock]) -> GpuResult<Vec<f32>>;

    /// Verify accuracy (GPU-side comparison)
    /// Returns true if max(abs(original - dequantized)) < 1e-4
    pub fn verify_accuracy(
        &self,
        original: &[f32],
        dequantized: &[f32],
    ) -> GpuResult<bool>;

    /// Complete roundtrip: quantize → dequantize → verify
    pub fn quantize_and_verify(&self, weights: &[f32]) -> GpuResult<bool>;
}
```

### 4. Q4_K Types Module

**File:** `src/gpu/quant/types.rs` (new module)

```rust
//! Q4_K quantization type definitions

use super::super::ffi::GpuQ4KBlock;

// Quantization constants (from llama.cpp)
pub const QK_K: usize = 256;           // Elements per block
pub const K_SCALE_SIZE: usize = 12;    // Scales array size
pub const Q4_K_BLOCK_SIZE: usize = 128 + 12 + 4; // Total bytes (qs + scales + d/dmin)

/// Rust-owned Q4_K block (can be stored on CPU or GPU)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q4KBlock {
    pub d: f16,              // delta/scale (2 bytes)
    pub dmin: f16,           // minimum scale (2 bytes)
    pub scales: [u8; 12],    // quantized scales (12 bytes)
    pub qs: [u8; 128],       // quants, 4-bit values (128 bytes)
}

impl From<GpuQ4KBlock> for Q4KBlock {
    fn from(gpu: GpuQ4KBlock) -> Self { ... }
}

impl From<Q4KBlock> for GpuQ4KBlock {
    fn from(rust: Q4KBlock) -> Self { ... }
}
```

## Data Flow

### Quantization Flow

1. **User calls:** `gpu_quant.quantize_q4_k(&weights_f32)`
2. **Validate:** `weights.len() % 256 == 0`
3. **Allocate:** GPU buffer for input (f32 array)
4. **Allocate:** GPU buffer for output (GpuQ4KBlock array)
5. **Copy:** `hipMemcpy HtoD` (input → GPU)
6. **Calculate launch config:**
   - `block_size = BLOCK_SIZE` (256)
   - `grid_size = (num_blocks + 255) / 256`
7. **Launch:** `quantize_q4_k_kernel<<<grid, block, 0, stream>>>()`
8. **Copy result:** GPU → CPU (optional, for testing)
9. **Return:** `GpuResult<Vec<GpuQ4KBlock>>`

### Dequantization Flow

1. **User calls:** `gpu_quant.dequantize_q4_k(&blocks)`
2. **Allocate:** GPU buffer for output (f32 array)
3. **Copy:** blocks to GPU if not already there
4. **Calculate launch config:** Based on num_blocks
5. **Launch:** `dequantize_q4_k_kernel<<<grid, block, 0, stream>>>()`
6. **Copy result:** GPU → CPU
7. **Return:** `GpuResult<Vec<f32>>`

### Verification Flow (All on GPU)

1. **After dequantize,** GPU has both original and dequantized arrays
2. **Allocate:** GPU memory for bool result (1 byte)
3. **Launch:** `verify_q4_k_kernel<<<grid, block, 0, stream>>>()`
   - Computes: `max_error = max(abs(original - dequantized))`
   - Writes: `*result = (max_error < tolerance)`
4. **Copy:** bool result GPU → CPU
5. **Return:** `GpuResult<bool>`

## Error Handling

### HIP Kernel Level

**All kernels use CHECK_HIP pattern:**
```cpp
extern "C" hipError_t hip_quantize_q4_k(...) {
    CHECK_HIP(hipSetDevice(device_id));

    quantize_q4_k_kernel<<<grid, block, 0, stream>>>(...);
    CHECK_LAST();

    return hipSuccess;
}
```

**Kernel bounds checking:**
```cpp
__global__ void quantize_q4_k_kernel(...) {
    int idx = get_global_id();
    CHECK_BOUNDS(idx, num_blocks);
    // Safe access guaranteed
}
```

### Rust FFI Level

**Wrap all FFI calls:**
```rust
pub fn quantize_q4_k(&self, weights: &[f32]) -> GpuResult<Vec<GpuQ4KBlock>> {
    // Validate input
    if weights.len() % QK_K != 0 {
        return Err(GpuError::InvalidInput(
            format!("weights length {} must be multiple of {}", weights.len(), QK_K)
        ));
    }

    unsafe {
        let result = hip_quantize_q4_k(d_input, d_output, num_blocks, self.stream);
        if result != hipError_t::hipSuccess {
            return Err(GpuError::HipError(result));
        }
    }
    // ... return result
}
```

### Error Types

| Error | Condition |
|-------|-----------|
| `HipError(hipError)` | HIP API failure |
| `InvalidInput(String)` | Wrong array size, empty input |
| `MemoryError` | Allocation failed |
| `VerificationFailed` | Accuracy threshold not met |
| `UnsupportedArchitecture` | Unknown GPU architecture |

**Never panic - always return GpuResult.**

## Testing Strategy

### Unit Tests (Add to `tests/quant_unit.rs`)

```rust
#[test]
fn test_q4_k_constants() {
    assert_eq!(QK_K, 256);
    assert_eq!(K_SCALE_SIZE, 12);
    assert_eq!(Q4_K_BLOCK_SIZE, 144); // 128 + 12 + 4
}

#[test]
fn test_q4_k_block_size() {
    assert_eq!(std::mem::size_of::<GpuQ4KBlock>(), 176);
}

#[test]
fn test_q4_k_simple_quantization() {
    // Test with known pattern: [1.0, 2.0, 3.0, ... 256.0]
    // Verify quantization produces expected result
}
```

### Integration Tests (Replace placeholders in `tests/quant_integration.rs`)

```rust
#[test]
fn test_q4_k_quantize_dequantize_roundtrip() {
    let weights = generate_test_weights(4096);

    let gpu = GpuDevice::init(0)?;
    let quant = GpuQuant::new(&gpu)?;

    // Quantize
    let blocks = quant.quantize_q4_k(&weights)?;

    // Dequantize
    let recovered = quant.dequantize_q4_k(&blocks)?;

    // Verify (GPU self-verification)
    let accurate = quant.verify_accuracy(&weights, &recovered)?;
    assert!(accurate, "Dequantized values should match within 1e-4");
}

#[test]
fn test_q4_k_realistic_weights() {
    // Load small real weight matrix
    // Compare GPU vs CPU quantization
}

#[test]
fn test_q4_k_vram_no_leak() {
    // Check VRAM before/after quantization
    // Verify memory returned
}
```

### Three-Tier Test Framework

| Tier | Tests | Purpose |
|------|-------|---------|
| Sanity | 8 existing | Build system, headers |
| Unit | +5 new | Constants, block sizes, patterns |
| Integration | Replace 7 placeholders | Full roundtrip, accuracy |

### Success Criteria

- ✅ All tests pass
- ✅ Accuracy: `max(abs(original - dequantized)) < 1e-4`
- ✅ Safety: No panics, proper error propagation
- ✅ VRAM: No leaks (verify with rocm-smi)
- ✅ Performance: Faster than CPU (can measure later)

## File Changes Summary

**New files:**
- `hip_kernels/quant/q4_k_quantize.hip`
- `hip_kernels/quant/q4_k_dequantize.hip`
- `hip_kernels/quant/q4_k_verify.hip`
- `src/gpu/quant.rs`
- `src/gpu/quant/mod.rs`
- `src/gpu/quant/types.rs`

**Modified files:**
- `hip_kernels/quant/CMakeLists.txt` - Add new kernels to build
  ```cmake
  # Add Q4_K kernels to existing test_quant library
  set(QUANT_SOURCES
      test_kernel.hip
      q4_k_quantize.hip    # NEW
      q4_k_dequantize.hip  # NEW
      q4_k_verify.hip      # NEW
  )
  ```
- `src/gpu/ffi.rs` - Add FFI declarations and GpuQ4KBlock struct
- `src/gpu/mod.rs` - Export `quant` module
- `tests/quant_unit.rs` - Add Q4_K unit tests
- `tests/quant_integration.rs` - Replace placeholders with real tests

**Constants:**
- `QK_K = 256` (from llama.cpp)
- `K_SCALE_SIZE = 12`
- `Q4_K_BLOCK_SIZE = 144` bytes

## Dependencies

**External:**
- ROCm/HIP (existing)
- llama.cpp reference (`/home/feanor/Projects/llama.cpp`)

**Internal:**
- Phase 1 foundation (CMake, safety macros, GpuArchitecture)
- Existing GPU infrastructure (GpuDevice, GpuBuffer, error handling)
- CPU quantization (for comparison testing)

## References

- llama.cpp: `ggml/src/ggml-common.h` - Block structures
- llama.cpp: `ggml/src/ggml-quants.c` - Quantization algorithms
- ROCm Documentation: https://rocm.docs.amd.com/
- Phase 1 Design: `docs/gpu_quant_phase1.md`

## Next Steps

After this design is implemented:
1. Run full test suite (sanity + unit + integration)
2. Verify accuracy against CPU implementation
3. Benchmark performance (optional)
4. Design Phase 3: Q8_0 implementation (similar pattern)
