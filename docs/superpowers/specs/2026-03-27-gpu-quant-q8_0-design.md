# GPU Quantization - Q8_0 Implementation Design

**Date:** 2026-03-27
**Status:** Design Approved
**Phase:** Q8_0 Quantization (Extension to Phase 2)

## Overview

Implement GPU-accelerated Q8_0 quantization with complete end-to-end pipeline:
- Quantization: f32 weights → Q8_0 blocks (8-bit format, 32 elements/block)
- Dequantization: Q8_0 blocks → f32 weights
- Self-verification: GPU-side accuracy check (no CPU round-trip)

**Scope:** Complete Q8_0 pipeline following Q4_K patterns. Extends existing GpuQuant with Q8_0 methods.

## Q8_0 Format Overview

Q8_0 is an 8-bit uniform quantization format that stores 32 f32 values in 34 bytes.

**Block Structure (from llama.cpp):**
```
Total: 34 bytes for 32 f32 values
├── d (f16): 2 bytes         - Scale factor
└── qs[32]: 32 bytes        - 8-bit quantized values (int8)
```

**Key Differences from Q4_K:**
| Aspect | Q4_K | Q8_0 |
|--------|------|------|
| Elements per block | 256 | 32 |
| Bits per value | 4 | 8 |
| Block size | 144 bytes | 34 bytes |
| Quantization | Non-uniform sub-blocks | Uniform |
| Complexity | High (scales[12], dmin) | Low (single scale) |

**Quantization Formula:**
```
For each element x:
    scale = max(abs(x)) / 127.0
    q = clamp(round(x / scale), -127, 127)
```

**Dequantization Formula:**
```
For each quantized value q:
    x = q * scale
```

## Architecture

### High-Level Data Flow

```
CPU f32 weights → GPU memory → Q8_0 quantization kernel → GPU quantized blocks
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
| Q8_0 Quantization Kernel | `hip_kernels/quant/q8_0_quantize.hip` | 32 f32 → quantized block |
| Q8_0 Dequantization Kernel | `hip_kernels/quant/q8_0_dequantize.hip` | Block → 32 f32 |
| Q8_0 Verification Kernel | `hip_kernels/quant/q8_0_verify.hip` | GPU-side accuracy check |
| Extended FFI Declarations | `src/gpu/kernels/quant.rs` | extern "C" functions |
| Extended GpuQuant Methods | `src/gpu/quant_wrapper.rs` | Safe wrappers |
| Q8_0 Types | `src/gpu/quant/types.rs` | Rust representations |

### Design Principles

- **Follow Q4_K patterns:** Reuse proven structures from Q4_K implementation
- **Wavefront-aware:** 32-element blocks map perfectly to RDNA wavefronts
- **GPU self-verification:** No CPU round-trip during accuracy checks
- **Safety-first:** Follow Phase 1's CHECK_HIP, CHECK_BOUNDS, CHECK_LAST patterns
- **Extend, don't duplicate:** Add to existing GpuQuant rather than create new struct

## Components

### 1. HIP Kernels

**File:** `hip_kernels/quant/q8_0_quantize.hip`

```cpp
#include "common.hip"

/// Quantize 32 f32 values to one Q8_0 block
/// Each thread block processes one 32-element block
__global__ void quantize_q8_0_kernel(
    const float* input,     // Input: 32 f32 values per block
    void* output,           // Output: Q8_0 block (34 bytes)
    int num_blocks
);

/// C wrapper for Rust FFI
extern "C" void quantize_q8_0_launch(
    const float* input,
    void* output,
    int num_blocks,
    hipStream_t stream
);
```

**File:** `hip_kernels/quant/q8_0_dequantize.hip`

```cpp
#include "common.hip"

/// Dequantize Q8_0 block to 32 f32 values
/// Each thread block processes one 32-element block
__global__ void dequantize_q8_0_kernel(
    const void* input,     // Input: Q8_0 block (34 bytes)
    float* output,         // Output: 32 f32 values
    int num_blocks
);

/// Batched dequantization for parallel processing
__global__ void dequantize_q8_0_batched_kernel(
    const void* input,
    float* output,
    int n,
    int batch_size
);
```

**File:** `hip_kernels/quant/q8_0_verify.hip`

```cpp
#include "common.hip"

/// Verify Q8_0 quantization accuracy on GPU
/// Computes max error, MSE, and relative error
__global__ void verify_q8_0_accuracy_kernel(
    const float* original,  // Original f32 values
    const void* quantized,  // Q8_0 quantized blocks
    float* errors,          // [4] Error metrics (atomic update)
    int n
);

/// Finalize verification metrics
__global__ void finalize_q8_0_metrics_kernel(
    const float* errors,    // [4] Intermediate metrics
    float* metrics,         // [3] Final metrics
    int n
);
```

### 2. Type Definitions

**File:** `src/gpu/quant/types.rs` (extend existing)

```rust
/// Number of elements per Q8_0 block (from llama.cpp)
pub const QK8_0: usize = 32;

/// Total bytes per Q8_0 block
pub const Q8_0_BLOCK_SIZE: usize = 34; // 2 (scale) + 32 (data)

/// Maximum quantized value for Q8_0
pub const Q8_0_MAX: f32 = 127.0;

/// Rust-owned Q8_0 block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q8_0Block {
    pub d: half::f16,        // scale (2 bytes)
    pub qs: [i8; 32],       // quantized values (32 bytes)
}

impl Default for Q8_0Block {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            qs: [0; 32],
        }
    }
}
```

### 3. FFI Declarations

**File:** `src/gpu/kernels/quant.rs` (extend existing)

```rust
/// Quantize f32 data to Q8_0 format
pub fn quantize_q8_k(
    input: *const f32,
    output: *mut u8,
    n: usize,
) -> GpuResult<()>;

/// Dequantize Q8_0 data to f32
pub fn dequantize_q8_k(
    input: *const u8,
    output: *mut f32,
    n: usize,
) -> GpuResult<()>;

/// Batched dequantize Q8_0 data
pub fn dequantize_q8_k_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()>;

/// Verify Q8_0 accuracy
pub fn verify_q8_k_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<(f32, f32, f32)>;

// FFI declarations
unsafe extern "C" {
    fn quantize_q8_0_kernel(...) -> hipError_t;
    fn dequantize_q8_0_kernel(...) -> hipError_t;
    fn dequantize_q8_0_batched_kernel(...) -> hipError_t;
    fn verify_q8_0_accuracy_kernel(...) -> hipError_t;
    fn finalize_q8_0_metrics_kernel(...) -> hipError_t;
}
```

### 4. GpuQuant Extension

**File:** `src/gpu/quant_wrapper.rs` (extend existing)

```rust
impl GpuQuant {
    /// Quantize f32 data to Q8_0 format
    pub fn quantize_q8_0(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        // Validate pointers
        // Call kernel
        // Synchronize
    }

    /// Dequantize Q8_0 data to f32
    pub fn dequantize_q8_0(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        // Validate pointers
        // Call kernel
        // Synchronize
    }

    /// Batched dequantize Q8_0 data
    pub fn dequantize_q8_0_batched(&self, input: *const u8, output: *mut f32, n: usize, batch_size: usize) -> GpuResult<()> {
        // Validate pointers
        // Call kernel
        // Synchronize
    }

    /// Verify Q8_0 quantization accuracy
    pub fn verify_accuracy_q8_0(&self, original: *const f32, quantized: *const u8, n: usize) -> GpuResult<(f32, f32, f32)> {
        // Allocate GPU memory for metrics
        // Run verification kernel
        // Finalize metrics
        // Copy results back
        // Cleanup
    }
}
```

### 5. CMake Build Integration

**File:** `hip_kernels/quant/CMakeLists.txt` (extend existing)

```cmake
# Q8_0 quantization kernel
add_library(q8_0_quantize STATIC
    q8_0_quantize.hip
)

# Link common library to Q8_0 kernel
target_link_libraries(q8_0_quantize
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q8_0_quantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q8_0 dequantization kernel
add_library(q8_0_dequantize STATIC ...)
# ... similar pattern for other kernels
```

### 6. Module Exports

**File:** `src/gpu/mod.rs` (extend existing)

```rust
pub use quant::{QK_K, K_SCALE_SIZE, Q4_K_BLOCK_SIZE, Q4KBlock};
pub use quant::{QK8_0, Q8_0_BLOCK_SIZE, Q8_0_MAX, Q8_0Block};
```

## Testing

### Unit Tests (tests/quant_unit.rs)

- `test_q8_0_constants`: Verify QK8_0=32, Q8_0_BLOCK_SIZE=34
- `test_q8_0_block_struct_size`: Verify sizeof(Q8_0Block)=34
- `test_q8_0_block_default`: Verify default array size

### Integration Tests (tests/quant_integration.rs)

```rust
#[test]
#[serial]
fn test_q8_0_quantization() {
    // Allocate GPU buffers
    // Prepare test data (256 elements = 8 Q8_0 blocks)
    // Quantize on GPU
    // Dequantize on GPU
    // Verify accuracy (tolerance for 8-bit: 0.01)
    // Test verify_accuracy_q8_0
}
```

### Test Pattern

Reuse Q4_K's 256-element test for consistency:
- 256 f32 values → 8 Q8_0 blocks (8 × 34 = 272 bytes)
- Same tolerance-based verification
- Same error metrics computation

## Kernel Implementation Details

### Quantization Kernel

```cpp
__global__ void quantize_q8_0_kernel(
    const float* input,
    void* output,
    int num_blocks
) {
    int block_idx = blockIdx.x;
    const float* x = &input[block_idx * QK8_0];

    // Shared memory for reduction
    __shared__ float s_max_val;
    __shared__ half s_scale;

    // Find max absolute value in block
    float max_val = 0.0f;
    for (int i = threadIdx.x; i < QK8_0; i += blockDim.x) {
        max_val = fmaxf(max_val, fabsf(x[i]));
    }

    // Warp reduction
    // ... (same pattern as Q4_K)

    // Compute scale
    if (threadIdx.x == 0) {
        s_scale = __float2half(fmaxf(max_val / 127.0f, 1e-30f));
    }
    __syncthreads();

    // Quantize values
    uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q8_0_BLOCK_SIZE;

    // Write scale
    if (threadIdx.x == 0) {
        half* d_ptr = reinterpret_cast<half*>(out);
        *d_ptr = s_scale;
    }

    // Write quantized values
    for (int i = threadIdx.x; i < QK8_0; i += blockDim.x) {
        int8_t q = static_cast<int8_t>(
            fminf(fmaxf(x[i] / __half2float(s_scale), -127.0f), 127.0f)
        );
        out[2 + i] = reinterpret_cast<uint8_t&>(q);
    }
}
```

### Dequantization Kernel

```cpp
__global__ void dequantize_q8_0_kernel(
    const void* input,
    float* output,
    int num_blocks
) {
    int block_idx = blockIdx.x;
    const uint8_t* in = static_cast<const uint8_t*>(input) + block_idx * Q8_0_BLOCK_SIZE;

    // Read scale
    half scale = *reinterpret_cast<const half*>(in);

    // Dequantize values
    const int8_t* qs = reinterpret_cast<const int8_t*>(in + 2);
    float* out = &output[block_idx * QK8_0];

    for (int i = threadIdx.x; i < QK8_0; i += blockDim.x) {
        out[i] = static_cast<float>(qs[i]) * __half2float(scale);
    }
}
```

## Success Criteria

1. **Compiles:** All Q8_0 kernels compile without errors
2. **Links:** Libraries properly linked via CMake and build.rs
3. **Tests Pass:** Unit tests validate constants and struct sizes
4. **Integration Test:** 256-element quantize/dequantize/verify cycle passes
5. **Accuracy:** 8-bit quantization achieves < 0.01 max error
6. **Consistent:** Follows all Q4_K patterns (error handling, memory management, etc.)

## Dependencies

- **Phase 1:** HIP infrastructure, error handling patterns
- **Phase 2 Q4_K:** CMake build system, FFI patterns, GpuQuant struct
- **llama.cpp:** Reference implementation for Q8_0 format

## Future Work (Out of Scope)

- Q4_K × Q8_0 matmul kernel (GEMM for inference)
- Shared infrastructure refactoring (common quantization utilities)
- Performance optimization (SIMD, shared memory tuning)
