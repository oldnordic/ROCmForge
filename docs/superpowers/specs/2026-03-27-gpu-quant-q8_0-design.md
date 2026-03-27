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
    float* errors,          // [4] Error metrics (atomic update):
                           //   errors[0]: max error (atomic max)
                           //   errors[1]: MSE (accumulated, atomic add)
                           //   errors[2]: sum of original values (atomic add)
                           //   errors[3]: sum of errors (atomic add)
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
///
/// # Arguments
/// * `input` - GPU pointer to f32 input data [n]
/// * `output` - GPU pointer to Q8_0 output data [n/32 * 34]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn quantize_q8_0(
    input: *const f32,
    output: *mut u8,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Ok(());
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe {
        quantize_q8_0_kernel(input, output, n as c_int)
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("quantize_q8_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dequantize Q8_0 data to f32
///
/// # Arguments
/// * `input` - GPU pointer to Q8_0 input data [n/32 * 34]
/// * `output` - GPU pointer to f32 output data [n]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q8_0(
    input: *const u8,
    output: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Ok(());
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe {
        dequantize_q8_0_kernel(input, output, n as c_int)
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q8_0 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched dequantize Q8_0 data to f32
///
/// # Arguments
/// * `input` - GPU pointer to Q8_0 input data [batch_size][n/32 * 34]
/// * `output` - GPU pointer to f32 output data [batch_size][n]
/// * `n` - Elements per batch
/// * `batch_size` - Number of batches
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn dequantize_q8_0_batched(
    input: *const u8,
    output: *mut f32,
    n: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n == 0 || batch_size == 0 {
        return Ok(());
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe {
        dequantize_q8_0_batched_kernel(input, output, n as c_int, batch_size as c_int)
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dequantize_q8_0_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Verify Q8_0 accuracy
///
/// # Arguments
/// * `original` - GPU pointer to original f32 data [n]
/// * `quantized` - GPU pointer to Q8_0 quantized data [n/32 * 34]
/// * `errors` - GPU pointer to error metrics [4]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn verify_q8_0_accuracy(
    original: *const f32,
    quantized: *const u8,
    errors: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Ok(());
    }

    let num_blocks = (n + 31) / 32;
    if num_blocks == 0 {
        return Ok(());
    }

    let result = unsafe {
        verify_q8_0_accuracy_kernel(original, quantized, errors, n as c_int)
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("verify_q8_0_accuracy kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Finalize Q8_0 accuracy metrics
///
/// # Arguments
/// * `errors` - GPU pointer to intermediate error metrics [4]
/// * `metrics` - GPU pointer to final metrics [3]
/// * `n` - Total number of elements
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn finalize_q8_0_metrics(
    errors: *const f32,
    metrics: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Ok(());
    }

    let result = unsafe {
        finalize_q8_0_metrics_kernel(errors, metrics, n as c_int)
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("finalize_q8_0_metrics kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// ── FFI Declarations ─────────────────────────────────────────────────────────────

/// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn quantize_q8_0_kernel(
        input: *const f32,
        output: *mut u8,
        n: c_int,
    ) -> hipError_t;

    fn dequantize_q8_0_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
    ) -> hipError_t;

    fn dequantize_q8_0_batched_kernel(
        input: *const u8,
        output: *mut f32,
        n: c_int,
        batch_size: c_int,
    ) -> hipError_t;

    fn verify_q8_0_accuracy_kernel(
        original: *const f32,
        quantized: *const u8,
        errors: *mut f32,
        n: c_int,
    ) -> hipError_t;

    fn finalize_q8_0_metrics_kernel(
        errors: *const f32,
        metrics: *mut f32,
        n: c_int,
    ) -> hipError_t;
}
```

### 4. GpuQuant Extension

**File:** `src/gpu/quant_wrapper.rs` (extend existing)

```rust
impl GpuQuant {
    /// Quantize f32 data to Q8_0 format
    pub fn quantize_q8_0(&self, input: *const f32, output: *mut u8, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q8_0: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q8_0: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "quantize_q8_0: output pointer is null".to_string(),
            });
        }

        // Call kernel
        quantize_q8_0(input, output, n)?;

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;

        Ok(())
    }

    /// Dequantize Q8_0 data to f32
    pub fn dequantize_q8_0(&self, input: *const u8, output: *mut f32, n: usize) -> GpuResult<()> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if input.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0: input pointer is null".to_string(),
            });
        }

        if output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0: output pointer is null".to_string(),
            });
        }

        // Call kernel
        dequantize_q8_0(input, output, n)?;

        // Synchronize
        self.device.synchronize()?;

        Ok(())
    }

    /// Batched dequantize Q8_0 data
    pub fn dequantize_q8_0_batched(&self, input: *const u8, output: *mut f32, n: usize, batch_size: usize) -> GpuResult<()> {
        if n == 0 || batch_size == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0_batched: n and batch_size cannot be zero".to_string(),
            });
        }

        // Validate pointers (same pattern as above)
        if input.is_null() || output.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "dequantize_q8_0_batched: pointer is null".to_string(),
            });
        }

        // Call kernel
        dequantize_q8_0_batched(input, output, n, batch_size)?;

        // Synchronize
        self.device.synchronize()?;

        Ok(())
    }

    /// Verify Q8_0 quantization accuracy
    pub fn verify_accuracy_q8_0(&self, original: *const f32, quantized: *const u8, n: usize) -> GpuResult<(f32, f32, f32)> {
        if n == 0 {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_accuracy_q8_0: n cannot be zero".to_string(),
            });
        }

        // Validate pointers
        if original.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_accuracy_q8_0: original pointer is null".to_string(),
            });
        }

        if quantized.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "verify_accuracy_q8_0: quantized pointer is null".to_string(),
            });
        }

        // Allocate GPU memory for error metrics
        let errors_gpu = unsafe {
            ffi::hip_malloc(4 * std::mem::size_of::<f32>())?
        };
        let metrics_gpu = unsafe {
            ffi::hip_malloc(3 * std::mem::size_of::<f32>())?
        };

        // Initialize errors to zero
        let zeros = vec![0.0f32; 4];
        unsafe {
            ffi::hip_memcpy_h2d(errors_gpu, zeros.as_ptr() as *const u8, 4 * std::mem::size_of::<f32>())?;
        }

        // Run verification kernel
        verify_q8_0_accuracy(original, quantized, errors_gpu as *mut f32, n)?;

        // Finalize metrics
        finalize_q8_0_metrics(errors_gpu as *const f32, metrics_gpu as *mut f32, n)?;

        // Synchronize
        self.device.synchronize()?;

        // Copy metrics back to host
        let mut metrics = [0.0f32; 3];
        unsafe {
            ffi::hip_memcpy_d2h(
                metrics.as_mut_ptr() as *mut u8,
                metrics_gpu as *const u8,
                3 * std::mem::size_of::<f32>()
            )?;
        }

        // Cleanup
        unsafe {
            ffi::hip_free(errors_gpu);
            ffi::hip_free(metrics_gpu);
        }

        Ok((metrics[0], metrics[1], metrics[2]))
    }
}
```

### 6. Build System Integration

**File:** `build.rs` (extend existing)

Add Q8_0 libraries to the `libs_to_copy` array:

```rust
let libs_to_copy = vec![
    ("libtest_quant.a", "test_quant"),
    ("libq4_k_quantize.a", "q4_k_quantize"),
    ("libq4_k_dequantize.a", "q4_k_dequantize"),
    ("libq4_k_verify.a", "q4_k_verify"),
    // Q8_0 libraries (new):
    ("libq8_0_quantize.a", "q8_0_quantize"),
    ("libq8_0_dequantize.a", "q8_0_dequantize"),
    ("libq8_0_verify.a", "q8_0_verify"),
];
```

### 7. CMake Build Integration

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

### 8. Module Exports

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
- Tolerance: 0.01 max error for 8-bit (vs 0.5 for 4-bit Q4_K)
- Same error metrics computation (max error, MSE, relative error)

### Expected Test Results

For typical random data in [-1, 1] range:
- Max error: < 0.01 (8-bit precision)
- MSE: < 0.0001
- Relative error: < 1% for values > 0.01

## Kernel Implementation Details

### Thread Strategy

**Key Optimization:** Q8_0's 32-element block size maps perfectly to RDNA wavefronts:
- One thread block = one Q8_0 block (32 elements)
- Use 32 threads per block (1 wavefront on RDNA)
- Each thread processes 1 element
- No loop striding needed for quantization/dequantization
- Warp reduction for max finding (single __shfl_down)

This is simpler than Q4_K which requires 256 elements with complex striding.

### Quantization Kernel

```cpp
__global__ void quantize_q8_0_kernel(
    const float* input,
    void* output,
    int num_blocks
) {
    int block_idx = blockIdx.x;
    const float* x = &input[block_idx * QK8_0];

    // Each thread loads one element (QK8_0 = 32 threads)
    float val = (threadIdx.x < QK8_0) ? x[threadIdx.x] : 0.0f;
    float abs_val = fabsf(val);

    // Warp reduction to find max (RDNA warp = 32 threads)
    // After reduction, threadIdx.x == 0 has the max
    float max_val = abs_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down(max_val, offset));
    }

    // Compute scale (thread 0 only)
    __shared__ half s_scale;
    if (threadIdx.x == 0) {
        // Avoid division by zero with max(1e-30f, ...)
        s_scale = __float2half(fmaxf(max_val / 127.0f, 1e-30f));
    }
    __syncthreads();

    // Quantize values (each thread processes its element)
    uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q8_0_BLOCK_SIZE;

    // Thread 0 writes scale (f16 = 2 bytes)
    if (threadIdx.x == 0) {
        half* d_ptr = reinterpret_cast<half*>(out);
        *d_ptr = s_scale;
    }

    // All threads write their quantized value
    // NOTE: Offset by 2 bytes for Q8_0 (vs 4 bytes for Q4_K)
    // Q8_0 block: [d: 2 bytes] [qs[0]: 1 byte] [qs[1]: 1 byte] ...
    // Q4_K block:  [scales: 4 bytes] [qs[0]: 1/2 byte] ...
    if (threadIdx.x < QK8_0) {
        int8_t q = static_cast<int8_t>(
            fminf(fmaxf(val / __half2float(s_scale), -127.0f), 127.0f)
        );
        out[2 + threadIdx.x] = reinterpret_cast<uint8_t&>(q);
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

    // Read scale (first 2 bytes of Q8_0 block)
    half scale = *reinterpret_cast<const half*>(in);

    // Dequantize values (each thread processes 1 element)
    const int8_t* qs = reinterpret_cast<const int8_t*>(in + 2);  // Skip 2-byte header
    float* out = &output[block_idx * QK8_0];

    // Each thread dequantizes its element (32 threads = 32 elements)
    if (threadIdx.x < QK8_0) {
        out[threadIdx.x] = static_cast<float>(qs[threadIdx.x]) * __half2float(scale);
    }
}
```

## Memory Alignment Considerations

**Block Alignment:**
- Q8_0 blocks are 34 bytes (not power-of-2 aligned)
- For optimal memory access, ensure block arrays start on 16-byte boundary
- ROCm typically handles unaligned loads/stores, but alignment improves performance

**Element Access Pattern:**
- Quantized values (int8) are accessed sequentially
- Scale (f16) is accessed once per block
- Coalesced reads: sequential threads reading sequential addresses

**Memory Layout:**
```
Block N: [d: f16][qs0: int8][qs1: int8]...[qs31: int8]
Block N+1: [d: f16][qs0: int8][qs1: int8]...[qs31: int8]
```

**Total size for N elements:** `(N / 32) * 34` bytes

## Performance Expectations

**Throughput (per SM on RDNA3):**
- Quantization: ~500-800 GB/s memory bandwidth limited
- Dequantization: ~600-900 GB/s (simpler than quantization)

**Latency:**
- Quantization per 256 elements: ~1-2 microseconds
- Dequantization per 256 elements: ~0.5-1 microsecond

**Comparison to Q4_K:**
- Q8_0 is ~2x faster than Q4_K for quantization (simpler algorithm)
- Q8_0 dequantization is ~1.5x faster (uniform vs non-uniform)
- Tradeoff: Q8_0 uses 2x memory vs Q4_K (34 vs 144 bytes for 256 elements)

**Optimization Opportunities:**
- Vector loads for quantized values (4 int8 = 32 bits per load)
- Pre-fetching next block during current block processing
- Batched kernels for multi-tensor processing

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
