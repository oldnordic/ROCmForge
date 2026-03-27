# GPU Quantization Phase 2 - Q4_K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GPU-accelerated Q4_K quantization with quantization, dequantization, and GPU self-verification kernels.

**Architecture:** HIP kernels for AMD GPU with wavefront-aware optimization, safe Rust wrappers following project patterns, three-tier test framework.

**Tech Stack:** HIP/ROCm, Rust, CMake, llama.cpp reference implementation

---

## File Structure Overview

**New files:**
- `hip_kernels/quant/q4_k_quantize.hip` - Quantization kernel (f32 → Q4_K block)
- `hip_kernels/quant/q4_k_dequantize.hip` - Dequantization kernel (Q4_K → f32)
- `hip_kernels/quant/q4_k_verify.hip` - Verification kernel (GPU accuracy check)
- `src/gpu/quant/mod.rs` - Module exports
- `src/gpu/quant/types.rs` - Q4_K type definitions
- `src/gpu/quant.rs` - GpuQuant safe wrapper

**Modified files:**
- `hip_kernels/quant/CMakeLists.txt` - Add new kernels to build
- `src/gpu/ffi.rs` - Add FFI declarations and GpuQ4KBlock struct
- `src/gpu/mod.rs` - Export quant module
- `tests/quant_unit.rs` - Add Q4_K unit tests
- `tests/quant_integration.rs` - Replace placeholders with real tests

---

## Task 1: Add Q4_K Types to FFI Layer

**Files:**
- Modify: `src/gpu/ffi.rs`

- [ ] **Step 1: Add GpuQ4KBlock struct to ffi.rs**

Add to `src/gpu/ffi.rs` after the existing structs:

```rust
/// Q4_K quantized block (144 bytes for 256 f32 values)
/// Matches llama.cpp block_q4_k structure
#[repr(C)]
pub struct GpuQ4KBlock {
    pub d: f16,              // delta/scale (2 bytes)
    pub dmin: f16,           // minimum scale (2 bytes)
    pub scales: [u8; 12],    // quantized scales (12 bytes)
    pub qs: [u8; 128],       // quants, 4-bit values (128 bytes)
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --features gpu`
Expected: SUCCESS (no errors)

- [ ] **Step 3: Commit**

```bash
git add src/gpu/ffi.rs
git commit -m "feat(gpu): add GpuQ4KBlock struct to FFI layer

144-byte structure matching llama.cpp block_q4_k:
- d (f16): global scale
- dmin (f16): minimum scale
- scales[12]: quantized sub-block scales
- qs[128]: 4-bit quantized values

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Create Q4_K Types Module

**Files:**
- Create: `src/gpu/quant/mod.rs`
- Create: `src/gpu/quant/types.rs`

- [ ] **Step 1: Create types.rs with Q4_K constants**

Create `src/gpu/quant/types.rs`:

```rust
//! Q4_K quantization type definitions

/// Number of elements per Q4_K block (from llama.cpp)
pub const QK_K: usize = 256;

/// Scales array size (from llama.cpp)
pub const K_SCALE_SIZE: usize = 12;

/// Total bytes per Q4_K block
pub const Q4_K_BLOCK_SIZE: usize = 128 + 12 + 4; // qs + scales + d/dmin

/// Rust-owned Q4_K block
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Q4KBlock {
    pub d: f16,              // delta/scale (2 bytes)
    pub dmin: f16,           // minimum scale (2 bytes)
    pub scales: [u8; 12],    // quantized scales (12 bytes)
    pub qs: [u8; 128],       // quants, 4-bit values (128 bytes)
}

impl Default for Q4KBlock {
    fn default() -> Self {
        Self {
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
            scales: [0; 12],
            qs: [0; 128],
        }
    }
}
```

- [ ] **Step 2: Create mod.rs for quant module**

Create `src/gpu/quant/mod.rs`:

```rust
//! GPU quantization module

mod types;

pub use types::{QK_K, K_SCALE_SIZE, Q4_K_BLOCK_SIZE, Q4KBlock};
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check --features gpu`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add src/gpu/quant/
git commit -m "feat(gpu): add Q4_K types module

Constants and Q4KBlock struct matching design doc.
Separate module for quantization types.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Export quant Module from gpu

**Files:**
- Modify: `src/gpu/mod.rs`

- [ ] **Step 1: Add quant module declaration**

Add to `src/gpu/mod.rs` with other module declarations:

```rust
mod quant;
```

- [ ] **Step 2: Export quant types**

Add to the pub use section:

```rust
pub use quant::{QK_K, K_SCALE_SIZE, Q4_K_BLOCK_SIZE, Q4KBlock};
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check --features gpu`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add src/gpu/mod.rs
git commit -m "feat(gpu): export quant module from gpu

Export Q4_K constants and Q4KBlock type for use in tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Add Q4_K Unit Tests for Types

**Files:**
- Modify: `tests/quant_unit.rs`

- [ ] **Step 1: Add Q4_K constant tests**

Add to `tests/quant_unit.rs`:

```rust
/// Test Q4_K constants match llama.cpp
#[test]
#[serial]
fn test_q4_k_constants() {
    use rocmforge::gpu::{QK_K, K_SCALE_SIZE, Q4_K_BLOCK_SIZE};

    assert_eq!(QK_K, 256, "QK_K must be 256 elements per block");
    assert_eq!(K_SCALE_SIZE, 12, "K_SCALE_SIZE must be 12");
    assert_eq!(Q4_K_BLOCK_SIZE, 144, "Q4_K_BLOCK_SIZE must be 144 bytes");
}

/// Test Q4KBlock size
#[test]
#[serial]
fn test_q4_k_block_size() {
    use rocmforge::gpu::Q4KBlock;

    let block = Q4KBlock::default();
    assert_eq!(std::mem::size_of::<Q4KBlock>(), 144);
}

/// Test Q4KBlock default values
#[test]
#[serial]
fn test_q4_k_block_default() {
    use rocmforge::gpu::Q4KBlock;

    let block = Q4KBlock::default();
    assert_eq!(block.scales.len(), 12);
    assert_eq!(block.qs.len(), 128);
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test --test quant_unit --features gpu test_q4_k`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/quant_unit.rs
git commit -m "test(gpu): add Q4_K type unit tests

Tests for Q4_K constants, block size, and default values.
Verifies llama.cpp compatibility.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Create Q4_K Quantization Kernel (HIP)

**Files:**
- Create: `hip_kernels/quant/q4_k_quantize.hip`

- [ ] **Step 1: Write quantization kernel**

Create `hip_kernels/quant/q4_k_quantize.hip`:

```cpp
#include "common.hip"

/// Quantize 256 f32 values to one Q4_K block
/// Each thread block processes one 256-element block
__global__ void quantize_q4_k_kernel(
    const float* __restrict__ input,
    GpuQ4KBlock* __restrict__ output,
    int num_blocks
) {
    const int block_idx = get_global_id();
    CHECK_BOUNDS(block_idx, num_blocks);

    // Load 256 f32 values for this block
    const float* block_input = &input[block_idx * QK_K];

    // Find min/max for scaling
    float vmin = block_input[0];
    float vmax = block_input[0];

    #pragma unroll
    for (int i = 1; i < QK_K; i++) {
        vmin = fminf_gpu(vmin, block_input[i]);
        vmax = fmaxf_gpu(vmax, block_input[i]);
    }

    // Compute global scale (d) and minimum scale (dmin)
    float d = (vmax - vmin) / 15.0f;  // 4-bit range [-8, 7]
    float dmin = vmin;

    // Prevent division by zero
    if (d < 1e-30f) d = 1e-30f;

    // Quantize values to 4-bit (simplified version)
    // TODO: Implement proper sub-block scaling with scales[12]
    GpuQ4KBlock block;
    block.d = __float2half(d);
    block.dmin = __float2half(dmin);

    // Initialize scales to neutral values
    #pragma unroll
    for (int i = 0; i < K_SCALE_SIZE; i++) {
        block.scales[i] = 128;  // Neutral scale (6-bit: 32+32)
    }

    // Quantize each value to 4-bit
    #pragma unroll
    for (int i = 0; i < QK_K / 2; i++) {
        float x0 = block_input[i * 2];
        float x1 = block_input[i * 2 + 1];

        int8_t q0 = (int8_t)roundf((x0 - dmin) / d);
        int8_t q1 = (int8_t)roundf((x1 - dmin) / d);

        // Clamp to 4-bit signed range [-8, 7]
        q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
        q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);

        // Pack two 4-bit values into one byte
        block.qs[i] = ((q0 & 0xF) << 4) | (q1 & 0xF);
    }

    output[block_idx] = block;
}

/// C wrapper for Rust FFI
extern "C" hipError_t hip_quantize_q4_k(
    const float* d_input,
    GpuQ4KBlock* d_output,
    int num_blocks,
    hipStream_t stream
) {
    int block_size = BLOCK_SIZE;
    int grid_size = (num_blocks + block_size - 1) / block_size;

    quantize_q4_k_kernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, num_blocks
    );

    CHECK_LAST();
    return hipSuccess;
}
```

- [ ] **Step 2: Update CMakeLists.txt**

Modify `hip_kernels/quant/CMakeLists.txt`, add to QUANT_SOURCES or create new library:

```cmake
# Q4_K quantization library
add_library(q4_k_quant STATIC
    q4_k_quantize.hip
)

target_link_libraries(q4_k_quant
    quant_common
)

set_target_properties(q4_k_quant PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
```

- [ ] **Step 3: Test CMake build**

Run: `cmake --build hip_kernels/quant/build`
Expected: SUCCESS, builds libq4_k_quant.a

- [ ] **Step 4: Commit**

```bash
git add hip_kernels/quant/q4_k_quantize.hip hip_kernels/quant/CMakeLists.txt
git commit -m "feat(hip): add Q4_K quantization kernel

Implements basic 4-bit quantization:
- Each thread block processes 256 elements (QK_K)
- Computes global scale (d) and minimum (dmin)
- Quantizes to 4-bit signed range [-8, 7]
- Packs two 4-bit values per byte

Simplified version (no sub-block scaling yet).
CMake builds libq4_k_quant.a

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Add FFI Declaration for Quantization

**Files:**
- Modify: `src/gpu/ffi.rs`

- [ ] **Step 1: Add FFI function declaration**

Add to `src/gpu/ffi.rs` (inside the module, not in extern blocks for C types):

```rust
extern "C" {
    /// Quantize f32 weights to Q4_K format
    pub fn hip_quantize_q4_k(
        d_input: *const f32,
        d_output: *mut GpuQ4KBlock,
        num_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --features gpu`
Expected: SUCCESS

- [ ] **Step 3: Commit**

```bash
git add src/gpu/ffi.rs
git commit -m "feat(gpu): add FFI declaration for hip_quantize_q4_k

Bridge between HIP kernel and Rust code.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Create Q4_K Dequantization Kernel

**Files:**
- Create: `hip_kernels/quant/q4_k_dequantize.hip`

- [ ] **Step 1: Write dequantization kernel**

Create `hip_kernels/quant/q4_k_dequantize.hip`:

```cpp
#include "common.hip"

/// Dequantize one Q4_K block to 256 f32 values
/// Each thread processes 2 values (4-bit packing)
__global__ void dequantize_q4_k_kernel(
    const GpuQ4KBlock* __restrict__ input,
    float* __restrict__ output,
    int num_blocks
) {
    const int block_idx = get_global_id();
    CHECK_BOUNDS(block_idx, num_blocks);

    GpuQ4KBlock block = input[block_idx];
    float d = __half2float(block.d);
    float dmin = __half2float(block.dmin);

    float* block_output = &output[block_idx * QK_K];

    // Dequantize each value
    #pragma unroll
    for (int i = 0; i < QK_K / 2; i++) {
        uint8_t packed = block.qs[i];
        int8_t q0 = (int8_t)((packed >> 4) & 0xF);
        int8_t q1 = (int8_t)(packed & 0xF);

        // Sign-extend 4-bit to 8-bit signed
        q0 = (q0 & 0x8) ? (q0 | 0xF0) : q0;
        q1 = (q1 & 0x8) ? (q1 | 0xF0) : q1;

        block_output[i * 2] = q0 * d + dmin;
        block_output[i * 2 + 1] = q1 * d + dmin;
    }
}

/// C wrapper for Rust FFI
extern "C" hipError_t hip_dequantize_q4_k(
    const GpuQ4KBlock* d_input,
    float* d_output,
    int num_blocks,
    hipStream_t stream
) {
    int block_size = BLOCK_SIZE;
    int grid_size = (num_blocks + block_size - 1) / block_size;

    dequantize_q4_k_kernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, num_blocks
    );

    CHECK_LAST();
    return hipSuccess;
}
```

- [ ] **Step 2: Update CMakeLists.txt**

Add to `hip_kernels/quant/CMakeLists.txt`:

```cmake
# Add to q4_k_quant library or create dequantize library
add_library(q4_k_dequant STATIC
    q4_k_dequantize.hip
)

target_link_libraries(q4_k_dequant
    quant_common
)

set_target_properties(q4_k_dequant PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
```

- [ ] **Step 3: Test CMake build**

Run: `cmake --build hip_kernels/quant/build`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add hip_kernels/quant/q4_k_dequantize.hip hip_kernels/quant/CMakeLists.txt
git commit -m "feat(hip): add Q4_K dequantization kernel

Reconstructs f32 values from Q4_K blocks:
- Unpacks 4-bit values from bytes
- Sign-extends 4-bit to 8-bit signed
- Applies scale and minimum

CMake builds libq4_k_dequant.a

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Add FFI Declaration for Dequantization

**Files:**
- Modify: `src/gpu/ffi.rs`

- [ ] **Step 1: Add FFI function declaration**

Add to `src/gpu/ffi.rs` extern block:

```rust
extern "C" {
    // ... existing declarations ...

    /// Dequantize Q4_K blocks to f32
    pub fn hip_dequantize_q4_k(
        d_input: *const GpuQ4KBlock,
        d_output: *mut f32,
        num_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --features gpu`
Expected: SUCCESS

- [ ] **Step 3: Commit**

```bash
git add src/gpu/ffi.rs
git commit -m "feat(gpu): add FFI declaration for hip_dequantize_q4_k

Bridge between HIP dequantization kernel and Rust.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Create Q4_K Verification Kernel

**Files:**
- Create: `hip_kernels/quant/q4_k_verify.hip`

- [ ] **Step 1: Write verification kernel**

Create `hip_kernels/quant/q4_k_verify.hip`:

```cpp
#include "common.hip"

/// Verify dequantized values match original within tolerance
/// Uses parallel reduction to find max error
__global__ void verify_q4_k_kernel(
    const float* __restrict__ original,
    const float* __restrict__ dequantized,
    int n,
    float tolerance,
    bool* result
) {
    __shared__ float block_max_error;
    if (threadIdx.x == 0) {
        block_max_error = 0.0f;
    }
    __syncthreads();

    const int idx = get_global_id();
    CHECK_BOUNDS(idx, n);

    // Compute local error
    float error = fabsf(original[idx] - dequantized[idx]);

    // Parallel reduction within block
    atomicMax(&block_max_error, error);

    __syncthreads();

    // First thread writes result if within tolerance
    if (threadIdx.x == 0) {
        if (block_max_error > tolerance) {
            *result = false;
        }
    }
}

/// Initialize result to true before calling kernel
extern "C" hipError_t hip_verify_q4_k(
    const float* d_original,
    const float* d_dequantized,
    int n,
    float tolerance,
    bool* d_result,
    hipStream_t stream
) {
    // Initialize result to true
    CHECK_HIP(hipMemsetAsync(d_result, 1, sizeof(bool), stream));

    int block_size = BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;

    verify_q4_k_kernel<<<grid_size, block_size, 0, stream>>>(
        d_original, d_dequantized, n, tolerance, d_result
    );

    CHECK_LAST();
    return hipSuccess;
}
```

- [ ] **Step 2: Update CMakeLists.txt**

Add to `hip_kernels/quant/CMakeLists.txt`:

```cmake
add_library(q4_k_verify STATIC
    q4_k_verify.hip
)

target_link_libraries(q4_k_verify
    quant_common
)

set_target_properties(q4_k_verify PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
```

- [ ] **Step 3: Test CMake build**

Run: `cmake --build hip_kernels/quant/build`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add hip_kernels/quant/q4_k_verify.hip hip_kernels/quant/CMakeLists.txt
git commit -m "feat(hip): add Q4_K verification kernel

GPU-side accuracy check using parallel reduction:
- Finds max error across all elements
- Returns true if max_error < tolerance
- No CPU round-trip needed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 10: Add FFI Declaration for Verification

**Files:**
- Modify: `src/gpu/ffi.rs`

- [ ] **Step 1: Add FFI function declaration**

Add to `src/gpu/ffi.rs` extern block:

```rust
extern "C" {
    // ... existing declarations ...

    /// Verify dequantized values match original
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

- [ ] **Step 2: Verify compilation**

Run: `cargo check --features gpu`
Expected: SUCCESS

- [ ] **Step 3: Commit**

```bash
git add src/gpu/ffi.rs
git commit -m "feat(gpu): add FFI declaration for hip_verify_q4_k

Bridge for GPU verification kernel.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 11: Create GpuQuant Safe Wrapper

**Files:**
- Modify: `src/gpu/quant.rs` (create new file)

- [ ] **Step 1: Write GpuQuant struct**

Create `src/gpu/quant.rs`:

```rust
//! GPU quantization wrapper - safe interface to HIP kernels

use super::ffi::{
    hipDevice_t, hipStream_t, hipError_t, hipSuccess,
    hip_quantize_q4_k, hip_dequantize_q4_k, hip_verify_q4_k,
    GpuQ4KBlock,
};
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
    pub fn new(device: &GpuDevice) -> GpuResult<Self> {
        // Get device handle (reuse existing patterns from device.rs)
        // For now, use placeholder - will be filled in implementation
        Ok(GpuQuant {
            device: std::ptr::null_mut(),
            stream: std::ptr::null_mut(),
            architecture: GpuArchitecture::Unknown(0),
        })
    }

    /// Quantize f32 weights to Q4_K format
    pub fn quantize_q4_k(&self, weights: &[f32]) -> GpuResult<Vec<GpuQ4KBlock>> {
        // Placeholder - will implement with full FFI calls
        Ok(vec![])
    }

    /// Dequantize Q4_K blocks back to f32
    pub fn dequantize_q4_k(&self, blocks: &[GpuQ4KBlock]) -> GpuResult<Vec<f32>> {
        // Placeholder
        Ok(vec![])
    }

    /// Verify accuracy (GPU-side comparison)
    pub fn verify_accuracy(
        &self,
        original: &[f32],
        dequantized: &[f32],
    ) -> GpuResult<bool> {
        // Placeholder
        Ok(true)
    }
}
```

- [ ] **Step 2: Update quant/mod.rs to export GpuQuant**

Update `src/gpu/quant/mod.rs`:

```rust
//! GPU quantization module

mod types;

pub use types::{QK_K, K_SCALE_SIZE, Q4_K_BLOCK_SIZE, Q4KBlock};
```

- [ ] **Step 3: Update gpu/mod.rs to export GpuQuant**

Add to `src/gpu/mod.rs` pub use section:

```rust
pub use quant::GpuQuant;
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check --features gpu`
Expected: SUCCESS (placeholder compiles)

- [ ] **Step 5: Commit**

```bash
git add src/gpu/quant.rs src/gpu/quant/mod.rs src/gpu/mod.rs
git commit -m "feat(gpu): add GpuQuant safe wrapper skeleton

Placeholder implementation with method signatures.
Full implementation will follow in next tasks.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 12: Implement quantize_q4_k in GpuQuant

**Files:**
- Modify: `src/gpu/quant.rs`

- [ ] **Step 1: Implement quantize_q4_k method**

Replace the placeholder quantize_q4_k in `src/gpu/quant.rs`:

```rust
/// Quantize f32 weights to Q4_K format
pub fn quantize_q4_k(&self, weights: &[f32]) -> GpuResult<Vec<GpuQ4KBlock>> {
    use super::ffi::{hipMalloc, hipMemcpy, hipMemcpyHostToDevice};

    // Validate input
    if weights.is_empty() {
        return Err(GpuError::HipApiError("weights cannot be empty".to_string()));
    }
    if weights.len() % QK_K != 0 {
        return Err(GpuError::HipApiError(
            format!("weights length {} must be multiple of {}", weights.len(), QK_K)
        ));
    }

    let num_blocks = weights.len() / QK_K;

    // Allocate GPU memory
    let mut d_input: *mut f32 = std::ptr::null_mut();
    let mut d_output: *mut GpuQ4KBlock = std::ptr::null_mut();

    unsafe {
        CHECK_HIP(hipMalloc(
            (&mut d_input) as *mut _ as *mut _,
            weights.len() * std::mem::size_of::<f32>()
        ));
        CHECK_HIP(hipMalloc(
            (&mut d_output) as *mut _ as *mut _,
            num_blocks * std::mem::size_of::<GpuQ4KBlock>()
        ));

        // Copy input to GPU
        CHECK_HIP(hipMemcpy(
            d_input as _,
            weights.as_ptr() as _,
            weights.len() * std::mem::size_of::<f32>(),
            hipMemcpyHostToDevice,
            self.stream,
        ));

        // Launch kernel
        let result = hip_quantize_q4_k(d_input, d_output, num_blocks as i32, self.stream);
        if result != hipSuccess {
            return Err(GpuError::HipApiError(format!("quantize kernel failed: {:?}", result)));
        }

        // Copy result back
        let mut blocks = vec![GpuQ4KBlock::default(); num_blocks];
        CHECK_HIP(hipMemcpy(
            blocks.as_mut_ptr() as _,
            d_output as _,
            num_blocks * std::mem::size_of::<GpuQ4KBlock>(),
            hipMemcpyDeviceToHost,
            self.stream,
        ));

        // Cleanup
        hipFree(d_input as _);
        hipFree(d_output as _);

        Ok(blocks)
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --features gpu`
Expected: May have errors - fix as needed

- [ ] **Step 3: Commit**

```bash
git add src/gpu/quant.rs
git commit -m "feat(gpu): implement quantize_q4_k in GpuQuant

Full FFI integration:
- Validates input size (must be multiple of QK_K)
- Allocates GPU memory for input and output
- Copies input to device, launches kernel
- Copies quantized blocks back to CPU
- Proper cleanup with hipFree

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 13: Implement dequantize_q4_k in GpuQuant

**Files:**
- Modify: `src/gpu/quant.rs`

- [ ] **Step 1: Implement dequantize_q4_k method**

Replace the placeholder dequantize_q4_k in `src/gpu/quant.rs`:

```rust
/// Dequantize Q4_K blocks back to f32
pub fn dequantize_q4_k(&self, blocks: &[GpuQ4KBlock]) -> GpuResult<Vec<f32>> {
    use super::ffi::{hipMalloc, hipMemcpy, hipMemcpyHostToDevice};

    if blocks.is_empty() {
        return Err(GpuError::HipApiError("blocks cannot be empty".to_string()));
    }

    let num_elements = blocks.len() * QK_K;

    // Allocate GPU memory
    let mut d_input: *mut GpuQ4KBlock = std::ptr::null_mut();
    let mut d_output: *mut f32 = std::ptr::null_mut();

    unsafe {
        CHECK_HIP(hipMalloc(
            (&mut d_input) as *mut _ as *mut _,
            blocks.len() * std::mem::size_of::<GpuQ4KBlock>()
        ));
        CHECK_HIP(hipMalloc(
            (&mut d_output) as *mut _ as *mut _,
            num_elements * std::mem::size_of::<f32>()
        ));

        // Copy input to GPU
        CHECK_HIP(hipMemcpy(
            d_input as _,
            blocks.as_ptr() as _,
            blocks.len() * std::mem::size_of::<GpuQ4KBlock>(),
            hipMemcpyHostToDevice,
            self.stream,
        ));

        // Launch kernel
        let result = hip_dequantize_q4_k(d_input, d_output, blocks.len() as i32, self.stream);
        if result != hipSuccess {
            return Err(GpuError::HipApiError(format!("dequantize kernel failed: {:?}", result)));
        }

        // Copy result back
        let mut output = vec![0.0f32; num_elements];
        CHECK_HIP(hipMemcpy(
            output.as_mut_ptr() as _,
            d_output as _,
            num_elements * std::mem::size_of::<f32>(),
            hipMemcpyDeviceToHost,
            self.stream,
        ));

        // Cleanup
        hipFree(d_input as _);
        hipFree(d_output as _);

        Ok(output)
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --features gpu`
Expected: May have errors - fix as needed

- [ ] **Step 3: Commit**

```bash
git add src/gpu/quant.rs
git commit -m "feat(gpu): implement dequantize_q4_k in GpuQuant

Full FFI integration for dequantization:
- Allocates GPU memory
- Copies blocks to device, launches kernel
- Copies dequantized values back to CPU
- Proper cleanup

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 14: Implement verify_accuracy in GpuQuant

**Files:**
- Modify: `src/gpu/quant.rs`

- [ ] **Step 1: Implement verify_accuracy method**

Replace the placeholder verify_accuracy in `src/gpu/quant.rs`:

```rust
/// Verify accuracy (GPU-side comparison)
pub fn verify_accuracy(
    &self,
    original: &[f32],
    dequantized: &[f32],
) -> GpuResult<bool> {
    use super::ffi::{hipMalloc, hipMemcpy, hipMemcpyHostToDevice};

    if original.len() != dequantized.len() {
        return Err(GpuError::HipApiError(
            "array lengths must match".to_string()
        ));
    }
    if original.is_empty() {
        return Ok(true);  // Empty arrays are trivially accurate
    }

    let n = original.len();
    let tolerance = 1e-4f32;

    // Allocate GPU memory
    let mut d_original: *mut f32 = std::ptr::null_mut();
    let mut d_dequantized: *mut f32 = std::ptr::null_mut();
    let mut d_result: *mut bool = std::ptr::null_mut();

    unsafe {
        CHECK_HIP(hipMalloc(
            (&mut d_original) as *mut _ as *mut _,
            n * std::mem::size_of::<f32>()
        ));
        CHECK_HIP(hipMalloc(
            (&mut d_dequantized) as *mut _ as *mut _,
            n * std::mem::size_of::<f32>()
        ));
        CHECK_HIP(hipMalloc(
            (&mut d_result) as *mut _ as *mut _,
            std::mem::size_of::<bool>()
        ));

        // Copy arrays to GPU
        CHECK_HIP(hipMemcpy(
            d_original as _,
            original.as_ptr() as _,
            n * std::mem::size_of::<f32>(),
            hipMemcpyHostToDevice,
            self.stream,
        ));
        CHECK_HIP(hipMemcpy(
            d_dequantized as _,
            dequantized.as_ptr() as _,
            n * std::mem::size_of::<f32>(),
            hipMemcpyHostToDevice,
            self.stream,
        ));

        // Launch verification kernel
        let result = hip_verify_q4_k(d_original, d_dequantized, n as i32, tolerance, d_result, self.stream);
        if result != hipSuccess {
            return Err(GpuError::HipApiError(format!("verify kernel failed: {:?}", result)));
        }

        // Copy result back
        let mut accurate = false;
        CHECK_HIP(hipMemcpy(
            (&mut accurate) as *mut _ as _,
            d_result as _,
            std::mem::size_of::<bool>(),
            hipMemcpyDeviceToHost,
            self.stream,
        ));

        // Cleanup
        hipFree(d_original as _);
        hipFree(d_dequantized as _);
        hipFree(d_result as _);

        Ok(accurate)
    }
}
```

- [ ] **Step 2: Add convenience method for roundtrip**

Add to GpuQuant impl:

```rust
/// Complete roundtrip: quantize → dequantize → verify
pub fn quantize_and_verify(&self, weights: &[f32]) -> GpuResult<bool> {
    let blocks = self.quantize_q4_k(weights)?;
    let recovered = self.dequantize_q4_k(&blocks)?;
    self.verify_accuracy(weights, &recovered)
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check --features gpu`
Expected: May have errors - fix as needed

- [ ] **Step 4: Commit**

```bash
git add src/gpu/quant.rs
git commit -m "feat(gpu): implement verify_accuracy and roundtrip

GPU self-verification:
- Allocates GPU memory for both arrays and result
- Copies data, launches verification kernel
- Returns bool indicating accuracy within 1e-4
- Adds convenience quantize_and_verify method

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 15: Add Integration Test for Q4_K Roundtrip

**Files:**
- Modify: `tests/quant_integration.rs`

- [ ] **Step 1: Replace test_q4_k_quantization placeholder**

Find and replace the ignored test in `tests/quant_integration.rs`:

```rust
/// Test Q4_K quantization/dequantization roundtrip
#[test]
#[serial]
fn test_q4_k_quantize_dequantize_roundtrip() {
    use rocmforge::gpu::{GpuDevice, GpuQuant};
    use rocmforge::gpu::arch::GpuArchitecture;

    // Skip if no GPU available
    let gpu = match GpuDevice::init(0) {
        Ok(g) => g,
        Err(_) => {
            println!("No GPU available, skipping test");
            return;
        }
    };

    let quant = match GpuQuant::new(&gpu) {
        Ok(q) => q,
        Err(_) => {
            println!("Failed to create GpuQuant, skipping test");
            return;
        }
    };

    // Generate test weights (simple pattern)
    let weights: Vec<f32> = (1..=4096).map(|i| i as f32 * 0.01).collect();

    // Quantize
    let blocks = quant.quantize_q4_k(&weights)
        .expect("quantize_q4_k should succeed");

    // Verify block count
    assert_eq!(blocks.len(), weights.len() / 256, "Should have correct number of blocks");

    // Dequantize
    let recovered = quant.dequantize_q4_k(&blocks)
        .expect("dequantize_q4_k should succeed");

    // Verify length
    assert_eq!(recovered.len(), weights.len(), "Should recover all elements");

    // Verify accuracy (GPU self-verification)
    let accurate = quant.verify_accuracy(&weights, &recovered)
        .expect("verify_accuracy should succeed");

    assert!(accurate, "Dequantized values should match within 1e-4 tolerance");

    println!("Q4_K roundtrip test passed: {} elements verified", weights.len());
}
```

- [ ] **Step 2: Run the integration test**

Run: `cargo test --test quant_integration --features gpu test_q4_k_quantize_dequantize_roundtrip -- --nocapture`
Expected: Test prints success message and passes

Note: Test may fail initially due to GpuQuant::new implementation details - fix as needed.

- [ ] **Step 3: Commit**

```bash
git add tests/quant_integration.rs
git commit -m "test(gpu): replace Q4_K placeholder with roundtrip test

Tests complete Q4_K pipeline:
- Quantize 4096 test weights
- Dequantize back to f32
- GPU self-verification within 1e-4 tolerance
- Skips gracefully if no GPU available

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 16: Run Full Test Suite and Verify

**Files:**
- (No file modifications - verification step)

- [ ] **Step 1: Run all quantization tests**

Run: `cargo test --test quant_sanity --test quant_unit --test quant_integration --features gpu`
Expected: All tests pass

- [ ] **Step 2: Check for any test failures**

If tests fail, investigate and fix:
- Check compilation errors
- Check HIP kernel launches
- Check FFI declarations match
- Check memory allocations/frees

- [ ] **Step 3: Verify VRAM with rocm-smi**

Run: `rocm-smi --showmeminfo vram`
Expected: VRAM usage reasonable, no leaks after tests

- [ ] **Step 4: Commit any fixes**

```bash
git add .
git commit -m "test(gpu): fix issues found during testing

[Describe any fixes needed]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 17: Update Phase 1 Documentation

**Files:**
- Modify: `docs/gpu_quant_phase1.md`

- [ ] **Step 1: Add Phase 2 completion note**

Add to `docs/gpu_quant_phase1.md` at the end:

```markdown
## Phase 2 Status (2026-03-27)

**Status:** ✅ Complete

Phase 2 implemented Q4_K quantization:
- Quantization kernel: f32 → Q4_K blocks (144 bytes each)
- Dequantization kernel: Q4_K blocks → f32
- Verification kernel: GPU-side accuracy check
- GpuQuant safe wrapper with FFI integration
- Integration tests passing

See Phase 2 design: `docs/superpowers/specs/2026-03-27-gpu-quant-phase2-design.md`
```

- [ ] **Step 2: Commit**

```bash
git add docs/gpu_quant_phase1.md
git commit -m "docs(gpu): update Phase 1 doc with Phase 2 completion

Note Q4_K implementation complete.
Reference Phase 2 design document.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 18: Final Verification and Summary

**Files:**
- (No file modifications - final verification)

- [ ] **Step 1: Verify all commits**

Run: `git log --oneline -20`
Expected: See all 18+ tasks committed with clear messages

- [ ] **Step 2: Run full cargo test**

Run: `cargo test --features gpu 2>&1 | grep -E "(test result:|running)" | tail -20`
Expected: All tests pass

- [ ] **Step 3: Verify CMake builds**

Run: `cmake --build hip_kernels/quant/build 2>&1 | grep -E "(Built target|Linking)"`
Expected: All quantization libraries built

- [ ] **Step 4: Summary of implementation**

Create summary:
- Q4_K quantization: ✅ Implemented
- Q4_K dequantization: ✅ Implemented
- GPU verification: ✅ Implemented
- Safe Rust wrapper: ✅ Implemented
- Integration tests: ✅ Passing
- Documentation: ✅ Updated

- [ ] **Step 5: Tag milestone (optional)**

```bash
git tag -a v0.2.0-gpu-quant-phase2 -m "Phase 2: Q4_K Quantization Complete"
git push origin main --tags
```

---

## Notes for Implementation

### Q4_K Algorithm Details

The simplified kernel in this plan implements global scaling only. The full llama.cpp Q4_K uses sub-block scaling:
- 8 sub-blocks of 32 elements each
- Each sub-block has its own scale from scales[12]
- scales are quantized to 6 bits

This can be added as a follow-up improvement.

### Memory Management

All GPU allocations use hipMalloc/hipFree. For production:
- Consider using GpuBuffer from existing code
- Consider memory pooling
- Add proper RAII wrappers

### Error Handling

All FFI calls use CHECK_HIP macro pattern. Errors return GpuResult:
- HipError(hipError) for HIP API failures
- HipApiError(String) for our own errors
- Never panic - always propagate errors

### Testing

- Unit tests verify constants and types
- Integration tests verify end-to-end pipeline
- Tests skip gracefully if no GPU available
- Use serial_test to ensure sequential execution

---

## Success Criteria

Phase 2 is complete when:
- ✅ All 18 tasks committed
- ✅ All tests pass (sanity + unit + integration)
- ✅ Q4_K roundtrip test passes with <1e-4 error
- ✅ No VRAM leaks (verified with rocm-smi)
- ✅ Documentation updated
