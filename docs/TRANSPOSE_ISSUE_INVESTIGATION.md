# GPU Transpose Issue Investigation

**Date:** 2026-01-20
**Phase:** v1.5 Planning
**Issue:** Inference fails during embedding transpose with `hipMemcpyDtoH failed with code 1`

## Problem Summary

Model loading succeeds, but inference fails when transposing the embedding tensor:

```
ERROR: Error processing request 0: Inference failed: Generic error: Failed to transpose embedding:
Memory copy failed: hipMemcpyDtoH failed with code 1
(base_ptr=0x7fe70fe00000, offset=1976131072, final_ptr=0x7fe785a95e00, size=519 MB)
```

### What Works

- Model loading: ✅ 291 tensors dequantized
- Memory arena: ✅ 2403 MB allocated, 0% fragmentation
- GPU upload: ✅ All tensors uploaded to GPU
- Engine start: ✅ Inference loop starts

### What Fails

- Embedding transpose: ❌ GPU→CPU copy fails for 519 MB tensor

## Root Cause

**Location:** `src/model/execution_plan/matmul.rs:114-145`

```rust
pub fn transpose_2d_tensor(backend: &HipBackend, tensor: &DeviceTensor) -> HipResult<DeviceTensor> {
    let mut host = vec![0.0f32; tensor.len()];
    backend.copy_from_device_safe(&tensor.buffer, &mut host)?;  // <-- FAILS HERE
    // CPU transpose...
    // Re-upload to GPU...
}
```

**Why it fails:**
1. Phase 22 introduced memory arena (single large allocation)
2. Tensors are now sub-buffers with offsets
3. The embedding is at offset 1,971,310,972 bytes (~1.84 GB into arena)
4. `hipMemcpyAsync` with sub-buffer pointer fails with error code 1 (`hipErrorInvalidValue`)

**Why embedding specifically:**
- Embedding is 519 MB - largest tensor needing transpose
- Other tensors (attention weights) are smaller and haven't hit this path yet
- This is the first large GPU→CPU→GPU round-trip after memory pool implementation

## Solutions Considered

### Option 1: GPU Transpose Kernel (Recommended)

**Reference:** AMD ROCm Examples `transpose_kernels.hpp`

```cpp
#define TILE_DIM 64

template<typename T>
__global__ void transposeLdsNoBankConflicts(T* odata, const T* idata, size_t size)
{
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];  // +1 avoids bank conflicts

    size_t idx_in   = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t idy_in   = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t index_in = idx_in + idy_in * size;

    size_t idx_out   = blockIdx.y * TILE_DIM + threadIdx.x;
    size_t idy_out   = blockIdx.x * TILE_DIM + threadIdx.y;
    size_t index_out = idx_out + idy_out * size;

    for(size_t y = 0; y < TILE_DIM; y += blockDim.y)
        tile[threadIdx.y + y][threadIdx.x] = idata[index_in + y * size];

    __syncthreads();

    for(size_t y = 0; y < TILE_DIM; y += blockDim.y)
        odata[index_out + y * size] = tile[threadIdx.x][threadIdx.y + y];
}
```

**Pros:**
- True GPU transpose, no CPU round-trip
- Fast with shared memory optimization
- Based on AMD reference code

**Cons:**
- Requires new HSACO kernel
- More code to maintain

### Option 2: Strided View (Fastest)

Instead of copying, create a view with swapped strides:

```rust
pub struct TensorView {
    buffer: HipBuffer,
    shape: TensorShape,
    strides: [usize; 2],  // [col_stride, row_stride]
}
```

For a `[H, V]` tensor viewed as `[V, H]`:
- Original strides: `[V, 1]`
- View strides: `[1, V]`

**Pros:**
- Zero memory copy
- Instant operation
- Works with any layout

**Cons:**
- Requires changes to all tensor consumers
- More complex abstraction

### Option 3: rocBLAS geam (Requires rocBLAS)

```rust
rocblas_geam(handle, rocblas_op_transpose, ...);
```

**Pros:**
- Vendor-optimized
- Already handles edge cases

**Cons:**
- Requires adding rocBLAS bindings
- Dependency on another library

### Option 4: Pre-transpose During Loading

Load the embedding already transposed from the GGUF file.

**Pros:**
- One-time cost at load time
- No runtime transpose needed

**Cons:**
- Doesn't work if model actually needs both layouts
- More complex loader logic

## Recommended Solution: Hybrid Approach

1. **Short-term:** Use GPU transpose kernel (Option 1)
   - Based on AMD reference code
   - Fix the immediate crash

2. **Long-term:** Implement strided views (Option 2)
   - Eliminates most transpose operations
   - More efficient overall

## Research Sources

- [AMD ROCm Examples - transpose_kernels.hpp](https://github.com/ROCm/rocm-examples) - Official AMD transpose kernels
- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html) - AMD's BLAS library
- [Deep Dive: Matrix Optimization on AMD GPUs](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html) - Recent RDNA3 optimization techniques

## Files to Modify

| File | Change |
|------|--------|
| `src/kernels/transpose/` | NEW: GPU transpose kernels |
| `src/backend/hip_backend/backend.rs` | Add transpose dispatch |
| `src/model/execution_plan/matmul.rs` | Replace CPU transpose with GPU kernel |
| `build.rs` | Add TRANSPOSE_HSACO env var |

## Testing Plan

1. Test with qwen2.5-0.5b.gguf (current failing case)
2. Verify transpose correctness with small test matrices
3. Benchmark CPU vs GPU transpose
4. Test with other models

## Resolution (Phase 27)

**Status:** FIXED 2026-01-20

GPU transpose kernel implemented based on AMD ROCm Examples.
Model inference now completes successfully.

**Files Changed:**
- src/kernels/transpose/mod.rs (NEW, 472 LOC including tests)
- kernels/transpose.hip (NEW, 131 LOC)
- build.rs (fixed: kernel path and name)
- src/model/execution_plan/types.rs:273 (uses GPU transpose)
- src/model/simple_transformer.rs (fixed: mut linear for reassignment)

**Before:**
```
ERROR: hipMemcpyDtoH failed with code 1 (offset=1.97GB into arena)
```

**After:**
```
INFO: Transposing embedding tensor from [896, 151936] to [151936, 896] on GPU
DEBUG: Transpose kernel launched: grid=[2374, 14, 1], block=[64, 64, 1]
INFO: Embedding weights loaded successfully
```

**Tests Added:**
- test_transpose_small_square_matrix: 8x8 matrix with exact value verification
- test_transpose_small_rectangular_matrix: 4x16 matrix with corner verification
- test_transpose_large_matrix: 512x1024 matrix for correctness
- test_transpose_embedding_sized_matrix: 128x1024 embedding simulation with vector verification

**Deviations:**
- build.rs: Fixed kernel path from "src/kernels/transpose/hip transpose.hip" to "kernels/transpose.hip"
- build.rs: Fixed kernel name from "transpose_kernel" to "transposeLdsNoBankConflicts"
- simple_transformer.rs: Added mut to let linear (compilation fix)
