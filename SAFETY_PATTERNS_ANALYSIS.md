# GPU Safety Patterns Analysis

## Overview

This document captures all safety patterns learned from studying working ROCm kernels (Q4_K, Q5_K, Q8_0) to prevent GPU crashes and resets.

## Critical Safety Issues That Caused GPU Crashes

### Problem: Q6_K kernels crashed GPU with memory access faults
- **Error**: "Page not present or supervisor privilege" 
- **Trigger**: Prompts ≥6 tokens (ncols_dst=151936)
- **Root Cause**: Missing safety patterns that working kernels use

## Safety Pattern Categories

### 1. CPU-Side Parameter Validation (Rust FFI Layer)

**Location**: `src/gpu/kernels/quant.rs`

All working kernels validate parameters BEFORE kernel launch:

```rust
// Check for zero values
if n_rows == 0 || ncols_dst == 0 {
    return Err(GpuError::HipApiError {
        code: -1,
        description: "gemv_q6_k_f32: n_rows and ncols_dst cannot be zero".to_string(),
    });
}

// Check alignment requirements
if n_rows % 256 != 0 {
    return Err(GpuError::HipApiError {
        code: -1,
        description: format!(
            "gemv_q6_k_f32: n_rows must be multiple of 256, got {}",
            n_rows
        ),
    });
}

// Validate pointers
if weights_q6_k.is_null() {
    return Err(GpuError::HipApiError {
        code: -1,
        description: "gemv_q6_k_f32: weights_q6_k pointer is null".to_string(),
    });
}

if input.is_null() {
    return Err(GpuError::HipApiError {
        code: -1,
        description: "gemv_q6_k_f32: input pointer is null".to_string(),
    });
}

if output.is_null() {
    return Err(GpuError::HipApiError {
        code: -1,
        description: "gemv_q6_k_f32: output pointer is null".to_string(),
    });
}
```

**Why This Matters**: 
- Prevents invalid kernel launches
- Catches errors before they reach GPU
- Provides clear error messages
- Avoids GPU resets from bad parameters

### 2. GPU-Side Bounds Checking

**Location**: `hip_kernels/common.hip` and all working kernels

```cpp
/// Kernel-safe bounds check macro
/// Use this to validate indices before memory access
#define CHECK_BOUNDS(idx, max) \
    if ((idx) >= (max)) { \
        return; \
    }
```

**Usage in kernels**:
```cpp
const int col = blockIdx.x;
if (col >= ncols_dst) return;  // Early exit on bounds violation
```

**Why This Matters**:
- Prevents out-of-bounds memory access
- Avoids "Page not present" errors
- Threads exit gracefully instead of crashing
- Critical for large grid sizes (151936+ blocks)

### 3. Shared Memory Safety

**Location**: `hip_kernels/quant/q8_0_gemv.hip`

**Critical Pattern**: Dynamic shared memory must match actual block size

```cpp
// Define shared memory limit
constexpr int Q8_0_LM_HEAD_SHARED_MEM_LIMIT = 32 * 1024;

// Calculate dynamic shared memory based on ACTUAL block size
const int block_size = 256;  // Must match kernel launch
const size_t dynamic_shared_mem = block_size * sizeof(float);

// Kernel launch with EXACT shared memory size
gemv_q8_0_f32_kernel<<<ncols_dst, block_size, dynamic_shared_mem, stream>>>()
```

**Inside kernel**:
```cpp
extern __shared__ float partial_sums[];  // Dynamic allocation
partial_sums[tid] = sum;
__syncthreads();  // REQUIRED before sharing

// Reduction with bounds checking
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {  // Bounds check
        partial_sums[tid] += partial_sums[tid + stride];
    }
    __syncthreads();  // REQUIRED after each iteration
}
```

**Block Size Selection Pattern**:
```cpp
static inline int select_lm_head_block_size(int n_rows) {
    const int n_blocks = n_rows / QK8_0;
    
    if (n_blocks <= 64) {
        return 64;
    }
    if (n_blocks <= 128) {
        return 128;
    }
    return 256;
}
```

**Why This Matters**:
- Mismatched shared memory causes silent corruption
- Fixed-size arrays with variable block sizes = crashes
- __syncthreads() REQUIRED before/after shared memory access
- Large shared memory can exceed device limits

### 4. HIP API Error Handling

**Location**: `hip_kernels/common.hip`

```cpp
/// CHECK_HIP macro for error checking
/// All HIP API calls MUST be wrapped with this macro
/// Returns error code on failure, never continues past errors
#define CHECK_HIP(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            return error; \
        } \
    } while (0)

/// Host-side null check macro for launchers
#define CHECK_NULL(ptr) \
    if ((ptr) == nullptr) { \
        return hipErrorInvalidValue; \
    }

/// CHECK_LAST macro for kernel launch error checking
/// Must be called immediately after kernel launch
#define CHECK_LAST() \
    do { \
        hipError_t error = hipGetLastError(); \
        if (error != hipSuccess) { \
            return error; \
        } \
    } while (0)
```

**Usage**:
```cpp
extern "C" hipError_t gemv_q8_0_f32_lm_head_launch(...) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK8_0 != 0) return hipErrorInvalidValue;
    
    CHECK_NULL(weights_q8_0);
    CHECK_NULL(input);
    CHECK_NULL(output);
    
    // ... kernel launch ...
    
    return hipGetLastError();  // CHECK_LAST pattern
}
```

**Why This Matters**:
- HIP API failures can corrupt GPU state
- Must check EVERY API call
- Never continue after error
- Prevents cascading failures

### 5. Thread Configuration Safety

**Location**: All working kernels

**Pattern 1: Simple Thread Configuration (Q4_K, Q5_K)**
```cpp
// Fixed 32 threads per block (warp size)
gemv_q5_k_f32_kernel<<<ncols_dst, 32, 0, stream>>>()
```

**Pattern 2: Variable Thread Configuration (Q8_0)**
```cpp
// Select block size based on work size
const int block_size = select_lm_head_block_size(n_rows);
const size_t dynamic_shared_mem = block_size * sizeof(float);

gemv_q8_0_f32_kernel<<<ncols_dst, block_size, dynamic_shared_mem, stream>>>()
```

**Pattern 3: Multi-Wavefront Configuration (Q8_0 LM Head)**
```cpp
// Calculate threads per block
const int n_subwaves = select_lm_head_subwaves(ncols_dst);
const int threads_per_block = n_subwaves * 32;  // 32 = warp size

// Calculate grid dimensions
const int blocks_x = (ncols_dst + (n_subwaves * Q8_0_LM_HEAD_COLS) - 1) / 
                     (n_subwaves * Q8_0_LM_HEAD_COLS);

gemv_q8_0_f32_lm_head_multi_row_kernel<8><<<blocks_x, threads_per_block, input_shared_mem, stream>>>()
```

**Why This Matters**:
- Thread configuration affects shared memory requirements
- Block size must match shared memory allocation
- Grid dimensions must handle all work items
- Large grids (151936+) work fine with proper bounds checking

### 6. Memory Access Safety

**Location**: All working kernels

**Pattern 1: Pointer Arithmetic with Bounds**
```cpp
const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_k) + 
                          col * n_blocks * Q5_K_BLOCK_SIZE;

for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
    sum += vec_dot_q5_k(col_base + block_idx * Q5_K_BLOCK_SIZE, input, block_idx * QK_K);
}
```

**Pattern 2: Struct Access with Proper Casting**
```cpp
const Q8_0_block* w_col = reinterpret_cast<const Q8_0_block*>(
    static_cast<const uint8_t*>(weights_q8_0) + col * n_blocks * Q8_0_BLOCK_SIZE
);
```

**Pattern 3: Safe float4 Usage**
```cpp
const float4* input_vec = reinterpret_cast<const float4*>(&s_input[row_offset]);
for (int i = 0; i < 8; ++i) {
    const float4 in = input_vec[i];
    dot += static_cast<float>(block->qs[q_offset + 0]) * in.x;
    dot += static_cast<float>(block->qs[q_offset + 1]) * in.y;
    dot += static_cast<float>(block->qs[q_offset + 2]) * in.z;
    dot += static_cast<float>(block->qs[q_offset + 3]) * in.w;
}
```

**Why This Matters**:
- Incorrect pointer arithmetic causes memory faults
- Misaligned accesses corrupt data
- Type casting must preserve alignment
- Vector loads require proper alignment

### 7. Warp Shuffle Reduction Safety

**Location**: `hip_kernels/quant/q4_k_gemv.hip`, `q5_k_gemv.hip`

```cpp
// Warp shuffle reduction (simpler than shared memory)
for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
}

if (tid == 0) {
    output[col] = sum;
}
```

**vs Shared Memory Reduction** (Q8_0):
```cpp
// Shared memory reduction (more complex)
extern __shared__ float partial_sums[];
partial_sums[tid] = sum;
__syncthreads();

for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial_sums[tid] += partial_sums[tid + stride];
    }
    __syncthreads();
}

if (tid == 0) {
    output[col] = partial_sums[0];
}
```

**Why This Matters**:
- Warp shuffle is simpler and faster for small reductions
- Shared memory works for any block size
- Warp shuffle limited to 32 threads per warp
- Must use correct reduction for block size

## Safety Pattern Checklist

Before launching any kernel:

### CPU-Side (Rust)
- [ ] Validate n_rows > 0 and ncols_dst > 0
- [ ] Check alignment (n_rows % 256 == 0 for Q6_K)
- [ ] Check all pointers are non-null
- [ ] Log parameters for debugging
- [ ] Check kernel launch result
- [ ] Return errors early, never continue on failure

### GPU-Side (HIP)
- [ ] Add bounds check: `if (col >= ncols_dst) return;`
- [ ] Use dynamic shared memory: `extern __shared__ float[]`
- [ ] Match shared memory size to block size
- [ ] Add __syncthreads() before/after shared memory access
- [ ] Add bounds checks in reduction loops: `if (tid < stride)`
- [ ] Use CHECK_HIP for all HIP API calls
- [ ] Use CHECK_NULL for pointer parameters
- [ ] Use CHECK_LAST after kernel launch
- [ ] Validate pointer arithmetic with bounds
- [ ] Use proper casting for struct access
- [ ] Ensure vector loads are aligned

## Why Q6_K Crashed But Q4_K/Q8_0 Worked

### Q4_K (Works with 151936 grid size)
- Uses simple 32-thread blocks
- Warp shuffle reduction (no shared memory)
- Proper bounds checking
- Simple pointer arithmetic

### Q8_0 (Works with 151936 grid size)
- Uses dynamic shared memory correctly
- Block size selection logic
- Shared memory limits
- Proper __syncthreads() usage

### Q6_K (Crashed with 151936 grid size)
- Attempted multiple threading patterns
- Missing proper bounds checking
- Shared memory size mismatch
- No CHECK_HIP/CHECK_NULL usage
- Attempted complex patterns without safety foundation

## Key Lessons

1. **Start Simple**: Q4_K pattern (32 threads + warp shuffle) is safest starting point
2. **Add Safety First**: Bounds checking before optimization
3. **Test Incrementally**: 1 token (4096 cols) → 2 tokens (151936 cols)
4. **Match Patterns**: Working kernels > theoretical optimization
5. **Never Skip Validation**: CPU checks + GPU checks + error handling
6. **Shared Memory is Complex**: Only use when necessary, match sizes exactly
7. **Large Grids Work**: Q4_K and Q8_0 handle 151936 grid size fine with proper safety

## Next Steps for Q6_K

1. Start with Q4_K pattern (32 threads, warp shuffle)
2. Add all safety patterns from this analysis
3. Test with 1 token (4096 cols) first
4. Then test with 2 tokens (151936 cols)
5. Only optimize after safety is verified

## References

- **common.hip**: Safety macros (CHECK_HIP, CHECK_NULL, CHECK_BOUNDS, CHECK_LAST)
- **q4_k_gemv.hip**: Simple working kernel with warp shuffle
- **q5_k_gemv.hip**: Intermediate complexity working kernel
- **q8_0_gemv.hip**: Complex kernel with shared memory
- **quant.rs**: CPU-side validation patterns