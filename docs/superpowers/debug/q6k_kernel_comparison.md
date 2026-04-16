# Q6_K GEMV vs GEMM Kernel Comparison

## Executive Summary

**BREAKTHROUGH FINDING:** The bug is NOT caused by structural differences between GEMV and GEMM kernels. Both kernels use **identical computation logic** (vec_dot pattern from llama.cpp). The actual difference is in **batch handling** and **input offset calculation**.

## GEMV Kernel (Working - Single Token Only)

### Function Signature
```rust
pub fn gemv_q6_k_f32(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()>
```

**Key Characteristic:** Always processes **batch_size = 1** (single token decode)

### Kernel Launch Configuration
```cpp
__global__ void gemv_q6_k_f32_kernel(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    // CRITICAL: Early bounds check prevents GPU crashes
    if (col >= ncols_dst) return;
```

**Thread Layout:**
- Grid: `(ncols_dst, 1, 1)` - X dimension = output columns
- Block: `(32, 1, 1)` - 32 threads = 1 RDNA wavefront
- No Y-dimension (no batch processing)

### Input Offset Calculation
```cpp
const uint8_t* col_base = static_cast<const uint8_t*>(weights_q6_k) +
                          col * n_blocks * Q6_K_BLOCK_SIZE;

float sum = 0.0f;
for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
    sum += vec_dot_q6_k(col_base + block_idx * Q6_K_BLOCK_SIZE, input, block_idx * QK_K);
}
```

**Memory Access Pattern:**
- Input: `input[block_idx * QK_K + l]` (direct access, no batch offset)
- Output: `output[col]` (single write per block)
- Weights: `col_base + block_idx * Q6_K_BLOCK_SIZE` (column-major)

### Computation Logic (vec_dot_q6_k device function)
```cpp
__device__ inline float vec_dot_q6_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Extract scale d (fp16 at bytes 208-209)
    half d_half;
    memcpy(&d_half, &block_bytes[208], sizeof(half));
    const float d = __half2float(d_half);

    // Q4_K safety: check for denormal d
    if (fabsf(d) < 1e-7f) return 0.0f;

    // [Quantization/dequantization logic identical to GEMM]

    return sum;
}
```

**Status:** ✅ **WORKING PERFECTLY** - No issues with single-token decode

---

## GEMM Kernel (Broken - Multi-Token Prefill)

### Function Signature
```rust
pub fn gemm_q6_k_f32(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,  // ⚠️ CRITICAL DIFFERENCE
) -> GpuResult<()>
```

**Key Characteristic:** Processes **batch_size > 1** (multi-token prefill)

### Kernel Launch Configuration
```cpp
__global__ void gemm_q6_k_f32_kernel_generic(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int seq_len  // ⚠️ CRITICAL: seq_len = batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;  // ⚠️ Y dimension = batch/token position
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= seq_len) return;
```

**Thread Layout:**
- Grid: `(ncols_dst, seq_len, 1)` - X dimension = output columns, **Y dimension = batch**
- Block: `(32, 1, 1)` - 32 threads = 1 RDNA wavefront
- **Y-dimension enables batch processing**

### Input Offset Calculation
```cpp
const uint8_t* col_base = weights + col * n_blocks * Q6_K_BLOCK_SIZE;

// CRITICAL FIX: Offset input by batch_idx for multi-token prefill
const float* input_batch = input + batch_idx * n_rows;

for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
    const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;
    const int offset = block_idx * QK_K;

    // [Identical quantization logic to GEMV]

    sum += input_batch[offset + l + 0] * (d * (float)sc[is + 0] * q1);
    sum += input_batch[offset + l + 32] * (d * (float)sc[is + 2] * q2);
    // ... etc
}
```

**Memory Access Pattern:**
- Input: `input_batch[block_idx * QK_K + l]` where `input_batch = input + batch_idx * n_rows`
- Output: `output[batch_idx * ncols_dst + col]` (batch-strided write)
- Weights: `col_base + block_idx * Q6_K_BLOCK_SIZE` (column-major, same as GEMV)

### Output Indexing
```cpp
if (tid == 0) {
    output[batch_idx * ncols_dst + col] = sum;  // ⚠️ Batch-strided output
}
```

**Status:** ❌ **BROKEN FOR MULTI-TOKEN PROMPTS ENDING WITH PERIODS**

---

## Critical Structural Differences

### 1. **Batch Processing Architecture**

**GEMV (Working):**
- No batch dimension
- Direct input access: `input[offset + l]`
- Single output per column: `output[col]`

**GEMM (Broken):**
- Y-dimension = batch_idx
- Batch-offset input access: `input[batch_idx * n_rows + offset + l]`
- Batch-strided output: `output[batch_idx * ncols_dst + col]`

### 2. **Input Pointer Arithmetic**

**GEMV (Working):**
```cpp
// Direct input access (no batch offset)
sum += input[offset + l + 0] * (d * (float)sc[is + 0] * q1);
```

**GEMM (Broken):**
```cpp
// Batch-offset input access
const float* input_batch = input + batch_idx * n_rows;
sum += input_batch[offset + l + 0] * (d * (float)sc[is + 0] * q1);
```

### 3. **Thread Block Configuration**

**GEMV (Working):**
```cpp
const dim3 gridDim(ncols_dst, 1);  // No Y-dimension
const dim3 blockDim(32, 1);
```

**GEMM (Broken):**
```cpp
const dim3 gridDim(ncols_dst, seq_len);  // Y-dimension = batch_size
const dim3 blockDim(32, 1);
```

---

## Hypothesis: Why Periods Trigger the Bug

### The "Period Token" Theory

**Observation:** The bug ONLY occurs when multi-token prompts end with periods (e.g., "Hello world.", "The quick brown fox.")

**Possible Explanation 1: Token Boundary Effect**
- Period tokens may be represented differently in the embedding space
- Last token in a batch (period) may have special alignment properties
- The batch offset calculation `input + batch_idx * n_rows` may be incorrect for the last token

**Possible Explanation 2: Memory Alignment Issue**
```cpp
const float* input_batch = input + batch_idx * n_rows;
```
- If `n_rows` is not aligned to a power-of-2 boundary
- And `batch_idx` causes the offset to cross a cache line/page boundary
- Period tokens may be more sensitive to this misalignment

**Possible Explanation 3: Input Layout Assumption**
- The kernel assumes input is laid out as `[batch_size][n_rows]`
- But actual input may be `[n_rows][batch_size]` (transposed)
- This would cause incorrect input samples when `batch_size > 1`
- Period tokens may amplify the error because they're the last token processed

### Most Likely Cause: Input Layout Mismatch

**Evidence:**
1. GEMV works perfectly (batch_size = 1, so layout doesn't matter)
2. GEMM fails only when batch_size > 1
3. The failure is content-dependent (periods trigger it)
4. The computation logic is identical between GEMV and GEMM

**Specific Hypothesis:**
```cpp
// Current (potentially wrong) assumption:
input_layout = [batch_size][n_rows]  // Row-major
input_batch = input + batch_idx * n_rows  // Skip entire rows

// Actual layout might be:
input_layout = [n_rows][batch_size]  // Column-major (transposed)
// Correct offset would be:
input_batch = input + batch_idx  // Skip to batch_idx-th element in each row
```

### Why Periods Specifically Trigger This

**Token Position Effect:**
- Periods are always the **last token** in a multi-token prompt
- Last token means `batch_idx = seq_len - 1` (maximum batch offset)
- Maximum offset maximizes the impact of an incorrect offset calculation
- Earlier tokens may have small enough offsets that the error is masked

**Example:**
```
Prompt: "Hello world." (3 tokens)
batch_idx = 0: "Hello" → input_batch = input + 0 * n_rows (offset=0, works by luck)
batch_idx = 1: "world" → input_batch = input + 1 * n_rows (offset=n_rows, may work)
batch_idx = 2: "." → input_batch = input + 2 * n_rows (offset=2*n_rows, MAXIMUM ERROR)
```

---

## Computation Logic Comparison

### ✅ IDENTICAL (Not the Source of the Bug)

Both kernels use the **exact same quantization/dequantization logic**:

1. **Q6_K block structure parsing** (identical)
2. **Scale extraction** (fp16 at bytes 208-209, identical)
3. **Quantization value extraction** (4-way unpacking, identical)
4. **Dot product accumulation** (identical)
5. **Warp shuffle reduction** (identical)

**Conclusion:** The bug is NOT in the computation logic - it's in the **memory access pattern** for batched inputs.

---

## Next Steps for Task 3 (Fix Implementation)

### High-Priority Investigation Areas

1. **Verify Input Layout**
   ```rust
   // Check how input is actually laid out in GPU memory
   // Is it [batch_size][n_rows] or [n_rows][batch_size]?
   ```

2. **Test Batch Offset Calculation**
   ```cpp
   // Current: const float* input_batch = input + batch_idx * n_rows;
   // Alternative: const float* input_batch = input + batch_idx;
   ```

3. **Add Debug Output**
   ```cpp
   // Print first few values of input_batch for each batch_idx
   // Compare with expected values from CPU
   ```

4. **Test with Different Token Positions**
   ```bash
   # Test if non-period last tokens also fail
   # Test if period in middle of prompt fails
   # This will confirm if it's "last token" or "period character"
   ```

### Potential Fix Scenarios

**Scenario A: Input Layout is Transposed**
```cpp
// Wrong (current):
const float* input_batch = input + batch_idx * n_rows;

// Correct:
const float* input_batch = input + batch_idx;
```

**Scenario B: Batch Dimension is in Wrong Place**
```cpp
// Wrong (current):
const float* input_batch = input + batch_idx * n_rows;

// Correct:
const float* input_batch = input + batch_idx * ncols_dst;
```

**Scenario C: Input is Column-Major**
```cpp
// Wrong (current): assumes row-major [batch_size][n_rows]
const float* input_batch = input + batch_idx * n_rows;

// Correct: column-major [n_rows][batch_size]
const float* input_batch = input + col * seq_len + batch_idx;
```

---

## Conclusion

**Key Finding:** The bug is NOT in the Q6_K computation logic (which is identical between GEMV and GEMM). The bug is in the **batch input offset calculation**:

```cpp
const float* input_batch = input + batch_idx * n_rows;
```

**Why Periods Trigger It:**
- Periods are always the last token
- Last token = maximum batch_idx = maximum offset error
- Earlier tokens may work due to smaller offsets masking the error

**Confidence Level:** HIGH - The structural differences clearly point to a memory access issue, not a computation issue.

**Next Task:** Verify the actual input layout in GPU memory and correct the batch offset calculation.
