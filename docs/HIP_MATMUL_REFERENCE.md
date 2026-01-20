# HIP Matrix Multiplication Reference

**Purpose:** Reference documentation for ROCm/HIP matrix multiplication patterns and validation.
**Last Updated:** 2026-01-20
**Sources:**
- [ROCm HIP Matrix Multiplication Tutorial](https://rocm.docs.amd.com/projects/HIP/en/develop/tutorial/programming-patterns/matrix_multiplication.html)
- [HIP-Examples MatrixMultiplication.cpp](https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/HIP-Examples-Applications/MatrixMultiplication/MatrixMultiplication.cpp)

## Standard MatMul Formula

For matrices A and B:
- A ∈ R^(m×k)  (m rows, k columns)
- B ∈ R^(k×n)  (k rows, n columns)
- **Requirement:** columns of A must equal rows of B (k == k)

Result C = A · B:
- C ∈ R^(m×n)  (m rows, n columns)
- Each element: C[i][j] = sum(A[i][r] * B[r][j] for r in 0..k)

## Dimension Validation

```rust
// Standard notation
let m = a_rows;        // rows of A
let k = a_cols;        // cols of A (must equal rows of B)
let n = b_cols;        // cols of B

// Validation checks
assert!(a_cols == b_rows, "Inner dimensions must match");
assert!(output_elements == m * n, "Output size must be m×n");
```

## Our Implementation

**File:** `src/ggml/hip_backend/mod.rs:260-282`

```rust
// Extract dimensions
let m = a_desc.shape[0] as i32;  // rows of A
let k = a_desc.shape[1] as i32;  // cols of A
let b_k = b_desc.shape[0] as i32; // rows of B
let n = b_desc.shape[1] as i32;  // cols of B

// Inner dimension check
if k != b_k {
    return Err(GgmlError::InvalidShape(format!(
        "MatMul dimension mismatch: k={} b_k={}", k, b_k
    )));
}

// Output size check
if output_desc.element_count() != (m as usize * n as usize) {
    return Err(GgmlError::InvalidShape(format!(
        "Output shape does not match matmul result: expected shape=[{}, {}] ({} elements), got shape={:?} ({} elements)",
        m, n, m as usize * n as usize, output_desc.shape, output_desc.element_count()
    )));
}
```

## Common Issues

### Issue 1: Zero-sized Tensors (KV Cache at current_len=0)

When KV cache has `current_len=0`, tensor shapes become `[0, heads, dim]`, creating zero-sized views.

**Fix:** Use underlying buffers for writes, not the zero-sized read views.

See: `src/model/kv_cache.rs:get_layer_buffers()`

### Issue 2: Output Shape Mismatch

Error: "Output shape does not match matmul result"

**Root causes to investigate:**
1. Graph construction created wrong output tensor shape
2. Tensor shape mutation (set_shape) caused drift
3. Layout/shape mismatch between GGUF and execution graph

**Debug:** Enable logging at `mod.rs:265-269` to see actual vs expected dimensions.

## GPU Memory Best Practices

From [ROCm Tutorial](https://rocm.docs.amd.com/projects/HIP/en/develop/tutorial/programming-patterns/matrix_multiplication.html):

1. **Boundary checking:** Always check thread boundaries to prevent invalid memory access
2. **Memory coalescing:** Arrange data access for contiguous memory locations
3. **Block sizes:** Use powers of two (16 or 32) for better occupancy
4. **Synchronization:** Use `hipDeviceSynchronize()` after kernel launches

## Related Files

- `src/ggml/hip_backend/mod.rs` - GGML backend with matmul validation
- `src/ggml/hip_backend/ops/matmul.rs` - Actual matmul implementation
- `src/model/kv_cache.rs` - KV cache with buffer management
- `src/model/execution_plan/execution_plan_src.rs` - Graph execution with KV binding
