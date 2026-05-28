#include <hip/hip_runtime.h>
#include <cmath>

#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

// ── Sparse CSR GEMV Kernel ───────────────────────────────────────────────────────
// y = A * x where A is in CSR format
// Each warp processes one row of A
// Use shared memory for x cache when beneficial

extern "C" __global__ void sparse_csr_gemv_f32_kernel(
    const float* __restrict__ values,
    const unsigned int* __restrict__ col_indices,
    const unsigned int* __restrict__ row_offsets,
    const int nnz,
    const int rows,
    const int cols,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    const int lane = threadIdx.x; // 0..31
    const int row_start = row_offsets[row];
    const int row_end = row_offsets[row + 1];

    float sum = 0.0f;

    // Each thread in the warp handles a subset of nonzeros in this row
    for (int idx = row_start + lane; idx < row_end; idx += 32) {
        const int col = col_indices[idx];
        // Bounds check: col must be within [0, cols)
        if (col >= 0 && col < cols) {
            sum += values[idx] * x[col];
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// ── Host-Side Launcher ───────────────────────────────────────────────────────────

extern "C" hipError_t gpu_sparse_csr_gemv_f32(
    const float* values,
    const unsigned int* col_indices,
    const unsigned int* row_offsets,
    int nnz,
    int rows,
    int cols,
    const float* x,
    float* y,
    hipStream_t stream
) {
    if (!values || !col_indices || !row_offsets || !x || !y) {
        return hipErrorInvalidValue;
    }
    if (nnz <= 0 || rows <= 0 || cols <= 0) {
        return hipErrorInvalidValue;
    }

    // One warp (32 threads) per row, 8 rows per block = 256 threads
    const int rows_per_block = 8;
    const int grid_x = (rows + rows_per_block - 1) / rows_per_block;

    hipLaunchKernelGGL(
        sparse_csr_gemv_f32_kernel,
        dim3(grid_x, 1, 1),
        dim3(32, rows_per_block, 1),
        0,
        stream,
        values,
        col_indices,
        row_offsets,
        nnz,
        rows,
        cols,
        x,
        y
    );

    return hipGetLastError();
}
