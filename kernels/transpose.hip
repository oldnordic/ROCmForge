/**
 * transpose.hip - GPU matrix transpose kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave64)
 * Block size: 16x16 threads (256 threads per block)
 * Tile size: 64x64 for RDNA3 optimization
 *
 * This kernel performs efficient matrix transposition on GPU using shared memory
 * tiling to avoid global memory bank conflicts. The +1 padding in the shared
 * memory tile array is critical for AMD GPU performance.
 *
 * Based on AMD ROCm Examples transpose implementation.
 *
 * Input/output layout: Row-major (C-style)
 * - Input:  [rows, cols] flattened to rows * cols
 * - Output: [cols, rows] flattened to cols * rows
 *
 * Grid dimension calculation:
 * - Grid: (ceil(cols / TILE_DIM), ceil(rows / TILE_DIM))
 * - Block: (TILE_DIM, TILE_DIM) threads
 */

#include <hip/hip_runtime.h>

// Tile dimension for shared memory blocking
// 64 is optimal for RDNA3 architecture (gfx1100)
#define TILE_DIM 64

/**
 * Optimized matrix transpose kernel with bank conflict avoidance
 *
 * Uses shared memory tiling with TILE_DIM x (TILE_DIM + 1) to avoid
 * bank conflicts on AMD GPUs. The +1 padding ensures that consecutive
 * threads access different banks when writing to shared memory.
 *
 * Algorithm:
 * 1. Each thread block loads a TILE_DIM x TILE_DIM tile from global memory
 * 2. Tile is stored in shared memory with +1 padding to avoid bank conflicts
 * 3. Threads synchronize via __syncthreads()
 * 4. Tile is written back to transposed position in global memory
 *
 * @param odata Output matrix [cols, rows] (transposed dimensions)
 * @param idata Input matrix [rows, cols] (original dimensions)
 * @param rows Number of rows in input matrix
 * @param cols Number of columns in input matrix
 */
extern "C" __global__ void transposeLdsNoBankConflicts(
    float* __restrict__ odata,
    const float* __restrict__ idata,
    const size_t rows,
    const size_t cols
) {
    // Shared memory tile with +1 padding to avoid bank conflicts
    // On AMD GPUs, shared memory is banked; consecutive 32-bit words
    // map to consecutive banks. The +1 padding ensures that when
    // threads in a wavefront access tile[threadIdx.x][threadIdx.y],
    // they access different banks.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Input matrix coordinates (row-major)
    const size_t idx_in = blockIdx.x * TILE_DIM + threadIdx.x;  // Column index in input
    const size_t idy_in = blockIdx.y * TILE_DIM + threadIdx.y;  // Row index in input

    // Output matrix coordinates (transposed: row becomes column, column becomes row)
    const size_t idx_out = blockIdx.y * TILE_DIM + threadIdx.x;  // Row index in output
    const size_t idy_out = blockIdx.x * TILE_DIM + threadIdx.y;  // Column index in output

    // Load tile from input matrix with boundary checks
    // Each thread loads potentially multiple elements if blockDim.y < TILE_DIM
    for (size_t y = 0; y < TILE_DIM; y += blockDim.y) {
        const size_t row = idy_in + y;
        const size_t col = idx_in;

        if (row < rows && col < cols) {
            // Linear index in row-major layout: row * cols + col
            const size_t index_in = row * cols + col;
            tile[threadIdx.y + y][threadIdx.x] = idata[index_in];
        } else {
            // Pad with zeros for out-of-bounds threads
            tile[threadIdx.y + y][threadIdx.x] = 0.0f;
        }
    }

    // Synchronize to ensure all threads have loaded their tile elements
    __syncthreads();

    // Write tile to output matrix with boundary checks
    for (size_t y = 0; y < TILE_DIM; y += blockDim.y) {
        const size_t row = idx_out;
        const size_t col = idy_out + y;

        if (row < cols && col < rows) {
            // Linear index in transposed layout: row * rows + col
            // (output dimensions are swapped: [cols, rows])
            const size_t index_out = row * rows + col;
            odata[index_out] = tile[threadIdx.x][threadIdx.y + y];
        }
    }
}

/**
 * Simple naive transpose kernel for reference/debugging
 *
 * This kernel performs transpose without shared memory optimization.
 * Useful for:
 * - Correctness verification
 * - Small matrices where shared memory overhead isn't justified
 * - Debugging transpose logic
 *
 * @param odata Output matrix [cols, rows] (transposed dimensions)
 * @param idata Input matrix [rows, cols] (original dimensions)
 * @param rows Number of rows in input matrix
 * @param cols Number of columns in input matrix
 */
extern "C" __global__ void transposeNaive(
    float* __restrict__ odata,
    const float* __restrict__ idata,
    const size_t rows,
    const size_t cols
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Input: [row, col] -> Output: [col, row]
        const size_t index_in = row * cols + col;
        const size_t index_out = col * rows + row;
        odata[index_out] = idata[index_in];
    }
}
