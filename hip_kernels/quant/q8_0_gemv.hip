#include "common.hip"

// Q8_0 constants (must match Rust types.rs)
#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34

// ── Type Definitions ───────────────────────────────────────────────────────────────

/// Q8_0 block layout (matches Rust Q8_0Block)
struct Q8_0_block {
    half d;          // scale (2 bytes)
    int8_t qs[32];   // quantized values (32 bytes)
};

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Load Q8_0 block and compute dot product with f32 vector
/// Device-side function for inline computation
__device__ inline float vec_dot_q8_0(const Q8_0_block* block, const float* vec, int offset) {
    // Load scale
    float d = __half2float(block->d);

    // Accumulate dot product
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK8_0; ++i) {
        sum += static_cast<float>(block->qs[i]) * vec[offset + i];
    }

    return d * sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

/// Template kernel for Q8_0 × f32 GEMV with ncols_dst as compile-time parameter
/// Follows llama.cpp pattern for optimal performance
template<int ncols_dst>
__launch_bounds__(512, 2)  // HIP-specific: max 512 threads, min 2 warps
__global__ void gemv_q8_0_f32_kernel(
    const void* __restrict__ weights_q8_0,  // [n_rows/32][ncols_dst][34] Q8_0 quantized weights
    const float* __restrict__ input,        // [n_rows] f32 input vector
    float* __restrict__ output,             // [ncols_dst] f32 output vector
    int n_rows                              // Number of rows (input dimension)
) {
    // Each block processes one column (ncols_dst = 1, 2, 4, or 8)
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst) return;

    // Calculate starting pointers for this column
    const Q8_0_block* w_col = reinterpret_cast<const Q8_0_block*>(
        static_cast<const uint8_t*>(weights_q8_0) + col * (n_rows / QK8_0) * Q8_0_BLOCK_SIZE
    );

    // Number of blocks per column
    const int n_blocks = (n_rows + QK8_0 - 1) / QK8_0;

    // Accumulator for this thread's partial sum
    float sum = 0.0f;

    // Stride across blocks with threads
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK8_0;
        sum += vec_dot_q8_0(&w_col[block_idx], input, offset);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    // Thread 0 writes the result
    if (tid == 0) {
        output[col] = sum;
    }
}

// ── Generic Kernel (non-template) ───────────────────────────────────────────────────

/// Generic kernel for arbitrary ncols_dst
/// Less optimized but handles any size
__global__ void gemv_q8_0_f32_generic_kernel(
    const void* __restrict__ weights_q8_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst) return;

    // Calculate starting pointers for this column
    const Q8_0_block* w_col = reinterpret_cast<const Q8_0_block*>(
        static_cast<const uint8_t*>(weights_q8_0) + col * (n_rows / QK8_0) * Q8_0_BLOCK_SIZE
    );

    // Number of blocks per column
    const int n_blocks = (n_rows + QK8_0 - 1) / QK8_0;

    // Accumulator for this thread's partial sum
    float sum = 0.0f;

    // Stride across blocks with threads
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK8_0;
        sum += vec_dot_q8_0(&w_col[block_idx], input, offset);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    // Thread 0 writes the result
    if (tid == 0) {
        output[col] = sum;
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t gemv_q8_0_f32_launch(
    const void* weights_q8_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    // Validate dimensions
    if (n_rows <= 0 || ncols_dst <= 0) {
        return hipErrorInvalidValue;
    }

    // Ensure n_rows is aligned to QK8_0
    if (n_rows % QK8_0 != 0) {
        return hipErrorInvalidValue;
    }

    // Use 256 threads per block (good balance for RDNA/CDNA)
    constexpr int threads_per_block = 256;

    // Dispatch based on ncols_dst (llama.cpp pattern)
    switch (ncols_dst) {
        case 1:
            gemv_q8_0_f32_kernel<1><<<1, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        case 2:
            gemv_q8_0_f32_kernel<2><<<2, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        case 3:
            gemv_q8_0_f32_kernel<3><<<3, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        case 4:
            gemv_q8_0_f32_kernel<4><<<4, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        case 5:
            gemv_q8_0_f32_kernel<5><<<5, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        case 6:
            gemv_q8_0_f32_kernel<6><<<6, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        case 7:
            gemv_q8_0_f32_kernel<7><<<7, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        case 8:
            gemv_q8_0_f32_kernel<8><<<8, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows);
            break;
        default:
            // Use generic kernel for larger sizes
            gemv_q8_0_f32_generic_kernel<<<ncols_dst, threads_per_block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, ncols_dst);
            break;
    }

    return hipGetLastError();
}
