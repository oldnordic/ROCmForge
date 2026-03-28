#include "common.hip"

// Q4_1 constants (must match Rust types.rs)
#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

// ── Type Definitions ───────────────────────────────────────────────────────────────

/// Q4_1 block layout (matches Rust Q4_1Block)
struct Q4_1_block {
    half d;          // scale (2 bytes)
    half m;          // min offset (2 bytes)
    uint8_t qs[16];  // quantized values packed as 2x4-bit per byte (16 bytes)
};

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Load Q4_1 block and compute dot product with f32 vector
/// Q4_1 formula: dequant = q * d + m
/// Device-side function for inline computation
__device__ inline float vec_dot_q4_1(const Q4_1_block* block, const float* vec, int offset) {
    // Load scale and min offset
    float d = __half2float(block->d);
    float m = __half2float(block->m);

    // Sequential unpacking: qs[i] low nibble → element i, high nibble → element i + 16
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t packed = block->qs[i];
        const float q0 = static_cast<float>(packed & 0x0F);          // Low 4 bits → element i
        const float q1 = static_cast<float>(packed >> 4);            // High 4 bits → element i + 16
        sum += (q0 * d + m) * vec[offset + i];
        sum += (q1 * d + m) * vec[offset + i + 16];
    }

    return sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

/// Template kernel for Q4_1 × f32 GEMV with ncols_dst as compile-time parameter
/// Follows llama.cpp pattern for optimal performance
template<int ncols_dst>
__launch_bounds__(512, 2)  // HIP-specific: max 512 threads, min 2 warps
__global__ void gemv_q4_1_f32_kernel(
    const void* __restrict__ weights_q4_1,  // [n_rows/32][ncols_dst][20] Q4_1 quantized weights
    const float* __restrict__ input,        // [n_rows] f32 input vector
    float* __restrict__ output,             // [ncols_dst] f32 output vector
    int n_rows                              // Number of rows (input dimension)
) {
    // Each block processes one column (ncols_dst = 1, 2, 3, ..., 8)
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst) return;

    // Calculate starting pointers for this column
    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * (n_rows / QK4_1) * Q4_1_BLOCK_SIZE
    );

    // Number of blocks per column
    const int n_blocks = (n_rows + QK4_1 - 1) / QK4_1;

    // Accumulator for this thread's partial sum
    float sum = 0.0f;

    // Stride across blocks with threads
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK4_1;
        sum += vec_dot_q4_1(&w_col[block_idx], input, offset);
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
__global__ void gemv_q4_1_f32_generic_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst) return;

    // Calculate starting pointers for this column
    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * (n_rows / QK4_1) * Q4_1_BLOCK_SIZE
    );

    // Number of blocks per column
    const int n_blocks = (n_rows + QK4_1 - 1) / QK4_1;

    // Accumulator for this thread's partial sum
    float sum = 0.0f;

    // Stride across blocks with threads
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK4_1;
        sum += vec_dot_q4_1(&w_col[block_idx], input, offset);
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

extern "C" hipError_t gemv_q4_1_f32_launch(
    const void* weights_q4_1,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    // Validate inputs
    if (weights_q4_1 == nullptr || input == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }

    // Validate dimensions
    if (n_rows <= 0 || ncols_dst <= 0) {
        return hipErrorInvalidValue;
    }

    // Ensure n_rows is aligned to QK4_1
    if (n_rows % QK4_1 != 0) {
        return hipErrorInvalidValue;
    }

    // Use 256 threads per block (good balance for RDNA/CDNA)
    constexpr int threads_per_block = 256;

    // Dispatch based on ncols_dst (llama.cpp pattern)
    switch (ncols_dst) {
        case 1:
            gemv_q4_1_f32_kernel<1><<<1, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        case 2:
            gemv_q4_1_f32_kernel<2><<<2, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        case 3:
            gemv_q4_1_f32_kernel<3><<<3, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        case 4:
            gemv_q4_1_f32_kernel<4><<<4, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        case 5:
            gemv_q4_1_f32_kernel<5><<<5, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        case 6:
            gemv_q4_1_f32_kernel<6><<<6, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        case 7:
            gemv_q4_1_f32_kernel<7><<<7, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        case 8:
            gemv_q4_1_f32_kernel<8><<<8, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows);
            break;
        default:
            // Use generic kernel for larger sizes
            gemv_q4_1_f32_generic_kernel<<<ncols_dst, threads_per_block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, ncols_dst);
            break;
    }

    return hipGetLastError();
}
