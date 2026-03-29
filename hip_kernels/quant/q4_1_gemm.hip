#include "common.hip"

// Q4_1 constants (must match Rust types.rs)
#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

// ── Type Definitions ───────────────────────────────────────────────────────────────

/// Q4_1 block layout (matches Rust Q4_1Block)
struct Q4_1_block {
    half d;          // scale (2 bytes)
    half m;          // min value (2 bytes)
    int8_t qs[16];   // quantized values packed as 2x4-bit per byte (16 bytes)
};

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Load Q4_1 block and compute dot product with f32 vector
__device__ inline float vec_dot_q4_1(const Q4_1_block* block, const float* vec, int offset) {
    // Load scale and min
    float d = __half2float(block->d);
    float m = __half2float(block->m);

    // Sequential layout: qs[i] low nibble → element i, high → element i+16
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = static_cast<uint8_t>(block->qs[i]);
        const int32_t q0 = q & 0x0F;
        const int32_t q1 = (q >> 4) & 0x0F;
        sum += (static_cast<float>(q0) * d + m) * vec[offset + i];
        sum += (static_cast<float>(q1) * d + m) * vec[offset + i + 16];
    }

    return sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

template<int ncols_dst>
__launch_bounds__(512, 2)
__global__ void gemm_q4_1_f32_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * (n_rows / QK4_1) * Q4_1_BLOCK_SIZE
    );

    const float* input_batch = input + batch_idx * n_rows;
    const int n_blocks = (n_rows + QK4_1 - 1) / QK4_1;

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK4_1;
        sum += vec_dot_q4_1(&w_col[block_idx], input_batch, offset);
    }

    __shared__ float partial_sums[BLOCK_SIZE];
    partial_sums[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = partial_sums[0];
    }
}

// ── Generic Kernel (non-template) ───────────────────────────────────────────────────

__global__ void gemm_q4_1_f32_generic_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * (n_rows / QK4_1) * Q4_1_BLOCK_SIZE
    );

    const float* input_batch = input + batch_idx * n_rows;
    const int n_blocks = (n_rows + QK4_1 - 1) / QK4_1;

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK4_1;
        sum += vec_dot_q4_1(&w_col[block_idx], input_batch, offset);
    }

    __shared__ float partial_sums[BLOCK_SIZE];
    partial_sums[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = partial_sums[0];
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t gemm_q4_1_f32_launch(
    const void* weights_q4_1,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int batch_size,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }

    if (n_rows % QK4_1 != 0) {
        return hipErrorInvalidValue;
    }

    constexpr int threads_per_block = 256;
    dim3 grid(ncols_dst, batch_size);
    dim3 block(threads_per_block);

    switch (ncols_dst) {
        case 1:
            gemm_q4_1_f32_kernel<1><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        case 2:
            gemm_q4_1_f32_kernel<2><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        case 3:
            gemm_q4_1_f32_kernel<3><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        case 4:
            gemm_q4_1_f32_kernel<4><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        case 5:
            gemm_q4_1_f32_kernel<5><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        case 6:
            gemm_q4_1_f32_kernel<6><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        case 7:
            gemm_q4_1_f32_kernel<7><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        case 8:
            gemm_q4_1_f32_kernel<8><<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, batch_size);
            break;
        default:
            gemm_q4_1_f32_generic_kernel<<<grid, block, 0, stream>>>(
                weights_q4_1, input, output, n_rows, ncols_dst, batch_size);
            break;
    }

    return hipGetLastError();
}
