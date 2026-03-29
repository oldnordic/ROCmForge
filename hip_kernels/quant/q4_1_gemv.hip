#include "common.hip"

// Q4_1 constants
#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

struct Q4_1_block {
    half d;
    half m;
    int8_t qs[16];
};

__device__ inline float vec_dot_q4_1(const Q4_1_block* block, const float* vec, int offset) {
    float d = __half2float(block->d);
    float m = __half2float(block->m);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = static_cast<uint8_t>(block->qs[i]);
        sum += (static_cast<float>(q & 0x0F) * d + m) * vec[offset + i];
        sum += (static_cast<float>(q >> 4) * d + m) * vec[offset + i + 16];
    }
    return sum;
}

__global__ void gemv_q4_1_f32_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK4_1;

    if (col >= ncols_dst) return;

    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * n_blocks * Q4_1_BLOCK_SIZE
    );

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        sum += vec_dot_q4_1(&w_col[block_idx], input, block_idx * QK4_1);
    }

    __shared__ float partial_sums[256];
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
}

extern "C" hipError_t gemv_q4_1_f32_launch(
    const void* weights_q4_1,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK4_1 != 0) return hipErrorInvalidValue;

    gemv_q4_1_f32_kernel<<<ncols_dst, 256, 0, stream>>>(
        weights_q4_1, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
