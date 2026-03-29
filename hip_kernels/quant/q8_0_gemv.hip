#include "common.hip"

// Q8_0 constants
#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34

struct Q8_0_block {
    half d;
    int8_t qs[32];
};

__device__ inline float vec_dot_q8_0(const Q8_0_block* block, const float* vec, int offset) {
    float d = __half2float(block->d);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        sum += static_cast<float>(block->qs[i]) * vec[offset + i];
    }
    return d * sum;
}

__global__ void gemv_q8_0_f32_kernel(
    const void* __restrict__ weights_q8_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK8_0;

    if (col >= ncols_dst) return;

    const Q8_0_block* w_col = reinterpret_cast<const Q8_0_block*>(
        static_cast<const uint8_t*>(weights_q8_0) + col * n_blocks * Q8_0_BLOCK_SIZE
    );

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        sum += vec_dot_q8_0(&w_col[block_idx], input, block_idx * QK8_0);
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

extern "C" hipError_t gemv_q8_0_f32_launch(
    const void* weights_q8_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK8_0 != 0) return hipErrorInvalidValue;

    gemv_q8_0_f32_kernel<<<ncols_dst, 256, 0, stream>>>(
        weights_q8_0, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
