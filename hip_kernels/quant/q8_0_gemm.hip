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
__device__ inline float vec_dot_q8_0(const Q8_0_block* block, const float* vec, int offset) {
    // Load scale
    float d = __half2float(block->d);

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        sum += static_cast<float>(block->qs[i]) * vec[offset + i];
    }

    return d * sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

template<int ncols_dst>
__launch_bounds__(512, 2)
__global__ void gemm_q8_0_f32_kernel(
    const void* __restrict__ weights_q8_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const Q8_0_block* w_col = reinterpret_cast<const Q8_0_block*>(
        static_cast<const uint8_t*>(weights_q8_0) + col * (n_rows / QK8_0) * Q8_0_BLOCK_SIZE
    );

    const float* input_batch = input + batch_idx * n_rows;
    const int n_blocks = (n_rows + QK8_0 - 1) / QK8_0;

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK8_0;
        sum += vec_dot_q8_0(&w_col[block_idx], input_batch, offset);
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

__global__ void gemm_q8_0_f32_generic_kernel(
    const void* __restrict__ weights_q8_0,
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

    const Q8_0_block* w_col = reinterpret_cast<const Q8_0_block*>(
        static_cast<const uint8_t*>(weights_q8_0) + col * (n_rows / QK8_0) * Q8_0_BLOCK_SIZE
    );

    const float* input_batch = input + batch_idx * n_rows;
    const int n_blocks = (n_rows + QK8_0 - 1) / QK8_0;

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK8_0;
        sum += vec_dot_q8_0(&w_col[block_idx], input_batch, offset);
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

extern "C" hipError_t gemm_q8_0_f32_launch(
    const void* weights_q8_0,
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

    if (n_rows % QK8_0 != 0) {
        return hipErrorInvalidValue;
    }

    constexpr int threads_per_block = 256;
    dim3 grid(ncols_dst, batch_size);
    dim3 block(threads_per_block);

    switch (ncols_dst) {
        case 1:
            gemm_q8_0_f32_kernel<1><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        case 2:
            gemm_q8_0_f32_kernel<2><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        case 3:
            gemm_q8_0_f32_kernel<3><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        case 4:
            gemm_q8_0_f32_kernel<4><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        case 5:
            gemm_q8_0_f32_kernel<5><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        case 6:
            gemm_q8_0_f32_kernel<6><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        case 7:
            gemm_q8_0_f32_kernel<7><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        case 8:
            gemm_q8_0_f32_kernel<8><<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, batch_size);
            break;
        default:
            gemm_q8_0_f32_generic_kernel<<<grid, block, 0, stream>>>(
                weights_q8_0, input, output, n_rows, ncols_dst, batch_size);
            break;
    }

    return hipGetLastError();
}
