#include "common.hip"
#include <hip/hip_fp16.h>

// Q5_0 constants
#define QK5_0 32
#define Q5_0_BLOCK_SIZE 22

__device__ inline float vec_dot_q5_0(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    const float d = __half2float(d_half);

    const uint8_t* qh = block_bytes + 2;
    const uint8_t* qs = block_bytes + 6;

    float sum = 0.0f;

    for (int i = 0; i < QK5_0; ++i) {
        const uint8_t high_bit = ((qh[i / 8] >> (i % 8)) & 1) << 4;
        const uint8_t low_bits = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
        const int q = (int)(high_bit | low_bits) - 16;
        sum += d * (float)q * vec[offset + i];
    }

    return sum;
}

template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemm_q5_0_f32_kernel(
    const void* __restrict__ weights_q5_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_0) + col * (n_rows / QK5_0) * Q5_0_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK5_0;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q5_0_BLOCK_SIZE;
        sum += vec_dot_q5_0(block_ptr, input_batch, block_idx * QK5_0);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

__global__ void gemm_q5_0_f32_generic_kernel(
    const void* __restrict__ weights_q5_0,
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

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_0) + col * (n_rows / QK5_0) * Q5_0_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK5_0;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q5_0_BLOCK_SIZE;
        sum += vec_dot_q5_0(block_ptr, input_batch, block_idx * QK5_0);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

extern "C" hipError_t gemm_q5_0_f32_launch(
    const void* weights_q5_0,
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
    if (n_rows % QK5_0 != 0) {
        return hipErrorInvalidValue;
    }

    dim3 grid(ncols_dst, batch_size);
    dim3 block(32);

    switch (ncols_dst) {
        case 1:
            gemm_q5_0_f32_kernel<1><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        case 2:
            gemm_q5_0_f32_kernel<2><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        case 3:
            gemm_q5_0_f32_kernel<3><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        case 4:
            gemm_q5_0_f32_kernel<4><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        case 5:
            gemm_q5_0_f32_kernel<5><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        case 6:
            gemm_q5_0_f32_kernel<6><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        case 7:
            gemm_q5_0_f32_kernel<7><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        case 8:
            gemm_q5_0_f32_kernel<8><<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, batch_size);
            break;
        default:
            gemm_q5_0_f32_generic_kernel<<<grid, block, 0, stream>>>(
                weights_q5_0, input, output, n_rows, ncols_dst, batch_size);
            break;
    }

    return hipGetLastError();
}
