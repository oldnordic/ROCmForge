#include "common.hip"
#include <hip/hip_fp16.h>

// Q5_1 constants
#define QK5_1 32
#define Q5_1_BLOCK_SIZE 24

__device__ inline float vec_dot_q5_1(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half, m_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&m_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float m = __half2float(m_half);

    const uint8_t* qh = block_bytes + 4;
    const uint8_t* qs = block_bytes + 8;

    float sum = 0.0f;

    // Q5_1 packs 32 values into 16 nibble bytes using the GGUF/llama.cpp
    // layout (same as Q4_1/Q5_0): byte i holds element i (low nibble) and
    // element i+16 (high nibble). Each value also has a high bit in qh.
    for (int i = 0; i < QK5_1 / 2; ++i) {
        // element i: low nibble of qs[i], high bit i of qh
        const uint8_t high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
        const uint8_t low_bits_0 = qs[i] & 0x0F;
        const int q0 = (int)(high_bit_0 | low_bits_0);

        // element i+16: high nibble of qs[i], high bit (i+16) of qh
        const uint8_t high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
        const uint8_t low_bits_1 = (qs[i] >> 4) & 0x0F;
        const int q1 = (int)(high_bit_1 | low_bits_1);

        sum += (d * (float)q0 + m) * vec[offset + i] +
               (d * (float)q1 + m) * vec[offset + i + 16];
    }

    return sum;
}

template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemm_q5_1_f32_kernel(
    const void* __restrict__ weights_q5_1,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_1) + col * (n_rows / QK5_1) * Q5_1_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK5_1;
    float sum = 0.0f;

    // Distribute Q5_1 blocks across the 32 lanes; vec_dot_q5_1 computes a
    // full block per thread, so each lane sums its assigned blocks and the
    // warp shuffle combines partials into the correct full-column result.
    for (int block_idx = tid; block_idx < n_blocks; block_idx += 32) {
        const void* block_ptr = col_base + block_idx * Q5_1_BLOCK_SIZE;
        sum += vec_dot_q5_1(block_ptr, input_batch, block_idx * QK5_1);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

__global__ void gemm_q5_1_f32_generic_kernel(
    const void* __restrict__ weights_q5_1,
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

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_1) + col * (n_rows / QK5_1) * Q5_1_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK5_1;
    float sum = 0.0f;

    // Distribute Q5_1 blocks across the 32 lanes; vec_dot_q5_1 computes a
    // full block per thread, so each lane sums its assigned blocks and the
    // warp shuffle combines partials into the correct full-column result.
    for (int block_idx = tid; block_idx < n_blocks; block_idx += 32) {
        const void* block_ptr = col_base + block_idx * Q5_1_BLOCK_SIZE;
        sum += vec_dot_q5_1(block_ptr, input_batch, block_idx * QK5_1);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

extern "C" hipError_t gemm_q5_1_f32_launch(
    const void* weights_q5_1,
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
    if (n_rows % QK5_1 != 0) {
        return hipErrorInvalidValue;
    }

    dim3 grid(ncols_dst, batch_size);
    dim3 block(32);

    switch (ncols_dst) {
        case 1:
            gemm_q5_1_f32_kernel<1><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        case 2:
            gemm_q5_1_f32_kernel<2><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        case 3:
            gemm_q5_1_f32_kernel<3><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        case 4:
            gemm_q5_1_f32_kernel<4><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        case 5:
            gemm_q5_1_f32_kernel<5><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        case 6:
            gemm_q5_1_f32_kernel<6><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        case 7:
            gemm_q5_1_f32_kernel<7><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        case 8:
            gemm_q5_1_f32_kernel<8><<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, batch_size);
            break;
        default:
            gemm_q5_1_f32_generic_kernel<<<grid, block, 0, stream>>>(
                weights_q5_1, input, output, n_rows, ncols_dst, batch_size);
            break;
    }

    return hipGetLastError();
}
