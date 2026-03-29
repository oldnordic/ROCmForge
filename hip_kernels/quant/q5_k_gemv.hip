#include "common.hip"
#include <hip/hip_fp16.h>

// Q5_K constants
#define QK_K 256
#define Q5_K_BLOCK_SIZE 176
#define K_SCALE_SIZE 12

__device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__device__ inline float vec_dot_q5_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half, dmin_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float dmin = __half2float(dmin_half);

    const uint8_t* scales = block_bytes + 4;
    const uint8_t* qh = block_bytes + 4 + K_SCALE_SIZE;
    const uint8_t* ql = block_bytes + 4 + K_SCALE_SIZE + QK_K / 8;

    float sum = 0.0f;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;

    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4(is + 0, scales, &sc1, &m1);
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        
        const float d1 = d * sc1, min1 = dmin * m1;
        const float d2 = d * sc2, min2 = dmin * m2;

        const uint8_t q4 = ql[tid] & 0x0F;
        const uint8_t high_bit = (qh[tid] & u1) ? 16 : 0;
        sum += (static_cast<float>(q4 + high_bit) / d1 + min1) * vec[offset + j + tid];

        const uint8_t q4_2 = (ql[tid] >> 4) & 0x0F;
        const uint8_t high_bit_2 = (qh[tid] & u2) ? 16 : 0;
        sum += (static_cast<float>(q4_2 + high_bit_2) / d2 + min2) * vec[offset + j + 32 + tid];

        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
    return sum;
}

__global__ void gemv_q5_k_f32_kernel(
    const void* __restrict__ weights_q5_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_k) + col * n_blocks * Q5_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q5_k(col_base + block_idx * Q5_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q5_k_f32_launch(
    const void* weights_q5_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    gemv_q5_k_f32_kernel<<<ncols_dst, 32, 0, stream>>>(
        weights_q5_k, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
