#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__device__ inline float vec_dot_q6_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Q6_K block layout: ql[0..128], qh[128..192], scales[192..208], d[208..210]
    const uint8_t* ql = &block_bytes[0];
    const uint8_t* qh = &block_bytes[128];
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block_bytes[192]);

    half d_half;
    memcpy(&d_half, &block_bytes[208], sizeof(half));
    const float d = __half2float(d_half);

    float sum = 0.0f;

    // for (int n = 0; n < QK_K; n += 128) {
    //     for (int l = 0; l < 32; ++l) {
    //         const int is = l / 16;
    //         const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    //         const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    //         const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    //         const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    //         sum += vec[offset + l + 0]  * (d * sc[is + 0] * q1);
    //         sum += vec[offset + l + 32] * (d * sc[is + 2] * q2);
    //         sum += vec[offset + l + 64] * (d * sc[is + 4] * q3);
    //         sum += vec[offset + l + 96] * (d * sc[is + 6] * q4);
    //     }
    //     offset += 128; ql += 64; qh += 32; sc += 8;
    // }

    // First 128 elements (n=0)
    for (int l = tid; l < 32; l += 32) {
        const int is = l / 16;
        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)(((ql[l + 0] >> 4) & 0xF) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)(((ql[l + 32] >> 4) & 0xF) | (((qh[l] >> 6) & 3) << 4)) - 32;

        const float d1 = d * (float)scales[is + 0];
        const float d2 = d * (float)scales[is + 2];
        const float d3 = d * (float)scales[is + 4];
        const float d4 = d * (float)scales[is + 6];

        sum += vec[offset + l + 0] * (d1 * (float)q1);
        sum += vec[offset + l + 32] * (d2 * (float)q2);
        sum += vec[offset + l + 64] * (d3 * (float)q3);
        sum += vec[offset + l + 96] * (d4 * (float)q4);
    }

    // Advance pointers (llama.cpp: y += 128; ql += 64; qh += 32; sc += 8;)
    ql += 64;
    qh += 32;
    scales += 8;

    // Second 128 elements (n=128)
    for (int l = tid; l < 32; l += 32) {
        const int is = l / 16;
        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)(((ql[l + 0] >> 4) & 0xF) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)(((ql[l + 32] >> 4) & 0xF) | (((qh[l] >> 6) & 3) << 4)) - 32;

        const float d1 = d * (float)scales[is + 0];
        const float d2 = d * (float)scales[is + 2];
        const float d3 = d * (float)scales[is + 4];
        const float d4 = d * (float)scales[is + 6];

        sum += vec[offset + 128 + l + 0] * (d1 * (float)q1);
        sum += vec[offset + 128 + l + 32] * (d2 * (float)q2);
        sum += vec[offset + 128 + l + 64] * (d3 * (float)q3);
        sum += vec[offset + 128 + l + 96] * (d4 * (float)q4);
    }

    return sum;
}

__global__ void gemv_q6_k_f32_kernel(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q6_k) + col * n_blocks * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q6_k(col_base + block_idx * Q6_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q6_k_f32_launch(
    const void* weights_q6_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    gemv_q6_k_f32_kernel<<<ncols_dst, 32, 0, stream>>>(
        weights_q6_k, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
