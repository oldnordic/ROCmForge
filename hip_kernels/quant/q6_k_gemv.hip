#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__device__ inline float vec_dot_q6_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block = static_cast<const uint8_t*>(block_ptr);

    // Q6_K block layout: ql[0..128], qh[128..192], scales[192..208], d[208..210]
    half d_half;
    memcpy(&d_half, &block[208], sizeof(half));
    const float d = __half2float(d_half);
    if (fabsf(d) < 1e-7f) return 0.0f;

    const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);
    const int l = tid;
    const int is = l / 16;
    float sum = 0.0f;

    // Group 0: values 0..127
    {
        const uint8_t* ql = &block[0];
        const uint8_t* qh = &block[128];
        const int8_t* sc = &scales[0];

        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

        sum += vec[offset + l + 0]   * (d * (float)sc[is + 0] * (float)q1);
        sum += vec[offset + l + 32]  * (d * (float)sc[is + 2] * (float)q2);
        sum += vec[offset + l + 64]  * (d * (float)sc[is + 4] * (float)q3);
        sum += vec[offset + l + 96]  * (d * (float)sc[is + 6] * (float)q4);
    }

    // Group 1: values 128..255
    {
        const uint8_t* ql = &block[64];
        const uint8_t* qh = &block[160];
        const int8_t* sc = &scales[8];

        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

        sum += vec[offset + l + 128] * (d * (float)sc[is + 0] * (float)q1);
        sum += vec[offset + l + 160] * (d * (float)sc[is + 2] * (float)q2);
        sum += vec[offset + l + 192] * (d * (float)sc[is + 4] * (float)q3);
        sum += vec[offset + l + 224] * (d * (float)sc[is + 6] * (float)q4);
    }

    return sum;
}

__launch_bounds__(32, 1)
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

    // CRITICAL: Early bounds check prevents GPU crashes
    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q6_k) +
                              col * n_blocks * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q6_k(col_base + block_idx * Q6_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    // Warp shuffle reduction (proven safe from Q4_K)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
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
    // Validate parameters
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    // Use safety macros
    CHECK_NULL(weights_q6_k);
    CHECK_NULL(input);
    CHECK_NULL(output);

    // Q4_K pattern: 32 threads per block (warp size)
    const int block_size = 32;

    gemv_q6_k_f32_kernel<<<ncols_dst, block_size, 0, stream>>>(
        weights_q6_k, input, output, n_rows, ncols_dst
    );

    // Check for launch errors
    CHECK_HIP(hipGetLastError());

    return hipSuccess;
}
