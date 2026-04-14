#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__global__ void gemm_q6_k_f32_kernel(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int seq_len
) {
    const int col = blockIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || row >= n_rows) return;

    const uint8_t* weights = static_cast<const uint8_t*>(weights_q6_k);
    const int n_blocks = n_rows / QK_K;

    float sum = 0.0f;
    const uint8_t* col_base = weights + col * n_blocks * Q6_K_BLOCK_SIZE;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;

        // Unpack Q6_K block
        const uint8_t* ql = &block[0];
        const uint8_t* qh = &block[128];
        const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);

        half d_half;
        memcpy(&d_half, &block[208], sizeof(half));
        const float d = __half2float(d_half);

        // Accumulate across block (same logic as GEMV)
        for (int n = 0; n < 2; n++) {
            for (int l = tid; l < 32; l += 32) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)(((ql[l + 0] >> 4) & 0xF) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)(((ql[l + 32] >> 4) & 0xF) | (((qh[l] >> 6) & 3) << 4)) - 32;

                const int base = n * 128;
                const float d1 = d * (float)scales[is + 0];
                const float d2 = d * (float)scales[is + 2];
                const float d3 = d * (float)scales[is + 4];
                const float d4 = d * (float)scales[is + 6];

                sum += input[block_idx * QK_K + base + l + 0] * (d1 * (float)q1);
                sum += input[block_idx * QK_K + base + l + 32] * (d2 * (float)q2);
                sum += input[block_idx * QK_K + base + l + 64] * (d3 * (float)q3);
                sum += input[block_idx * QK_K + base + l + 96] * (d4 * (float)q4);
            }
        }
    }

    // Reduction within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col * n_rows + row] = sum;
    }
}

extern "C" hipError_t gemm_q6_k_f32_launch(
    const void* weights_q6_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int seq_len,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || seq_len <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    dim3 blockDim(32, 4); // 32 threads x 4 rows per block
    dim3 gridDim(ncols_dst, (n_rows + blockDim.y - 1) / blockDim.y);

    gemm_q6_k_f32_kernel<<<gridDim, blockDim, 0, stream>>>(
        weights_q6_k, input, output, n_rows, ncols_dst, seq_len
    );

    return hipGetLastError();
}
