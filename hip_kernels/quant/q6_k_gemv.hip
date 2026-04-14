#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__device__ inline float vec_dot_q6_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Extract scale d (fp16 at bytes 208-209)
    half d_half;
    memcpy(&d_half, &block_bytes[208], sizeof(half));
    const float d = __half2float(d_half);

    // Pointer to scales (int8_t at bytes 192-207)
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block_bytes[192]);

    float sum = 0.0f;

    // Linear processing: each thread handles 8 elements (32 threads * 8 = 256)
    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;  // Linear index 0-255

        // Compute group (0 or 1) and position within group
        const int group = i / 128;           // Which group of 128
        const int pos_in_group = i % 128;    // 0-127
        const int l_base = pos_in_group % 32; // 0-31
        const int quadrant = pos_in_group / 32; // 0, 1, 2, or 3

        // Compute scale index: group * 8 + is * 2 + quadrant
        const int is = l_base / 16;  // 0 or 1
        const int scale_idx = group * 8 + is * 2 + quadrant;
        const float scale = d * (float)scales[scale_idx];

        // Compute ql offset
        const int ql_offset = group * 64 + (l_base % 16) + (l_base / 16) * 32;
        const uint8_t ql_byte = block_bytes[ql_offset];

        // Compute qh offset
        const int qh_offset = 128 + group * 32 + l_base;
        const uint8_t qh_byte = block_bytes[qh_offset];

        // Determine which pair and which value in pair
        const int pair_idx = pos_in_group % 32;  // 0-31
        const int is_low_half = (pair_idx < 16) ? 0 : 1;
        const int shift = is_low_half ? 0 : 4;

        // Unpack Q6_K value (6 bits: 4 from ql, 2 from qh)
        const uint8_t ql_4bits = (ql_byte >> shift) & 0x0F;
        const uint8_t qh_2bits = (qh_byte >> (is_low_half ? 0 : 2)) & 0x03;
        const int8_t q = (int8_t)(ql_4bits | (qh_2bits << 4)) - 32;

        // Direct memory access - no computed offsets in array indexing
        sum += vec[offset + i] * (scale * (float)q);
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
