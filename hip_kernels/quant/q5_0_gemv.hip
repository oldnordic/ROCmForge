#include "common.hip"
#include <hip/hip_fp16.h>

// Q5_0 constants
#define QK5_0 32
#define Q5_0_BLOCK_SIZE 22

__device__ inline float vec_dot_q5_0(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Extract scale d (fp16 at bytes 0-1)
    half d_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    const float d = __half2float(d_half);

    // qh: 4 bytes of high bits (1 bit per value)
    const uint8_t* qh = block_bytes + 2;
    // qs: 16 bytes of low 4 bits (2 values per byte)
    const uint8_t* qs = block_bytes + 6;

    float sum = 0.0f;

    // Each thread processes all 32 elements of the block
    // (unlike K-quants which use 1 thread per 8 elements)
    for (int i = 0; i < QK5_0; ++i) {
        // Get high bit from qh
        const uint8_t high_bit = ((qh[i / 8] >> (i % 8)) & 1) << 4;
        // Get low 4 bits from qs
        const uint8_t low_bits = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
        // Combine to get 5-bit value (0-31), then shift to signed range (-16 to 15)
        const int q = (int)(high_bit | low_bits) - 16;

        sum += d * (float)q * vec[offset + i];
    }

    return sum;
}

__launch_bounds__(32, 1)
__global__ void gemv_q5_0_f32_kernel(
    const void* __restrict__ weights_q5_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK5_0;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_0) + col * n_blocks * Q5_0_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q5_0(col_base + block_idx * Q5_0_BLOCK_SIZE, input, block_idx * QK5_0);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q5_0_f32_launch(
    const void* weights_q5_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK5_0 != 0) return hipErrorInvalidValue;

    CHECK_NULL(weights_q5_0);
    CHECK_NULL(input);
    CHECK_NULL(output);

    gemv_q5_0_f32_kernel<<<ncols_dst, 32, 0, stream>>>(
        weights_q5_0, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
