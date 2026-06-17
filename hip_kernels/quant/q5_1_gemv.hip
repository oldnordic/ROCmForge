#include "common.hip"
#include <hip/hip_fp16.h>

// Q5_1 constants
#define QK5_1 32
#define Q5_1_BLOCK_SIZE 24

__device__ inline float vec_dot_q5_1(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Q5_1 block: [d f16 | m f16 | qh[4] | qs[16]] = 24 bytes for 32 values
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

__launch_bounds__(32, 1)
__global__ void gemv_q5_1_f32_kernel(
    const void* __restrict__ weights_q5_1,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK5_1;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_1) + col * n_blocks * Q5_1_BLOCK_SIZE;

    float sum = 0.0f;
    // Distribute Q5_1 blocks across the 32 lanes; each lane computes a
    // partial dot product for its assigned blocks, then the warp shuffle
    // reduction sums the partials into the correct full-column result.
    for (int block_idx = tid; block_idx < n_blocks; block_idx += 32) {
        sum += vec_dot_q5_1(col_base + block_idx * Q5_1_BLOCK_SIZE, input, block_idx * QK5_1);
    }

    // Warp reduction across lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q5_1_f32_launch(
    const void* weights_q5_1,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK5_1 != 0) return hipErrorInvalidValue;

    CHECK_NULL(weights_q5_1);
    CHECK_NULL(input);
    CHECK_NULL(output);

    gemv_q5_1_f32_kernel<<<ncols_dst, 32, 0, stream>>>(
        weights_q5_1, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
