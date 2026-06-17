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

    // Q5_0 packs 32 values into 16 nibble bytes using the GGUF/llama.cpp
    // layout: byte i holds element i (low nibble) and element i+16 (high
    // nibble) — identical to Q4_0 packing. Each value also has a high bit
    // stored in the 4-byte qh array (bit i => element i).
    for (int i = 0; i < QK5_0 / 2; ++i) {
        // element i: low nibble of qs[i], high bit i of qh
        const uint8_t high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
        const uint8_t low_bits_0 = qs[i] & 0x0F;
        const int q0 = (int)(high_bit_0 | low_bits_0) - 16;

        // element i+16: high nibble of qs[i], high bit (i+16) of qh
        const uint8_t high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
        const uint8_t low_bits_1 = (qs[i] >> 4) & 0x0F;
        const int q1 = (int)(high_bit_1 | low_bits_1) - 16;

        sum += d * (float)q0 * vec[offset + i] +
               d * (float)q1 * vec[offset + i + 16];
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
    // Distribute Q5_0 blocks across the 32 lanes; each lane computes a
    // partial dot product for its assigned blocks, then the warp shuffle
    // reduction sums the partials into the correct full-column result.
    for (int block_idx = tid; block_idx < n_blocks; block_idx += 32) {
        sum += vec_dot_q5_0(col_base + block_idx * Q5_0_BLOCK_SIZE, input, block_idx * QK5_0);
    }

    // Warp reduction across lanes
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
