#ifndef ROCMFORGE_Q6_K_TEST_HIP
#define ROCMFORGE_Q6_K_TEST_HIP

#include "common.hip"

/// Reference Q6_K dequantization (scalar, tested implementation)
/// This is the baseline we know produces correct results
extern "C" __global__ void q6_k_dequant_reference(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = weights + col * (n_rows / QK_K) * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;
    const int n_blocks = n_rows / QK_K;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;

        const uint8_t* ql = &block[0];
        const uint8_t* qh = &block[128];
        const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);

        half d_half;
        memcpy(&d_half, &block[208], sizeof(half));
        const float d = __half2float(d_half);

        for (int l = 0; l < 8; ++l) {
            const int i = tid * 8 + l;

            const int pos_in_group = i % 128;
            const int group = i / 128;
            const int l_base = pos_in_group % 32;
            const int quadrant = pos_in_group / 32;
            const int is = l_base / 16;

            const int scale_idx = group * 8 + is * 2 + quadrant;
            const float scale = d * static_cast<float>(scales[scale_idx]);

            const int ql_local_idx = l_base + quadrant * 32;
            const int ql_offset = group * 64 + ql_local_idx;
            const uint8_t ql_byte = block[ql_offset];

            const int qh_offset = 128 + group * 32 + l_base;
            const uint8_t qh_byte = block[qh_offset];

            const int is_low_half = (quadrant < 2) ? 0 : 1;
            const int shift = is_low_half ? 0 : 4;
            const int qh_shift = is_low_half ? (quadrant % 2) * 2 : (quadrant % 2) * 2 + 4;

            const uint8_t ql_4bits = (ql_byte >> shift) & 0x0F;
            const uint8_t qh_2bits = (qh_byte >> qh_shift) & 0x03;
            const int8_t q = static_cast<int8_t>(ql_4bits | (qh_2bits << 4)) - 32;

            const int vec_offset = block_idx * QK_K + i;
            sum += input[vec_offset] * (scale * static_cast<float>(q));
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

/// Optimized Q6_K dequantization (tile-based, to be implemented)
/// This will be the target for our optimization work
extern "C" __global__ void q6_k_dequant_optimized(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    // For now, just call the reference implementation
    // After tile-based optimization, this will diverge

    // Simple kernel that just processes first column for testing
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || col != 0) return;

    const uint8_t* col_base = weights + col * (n_rows / QK_K) * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;
    const int n_blocks = n_rows / QK_K;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;

        // ... (simplified for now - just copy reference logic)
        sum += 1.0f;  // Placeholder
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

#endif // ROCMFORGE_Q6_K_TEST_HIP
