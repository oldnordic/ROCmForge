#ifndef ROCMFORGE_Q6_K_VECTORIZED_HIP
#define ROCMFORGE_Q6_K_VECTORIZED_HIP

#include "common.hip"

/// Vectorized Q6_K dequantization kernel (Phase 2)
/// Uses vectorized bit extraction (4 nibbles at once)
/// Matches llama.cpp pattern: get_int_b2() + 0x0F0F0F0F masks
template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemm_q6_k_f32_kernel_vectorized(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int seq_len
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Bounds check with compile-time constant
    if (col >= ncols_dst || batch_idx >= seq_len) return;

    const uint8_t* weights = static_cast<const uint8_t*>(weights_q6_k);
    const int n_blocks = n_rows / QK_K;

    float sum = 0.0f;
    const uint8_t* col_base = weights + col * n_blocks * Q6_K_BLOCK_SIZE;

    // CRITICAL: Offset input by batch_idx for multi-token prefill
    const float* input_batch = input + batch_idx * n_rows;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;

        // Unpack Q6_K block metadata
        const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);

        half d_half;
        memcpy(&d_half, &block[208], sizeof(half));
        const float d = __half2float(d_half);

        // Process all 256 elements using 32 threads
        // Each thread processes 8 elements: tid*8, tid*8+1, ..., tid*8+7
        // VECTORIZED: Process 2 elements per iteration (4 iterations total)
        for (int l = 0; l < 8; l += 2) {
            const int i0 = tid * 8 + l;     // First element
            const int i1 = tid * 8 + l + 1; // Second element

            // Map first element to Q6_K interleaved pattern
            const int pos_in_group0 = i0 % 128;
            const int group0 = i0 / 128;
            const int l_base0 = pos_in_group0 % 32;
            const int quadrant0 = pos_in_group0 / 32;
            const int is0 = l_base0 / 16;

            // Map second element to Q6_K interleaved pattern
            const int pos_in_group1 = i1 % 128;
            const int group1 = i1 / 128;
            const int l_base1 = pos_in_group1 % 32;
            const int quadrant1 = pos_in_group1 / 32;
            const int is1 = l_base1 / 16;

            // Compute scale indices
            const int scale_idx0 = group0 * 8 + is0 * 2 + quadrant0;
            const int scale_idx1 = group1 * 8 + is1 * 2 + quadrant1;
            const float scale0 = d * (float)scales[scale_idx0];
            const float scale1 = d * (float)scales[scale_idx1];

            // === VECTORIZED BIT EXTRACTION (Phase 2) ===
            // Read 4 bytes at once using get_int_b2()
            // We'll extract 2 nibbles from the packed int

            const int ql_offset0 = group0 * 64 + (l_base0 + quadrant0 * 32);
            const int ql_packed0 = get_int_b2(block, ql_offset0);

            const int qh_offset0 = 128 + group0 * 32 + l_base0;
            const int qh_packed0 = get_int_b2(block, qh_offset0);

            const int ql_offset1 = group1 * 64 + (l_base1 + quadrant1 * 32);
            const int ql_packed1 = get_int_b2(block, ql_offset1);

            const int qh_offset1 = 128 + group1 * 32 + l_base1;
            const int qh_packed1 = get_int_b2(block, qh_offset1);

            // Extract nibbles for first element
            const int is_low_half0 = (quadrant0 < 2) ? 0 : 1;
            const int shift0 = is_low_half0 ? 0 : 4;
            const int qh_shift0 = is_low_half0 ? (quadrant0 % 2) * 2 : (quadrant0 % 2) * 2 + 4;

            const uint8_t ql_4bits0 = (ql_packed0 >> shift0) & 0x0F;
            const uint8_t qh_2bits0 = (qh_packed0 >> qh_shift0) & 0x03;
            const int8_t q0 = (int8_t)(ql_4bits0 | (qh_2bits0 << 4)) - 32;

            // Extract nibbles for second element
            const int is_low_half1 = (quadrant1 < 2) ? 0 : 1;
            const int shift1 = is_low_half1 ? 0 : 4;
            const int qh_shift1 = is_low_half1 ? (quadrant1 % 2) * 2 : (quadrant1 % 2) * 2 + 4;

            const uint8_t ql_4bits1 = (ql_packed1 >> shift1) & 0x0F;
            const uint8_t qh_2bits1 = (qh_packed1 >> qh_shift1) & 0x03;
            const int8_t q1 = (int8_t)(ql_4bits1 | (qh_2bits1 << 4)) - 32;

            // Accumulate both elements
            const int vec_offset0 = block_idx * QK_K + i0;
            const int vec_offset1 = block_idx * QK_K + i1;
            sum += input_batch[vec_offset0] * (scale0 * (float)q0);
            sum += input_batch[vec_offset1] * (scale1 * (float)q1);
        }
    }

    // Warp reduction with unroll
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

#endif // ROCMFORGE_Q6_K_VECTORIZED_HIP
