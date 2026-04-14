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

// ── Vector Intrinsic Tests (Phase 1: Zero Risk) ───────────────────────────────────────

/// Test get_int_b2() against scalar byte reads
/// Verifies that 32-bit reads match 4 individual byte reads
extern "C" __global__ void test_get_int_b2(
    const uint8_t* __restrict__ input,
    int* __restrict__ errors,
    int n_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements / 4) return;

    // Test: Read 4 bytes as 32-bit integer
    const int vector_result = get_int_b2(input, tid);

    // Compare against scalar byte reads
    int scalar_result = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        scalar_result |= (input[tid * 4 + i] << (i * 8));
    }

    // Report error if mismatch
    if (vector_result != scalar_result) {
        atomicAdd(errors, 1);
    }
}

/// Test __vsubss4_gpu() against scalar subtraction
/// Verifies vector subtract with saturation produces correct results
extern "C" __global__ void test_vsubss4(
    const int* __restrict__ input_a,
    const int* __restrict__ input_b,
    int* __restrict__ errors,
    int n_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    const int a = input_a[tid];
    const int b = input_b[tid];

    // Test: Vector subtract with saturation (4 int8 values)
    const int vector_result = __vsubss4_gpu(a, b);

    // Compare against scalar byte-by-byte subtraction with saturation
    int scalar_result = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Extract each byte
        const int byte_a = (a >> (i * 8)) & 0xFF;
        const int byte_b = (b >> (i * 8)) & 0xFF;

        // Reinterpret as signed int8
        const int8_t signed_a = static_cast<int8_t>(byte_a);
        const int8_t signed_b = static_cast<int8_t>(byte_b);

        // Subtract with saturation
        int16_t result = static_cast<int16_t>(signed_a) - static_cast<int16_t>(signed_b);
        if (result < -128) result = -128;
        if (result > 127) result = 127;

        // Pack back into result
        scalar_result |= (static_cast<uint8_t>(result) << (i * 8));
    }

    // Report error if mismatch
    if (vector_result != scalar_result) {
        atomicAdd(errors, 1);
    }
}

#endif // ROCMFORGE_Q6_K_TEST_HIP
