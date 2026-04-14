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

    // Q4_K safety: check for denormal d
    if (fabsf(d) < 1e-7f) return 0.0f;

    // Pointer to scales (int8_t at bytes 192-207)
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block_bytes[192]);

    float sum = 0.0f;

    // Q6_K uses INTERLEAVED element distribution, not linear
    // Thread T processes elements: T, T+32, T+64, T+96 (group 0) and T+128, T+160, T+192, T+224 (group 1)
    // Each thread processes 8 elements total, but they're strided by 32, not contiguous

    for (int group = 0; group < 2; ++group) {
        const int base_offset = group * 128;  // 0 for group 0, 128 for group 1

        // Process 4 elements per group: tid + base_offset + {0, 32, 64, 96}
        for (int s = 0; s < 4; ++s) {
            const int l = tid;  // Thread ID (0-31)
            const int vec_offset = base_offset + l + s * 32;  // Actual element index in vec

            // Map linear index (0-255) to Q6_K parameters
            const int i = base_offset + l + s * 32;
            const int pos_in_group = i % 128;    // 0-127
            const int l_base = pos_in_group % 32; // 0-31 (= l for group 0, = l for group 1)
            const int quadrant = pos_in_group / 32; // 0, 1, 2, or 3 (= s)
            const int is = l_base / 16;  // 0 or 1

            // Compute scale index: group * 8 + is * 2 + quadrant
            const int scale_idx = group * 8 + is * 2 + quadrant;
            const float scale = d * (float)scales[scale_idx];

            // Compute ql offset (matching original kernel: ql[l + 0] and ql[l + 32])
            const int ql_local_idx = l + s * 32;
            const int ql_offset = group * 64 + ql_local_idx;
            const uint8_t ql_byte = block_bytes[ql_offset];

            // Compute qh offset (matching original kernel: qh[l])
            const int qh_offset = 128 + group * 32 + l;
            const uint8_t qh_byte = block_bytes[qh_offset];

            // Unpack Q6_K value (matching original kernel bit operations)
            // Determine which half of ql byte and corresponding qh bits
            const int is_low_half = (s < 2) ? 0 : 1;  // s=0,1 use low half, s=2,3 use high half
            const int shift = is_low_half ? 0 : 4;
            const int qh_shift = is_low_half ? (s % 2) * 2 : (s % 2) * 2 + 4;

            const uint8_t ql_4bits = (ql_byte >> shift) & 0x0F;
            const uint8_t qh_2bits = (qh_byte >> qh_shift) & 0x03;
            const int8_t q = (int8_t)(ql_4bits | (qh_2bits << 4)) - 32;

            // Q4_K safety: simple linear access without complex validation
            // The Q4_K pattern proves this works: just access the memory directly
            sum += vec[offset + vec_offset] * (scale * (float)q);
        }
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
