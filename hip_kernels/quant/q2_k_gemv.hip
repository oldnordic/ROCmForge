#include "common.hip"

// Q2_K constants
#define QK_K 256
#define Q2_K_BLOCK_SIZE 84

__device__ inline float vec_dot_q2_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half, dmin_half;
    memcpy(&d_half, block_bytes + 80, sizeof(half));
    memcpy(&dmin_half, block_bytes + 82, sizeof(half));
    const float d = __half2float(d_half);
    const float dmin = __half2float(dmin_half);

    const uint8_t* scales = block_bytes;
    const uint8_t* qs = block_bytes + 16;

    if (fabsf(d) < 1e-7f && fabsf(dmin) < 1e-7f) return 0.0f;

    float sum = 0.0f;
    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;
        const int q_chunk = i / 128;
        const int i_chunk = i % 128;
        const int group_32 = i_chunk / 32;
        const int group_16 = (i_chunk % 32) / 16;
        const int scale_idx = (q_chunk * 8) + (group_32 * 2) + group_16;
        const int shift = group_32 * 2;
        const int qs_idx = q_chunk * 32 + (i_chunk % 32);

        const uint8_t q_val = (qs[qs_idx] >> shift) & 0x03;
        const uint8_t sc = scales[scale_idx];
        const float dl = d * (sc & 0x0F);
        const float ml = dmin * (sc >> 4);

        sum += (dl * q_val - ml) * vec[offset + i];
    }
    return sum;
}

__launch_bounds__(32, 1)
__global__ void gemv_q2_k_f32_kernel(
    const void* __restrict__ weights_q2_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q2_k) + col * n_blocks * Q2_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q2_k(col_base + block_idx * Q2_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q2_k_f32_launch(
    const void* weights_q2_k,
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
    CHECK_NULL(weights_q2_k);
    CHECK_NULL(input);
    CHECK_NULL(output);

    // Q4_K pattern: 32 threads per block (warp size)
    const int block_size = 32;

    gemv_q2_k_f32_kernel<<<ncols_dst, block_size, 0, stream>>>(
        weights_q2_k, input, output, n_rows, ncols_dst
    );

    // Check for launch errors
    CHECK_HIP(hipGetLastError());

    return hipSuccess;
}
