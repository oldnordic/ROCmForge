#include "common.hip"

// Q4_K constants
#define QK_K 256
#define Q4_K_BLOCK_SIZE 144

__device__ inline float vec_dot_q4_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half, dmin_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float dmin = __half2float(dmin_half);

    const uint8_t* qs = block_bytes + 16;
    if (fabsf(d) < 1e-7f) return 0.0f;

    float sum = 0.0f;
    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;
        const uint8_t packed = qs[i / 2];
        const uint8_t q4 = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        sum += (static_cast<float>(q4) / d + dmin) * vec[offset + i];
    }
    return sum;
}

__global__ void gemv_q4_k_f32_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q4_k) + col * n_blocks * Q4_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q4_k(col_base + block_idx * Q4_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q4_k_f32_launch(
    const void* weights_q4_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    gemv_q4_k_f32_kernel<<<ncols_dst, 32, 0, stream>>>(
        weights_q4_k, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
