#include "common.hip"

// Q3_K constants
#define QK_K 256
#define Q3_K_BLOCK_SIZE 110

__device__ inline float unpack_scale_q3_k(const uint8_t* scales_raw, int scale_idx) {
    const int i = scale_idx / 4;
    const int j = scale_idx % 4;
    uint8_t scale = 0;
    if (i == 0) {
        scale = (scales_raw[j] & 0x0F) | ((scales_raw[8 + j] & 0x03) << 4);
    } else if (i == 1) {
        scale = (scales_raw[4 + j] & 0x0F) | (((scales_raw[8 + j] >> 2) & 0x03) << 4);
    } else if (i == 2) {
        scale = ((scales_raw[j] >> 4) & 0x0F) | (((scales_raw[8 + j] >> 4) & 0x03) << 4);
    } else {
        scale = ((scales_raw[4 + j] >> 4) & 0x0F) | (((scales_raw[8 + j] >> 6) & 0x03) << 4);
    }
    return static_cast<float>(scale) - 32.0f;
}

__device__ inline float vec_dot_q3_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half;
    memcpy(&d_half, block_bytes + 108, sizeof(half));
    const float d = __half2float(d_half);

    if (fabsf(d) < 1e-7f) return 0.0f;

    const uint8_t* hmask = block_bytes;
    const uint8_t* qs = block_bytes + 32;
    const uint8_t* scales = block_bytes + 96;

    float sum = 0.0f;
    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;
        const int chunk = i / 128;
        const int i_chunk = i % 128;
        const int group_32 = i_chunk / 32;
        const int group_16 = (i_chunk % 32) / 16;
        const int scale_idx = chunk * 8 + group_32 * 2 + group_16;
        const int shift = group_32 * 2;
        const int qs_idx = chunk * 32 + (i_chunk % 32);
        const int hmask_idx = chunk * 16 + (i_chunk % 32);

        const float dl = d * unpack_scale_q3_k(scales, scale_idx);
        const uint8_t ql = (qs[qs_idx] >> shift) & 0x03;
        const uint8_t hbit = ((hmask[hmask_idx] & (1 << group_32)) != 0) ? 0 : 4;

        sum += dl * (static_cast<float>(ql) - static_cast<float>(hbit)) * vec[offset + i];
    }
    return sum;
}

__launch_bounds__(32, 1)
__global__ void gemv_q3_k_f32_kernel(
    const void* __restrict__ weights_q3_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q3_k) + col * n_blocks * Q3_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q3_k(col_base + block_idx * Q3_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q3_k_f32_launch(
    const void* weights_q3_k,
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
    CHECK_NULL(weights_q3_k);
    CHECK_NULL(input);
    CHECK_NULL(output);

    // Q4_K pattern: 32 threads per block (warp size)
    const int block_size = 32;

    gemv_q3_k_f32_kernel<<<ncols_dst, block_size, 0, stream>>>(
        weights_q3_k, input, output, n_rows, ncols_dst
    );

    // Check for launch errors
    CHECK_HIP(hipGetLastError());

    return hipSuccess;
}
