#include "common.hip"
#include <hip/hip_fp16.h>

// Q5_K constants (must match Rust types.rs)
#define QK_K 256
#define Q5_K_BLOCK_SIZE 176
#define K_SCALE_SIZE 12

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Extract scale and min for sub-block j (llama.cpp get_scale_min_k4 pattern)
__device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/// Vector dot product: Q5_K block × f32 vector segment
__device__ inline float vec_dot_q5_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half, dmin_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float dmin = __half2float(dmin_half);

    const uint8_t* scales = block_bytes + 4;
    const uint8_t* qh = block_bytes + 4 + K_SCALE_SIZE;
    const uint8_t* ql = block_bytes + 4 + K_SCALE_SIZE + QK_K / 8;

    float sum = 0.0f;

    int is = 0;
    uint8_t u1 = 1, u2 = 2;

    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc1, m1;
        get_scale_min_k4(is + 0, scales, &sc1, &m1);
        const float d1 = d * sc1;
        const float min1 = dmin * m1;

        uint8_t sc2, m2;
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        const float d2 = d * sc2;
        const float min2 = dmin * m2;

        const int l = tid;
        const uint8_t q4 = ql[l] & 0x0F;
        const uint8_t high_bit = (qh[l] & u1) ? 16 : 0;
        const float val1 = static_cast<float>(q4 + high_bit) / d1 + min1;
        sum += val1 * vec[offset + j + l];

        const uint8_t q4_2 = (ql[l] >> 4) & 0x0F;
        const uint8_t high_bit_2 = (qh[l] & u2) ? 16 : 0;
        const float val2 = static_cast<float>(q4_2 + high_bit_2) / d2 + min2;
        sum += val2 * vec[offset + j + 32 + l];

        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }

    return sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemm_q5_k_f32_kernel(
    const void* __restrict__ weights_q5_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_k) + col * (n_rows / QK_K) * Q5_K_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q5_K_BLOCK_SIZE;
        sum += vec_dot_q5_k(block_ptr, input_batch, block_idx * QK_K);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Generic Kernel (non-template) ───────────────────────────────────────────────────

__global__ void gemm_q5_k_f32_generic_kernel(
    const void* __restrict__ weights_q5_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_k) + col * (n_rows / QK_K) * Q5_K_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q5_K_BLOCK_SIZE;
        sum += vec_dot_q5_k(block_ptr, input_batch, block_idx * QK_K);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t gemm_q5_k_f32_launch(
    const void* weights_q5_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int batch_size,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK_K != 0) {
        return hipErrorInvalidValue;
    }

    dim3 grid(ncols_dst, batch_size);
    dim3 block(32);

    switch (ncols_dst) {
        case 1:
            gemm_q5_k_f32_kernel<1><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        case 2:
            gemm_q5_k_f32_kernel<2><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        case 3:
            gemm_q5_k_f32_kernel<3><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        case 4:
            gemm_q5_k_f32_kernel<4><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        case 5:
            gemm_q5_k_f32_kernel<5><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        case 6:
            gemm_q5_k_f32_kernel<6><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        case 7:
            gemm_q5_k_f32_kernel<7><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        case 8:
            gemm_q5_k_f32_kernel<8><<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, batch_size);
            break;
        default:
            gemm_q5_k_f32_generic_kernel<<<grid, block, 0, stream>>>(
                weights_q5_k, input, output, n_rows, ncols_dst, batch_size);
            break;
    }

    return hipGetLastError();
}
