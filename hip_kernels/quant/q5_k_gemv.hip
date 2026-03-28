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
/// Thread layout: 32 threads, each processes 8 elements using 2 different scales
__device__ inline float vec_dot_q5_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;

    // Access block as raw bytes to avoid struct padding issues
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Load d and dmin (bytes 0-3)
    half d_half, dmin_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float dmin = __half2float(dmin_half);

    // Get pointers to arrays
    const uint8_t* scales = block_bytes + 4;
    const uint8_t* qh = block_bytes + 4 + K_SCALE_SIZE;
    const uint8_t* ql = block_bytes + 4 + K_SCALE_SIZE + QK_K / 8;

    float sum = 0.0f;

    // Process 64 elements at a time (2 sub-blocks)
    // Each thread processes 8 elements per iteration
    int is = 0;
    uint8_t u1 = 1, u2 = 2;

    for (int j = 0; j < QK_K; j += 64) {
        // Get scale and min for first sub-block
        uint8_t sc1, m1;
        get_scale_min_k4(is + 0, scales, &sc1, &m1);
        const float d1 = d * sc1;
        const float min1 = dmin * m1;

        // Get scale and min for second sub-block
        uint8_t sc2, m2;
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        const float d2 = d * sc2;
        const float min2 = dmin * m2;

        // Each thread processes 8 elements (4 from each sub-block)
        for (int l = 0; l < 8; ++l) {
            const int i = tid * 8 + l;  // Global element index

            if (i < 32 && (j + i) < QK_K) {
                // First sub-block
                const uint8_t q4 = ql[i] & 0x0F;
                const uint8_t high_bit = (qh[i] & u1) ? 16 : 0;
                const float val = static_cast<float>(q4 + high_bit) / d1 + min1;
                sum += val * vec[offset + j + i];
            } else if (i >= 32 && (j + i) < QK_K) {
                // Second sub-block
                const int i2 = i - 32;
                const uint8_t q4 = (ql[i2] >> 4) & 0x0F;
                const uint8_t high_bit = (qh[i2] & u2) ? 16 : 0;
                const float val = static_cast<float>(q4 + high_bit) / d2 + min2;
                sum += val * vec[offset + j + i];
            }
        }

        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }

    return sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

/// Template kernel for Q5_K × f32 GEMV with ncols_dst as compile-time parameter
/// Follows llama.cpp pattern for optimal performance
template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemv_q5_k_f32_kernel(
    const void* __restrict__ weights_q5_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    // Bounds check
    if (col >= ncols_dst) return;

    // Calculate column base pointer (interleaved layout)
    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_k) + col * (n_rows / QK_K) * Q5_K_BLOCK_SIZE;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    // Process each block - all threads collaborate on each block
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q5_K_BLOCK_SIZE;
        sum += vec_dot_q5_k(block_ptr, input, block_idx * QK_K);
    }

    // Warp reduction (RDNA: 32 threads = 1 wavefront)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    // Thread 0 writes result
    if (tid == 0) {
        output[col] = sum;
    }
}

// ── Generic Kernel (non-template) ───────────────────────────────────────────────────

/// Generic kernel for arbitrary ncols_dst
/// Less optimized but handles any size
__global__ void gemv_q5_k_f32_generic_kernel(
    const void* __restrict__ weights_q5_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q5_k) + col * (n_rows / QK_K) * Q5_K_BLOCK_SIZE;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    // Process each block - all threads collaborate on each block
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q5_K_BLOCK_SIZE;
        sum += vec_dot_q5_k(block_ptr, input, block_idx * QK_K);
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

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t gemv_q5_k_f32_launch(
    const void* weights_q5_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    // Input validation
    if (!weights_q5_k || !input || !output) {
        return hipErrorInvalidValue;
    }
    if (n_rows <= 0 || ncols_dst <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK_K != 0) {
        return hipErrorInvalidValue;
    }
    if (ncols_dst > 1024) {
        return hipErrorInvalidValue;
    }

    // Dispatch based on ncols_dst (llama.cpp pattern)
    switch (ncols_dst) {
        case 1:
            gemv_q5_k_f32_kernel<1><<<dim3(1), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        case 2:
            gemv_q5_k_f32_kernel<2><<<dim3(2), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        case 3:
            gemv_q5_k_f32_kernel<3><<<dim3(3), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        case 4:
            gemv_q5_k_f32_kernel<4><<<dim3(4), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        case 5:
            gemv_q5_k_f32_kernel<5><<<dim3(5), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        case 6:
            gemv_q5_k_f32_kernel<6><<<dim3(6), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        case 7:
            gemv_q5_k_f32_kernel<7><<<dim3(7), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        case 8:
            gemv_q5_k_f32_kernel<8><<<dim3(8), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows);
            break;
        default:
            gemv_q5_k_f32_generic_kernel<<<dim3(ncols_dst), dim3(32), 0, stream>>>(
                weights_q5_k, input, output, n_rows, ncols_dst);
            break;
    }

    return hipGetLastError();
}
