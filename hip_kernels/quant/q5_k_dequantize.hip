//! Q5_K dequantization kernel
//!
//! Dequantizes Q5_K blocks to f32 values.
//! Uses non-uniform quantization with get_scale_min_k4 pattern.

#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
#define Q5_K_BLOCK_SIZE 176
#define K_SCALE_SIZE 12

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

/// Dequantize Q5_K block to f32 (device kernel)
/// Follows llama.cpp pattern: processes 64 elements at a time (2 sub-blocks)
__global__ void dequantize_q5_k_device(
    const void* input,   // [n/256 * 176] Q5_K quantized input
    float* output,      // [n] f32 output data
    int n               // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;
    const int block_end = min(block_start + QK_K, n);

    // Access block as raw bytes to avoid struct padding issues
    const uint8_t* block_bytes = static_cast<const uint8_t*>(input) + block_idx * Q5_K_BLOCK_SIZE;

    // Load d and dmin
    half d_half, dmin_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float min = __half2float(dmin_half);

    // Get pointers to arrays
    const uint8_t* scales = block_bytes + 4;
    const uint8_t* qh = block_bytes + 4 + K_SCALE_SIZE;
    const uint8_t* ql = block_bytes + 4 + K_SCALE_SIZE + QK_K / 8;

    // Process 64 elements at a time (2 sub-blocks of 32 each)
    // llama.cpp pattern: lines 1490-1499
    int is = 0;
    uint8_t u1 = 1, u2 = 2;

    for (int j = 0; j < QK_K; j += 64) {
        // Get scale and min for first sub-block
        uint8_t sc1, m1;
        get_scale_min_k4(is + 0, scales, &sc1, &m1);
        const float d1 = d * sc1;
        const float min1 = min * m1;

        // Get scale and min for second sub-block
        uint8_t sc2, m2;
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        const float d2 = d * sc2;
        const float min2 = min * m2;

        // Process 32 elements from first sub-block
        for (int l = threadIdx.x; l < 32 && (j + l) < (block_end - block_start); l += 32) {
            const uint8_t q4 = ql[l] & 0x0F;
            const uint8_t high_bit = (qh[l] & u1) ? 16 : 0;
            const float val = static_cast<float>(q4 + high_bit) / d1 + min1;
            output[block_start + j + l] = val;
        }

        // Process 32 elements from second sub-block
        for (int l = threadIdx.x; l < 32 && (j + 32 + l) < (block_end - block_start); l += 32) {
            const uint8_t q4 = (ql[l] >> 4) & 0x0F;
            const uint8_t high_bit = (qh[l] & u2) ? 16 : 0;
            const float val = static_cast<float>(q4 + high_bit) / d2 + min2;
            output[block_start + j + 32 + l] = val;
        }

        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Batched dequantization kernel
__global__ void dequantize_q5_k_batched_device(
    const void* input,   // [batch_size][n/256 * 176]
    float* output,      // [batch_size][n]
    int n,
    int batch_size
) {
    const int batch_idx = blockIdx.y;
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;
    const int block_end = min(block_start + QK_K, n);

    const uint8_t* batch_input = static_cast<const uint8_t*>(input) + batch_idx * ((n / QK_K) * Q5_K_BLOCK_SIZE);
    const uint8_t* block_bytes = batch_input + block_idx * Q5_K_BLOCK_SIZE;

    // Load d and dmin
    half d_half, dmin_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d = __half2float(d_half);
    const float min_val = __half2float(dmin_half);

    const uint8_t* scales = block_bytes + 4;
    const uint8_t* qh = block_bytes + 4 + K_SCALE_SIZE;
    const uint8_t* ql = block_bytes + 4 + K_SCALE_SIZE + QK_K / 8;

    float* batch_output = output + batch_idx * n;

    int is = 0;
    uint8_t u1 = 1, u2 = 2;

    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc1, m1;
        get_scale_min_k4(is + 0, scales, &sc1, &m1);
        const float d1 = d * sc1;
        const float min1 = min_val * m1;

        uint8_t sc2, m2;
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        const float d2 = d * sc2;
        const float min2 = min_val * m2;

        for (int l = threadIdx.x; l < 32 && (j + l) < (block_end - block_start); l += 32) {
            const uint8_t q4 = ql[l] & 0x0F;
            const uint8_t high_bit = (qh[l] & u1) ? 16 : 0;
            batch_output[block_start + j + l] = static_cast<float>(q4 + high_bit) / d1 + min1;
        }

        for (int l = threadIdx.x; l < 32 && (j + 32 + l) < (block_end - block_start); l += 32) {
            const uint8_t q4 = (ql[l] >> 4) & 0x0F;
            const uint8_t high_bit = (qh[l] & u2) ? 16 : 0;
            batch_output[block_start + j + 32 + l] = static_cast<float>(q4 + high_bit) / d2 + min2;
        }

        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// Dispatch functions (called from Rust)
extern "C" hipError_t dequantize_q5_k_kernel(
    const void* input,
    float* output,
    int n
) {
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK_K - 1) / QK_K;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q5_k_device),
        dim3(num_blocks),
        dim3(32),
        0, 0,
        input, output, n
    );

    return hipGetLastError();
}

extern "C" hipError_t dequantize_q5_k_batched_kernel(
    const void* input,
    float* output,
    int n,
    int batch_size
) {
    if (n <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK_K - 1) / QK_K;

    dim3 grid(num_blocks, batch_size);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q5_k_batched_device),
        grid,
        dim3(32),
        0, 0,
        input, output, n, batch_size
    );

    return hipGetLastError();
}
