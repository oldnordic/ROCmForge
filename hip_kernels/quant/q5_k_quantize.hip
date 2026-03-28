//! Q5_K quantization kernel
//!
//! Quantizes f32 weights to Q5_K format (5-bit quantization with 256-element blocks).
//! Follows llama.cpp pattern with non-uniform quantization using get_scale_min_k4.
//!
//! Algorithm:
//! 1. Compute per-32-element scales and mins (8 sub-blocks)
//! 2. Quantize the scales themselves to 6-bit values
//! 3. Pack scales into scales[12] using llama.cpp bit packing
//! 4. Quantize values using per-sub-block scales
//! 5. Pack 5-bit values (4 low bits in qs, high bit in qh)

#include "common.hip"
#include <hip/hip_fp16.h>

// Q5_K constants
#define QK_K 256
#define Q5_K_BLOCK_SIZE 176  // 2+2+12+32+128
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

/// Quantize f32 data to Q5_K format (device kernel)
/// Each thread block processes one 256-element block with 32 threads
__global__ void quantize_q5_k_device(
    const float* input,   // [n] f32 input data
    void* output,         // [n/256 * 176] Q5_K quantized output
    int n                 // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;
    const float* x = &input[block_start];

    // Shared memory for quantization
    __shared__ float s_subblock_scales[8];  // 8 sub-blocks
    __shared__ float s_subblock_mins[8];    // 8 sub-blocks
    __shared__ float s_max_scale;
    __shared__ float s_max_min;
    __shared__ half s_d;
    __shared__ half s_dmin;
    __shared__ uint8_t s_scales[K_SCALE_SIZE];
    __shared__ uint8_t s_L[QK_K];  // Temporary quantized values
    __shared__ uint8_t s_qh[QK_K / 8];
    __shared__ uint8_t s_qs[QK_K / 2];

    // Each thread processes one sub-block (32 elements)
    // 32 threads = 8 sub-blocks * 4 threads per sub-block
    const int subblock_idx = threadIdx.x / 4;  // 0-7
    const int lane_idx = threadIdx.x % 4;      // 0-3

    if (subblock_idx < 8) {
        const int subblock_start = subblock_idx * 32;
        const float* subblock_x = &x[subblock_start];

        // Find min/max for this sub-block (each thread handles 8 elements)
        float local_min = 1e38f;
        float local_max = -1e38f;
        for (int i = lane_idx * 8; i < (lane_idx + 1) * 8 && i < 32; i++) {
            local_min = fminf(local_min, subblock_x[i]);
            local_max = fmaxf(local_max, subblock_x[i]);
        }

        // Reduce across 4 threads to get sub-block min/max
        for (int offset = 2; offset > 0; offset >>= 1) {
            float other_min = __shfl_down(local_min, offset);
            float other_max = __shfl_down(local_max, offset);
            local_min = fminf(local_min, other_min);
            local_max = fmaxf(local_max, other_max);
        }

        // Thread 0 of each group writes to shared memory
        if (lane_idx == 0) {
            // Simplified: direct scaling (llama.cpp uses more complex make_qkx2_quants)
            float range = local_max - local_min;
            float scale = (range < 1e-30f) ? 1.0f : 31.0f / range;
            float min = local_min;

            s_subblock_scales[subblock_idx] = scale;
            s_subblock_mins[subblock_idx] = min;
        }
    }
    __syncthreads();

    // Find max_scale and max_min across all sub-blocks
    if (threadIdx.x == 0) {
        s_max_scale = 0;
        s_max_min = 0;
        for (int i = 0; i < 8; i++) {
            if (s_subblock_scales[i] > s_max_scale) s_max_scale = s_subblock_scales[i];
            if (s_subblock_mins[i] > s_max_min) s_max_min = s_subblock_mins[i];
        }
    }
    __syncthreads();

    // Thread 0 computes global d and dmin, and quantizes scales
    if (threadIdx.x == 0) {
        float inv_scale = (s_max_scale > 0) ? 63.0f / s_max_scale : 0.0f;
        float inv_min = (s_max_min > 0) ? 63.0f / s_max_min : 0.0f;

        s_d = __float2half(s_max_scale / 63.0f);
        s_dmin = __float2half(s_max_min / 63.0f);

        // Quantize and pack scales (llama.cpp lines 1421-1434)
        for (int j = 0; j < 8; j++) {
            uint8_t ls = (uint8_t)roundf(inv_scale * s_subblock_scales[j]);
            uint8_t lm = (uint8_t)roundf(inv_min * s_subblock_mins[j]);
            ls = (ls > 63) ? 63 : ls;
            lm = (lm > 63) ? 63 : lm;

            if (j < 4) {
                s_scales[j] = ls;
                s_scales[j + 4] = lm;
            } else {
                s_scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                s_scales[j - 4] |= ((ls >> 4) << 6);
                s_scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
    }
    __syncthreads();

    // Quantize values using per-sub-block scales
    if (subblock_idx < 8) {
        const int subblock_start = subblock_idx * 32;
        const float* subblock_x = &x[subblock_start];

        // Get scale and min for this sub-block
        uint8_t sc, m;
        get_scale_min_k4(subblock_idx, s_scales, &sc, &m);

        const float d = __half2float(s_d) * sc;
        const float dm = __half2float(s_dmin) * m;

        // Quantize 8 elements per thread
        for (int i = lane_idx * 8; i < (lane_idx + 1) * 8 && i < 32; i++) {
            int l = (int)roundf((subblock_x[i] + dm) / d);
            l = (l < 0) ? 0 : (l > 31 ? 31 : l);
            s_L[subblock_start + i] = (uint8_t)l;
        }
    }
    __syncthreads();

    // Pack values into qs and qh (llama.cpp lines 1451-1470)
    if (threadIdx.x == 0) {
        memset(s_qh, 0, QK_K / 8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; j++) {
                int l1 = s_L[n + j];
                if (l1 > 15) {
                    l1 -= 16;
                    s_qh[j] |= m1;
                }
                int l2 = s_L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16;
                    s_qh[j] |= m2;
                }
                s_qs[n / 2 + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
    }
    __syncthreads();

    // Thread 0 writes output
    if (threadIdx.x == 0) {
        uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q5_K_BLOCK_SIZE;

        // Write d and dmin (use memcpy for safety)
        memcpy(out, &s_d, sizeof(half));
        memcpy(out + 2, &s_dmin, sizeof(half));

        // Write scales
        for (int i = 0; i < K_SCALE_SIZE; i++) {
            out[4 + i] = s_scales[i];
        }

        // Write qh (high bits)
        for (int i = 0; i < QK_K / 8; i++) {
            out[4 + K_SCALE_SIZE + i] = s_qh[i];
        }

        // Write qs (low 4 bits)
        for (int i = 0; i < QK_K / 2; i++) {
            out[4 + K_SCALE_SIZE + QK_K / 8 + i] = s_qs[i];
        }
    }
}

// Dispatch function (called from Rust)
extern "C" hipError_t quantize_q5_k_kernel(
    const float* input,
    void* output,
    int n
) {
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK_K - 1) / QK_K;

    // Launch with 32 threads per block (one warp)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize_q5_k_device),
        dim3(num_blocks),
        dim3(32),
        0, 0,
        input, output, n
    );

    return hipGetLastError();
}
