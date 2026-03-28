//! Q4_K quantization kernel
//!
//! Quantizes f32 weights to Q4_K format (4-bit quantization with 256-element blocks).
//! Simplified approach using 32 threads per block (one warp).
//!
//! Q4_K format (144 bytes for 256 f32 values):
//! - d: f16 scale (2 bytes)
//! - dmin: f16 minimum scale (2 bytes)
//! - scales[12]: quantized scales (12 bytes)
//! - qs[128]: 4-bit quantized values packed as bytes (128 bytes)

#include "common.hip"
#include <hip/hip_fp16.h>

// Q4_K constants
#define QK_K 256
#define Q4_K_BLOCK_SIZE 144
#define K_SCALE_SIZE 12

/// Quantize f32 data to Q4_K format (device kernel)
/// Each thread block processes one 256-element block with 32 threads
/// Each thread processes 8 elements sequentially
__global__ void quantize_q4_k_device(
    const float* input,   // [n] f32 input data
    void* output,         // [n/256 * 144] Q4_K quantized output
    int n                 // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;
    const float* x = &input[block_start];

    // Shared memory for quantized data
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ half s_d;
    __shared__ half s_dmin;
    __shared__ uint8_t s_scales[K_SCALE_SIZE];
    __shared__ uint8_t s_qs[QK_K / 2]; // 128 bytes for 256 4-bit values

    // Each thread finds min/max across its 8 elements
    float min_val = 1e38f;
    float max_val = -1e38f;

    // Process 8 elements per thread (256 / 32 = 8)
    for (int i = threadIdx.x; i < QK_K; i += 32) {
        float val = x[i];
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
    }

    // Warp reduction to find global min/max
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_min = __shfl_down(min_val, offset);
        float other_max = __shfl_down(max_val, offset);
        min_val = fminf(min_val, other_min);
        max_val = fmaxf(max_val, other_max);
    }

    // Thread 0 writes results to shared memory
    if (threadIdx.x == 0) {
        s_min = min_val;
        s_max = max_val;
    }
    __syncthreads();

    // Thread 0 computes scale and initializes quantized array
    if (threadIdx.x == 0) {
        float range = s_max - s_min;
        if (range < 1e-30f) {
            s_d = __float2half(1.0f);
            s_dmin = __float2half(s_min);
        } else {
            s_d = __float2half(15.0f / range);
            s_dmin = __float2half(s_min);
        }

        // Initialize scales to zero (uniform quantization for now)
        for (int i = 0; i < K_SCALE_SIZE; i++) {
            s_scales[i] = 0;
        }

        // Initialize quantized values to zero BEFORE other threads write
        for (int i = 0; i < QK_K / 2; i++) {
            s_qs[i] = 0;
        }
    }
    __syncthreads();

    // Each thread quantizes its 8 elements
    // Process in two phases to avoid race conditions on shared memory
    // Phase 1: Even indices (lower 4 bits)
    for (int i = threadIdx.x; i < QK_K; i += 32) {
        if (i % 2 == 0) {
            float val = x[i];
            float q = __half2float(s_d) * (val - __half2float(s_dmin));
            int iq = (int)(q + 0.5f);
            if (iq > 15) iq = 15;
            if (iq < 0) iq = 0;
            s_qs[i / 2] = iq;
        }
    }
    __syncthreads();

    // Phase 2: Odd indices (upper 4 bits)
    for (int i = threadIdx.x; i < QK_K; i += 32) {
        if (i % 2 == 1) {
            float val = x[i];
            float q = __half2float(s_d) * (val - __half2float(s_dmin));
            int iq = (int)(q + 0.5f);
            if (iq > 15) iq = 15;
            if (iq < 0) iq = 0;
            s_qs[i / 2] |= (iq << 4);
        }
    }
    __syncthreads();

    // Thread 0 writes output
    if (threadIdx.x == 0) {
        uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q4_K_BLOCK_SIZE;

        // Write d (f16 scale) - use memcpy for safety
        memcpy(out, &s_d, sizeof(half));

        // Write dmin (f16 minimum)
        memcpy(out + 2, &s_dmin, sizeof(half));

        // Write scales
        for (int i = 0; i < K_SCALE_SIZE; i++) {
            out[4 + i] = s_scales[i];
        }

        // Write quantized values
        for (int i = 0; i < QK_K / 2; i++) {
            out[4 + K_SCALE_SIZE + i] = s_qs[i];
        }
    }
}

// Dispatch Function
extern "C" hipError_t quantize_q4_k_kernel(
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
        HIP_KERNEL_NAME(quantize_q4_k_device),
        dim3(num_blocks),
        dim3(32),  // 32 threads = one warp on RDNA
        0, 0,
        input, output, n
    );

    return hipGetLastError();
}
