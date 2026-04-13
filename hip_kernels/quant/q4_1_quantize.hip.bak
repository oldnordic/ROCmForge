#include "common.hip"
#include <hip/hip_fp16.h>

// Q4_1 constants (must match Rust types.rs)
// Block layout: d (half, 2 bytes) + m (half, 2 bytes) + qs (16 bytes) = 20 bytes
// Quantizes 32 f32 values into one 20-byte block
#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

/// Quantize f32 data to Q4_1 format (device kernel)
/// Each thread block processes one 32-element block
///
/// Q4_1 differs from Q4_0:
///   - Stores both scale (d) and min (m) instead of just scale
///   - Unsigned quant: q = clamp(round((x - m) / d), 0, 15)
///   - Block size: 20 bytes vs Q4_0's 18 bytes
///   - Sequential packing: first 16 values in low nibbles, next 16 in high nibbles
__global__ void quantize_q4_1_device(
    const float* __restrict__ input,   // [n] f32 input data
    void* __restrict__ output,         // [n/32 * 20] Q4_1 quantized output
    int n                              // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const float* x = &input[block_idx * QK4_1];

    // Each thread loads one element (QK4_1 = 32 threads)
    float val = x[threadIdx.x];

    // Warp reduction for min and max
    float block_min = val;
    float block_max = val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_min = __shfl_down(block_min, offset);
        float other_max = __shfl_down(block_max, offset);
        if (other_min < block_min) block_min = other_min;
        if (other_max > block_max) block_max = other_max;
    }

    // Thread 0 computes scale (d) and min (m)
    __shared__ half s_d;
    __shared__ half s_m;
    if (threadIdx.x == 0) {
        float d = (block_max - block_min) / 15.0f;
        s_d = __float2half(d);
        s_m = __float2half(block_min);
    }
    __syncthreads();

    // Quantize values: q = clamp(round((x - m) / d), 0, 15)
    float d = __half2float(s_d);
    float m = __half2float(s_m);
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    int q = (int)roundf((val - m) * id);
    q = (q < 0) ? 0 : (q > 15) ? 15 : q;

    // Pack sequential: threads 0-15 write low nibbles, threads 16-31 write high nibbles
    __shared__ uint8_t s_qs[QK4_1 / 2];
    if (threadIdx.x < 16) {
        s_qs[threadIdx.x] = (uint8_t)q;  // Low nibble
    } else {
        s_qs[threadIdx.x - 16] |= ((uint8_t)q) << 4;  // High nibble
    }
    __syncthreads();

    // Thread 0 writes output: d (2) + m (2) + qs (16) = 20 bytes
    if (threadIdx.x == 0) {
        uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q4_1_BLOCK_SIZE;
        memcpy(out, &s_d, sizeof(half));       // Write d (2 bytes)
        memcpy(out + 2, &s_m, sizeof(half));   // Write m (2 bytes)
        memcpy(out + 4, s_qs, QK4_1 / 2);     // Write qs (16 bytes)
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t quantize_q4_1_kernel(
    const float* input,
    void* output,
    int n
) {
    // Validate dimensions
    if (n <= 0) {
        return hipErrorInvalidValue;
    }
    if (n % QK4_1 != 0) {
        return hipErrorInvalidValue;
    }

    // Calculate number of blocks
    int num_blocks = n / QK4_1;

    // Launch kernel with 32 threads per block (QK4_1)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize_q4_1_device),
        dim3(num_blocks),
        dim3(QK4_1),
        20,  // shared memory (2 bytes s_d + 2 bytes s_m + 16 bytes s_qs)
        0,   // stream
        input,
        output,
        n
    );

    return hipGetLastError();
}
