#include "common.hip"
#include <hip/hip_fp16.h>

// Q4_0 constants (must match Rust types.rs)
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

/// Quantize f32 data to Q4_0 format (device kernel)
/// Each thread block processes one 32-element block
__global__ void quantize_q4_0_device(
    const float* __restrict__ input,   // [n] f32 input data
    void* __restrict__ output,         // [n/32 * 18] Q4_0 quantized output
    int n                              // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const float* x = &input[block_idx * QK4_0];

    // Each thread loads one element (QK4_0 = 32 threads)
    float val = x[threadIdx.x];
    float abs_val = fabsf(val);

    // Warp reduction to find max - track both abs and signed value
    float max_abs = abs_val;
    float max_signed = val;  // Track signed value corresponding to abs max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_abs = __shfl_down(max_abs, offset);
        if (other_abs > max_abs) {
            max_abs = other_abs;
            max_signed = __shfl_down(val, offset);  // Get corresponding signed value
        }
    }

    // Thread 0 computes scale (d = max / -8 from llama.cpp)
    __shared__ half s_d;
    if (threadIdx.x == 0) {
        // Use signed max for scale calculation
        float d = (max_abs != 0.0f) ? (max_signed / -8.0f) : -1.0f;
        s_d = __float2half(d);
    }
    __syncthreads();

    // Quantize values (llama.cpp formula: q = round(x/d + 8.5))
    float d = __half2float(s_d);
    int q = (int)roundf(val / d + 8.5f);
    q = (q < 0) ? 0 : (q > 15) ? 15 : q;

    // Pack pairwise into shared memory (2 threads per byte)
    __shared__ uint8_t s_qs[QK4_0 / 2];
    if (threadIdx.x % 2 == 0) {
        s_qs[threadIdx.x / 2] = (uint8_t)q;  // Low 4 bits
    } else {
        s_qs[threadIdx.x / 2] |= ((uint8_t)q) << 4;  // High 4 bits
    }
    __syncthreads();

    // Thread 0 writes output
    if (threadIdx.x == 0) {
        uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q4_0_BLOCK_SIZE;
        memcpy(out, &s_d, sizeof(half));  // Write d (2 bytes)
        memcpy(out + 2, s_qs, QK4_0 / 2);  // Write qs (16 bytes)
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t quantize_q4_0_kernel(
    const float* input,
    void* output,
    int n
) {
    // Validate dimensions
    if (n <= 0) {
        return hipErrorInvalidValue;
    }
    if (n % QK4_0 != 0) {
        return hipErrorInvalidValue;
    }

    // Calculate number of blocks
    int num_blocks = (n + QK4_0 - 1) / QK4_0;

    // Launch kernel with 32 threads per block (QK4_0)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize_q4_0_device),
        dim3(num_blocks),
        dim3(QK4_0),
        18,  // shared memory (2 bytes half + 16 bytes qs)
        0,  // stream
        input,
        output,
        n
    );

    return hipGetLastError();
}
