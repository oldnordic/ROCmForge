#include "common.hip"
#include <hip/hip_fp16.h>

#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

/// Dequantize Q4_1 block to f32 (device kernel)
/// Q4_1 block layout: d (half 2B) + m (half 2B) + qs[16] (16B) = 20 bytes
/// Formula: y = q * d + m (NOT offset-subtraction like Q4_0)
__global__ void dequantize_q4_1_device(
    const void* __restrict__ input,   // [n/32 * 20] Q4_1 quantized input
    float* __restrict__ output,       // [n] f32 output data
    int n                              // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const uint8_t* in = static_cast<const uint8_t*>(input) + block_idx * Q4_1_BLOCK_SIZE;

    // Read scale (d) and min (m) — first 4 bytes
    half d_half, m_half;
    memcpy(&d_half, in, sizeof(half));
    memcpy(&m_half, in + 2, sizeof(half));
    float d = __half2float(d_half);
    float m = __half2float(m_half);

    // Read quantized values (bytes 4-19)
    const uint8_t* qs = in + 4;
    float* out = &output[block_idx * QK4_1];

    // Sequential nibble unpacking: low nibble = first 16 values, high nibble = next 16
    if (threadIdx.x < 16) {
        uint8_t q = qs[threadIdx.x] & 0x0F;
        out[threadIdx.x] = (float)q * d + m;
    } else {
        uint8_t q = qs[threadIdx.x - 16] >> 4;
        out[threadIdx.x] = (float)q * d + m;
    }
}

/// Batched dequantization for parallel processing
__global__ void dequantize_q4_1_batched_device(
    const void* __restrict__ input,   // [batch_size][n/32 * 20] Q4_1 quantized input
    float* __restrict__ output,       // [batch_size][n] f32 output data
    int n,                             // Elements per batch
    int batch_size                     // Number of batches
) {
    const int batch_idx = blockIdx.x;
    const int block_idx = threadIdx.x;  // Each thread processes one block

    const int num_blocks = (n + QK4_1 - 1) / QK4_1;
    if (block_idx >= num_blocks) return;

    const int global_block_idx = batch_idx * num_blocks + block_idx;
    const uint8_t* in = static_cast<const uint8_t*>(input) + global_block_idx * Q4_1_BLOCK_SIZE;

    // Read scale (d) and min (m)
    half d_half, m_half;
    memcpy(&d_half, in, sizeof(half));
    memcpy(&m_half, in + 2, sizeof(half));
    float d = __half2float(d_half);
    float m = __half2float(m_half);

    // Read and dequantize values
    const uint8_t* qs = in + 4;
    float* out = &output[global_block_idx * QK4_1];

    // Sequential unpacking: qs[i] low nibble → element i, high nibble → element i + 16
    for (int i = 0; i < QK4_1 / 2; ++i) {
        uint8_t q0 = qs[i] & 0x0F;
        uint8_t q1 = qs[i] >> 4;
        out[i]                = (float)q0 * d + m;  // First half (elements 0..15)
        out[i + QK4_1 / 2]   = (float)q1 * d + m;  // Second half (elements 16..31)
    }
}

// ── Dispatch Functions (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t dequantize_q4_1_kernel(
    const void* input,
    float* output,
    int n
) {
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK4_1 - 1) / QK4_1;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q4_1_device),
        dim3(num_blocks),
        dim3(QK4_1),
        0, 0,
        input,
        output,
        n
    );

    return hipGetLastError();
}

extern "C" hipError_t dequantize_q4_1_batched_kernel(
    const void* input,
    float* output,
    int n,
    int batch_size
) {
    if (n <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK4_1 - 1) / QK4_1;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q4_1_batched_device),
        dim3(batch_size),
        dim3(num_blocks),
        0, 0,
        input,
        output,
        n,
        batch_size
    );

    return hipGetLastError();
}
