#include "common.hip"
#include <hip/hip_fp16.h>

// Q4_0 constants (must match Rust types.rs)
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

/// Dequantize Q4_0 block to f32 (device kernel)
__global__ void dequantize_q4_0_device(
    const void* __restrict__ input,   // [n/32 * 18] Q4_0 quantized input
    float* __restrict__ output,       // [n] f32 output data
    int n                              // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const uint8_t* in = static_cast<const uint8_t*>(input) + block_idx * Q4_0_BLOCK_SIZE;

    // Read scale (first 2 bytes)
    half d_half;
    memcpy(&d_half, in, sizeof(half));
    float d = __half2float(d_half);

    // Read quantized values (bytes 2-17)
    const uint8_t* qs = in + 2;
    float* out = &output[block_idx * QK4_0];

    uint8_t q = unpack_nibble_sequential(qs, threadIdx.x);

    // Apply offset and scale (llama.cpp: y = (q - 8) * d)
    out[threadIdx.x] = (float(q) - 8.0f) * d;
}

/// Batched dequantization for parallel processing
__global__ void dequantize_q4_0_batched_device(
    const void* __restrict__ input,   // [batch_size][n/32 * 18] Q4_0 quantized input
    float* __restrict__ output,       // [batch_size][n] f32 output data
    int n,                             // Elements per batch
    int batch_size                     // Number of batches
) {
    const int batch_idx = blockIdx.x;
    const int block_idx = threadIdx.x;  // Each thread processes one block

    const int num_blocks = (n + QK4_0 - 1) / QK4_0;
    if (block_idx >= num_blocks) return;

    const int global_block_idx = batch_idx * num_blocks + block_idx;
    const uint8_t* in = static_cast<const uint8_t*>(input) + global_block_idx * Q4_0_BLOCK_SIZE;

    // Read scale
    half d_half;
    memcpy(&d_half, in, sizeof(half));
    float d = __half2float(d_half);

    // Read and dequantize values
    const uint8_t* qs = in + 2;
    float* out = &output[global_block_idx * QK4_0];

    for (int i = 0; i < QK4_0 / 2; ++i) {
        uint8_t q0 = qs[i] & 0x0F;
        uint8_t q1 = qs[i] >> 4;

        out[i] = (float(q0) - 8.0f) * d;
        out[i + QK4_0 / 2] = (float(q1) - 8.0f) * d;
    }
}

// ── Dispatch Functions (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t dequantize_q4_0_kernel(
    const void* input,
    float* output,
    int n
) {
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK4_0 - 1) / QK4_0;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q4_0_device),
        dim3(num_blocks),
        dim3(QK4_0),
        0, 0,
        input,
        output,
        n
    );

    return hipGetLastError();
}

extern "C" hipError_t dequantize_q4_0_batched_kernel(
    const void* input,
    float* output,
    int n,
    int batch_size
) {
    if (n <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK4_0 - 1) / QK4_0;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q4_0_batched_device),
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
