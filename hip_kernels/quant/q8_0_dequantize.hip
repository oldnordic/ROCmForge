#include "common.hip"

#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34

//! Q8_0 dequantization kernel
//!
//! Dequantizes Q8_0 format back to f32.
//! Reference-inspired from llama.cpp ggml_quants.c dequantize_row_q8_0.
//!
//! Q8_0 format (34 bytes for 32 f32 values):
//! - d: f16 scale (2 bytes)
//! - qs[32]: 8-bit quantized values (32 bytes)
//!
//! Algorithm:
//! 1. Read block header (d = scale)
//! 2. Reconstruct f32 values: val = qs[i] * d

/// Dequantize Q8_0 block to 32 f32 values (device kernel)
__global__ void dequantize_q8_0_device(
    const void* __restrict__ input,   // [n/32 * 34] Q8_0 quantized input
    float* __restrict__ output,       // [n] f32 output data
    int n                              // Total number of elements
) {
    // Each thread block processes one 32-element block
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK8_0;
    const int block_end = min(block_start + QK8_0, n);

    // Shared memory for scale (all threads need it)
    __shared__ half s_scale;

    // Thread 0 reads the scale from block header
    if (threadIdx.x == 0) {
        const uint8_t* in = static_cast<const uint8_t*>(input) + block_idx * Q8_0_BLOCK_SIZE;
        s_scale = *reinterpret_cast<const half*>(in);  // First 2 bytes = scale (f16)
    }
    __syncthreads();

    // Convert scale to f32 once
    const float scale = __half2float(s_scale);

    // Read quantized values and dequantize
    const int8_t* qs = reinterpret_cast<const int8_t*>(
        static_cast<const uint8_t*>(input) + block_idx * Q8_0_BLOCK_SIZE + 2
    );

    // Each thread processes one element (32 threads = QK8_0)
    if (threadIdx.x < (block_end - block_start)) {
        output[block_start + threadIdx.x] = static_cast<float>(qs[threadIdx.x]) * scale;
    }
}

/// Batched dequantization for parallel processing (device kernel)
__global__ void dequantize_q8_0_batched_device(
    const void* __restrict__ input,   // [batch_size][n/32 * 34]
    float* __restrict__ output,       // [batch_size][n]
    int n,                             // Elements per batch
    int batch_size                     // Number of batches
) {
    const int batch_idx = blockIdx.y;
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK8_0;
    const int block_end = min(block_start + QK8_0, n);

    // Calculate batch offsets
    const size_t blocks_per_batch = (n + QK8_0 - 1) / QK8_0;
    const size_t batch_input_offset = static_cast<size_t>(batch_idx) * blocks_per_batch * Q8_0_BLOCK_SIZE;
    const size_t batch_output_offset = static_cast<size_t>(batch_idx) * n;

    // Shared memory for scale
    __shared__ half s_scale;

    // Thread 0 reads the scale
    if (threadIdx.x == 0) {
        const uint8_t* in = static_cast<const uint8_t*>(input) + batch_input_offset + block_idx * Q8_0_BLOCK_SIZE;
        s_scale = *reinterpret_cast<const half*>(in);
    }
    __syncthreads();

    const float scale = __half2float(s_scale);

    // Read and dequantize values
    const int8_t* qs = reinterpret_cast<const int8_t*>(
        static_cast<const uint8_t*>(input) + batch_input_offset + block_idx * Q8_0_BLOCK_SIZE + 2
    );
    float* out_ptr = output + batch_output_offset;

    // Each thread processes one element
    if (threadIdx.x < (block_end - block_start)) {
        out_ptr[block_start + threadIdx.x] = static_cast<float>(qs[threadIdx.x]) * scale;
    }
}

// ── Dispatch Functions (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t dequantize_q8_0_kernel(
    const void* input,
    float* output,
    int n
) {
    // Validate dimensions
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    // Calculate number of blocks
    int num_blocks = (n + QK8_0 - 1) / QK8_0;

    // Launch kernel with 32 threads per block (QK8_0)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q8_0_device),
        dim3(num_blocks),
        dim3(QK8_0),
        0,  // shared memory
        0,  // stream
        input,
        output,
        n
    );

    return hipGetLastError();
}

extern "C" hipError_t dequantize_q8_0_batched_kernel(
    const void* input,
    float* output,
    int n,
    int batch_size
) {
    // Validate dimensions
    if (n <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }

    // Calculate grid dimensions
    int blocks_per_batch = (n + QK8_0 - 1) / QK8_0;
    dim3 grid(blocks_per_batch, batch_size);

    // Launch kernel with 32 threads per block (QK8_0)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dequantize_q8_0_batched_device),
        grid,
        dim3(QK8_0),
        0,  // shared memory
        0,  // stream
        input,
        output,
        n,
        batch_size
    );

    return hipGetLastError();
}
