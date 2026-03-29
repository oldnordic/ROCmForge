//! Q4_1 accuracy verification kernel
//!
//! Verifies quantization accuracy by computing error metrics.
//! Compares original f32 data with quantize-dequantize round-trip.
//!
//! Q4_1 block layout (20 bytes):
//!   d  (half, 2B) - scale
//!   m  (half, 2B) - minimum / offset
//!   qs (16B)      - 32 nibbles packed into 16 bytes
//!
//! Dequantization formula: y = q * d + m
//!
//! Metrics computed:
//! - Max absolute error
//! - Mean squared error (MSE)
//! - Relative error

#include "common.hip"
#include <hip/hip_fp16.h>

// ── Constants ─────────────────────────────────────────────────────────────────

#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

// ── Main Kernel ─────────────────────────────────────────────────────────────────

__global__ void verify_q4_1_accuracy_device(
    const float* __restrict__ original,  // [n] Original f32 data
    const void* __restrict__ quantized,   // [n/32 * 20] Q4_1 quantized data
    float* __restrict__ errors,           // [4] Output: [max_error, mse, sum_original, sum_error]
    int n                                 // Total number of elements
) {
    // Each thread block processes one 32-element block
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK4_1;
    const int block_end = min(block_start + QK4_1, n);

    // Shared memory for reduction
    __shared__ float s_max_error;
    __shared__ float s_mse;
    __shared__ float s_sum_original;
    __shared__ float s_sum_error;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        s_max_error = 0.0f;
        s_mse = 0.0f;
        s_sum_original = 0.0f;
        s_sum_error = 0.0f;
    }
    __syncthreads();

    // Read quantized block and dequantize on-the-fly
    const uint8_t* quant_ptr = static_cast<const uint8_t*>(quantized) + block_idx * Q4_1_BLOCK_SIZE;

    // Read scale (d) and offset (m) — header is 4 bytes: d (half) + m (half)
    __shared__ float s_d;
    __shared__ float s_m;

    if (threadIdx.x == 0) {
        half d_half;
        half m_half;
        memcpy(&d_half, quant_ptr, sizeof(half));
        memcpy(&m_half, quant_ptr + 2, sizeof(half));
        s_d = __half2float(d_half);
        s_m = __half2float(m_half);
    }
    __syncthreads();

    // Dequantize and compute errors
    const uint8_t* qs = quant_ptr + 4;

    float local_max_error = 0.0f;
    float local_mse = 0.0f;
    float local_sum_original = 0.0f;
    float local_sum_error = 0.0f;

    // Each thread processes one element (32 threads = QK4_1 = 1 wavefront)
    if (threadIdx.x < (block_end - block_start)) {
        int idx = block_start + threadIdx.x;

        uint8_t q = unpack_nibble_sequential(qs, threadIdx.x);
        float dequantized = float(q) * s_d + s_m;

        float original_val = original[idx];

        // Compute error
        float error = fabsf(original_val - dequantized);
        local_max_error = error;
        local_mse = error * error;
        local_sum_original = fabsf(original_val);
        local_sum_error = error;
    }

    // Warp reduction (32 threads = 1 RDNA wavefront)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down(local_max_error, offset);
        float other_mse = __shfl_down(local_mse, offset);
        float other_sum_orig = __shfl_down(local_sum_original, offset);
        float other_sum_err = __shfl_down(local_sum_error, offset);

        local_max_error = fmaxf(local_max_error, other_max);
        local_mse += other_mse;
        local_sum_original += other_sum_orig;
        local_sum_error += other_sum_err;
    }

    // First thread writes global results
    if (threadIdx.x == 0) {
        atomicMax(&errors[0], local_max_error);      // max_error
        atomicAdd(&errors[1], local_mse);            // mse
        atomicAdd(&errors[2], local_sum_original);   // sum_original
        atomicAdd(&errors[3], local_sum_error);      // sum_error
    }
}

//! Compute final accuracy metrics (device kernel)
//!
//! Must be called after verify_q4_1_accuracy to compute final values.
//! Output: [max_error, mse, relative_error]
__global__ void finalize_q4_1_metrics_device(
    float* __restrict__ errors,     // [4] Input: [max_error, mse, sum_original, sum_error]
    float* __restrict__ metrics,    // [3] Output: [max_error, mse, relative_error]
    int n                           // Total number of elements (for MSE normalization)
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_error = errors[0];
        float mse = errors[1] / n;  // Normalize MSE by element count
        float relative_error = errors[2] > 0.0f ? errors[3] / errors[2] : 0.0f;

        metrics[0] = max_error;
        metrics[1] = mse;
        metrics[2] = relative_error;
    }
}

// ── Dispatch Functions (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t verify_q4_1_accuracy_kernel(
    const float* original,
    const void* quantized,
    float* errors,
    int n
) {
    // Validate dimensions
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    // Calculate number of blocks
    int num_blocks = (n + QK4_1 - 1) / QK4_1;

    // Launch kernel with 32 threads per block (QK4_1)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(verify_q4_1_accuracy_device),
        dim3(num_blocks),
        dim3(QK4_1),
        0,  // shared memory
        0,  // stream
        original,
        quantized,
        errors,
        n
    );

    return hipGetLastError();
}

extern "C" hipError_t finalize_q4_1_metrics_kernel(
    const float* errors,
    float* metrics,
    int n
) {
    // Validate dimensions
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    // Launch kernel with 1 block, 1 thread (scalar operation)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(finalize_q4_1_metrics_device),
        dim3(1),
        dim3(1),
        0,  // shared memory
        0,  // stream
        const_cast<float*>(errors),
        metrics,
        n
    );

    return hipGetLastError();
}
