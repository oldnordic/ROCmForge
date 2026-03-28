//! Q5_K accuracy verification kernel
//!
//! Verifies quantization accuracy by computing error metrics.
//! Compares original f32 data with quantize-dequantize round-trip.

#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
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

/// Verify Q5_K quantization accuracy (device kernel)
__global__ void verify_q5_k_accuracy_device(
    const float* __restrict__ original,  // [n] Original f32 data
    const void* __restrict__ quantized,   // [n/256 * 176] Q5_K quantized data
    float* __restrict__ errors,           // [4] Output: [max_error, mse, sum_original, sum_error]
    int n                                 // Total number of elements
) {
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;
    const int block_end = min(block_start + QK_K, n);

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
    const uint8_t* quant_ptr = static_cast<const uint8_t*>(quantized) + block_idx * 176;

    // Read header
    half d_val, dmin_val;
    memcpy(&d_val, quant_ptr, sizeof(half));
    memcpy(&dmin_val, quant_ptr + 2, sizeof(half));

    const float d = __half2float(d_val);
    const float dmin = __half2float(dmin_val);

    const uint8_t* scales = quant_ptr + 4;
    const uint8_t* qh = quant_ptr + 4 + K_SCALE_SIZE;
    const uint8_t* ql = quant_ptr + 4 + K_SCALE_SIZE + QK_K / 8;

    float local_max_error = 0.0f;
    float local_mse = 0.0f;
    float local_sum_original = 0.0f;
    float local_sum_error = 0.0f;

    // Dequantize using non-uniform pattern (same as dequantization kernel)
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

        // Process elements for this sub-block pair
        for (int l = threadIdx.x; l < 64 && (j + l) < (block_end - block_start); l += 32) {
            int idx = block_start + j + l;
            float original_val = original[idx];
            float dequantized;

            if (l < 32) {
                // First sub-block
                const uint8_t q4 = ql[l] & 0x0F;
                const uint8_t high_bit = (qh[l] & u1) ? 16 : 0;
                dequantized = d1 * (q4 + high_bit) - min1;
            } else {
                // Second sub-block
                const uint8_t q4 = (ql[l - 32] >> 4) & 0x0F;
                const uint8_t high_bit = (qh[l - 32] & u2) ? 16 : 0;
                dequantized = d2 * (q4 + high_bit) - min2;
            }

            // Compute error
            float error = fabsf(original_val - dequantized);
            local_max_error = fmaxf(local_max_error, error);
            local_mse += error * error;
            local_sum_original += fabsf(original_val);
            local_sum_error += error;
        }

        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }

    // Reduce within warp
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

    // First thread in warp writes to shared memory
    __shared__ float warp_max_errors[32];
    __shared__ float warp_mses[32];
    __shared__ float warp_sum_originals[32];
    __shared__ float warp_sum_errors[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane_id == 0) {
        warp_max_errors[warp_id] = local_max_error;
        warp_mses[warp_id] = local_mse;
        warp_sum_originals[warp_id] = local_sum_original;
        warp_sum_errors[warp_id] = local_sum_error;
    }
    __syncthreads();

    // Final reduction across warps (only first warp)
    if (warp_id == 0) {
        local_max_error = (lane_id < blockDim.x / 32) ? warp_max_errors[lane_id] : 0.0f;
        local_mse = (lane_id < blockDim.x / 32) ? warp_mses[lane_id] : 0.0f;
        local_sum_original = (lane_id < blockDim.x / 32) ? warp_sum_originals[lane_id] : 0.0f;
        local_sum_error = (lane_id < blockDim.x / 32) ? warp_sum_errors[lane_id] : 0.0f;

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
            atomicMax(&errors[0], local_max_error);
            atomicAdd(&errors[1], local_mse);
            atomicAdd(&errors[2], local_sum_original);
            atomicAdd(&errors[3], local_sum_error);
        }
    }
}

/// Compute final accuracy metrics
__global__ void finalize_q5_k_metrics_device(
    const float* __restrict__ errors,
    float* __restrict__ metrics,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_error = errors[0];
        float mse = errors[1] / n;
        float relative_error = errors[2] > 0.0f ? errors[3] / errors[2] : 0.0f;

        metrics[0] = max_error;
        metrics[1] = mse;
        metrics[2] = relative_error;
    }
}

// Dispatch functions (called from Rust)
extern "C" hipError_t verify_q5_k_accuracy_kernel(
    const float* original,
    const void* quantized,
    float* errors,
    int n
) {
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    int num_blocks = (n + QK_K - 1) / QK_K;

    hipError_t memset_result = hipMemset(errors, 0, 4 * sizeof(float));
    if (memset_result != hipSuccess) {
        return memset_result;
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(verify_q5_k_accuracy_device),
        dim3(num_blocks),
        dim3(32),
        0, 0,
        original, quantized, errors, n
    );

    return hipGetLastError();
}

extern "C" hipError_t finalize_q5_k_metrics_kernel(
    const float* errors,
    float* metrics,
    int n
) {
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(finalize_q5_k_metrics_device),
        dim3(1),
        dim3(32),
        0, 0,
        errors, metrics, n
    );

    return hipGetLastError();
}
