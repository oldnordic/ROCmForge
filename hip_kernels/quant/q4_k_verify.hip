//! Q4_K accuracy verification kernel
//!
//! Verifies quantization accuracy by computing error metrics.
//! Compares original f32 data with quantize-dequantize round-trip.
//!
//! Metrics computed:
//! - Max absolute error
//! - Mean squared error (MSE)
//! - Relative error

#include "common.hip"
#include <hip/hip_fp16.h>

// ── Constants ─────────────────────────────────────────────────────────────────

#define QK_K 256
#define K_SCALE_SIZE 12

// ── Main Kernel ─────────────────────────────────────────────────────────────────

extern "C" __global__ void verify_q4_k_accuracy_kernel(
    const float* __restrict__ original,  // [n] Original f32 data
    const void* __restrict__ quantized,   // [n/256 * 144] Q4_K quantized data
    float* __restrict__ errors,           // [3] Output: [max_error, mse, relative_error]
    int n                                 // Total number of elements
) {
    // Each thread block processes one 256-element block
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
    const uint8_t* quant_ptr = static_cast<const uint8_t*>(quantized) + block_idx * 144;

    // Read header (only first thread)
    __shared__ float s_d;
    __shared__ float s_dmin;

    if (threadIdx.x == 0) {
        half d_val = *reinterpret_cast<const half*>(quant_ptr);
        s_d = __half2float(d_val);

        half dmin_val = *reinterpret_cast<const half*>(quant_ptr + 2);
        s_dmin = __half2float(dmin_val);
    }
    __syncthreads();

    // Dequantize and compute errors
    const uint8_t* qs_ptr = quant_ptr + 4 + K_SCALE_SIZE;

    float local_max_error = 0.0f;
    float local_mse = 0.0f;
    float local_sum_original = 0.0f;
    float local_sum_error = 0.0f;

    for (int i = threadIdx.x; i < (block_end - block_start); i += blockDim.x) {
        int idx = block_start + i;

        // Dequantize value
        uint8_t packed = qs_ptr[i / 2];
        uint8_t q4;

        if (i % 2 == 0) {
            q4 = packed & 0x0F;
        } else {
            q4 = (packed >> 4) & 0x0F;
        }

        float dequantized = static_cast<float>(q4) / s_d + s_dmin;
        float original_val = original[idx];

        // Compute error
        float error = fabsf(original_val - dequantized);
        local_max_error = fmaxf(local_max_error, error);
        local_mse += error * error;
        local_sum_original += fabsf(original_val);
        local_sum_error += error;
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
            atomicMax(&errors[0], local_max_error);  // max_error
            atomicAdd(&errors[1], local_mse);        // mse
            atomicAdd(&errors[2], local_sum_original);  // sum_original
            atomicAdd(&errors[3], local_sum_error);  // sum_error
        }
    }
}

//! Compute final accuracy metrics
//!
//! Must be called after verify_q4_k_accuracy to compute final values.
//! Output: [max_error, mse, relative_error]
extern "C" __global__ void finalize_q4_k_metrics_kernel(
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
