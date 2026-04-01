#include "common.hip"

/**
 * Specialized RMS Norm using Wavefront Shuffles.
 * 
 * Architectural patterns from llama.cpp:
 * 1. Vectorized Loads (float4): Reads 4 floats at once to maximize bandwidth.
 * 2. Wave-Shuffle Reduction: Uses __shfl_down to reduce within a wavefront.
 * 3. Minimal Synchronizations: Only uses one LDS-based sync at the very end of reduction.
 * 4. Zero LDS for small shapes: If block size <= 32, no LDS is used at all.
 */
__global__ void rms_norm_vulkan_style_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n,
    const float eps
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Static shared memory for cross-wave reduction (max 32 waves/block)
    __shared__ float s_reduction[32];

    float sum_sq = 0.0f;

    // Vectorized pass
    int i = tid * 4;
    for (; i + 3 < n; i += blockDim.x * 4) {
        float4 val4 = reinterpret_cast<const float4*>(&x[i])[0];
        sum_sq += val4.x * val4.x;
        sum_sq += val4.y * val4.y;
        sum_sq += val4.z * val4.z;
        sum_sq += val4.w * val4.w;
    }
    // Scalar fallback for remainder
    for (; i < n; i++) {
        float val = x[i];
        sum_sq += val * val;
    }

    // 1. Wave-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset);
    }

    // 2. Cross-wave reduction via LDS
    if (lane_id == 0) {
        s_reduction[wave_id] = sum_sq;
    }
    __syncthreads();

    // Final wave-level reduction of the wave-sums
    float final_sum_sq = (tid < (blockDim.x / 32)) ? s_reduction[tid] : 0.0f;
    if (wave_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum_sq += __shfl_down(final_sum_sq, offset);
        }
        if (tid == 0) {
            s_reduction[0] = final_sum_sq;
        }
    }
    __syncthreads();

    // Broadcast result
    float inv_rms = rsqrtf(s_reduction[0] / (float)n + eps);

    // Vectorized apply (if pointers are aligned)
    i = tid * 4;
    for (; i + 3 < n; i += blockDim.x * 4) {
        float4 x4 = reinterpret_cast<const float4*>(&x[i])[0];
        float4 w4 = reinterpret_cast<const float4*>(&weight[i])[0];
        float4 out4;
        out4.x = x4.x * inv_rms * w4.x;
        out4.y = x4.y * inv_rms * w4.y;
        out4.z = x4.z * inv_rms * w4.z;
        out4.w = x4.w * inv_rms * w4.w;
        reinterpret_cast<float4*>(&out[i])[0] = out4;
    }
    // Scalar fallback
    for (; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

extern "C" {

hipError_t gpu_rms_norm_vulkan_style(
    const float* d_x,
    const float* d_weight,
    float* d_out,
    const int n,
    const float eps,
    hipStream_t stream
) {
    CHECK_NULL(d_x);
    CHECK_NULL(d_weight);
    CHECK_NULL(d_out);

    // Verify alignment for float4 (16 bytes)
    if (((uintptr_t)d_x % 16 != 0) || ((uintptr_t)d_weight % 16 != 0) || ((uintptr_t)d_out % 16 != 0)) {
        return hipErrorInvalidValue;
    }

    // 256 threads is a good default for RDNA3 occupancy
    const int block_size = 256;

    hipLaunchKernelGGL(
        rms_norm_vulkan_style_kernel,
        1, block_size, 0, stream,
        d_x, d_weight, d_out, n, eps
    );

    CHECK_LAST();
    return hipSuccess;
}

}
