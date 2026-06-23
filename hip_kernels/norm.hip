#include "common.hip"

/// RMS normalization kernel
///
/// Computes: out = x * rsqrt(mean(x^2) + eps) * weight
///
/// # Safety
/// - All threads participate in shared memory reduction
/// - Bounds checking via blockDim/gridDim
/// - Uses __syncthreads() for barrier synchronization
///
/// # Arguments
/// * x - Input tensor [n]
/// * weight - Scale/weight tensor [n]
/// * out - Output tensor [n]
/// * n - Number of elements
/// * eps - Epsilon for numerical stability
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n,
    const float eps
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE_COMPTIME;
    const int lane_id = tid % WARP_SIZE_COMPTIME;

    // Shared memory for cross-wave reduction
    // Number of waves = blockDim.x / WARP_SIZE_COMPTIME
    // For wave32 (block_size=256): 256 / 32 = 8 waves
    constexpr int kWavesPerBlock = 256 / 32;
    __shared__ float s_reduction[kWavesPerBlock];

    float sum_sq = 0.0f;

    int i = tid * 4;
    for (; i + 3 < n; i += blockDim.x * 4) {
        float4 val4 = reinterpret_cast<const float4*>(&x[i])[0];
        sum_sq += val4.x * val4.x;
        sum_sq += val4.y * val4.y;
        sum_sq += val4.z * val4.z;
        sum_sq += val4.w * val4.w;
    }
    for (; i < n; i++) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Wave-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE_COMPTIME / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, WARP_SIZE_COMPTIME);
    }

    if (lane_id == 0) {
        s_reduction[wave_id] = sum_sq;
    }
    __syncthreads();

    // Cross-wave reduction
    float final_sum_sq = (tid < (blockDim.x / WARP_SIZE_COMPTIME)) ? s_reduction[tid] : 0.0f;
    if (wave_id == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE_COMPTIME / 2; offset > 0; offset >>= 1) {
            final_sum_sq += __shfl_down(final_sum_sq, offset, WARP_SIZE_COMPTIME);
        }
        if (tid == 0) {
            s_reduction[0] = final_sum_sq;
        }
    }
    __syncthreads();

    float inv_rms = rsqrtf(s_reduction[0] / (float)n + eps);

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
    for (; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/// Batched RMS norm for prefill (multiple sequences)
///
/// # Arguments
/// * x - Input tensor [seq_len, n]
/// * weight - Scale/weight tensor [n]
/// * out - Output tensor [seq_len, n]
/// * n - Hidden size
/// * eps - Epsilon for numerical stability
/// * seq_len - Number of sequences to process
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rms_norm_batched_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n,
    const float eps,
    const int seq_len
) {
    extern __shared__ float shared_sq[];

    const int s = blockIdx.x;
    const int tid = threadIdx.x;

    // Bounds check
    if (s >= seq_len) return;

    // Get pointers for this sequence
    const float* row_x = x + (size_t)s * n;
    float* row_out = out + (size_t)s * n;

    // Compute sum of squares for this row
    float sum_sq = 0.0f;
    if (n % 4 == 0) {
        int i = tid * 4;
        for (; i + 3 < n; i += blockDim.x * 4) {
            float4 val4 = reinterpret_cast<const float4*>(&row_x[i])[0];
            sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
        }
    } else {
        for (int i = tid; i < n; i += blockDim.x) {
            float val = row_x[i];
            sum_sq += val * val;
        }
    }
    shared_sq[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            shared_sq[tid] += shared_sq[tid + step];
        }
        __syncthreads();
    }

    // Compute and apply normalization
    float inv_n = 1.0f / (float)n;
    float inv_rms = rsqrt(shared_sq[0] * inv_n + eps);
    if (n % 4 == 0) {
        int i = tid * 4;
        for (; i + 3 < n; i += blockDim.x * 4) {
            float4 x4 = reinterpret_cast<const float4*>(&row_x[i])[0];
            float4 w4 = reinterpret_cast<const float4*>(&weight[i])[0];
            float4 out4 = make_float4(x4.x * inv_rms * w4.x, x4.y * inv_rms * w4.y, x4.z * inv_rms * w4.z, x4.w * inv_rms * w4.w);
            reinterpret_cast<float4*>(&row_out[i])[0] = out4;
        }
    } else {
        for (int i = tid; i < n; i += blockDim.x) {
            row_out[i] = row_x[i] * inv_rms * weight[i];
        }
    }
}

// C wrapper functions for dynamic linking
extern "C" {

hipError_t gpu_rms_norm(
    const float* d_x,
    const float* d_weight,
    float* d_out,
    const int n,
    const float eps,
    hipStream_t stream
) {
    const int block_size = 256;

    hipLaunchKernelGGL(
        rms_norm_kernel,
        1, block_size, 0, stream,
        d_x, d_weight, d_out, n, eps
    );

    return hipGetLastError();
}

/// Reference RMS normalization kernel.
///
/// Matches the CPU scalar path exactly: a single thread computes a serial sum
/// of squares, then `rsqrtf(mean + eps)`, then scales elementwise.  This avoids
/// any wave-reduction ordering differences between GPU implementations.
__global__ void rms_norm_reference_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n,
    const float eps
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        const float v = x[i];
        sum_sq += v * v;
    }
    const float inv_rms = rsqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

hipError_t gpu_rms_norm_reference(
    const float* d_x,
    const float* d_weight,
    float* d_out,
    const int n,
    const float eps,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        rms_norm_reference_kernel,
        1, 1, 0, stream,
        d_x, d_weight, d_out, n, eps
    );
    return hipGetLastError();
}

hipError_t gpu_rms_norm_batched(
    const float* d_x,
    const float* d_weight,
    float* d_out,
    const int n,
    const float eps,
    const int seq_len
) {
    const int block_size = 256;
    const int smem_size = block_size * sizeof(float);

    hipLaunchKernelGGL(
        rms_norm_batched_kernel,
        seq_len, block_size, smem_size, 0,
        d_x, d_weight, d_out, n, eps, seq_len
    );

    return hipGetLastError();
}

} // extern "C"
