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
    // Shared memory for parallel reduction
    extern __shared__ float shared_sq[];

    const int tid = threadIdx.x;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }
    shared_sq[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq[tid] += shared_sq[tid + s];
        }
        __syncthreads();
    }

    // Compute inverse RMS
    float inv_rms = rsqrt(shared_sq[0] / (float)n + eps);

    // Apply normalization
    for (int i = tid; i < n; i += blockDim.x) {
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
    for (int i = tid; i < n; i += blockDim.x) {
        float val = row_x[i];
        sum_sq += val * val;
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
    float inv_rms = rsqrt(shared_sq[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        row_out[i] = row_x[i] * inv_rms * weight[i];
    }
}

// C wrapper functions for dynamic linking
extern "C" {

hipError_t gpu_rms_norm(
    const float* d_x,
    const float* d_weight,
    float* d_out,
    const int n,
    const float eps
) {
    const int block_size = 256;
    const int smem_size = block_size * sizeof(float);

    hipLaunchKernelGGL(
        rms_norm_kernel,
        1, block_size, smem_size, 0,
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
