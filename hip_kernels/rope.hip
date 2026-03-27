#include "common.hip"

/// Rotary Position Embedding (RoPE) kernel
///
/// Applies rotary position embeddings to query/key tensors.
/// For each pair (x[2i], x[2i+1]):
///   x[2i]   = x[2i] * cos(pos * theta_i) - x[2i+1] * sin(pos * theta_i)
///   x[2i+1] = x[2i] * sin(pos * theta_i) + x[2i+1] * cos(pos * theta_i)
///
/// # Arguments
/// * x - Input/output tensor [hidden_size] (modified in-place)
/// * pos - Position index for this token
/// * dim - Hidden size (must be even)
/// * theta_base - Base frequency (typically 10000.0)
__global__ void rope_kernel(
    float* __restrict__ x,
    const int pos,
    const int dim,
    const float theta_base
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= dim / 2) return;

    // Compute theta_i = 1 / (theta_base^(2i/dim))
    const float exponent = (float)(2 * i) / (float)dim;
    const float theta = 1.0f / pow(theta_base, exponent);

    const float angle = (float)pos * theta;
    const float cos_val = cos(angle);
    const float sin_val = sin(angle);

    const int idx0 = 2 * i;
    const int idx1 = 2 * i + 1;

    const float x0 = x[idx0];
    const float x1 = x[idx1];

    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

/// Batched RoPE for prefill (multiple tokens)
///
/// # Arguments
/// * x - Input/output tensor [seq_len, hidden_size] (modified in-place)
/// * start_pos - Starting position for this batch
/// * dim - Hidden size (must be even)
/// * theta_base - Base frequency (typically 10000.0)
/// * seq_len - Number of tokens in this batch
__global__ void rope_batched_kernel(
    float* __restrict__ x,
    const int start_pos,
    const int dim,
    const float theta_base,
    const int seq_len
) {
    const int s = blockIdx.y;  // Sequence index from second dimension
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // Pair index from first dimension

    if (s >= seq_len || i >= dim / 2) return;

    // Get pointer to this sequence's data
    float* row_x = x + (size_t)s * dim;
    const int pos = start_pos + s;

    // Compute theta_i = 1 / (theta_base^(2i/dim))
    const float exponent = (float)(2 * i) / (float)dim;
    const float theta = 1.0f / pow(theta_base, exponent);

    const float angle = (float)pos * theta;
    const float cos_val = cos(angle);
    const float sin_val = sin(angle);

    const int idx0 = 2 * i;
    const int idx1 = 2 * i + 1;

    const float x0 = row_x[idx0];
    const float x1 = row_x[idx1];

    row_x[idx0] = x0 * cos_val - x1 * sin_val;
    row_x[idx1] = x0 * sin_val + x1 * cos_val;
}

// C wrapper functions for dynamic linking
extern "C" {

hipError_t gpu_rope(
    float* d_x,
    const int pos,
    const int dim,
    const float theta_base
) {
    if (dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int block_size = 256;
    const int grid_size = (dim / 2 + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        rope_kernel,
        grid_size, block_size, 0, 0,
        d_x, pos, dim, theta_base
    );

    return hipGetLastError();
}

hipError_t gpu_rope_batched(
    float* d_x,
    const int start_pos,
    const int dim,
    const float theta_base,
    const int seq_len
) {
    if (dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int block_size = 256;
    const int pairs_per_seq = dim / 2;
    const int grid_x = (pairs_per_seq + block_size - 1) / block_size;

    dim3 grid(grid_x, seq_len);

    hipLaunchKernelGGL(
        rope_batched_kernel,
        grid, block_size, 0, 0,
        d_x, start_pos, dim, theta_base, seq_len
    );

    return hipGetLastError();
}

} // extern "C"
