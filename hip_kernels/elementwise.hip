#include "common.hip"

/// Element-wise add kernel: out = x + y
__global__ void add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}

/// Element-wise multiply kernel: out = x * y
__global__ void mul_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * y[i];
    }
}

/// Element-wise scale kernel: out = x * scale
__global__ void scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float scale,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * scale;
    }
}

/// Element-wise add scalar: out = x + scalar
__global__ void add_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float scalar,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + scalar;
    }
}

/// Gelu activation kernel
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void gelu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float val = x[i];
        const float cube = val * val * val;
        const float tanh_arg = 0.7978845608028654f * (val + 0.044715f * cube);
        out[i] = 0.5f * val * (1.0f + tanh(tanh_arg));
    }
}

/// Silu activation kernel: silu(x) = x / (1 + exp(-x))
__global__ void silu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float val = x[i];
        out[i] = val / (1.0f + exp(-val));
    }
}

/// Batched element-wise add for prefill: out[s, i] = x[s, i] + y[i]
/// where x is [seq_len, n] and y is [n] (broadcast)
__global__ void add_batched_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    const int n,
    const int seq_len
) {
    const int s = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;

    if (s < seq_len && i < n) {
        const size_t idx = (size_t)s * n + i;
        out[idx] = x[idx] + y[i];
    }
}

/// Batched element-wise multiply for prefill: out[s, i] = x[s, i] * y[i]
__global__ void mul_batched_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    const int n,
    const int seq_len
) {
    const int s = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;

    if (s < seq_len && i < n) {
        const size_t idx = (size_t)s * n + i;
        out[idx] = x[idx] * y[i];
    }
}

// C wrapper functions for dynamic linking
extern "C" {

hipError_t gpu_add(
    const float* d_x,
    const float* d_y,
    float* d_out,
    const int n
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        add_kernel,
        grid_size, block_size, 0, 0,
        d_x, d_y, d_out, n
    );

    return hipGetLastError();
}

hipError_t gpu_mul(
    const float* d_x,
    const float* d_y,
    float* d_out,
    const int n
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        mul_kernel,
        grid_size, block_size, 0, 0,
        d_x, d_y, d_out, n
    );

    return hipGetLastError();
}

hipError_t gpu_scale(
    const float* d_x,
    float* d_out,
    const float scale,
    const int n
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        scale_kernel,
        grid_size, block_size, 0, 0,
        d_x, d_out, scale, n
    );

    return hipGetLastError();
}

hipError_t gpu_gelu(
    const float* d_x,
    float* d_out,
    const int n
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        gelu_kernel,
        grid_size, block_size, 0, 0,
        d_x, d_out, n
    );

    return hipGetLastError();
}

hipError_t gpu_silu(
    const float* d_x,
    float* d_out,
    const int n
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        silu_kernel,
        grid_size, block_size, 0, 0,
        d_x, d_out, n
    );

    return hipGetLastError();
}

hipError_t gpu_add_batched(
    const float* d_x,
    const float* d_y,
    float* d_out,
    const int n,
    const int seq_len
) {
    const int block_size = 256;
    const int grid_x = (n + block_size - 1) / block_size;

    dim3 grid(grid_x, seq_len);

    hipLaunchKernelGGL(
        add_batched_kernel,
        grid, block_size, 0, 0,
        d_x, d_y, d_out, n, seq_len
    );

    return hipGetLastError();
}

hipError_t gpu_mul_batched(
    const float* d_x,
    const float* d_y,
    float* d_out,
    const int n,
    const int seq_len
) {
    const int block_size = 256;
    const int grid_x = (n + block_size - 1) / block_size;

    dim3 grid(grid_x, seq_len);

    hipLaunchKernelGGL(
        mul_batched_kernel,
        grid, block_size, 0, 0,
        d_x, d_y, d_out, n, seq_len
    );

    return hipGetLastError();
}

} // extern "C"
