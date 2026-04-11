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

/// Zero-fill kernel: initialize GPU memory to zeros
__global__ void zero_fill_kernel(
    float* __restrict__ ptr,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[i] = 0.0f;
    }
}

/// Increment decode state in place: state = [pos, seq_len] -> [pos + 1, seq_len + 1]
__global__ void increment_decode_state_kernel(
    int* __restrict__ state
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        state[0] += 1;
        state[1] += 1;
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;

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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;

    if (s < seq_len && i < n) {
        const size_t idx = (size_t)s * n + i;
        out[idx] = x[idx] * y[i];
    }
}

constexpr int ARGMAX_ITEMS_PER_THREAD = 4;
constexpr int ARGMAX_ITEMS_PER_BLOCK = BLOCK_SIZE * ARGMAX_ITEMS_PER_THREAD;
constexpr float ARGMAX_NEG_INF = -3.402823466e+38F;
constexpr int Q4_0_EMBED_BLOCK_ELEMS = 32;
constexpr int Q4_0_EMBED_BLOCK_BYTES = 18;
constexpr int Q8_0_EMBED_BLOCK_ELEMS = 32;
constexpr int Q8_0_EMBED_BLOCK_BYTES = 34;

__device__ inline void update_argmax(
    const float candidate_value,
    const int candidate_index,
    float* best_value,
    int* best_index
) {
    if (candidate_value > *best_value ||
        (candidate_value == *best_value && candidate_index > *best_index)) {
        *best_value = candidate_value;
        *best_index = candidate_index;
    }
}

__global__ void argmax_stage1_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_values,
    int* __restrict__ partial_indices,
    const int n
) {
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * ARGMAX_ITEMS_PER_BLOCK;

    float best_value = ARGMAX_NEG_INF;
    int best_index = 0;

    #pragma unroll
    for (int item = 0; item < ARGMAX_ITEMS_PER_THREAD; ++item) {
        const int idx = block_start + tid + item * BLOCK_SIZE;
        if (idx < n) {
            update_argmax(input[idx], idx, &best_value, &best_index);
        }
    }

    __shared__ float shared_values[BLOCK_SIZE];
    __shared__ int shared_indices[BLOCK_SIZE];
    shared_values[tid] = best_value;
    shared_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_argmax(
                shared_values[tid + stride],
                shared_indices[tid + stride],
                &shared_values[tid],
                &shared_indices[tid]
            );
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_values[blockIdx.x] = shared_values[0];
        partial_indices[blockIdx.x] = shared_indices[0];
    }
}

__global__ void argmax_finalize_kernel(
    const float* __restrict__ partial_values,
    const int* __restrict__ partial_indices,
    int* __restrict__ output_index,
    const int n
) {
    const int tid = threadIdx.x;

    float best_value = ARGMAX_NEG_INF;
    int best_index = 0;

    for (int idx = tid; idx < n; idx += blockDim.x) {
        update_argmax(partial_values[idx], partial_indices[idx], &best_value, &best_index);
    }

    __shared__ float shared_values[BLOCK_SIZE];
    __shared__ int shared_indices[BLOCK_SIZE];
    shared_values[tid] = best_value;
    shared_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_argmax(
                shared_values[tid + stride],
                shared_indices[tid + stride],
                &shared_values[tid],
                &shared_indices[tid]
            );
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_index[0] = shared_indices[0];
    }
}

__global__ void embed_q8_0_token_kernel(
    const uint8_t* __restrict__ embedding,
    float* __restrict__ out,
    const int hidden_size,
    const int vocab_size,
    const int token_id
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size || token_id < 0 || token_id >= vocab_size) {
        return;
    }

    const int blocks_per_token = hidden_size / Q8_0_EMBED_BLOCK_ELEMS;
    const int block_idx = i / Q8_0_EMBED_BLOCK_ELEMS;
    const int elem_idx = i % Q8_0_EMBED_BLOCK_ELEMS;
    const size_t token_offset =
        static_cast<size_t>(token_id) * static_cast<size_t>(blocks_per_token) * Q8_0_EMBED_BLOCK_BYTES;
    const uint8_t* block = embedding + token_offset + block_idx * Q8_0_EMBED_BLOCK_BYTES;

    const half scale_half = *reinterpret_cast<const half*>(block);
    const float scale = __half2float(scale_half);
    const int8_t q = reinterpret_cast<const int8_t*>(block + 2)[elem_idx];
    out[i] = static_cast<float>(q) * scale;
}

__global__ void embed_q8_0_batch_kernel(
    const uint8_t* __restrict__ embedding,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    const int hidden_size,
    const int vocab_size,
    const int seq_len
) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq_len || i >= hidden_size) {
        return;
    }

    const int token_id = token_ids[s];
    if (token_id < 0 || token_id >= vocab_size) {
        return;
    }

    const int blocks_per_token = hidden_size / Q8_0_EMBED_BLOCK_ELEMS;
    const int block_idx = i / Q8_0_EMBED_BLOCK_ELEMS;
    const int elem_idx = i % Q8_0_EMBED_BLOCK_ELEMS;
    const size_t token_offset =
        static_cast<size_t>(token_id) * static_cast<size_t>(blocks_per_token) * Q8_0_EMBED_BLOCK_BYTES;
    const uint8_t* block = embedding + token_offset + block_idx * Q8_0_EMBED_BLOCK_BYTES;
    const half scale_half = *reinterpret_cast<const half*>(block);
    const float scale = __half2float(scale_half);
    const int8_t q = reinterpret_cast<const int8_t*>(block + 2)[elem_idx];

    out[static_cast<size_t>(s) * hidden_size + i] = static_cast<float>(q) * scale;
}

__global__ void embed_q4_0_token_kernel(
    const uint8_t* __restrict__ embedding,
    float* __restrict__ out,
    const int hidden_size,
    const int vocab_size,
    const int token_id
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size || token_id < 0 || token_id >= vocab_size) {
        return;
    }

    const int blocks_per_token = hidden_size / Q4_0_EMBED_BLOCK_ELEMS;
    const int block_idx = i / Q4_0_EMBED_BLOCK_ELEMS;
    const int elem_idx = i % Q4_0_EMBED_BLOCK_ELEMS;
    const size_t token_offset =
        static_cast<size_t>(token_id) * static_cast<size_t>(blocks_per_token) * Q4_0_EMBED_BLOCK_BYTES;
    const uint8_t* block = embedding + token_offset + block_idx * Q4_0_EMBED_BLOCK_BYTES;

    const half scale_half = *reinterpret_cast<const half*>(block);
    const float scale = __half2float(scale_half);
    const uint8_t packed = block[2 + (elem_idx & 15)];
    const int q = (elem_idx < 16) ? ((packed & 0x0F) - 8) : ((packed >> 4) - 8);
    out[i] = static_cast<float>(q) * scale;
}

__global__ void embed_q4_0_batch_kernel(
    const uint8_t* __restrict__ embedding,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    const int hidden_size,
    const int vocab_size,
    const int seq_len
) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq_len || i >= hidden_size) {
        return;
    }

    const int token_id = token_ids[s];
    if (token_id < 0 || token_id >= vocab_size) {
        return;
    }

    const int blocks_per_token = hidden_size / Q4_0_EMBED_BLOCK_ELEMS;
    const int block_idx = i / Q4_0_EMBED_BLOCK_ELEMS;
    const int elem_idx = i % Q4_0_EMBED_BLOCK_ELEMS;
    const size_t token_offset =
        static_cast<size_t>(token_id) * static_cast<size_t>(blocks_per_token) * Q4_0_EMBED_BLOCK_BYTES;
    const uint8_t* block = embedding + token_offset + block_idx * Q4_0_EMBED_BLOCK_BYTES;

    const half scale_half = *reinterpret_cast<const half*>(block);
    const float scale = __half2float(scale_half);
    const uint8_t packed = block[2 + (elem_idx & 15)];
    const int q = (elem_idx < 16) ? ((packed & 0x0F) - 8) : ((packed >> 4) - 8);
    out[static_cast<size_t>(s) * hidden_size + i] = static_cast<float>(q) * scale;
}

// C wrapper functions for dynamic linking
extern "C" {

hipError_t gpu_mul_on_stream(
    const float* d_x,
    const float* d_y,
    float* d_out,
    const int n,
    hipStream_t stream
);

hipError_t gpu_silu_on_stream(
    const float* d_x,
    float* d_out,
    const int n,
    hipStream_t stream
);

hipError_t gpu_add(
    const float* d_x,
    const float* d_y,
    float* d_out,
    const int n,
    hipStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        add_kernel,
        grid_size, block_size, 0, stream,
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
    return gpu_mul_on_stream(d_x, d_y, d_out, n, 0);
}

hipError_t gpu_mul_on_stream(
    const float* d_x,
    const float* d_y,
    float* d_out,
    const int n,
    hipStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        mul_kernel,
        grid_size, block_size, 0, stream,
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
    return gpu_silu_on_stream(d_x, d_out, n, 0);
}

hipError_t gpu_silu_on_stream(
    const float* d_x,
    float* d_out,
    const int n,
    hipStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        silu_kernel,
        grid_size, block_size, 0, stream,
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

hipError_t gpu_argmax_f32_on_stream(
    const float* d_input,
    float* d_partial_values,
    int* d_partial_indices,
    int* d_output_index,
    const int n,
    hipStream_t stream
) {
    if (d_input == nullptr || d_partial_values == nullptr ||
        d_partial_indices == nullptr || d_output_index == nullptr || n <= 0) {
        return hipErrorInvalidValue;
    }

    const int block_size = BLOCK_SIZE;
    const int grid_size = (n + ARGMAX_ITEMS_PER_BLOCK - 1) / ARGMAX_ITEMS_PER_BLOCK;

    hipLaunchKernelGGL(
        argmax_stage1_kernel,
        grid_size, block_size, 0, stream,
        d_input, d_partial_values, d_partial_indices, n
    );

    hipError_t launch_error = hipGetLastError();
    if (launch_error != hipSuccess) {
        return launch_error;
    }

    hipLaunchKernelGGL(
        argmax_finalize_kernel,
        1, block_size, 0, stream,
        d_partial_values, d_partial_indices, d_output_index, grid_size
    );

    return hipGetLastError();
}

hipError_t gpu_argmax_f32(
    const float* d_input,
    float* d_partial_values,
    int* d_partial_indices,
    int* d_output_index,
    const int n
) {
    return gpu_argmax_f32_on_stream(
        d_input,
        d_partial_values,
        d_partial_indices,
        d_output_index,
        n,
        0
    );
}

hipError_t gpu_embed_q8_0_token(
    const uint8_t* d_embedding,
    float* d_out,
    const int hidden_size,
    const int vocab_size,
    const int token_id
) {
    if (d_embedding == nullptr || d_out == nullptr ||
        hidden_size <= 0 || vocab_size <= 0 ||
        hidden_size % Q8_0_EMBED_BLOCK_ELEMS != 0 ||
        token_id < 0 || token_id >= vocab_size) {
        return hipErrorInvalidValue;
    }

    const int block_size = BLOCK_SIZE;
    const int grid_size = (hidden_size + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        embed_q8_0_token_kernel,
        grid_size, block_size, 0, 0,
        d_embedding, d_out, hidden_size, vocab_size, token_id
    );

    return hipGetLastError();
}

hipError_t gpu_embed_q8_0_batch(
    const uint8_t* d_embedding,
    const int* d_token_ids,
    float* d_out,
    const int hidden_size,
    const int vocab_size,
    const int seq_len
) {
    if (d_embedding == nullptr || d_token_ids == nullptr || d_out == nullptr ||
        hidden_size <= 0 || vocab_size <= 0 || seq_len <= 0 ||
        hidden_size % Q8_0_EMBED_BLOCK_ELEMS != 0) {
        return hipErrorInvalidValue;
    }

    const int block_size = BLOCK_SIZE;
    const int grid_x = (hidden_size + block_size - 1) / block_size;
    dim3 grid(grid_x, seq_len);

    hipLaunchKernelGGL(
        embed_q8_0_batch_kernel,
        grid, block_size, 0, 0,
        d_embedding, d_token_ids, d_out, hidden_size, vocab_size, seq_len
    );

    return hipGetLastError();
}

hipError_t gpu_zero_fill(
    float* d_ptr,
    const int n,
    hipStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        zero_fill_kernel,
        grid_size, block_size, 0, stream,
        d_ptr, n
    );

    return hipGetLastError();
}

hipError_t gpu_increment_decode_state_on_stream(
    int* d_state,
    hipStream_t stream
) {
    if (d_state == nullptr) {
        return hipErrorInvalidValue;
    }

    hipLaunchKernelGGL(
        increment_decode_state_kernel,
        1, 1, 0, stream,
        d_state
    );

    return hipGetLastError();
}

hipError_t gpu_embed_q4_0_token(
    const uint8_t* d_embedding,
    float* d_out,
    const int hidden_size,
    const int vocab_size,
    const int token_id
) {
    if (d_embedding == nullptr || d_out == nullptr ||
        hidden_size <= 0 || vocab_size <= 0 || token_id < 0 || token_id >= vocab_size ||
        (hidden_size % Q4_0_EMBED_BLOCK_ELEMS) != 0) {
        return hipErrorInvalidValue;
    }

    const int block_size = 256;
    const int grid_size = (hidden_size + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        embed_q4_0_token_kernel,
        dim3(grid_size), dim3(block_size), 0, 0,
        d_embedding, d_out, hidden_size, vocab_size, token_id
    );

    return hipGetLastError();
}

hipError_t gpu_embed_q4_0_batch(
    const uint8_t* d_embedding,
    const int* d_token_ids,
    float* d_out,
    const int hidden_size,
    const int vocab_size,
    const int seq_len
) {
    if (d_embedding == nullptr || d_token_ids == nullptr || d_out == nullptr ||
        hidden_size <= 0 || vocab_size <= 0 || seq_len <= 0 ||
        (hidden_size % Q4_0_EMBED_BLOCK_ELEMS) != 0) {
        return hipErrorInvalidValue;
    }

    const int block_size = 256;
    const int grid_x = (hidden_size + block_size - 1) / block_size;
    dim3 grid(grid_x, seq_len);

    hipLaunchKernelGGL(
        embed_q4_0_batch_kernel,
        grid, dim3(block_size), 0, 0,
        d_embedding, d_token_ids, d_out, hidden_size, vocab_size, seq_len
    );

    return hipGetLastError();
}

} // extern "C"
