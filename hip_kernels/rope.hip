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
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rope_kernel(
    float* __restrict__ x,
    const int pos,
    const int dim,
    const float theta_base
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= dim / 2) return;

    // Compute theta_i = 1 / (theta_base^(2i/dim))
    const float inv_dim = 1.0f / (float)dim;
    const float exponent = (float)(2 * i) * inv_dim;
    const float theta = 1.0f / powf(theta_base, exponent);

    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const int idx0 = 2 * i;
    const int idx1 = 2 * i + 1;

    const float x0 = x[idx0];
    const float x1 = x[idx1];

    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

/// Multi-head RoPE kernel for decode.
///
/// Supports both classic consecutive-pair RoPE and NeoX split-half pairing.
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rope_heads_kernel(
    float* __restrict__ x,
    const int pos,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    const int neox
) {
    const int half = head_dim / 2;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = num_heads * half;

    if (pair_idx >= total_pairs) return;

    const int head = pair_idx / half;
    const int i = pair_idx - head * half;
    const int base = head * head_dim;

    const float inv_head_dim = 1.0f / (float)head_dim;
    const float exponent = (float)(2 * i) * inv_head_dim;
    const float theta = 1.0f / powf(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    int idx0;
    int idx1;
    if (neox) {
        idx0 = base + i;
        idx1 = base + i + half;
    } else {
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }

    const float x0 = x[idx0];
    const float x1 = x[idx1];

    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rope_heads_state_kernel(
    float* __restrict__ x,
    const int* __restrict__ pos_ptr,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    const int neox
) {
    const int pos = *pos_ptr;
    const int half = head_dim / 2;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = num_heads * half;

    if (pair_idx >= total_pairs) return;

    const int head = pair_idx / half;
    const int i = pair_idx - head * half;
    const int base = head * head_dim;

    const float inv_head_dim = 1.0f / (float)head_dim;
    const float exponent = (float)(2 * i) * inv_head_dim;
    const float theta = 1.0f / powf(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    int idx0;
    int idx1;
    if (neox) {
        idx0 = base + i;
        idx1 = base + i + half;
    } else {
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }

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
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
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
    const float inv_dim = 1.0f / (float)dim;
    const float exponent = (float)(2 * i) * inv_dim;
    const float theta = 1.0f / powf(theta_base, exponent);

    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const int idx0 = 2 * i;
    const int idx1 = 2 * i + 1;

    const float x0 = row_x[idx0];
    const float x1 = row_x[idx1];

    row_x[idx0] = x0 * cos_val - x1 * sin_val;
    row_x[idx1] = x0 * sin_val + x1 * cos_val;
}

__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rope_heads_batched_kernel(
    float* __restrict__ x,
    const int start_pos,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    const int seq_len,
    const int neox
) {
    const int s = blockIdx.y;
    const int half = head_dim / 2;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = num_heads * half;

    if (s >= seq_len || pair_idx >= total_pairs) {
        return;
    }

    const int head = pair_idx / half;
    const int i = pair_idx - head * half;
    const int row_stride = num_heads * head_dim;
    float* row_x = x + static_cast<size_t>(s) * row_stride;
    const int base = head * head_dim;
    const int pos = start_pos + s;

    const float inv_head_dim = 1.0f / (float)head_dim;
    const float exponent = (float)(2 * i) * inv_head_dim;
    const float theta = 1.0f / powf(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    int idx0;
    int idx1;
    if (neox) {
        idx0 = base + i;
        idx1 = base + i + half;
    } else {
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }

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

hipError_t gpu_rope_heads(
    float* d_x,
    const int pos,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    const int neox,
    hipStream_t stream
) {
    if (num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int total_pairs = num_heads * (head_dim / 2);
    const int block_size = 256;
    const int grid_size = (total_pairs + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        rope_heads_kernel,
        grid_size, block_size, 0, stream,
        d_x, pos, num_heads, head_dim, theta_base, neox
    );

    return hipGetLastError();
}

hipError_t gpu_rope_heads_state(
    float* d_x,
    const int* d_pos,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    const int neox,
    hipStream_t stream
) {
    if (d_pos == nullptr || num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int total_pairs = num_heads * (head_dim / 2);
    const int block_size = 256;
    const int grid_size = (total_pairs + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        rope_heads_state_kernel,
        grid_size, block_size, 0, stream,
        d_x, d_pos, num_heads, head_dim, theta_base, neox
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

hipError_t gpu_rope_heads_batched(
    float* d_x,
    const int start_pos,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    const int seq_len,
    const int neox
) {
    if (num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 || seq_len <= 0) {
        return hipErrorInvalidValue;
    }

    const int total_pairs = num_heads * (head_dim / 2);
    const int block_size = 256;
    const int grid_x = (total_pairs + block_size - 1) / block_size;
    dim3 grid(grid_x, seq_len);

    hipLaunchKernelGGL(
        rope_heads_batched_kernel,
        grid, block_size, 0, 0,
        d_x, start_pos, num_heads, head_dim, theta_base, seq_len, neox
    );

    return hipGetLastError();
}

// Reference RoPE path: uses a pre-computed frequency table (matching the CPU
// config) instead of recomputing theta with powf in the kernel.  This removes
// one source of CPU/GPU divergence for the reference decode route.
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rope_heads_state_from_freq_kernel(
    float* __restrict__ x,
    const int* __restrict__ pos_ptr,
    const float* __restrict__ freq,
    const int num_heads,
    const int head_dim,
    const int neox
) {
    const int pos = *pos_ptr;
    const int half = head_dim / 2;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = num_heads * half;

    if (pair_idx >= total_pairs) return;

    const int head = pair_idx / half;
    const int i = pair_idx - head * half;
    const int base = head * head_dim;

    const float angle = (float)pos * freq[i];
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    int idx0;
    int idx1;
    if (neox) {
        idx0 = base + i;
        idx1 = base + i + half;
    } else {
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }

    const float x0 = x[idx0];
    const float x1 = x[idx1];

    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

// Reference KV write path that applies the same pre-computed frequency table.
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void kv_write_rope_from_freq_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const float* __restrict__ freq,
    const int num_kv_heads,
    const int head_dim,
    const int neox,
    const int max_seq_len
) {
    const int pos = *pos_ptr;
    const int half = head_dim / 2;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = num_kv_heads * half;

    if (pair_idx >= total_pairs) return;

    const int head = pair_idx / half;
    const int i = pair_idx - head * half;
    const int base = head * head_dim;

    const float angle = (float)pos * freq[i];
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    int idx0, idx1;
    if (neox) {
        idx0 = base + i;
        idx1 = base + i + half;
    } else {
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }

    const float x0 = k[idx0];
    const float x1 = k[idx1];
    const float rotated0 = x0 * cos_val - x1 * sin_val;
    const float rotated1 = x0 * sin_val + x1 * cos_val;

    const size_t cache_offset = (size_t)pos * num_kv_heads * head_dim + base;
    k_cache[cache_offset + idx0 - base] = rotated0;
    k_cache[cache_offset + idx1 - base] = rotated1;

    // V is stored without rotation.
    v_cache[cache_offset + idx0 - base] = v[idx0];
    v_cache[cache_offset + idx1 - base] = v[idx1];
}

hipError_t gpu_rope_heads_state_from_freq(
    float* d_x,
    const int* d_pos,
    const float* d_freq,
    const int num_heads,
    const int head_dim,
    const int neox,
    hipStream_t stream
) {
    if (d_pos == nullptr || d_freq == nullptr || num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int total_pairs = num_heads * (head_dim / 2);
    const int block_size = 256;
    const int grid_size = (total_pairs + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        rope_heads_state_from_freq_kernel,
        grid_size, block_size, 0, stream,
        d_x, d_pos, d_freq, num_heads, head_dim, neox
    );

    return hipGetLastError();
}

hipError_t gpu_kv_write_rope_from_freq(
    float* d_k_cache,
    float* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int* d_pos,
    const float* d_freq,
    const int num_kv_heads,
    const int head_dim,
    const int neox,
    const int max_seq_len,
    hipStream_t stream
) {
    if (d_pos == nullptr || d_freq == nullptr || num_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int total_pairs = num_kv_heads * (head_dim / 2);
    const int block_size = 256;
    const int grid_size = (total_pairs + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        kv_write_rope_from_freq_kernel,
        grid_size, block_size, 0, stream,
        d_k_cache, d_v_cache, d_k, d_v, d_pos, d_freq,
        num_kv_heads, head_dim, neox, max_seq_len
    );

    return hipGetLastError();
}

// Reference RoPE path using CPU-computed sin/cos tables.  This removes the
// GPU/CPU libm difference for sin/cos, which is the dominant source of drift
// in long-context decode when matching the CPU reference.
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rope_heads_state_from_sin_cos_kernel(
    float* __restrict__ x,
    const int* __restrict__ pos_ptr,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    const int num_heads,
    const int head_dim,
    const int neox
) {
    const int pos = *pos_ptr;
    const int half = head_dim / 2;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = num_heads * half;

    if (pair_idx >= total_pairs) return;

    const int head = pair_idx / half;
    const int i = pair_idx - head * half;
    const int base = head * head_dim;

    const float s = sin_table[i];
    const float c = cos_table[i];

    int idx0;
    int idx1;
    if (neox) {
        idx0 = base + i;
        idx1 = base + i + half;
    } else {
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }

    const float x0 = x[idx0];
    const float x1 = x[idx1];

    x[idx0] = x0 * c - x1 * s;
    x[idx1] = x0 * s + x1 * c;
}

__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void kv_write_rope_from_sin_cos_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    const int num_kv_heads,
    const int head_dim,
    const int neox,
    const int max_seq_len
) {
    const int pos = *pos_ptr;
    const int half = head_dim / 2;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = num_kv_heads * half;

    if (pair_idx >= total_pairs) return;

    const int head = pair_idx / half;
    const int i = pair_idx - head * half;
    const int base = head * head_dim;

    const float s = sin_table[i];
    const float c = cos_table[i];

    int idx0, idx1;
    if (neox) {
        idx0 = base + i;
        idx1 = base + i + half;
    } else {
        idx0 = base + 2 * i;
        idx1 = base + 2 * i + 1;
    }

    const float x0 = k[idx0];
    const float x1 = k[idx1];
    const float rotated0 = x0 * c - x1 * s;
    const float rotated1 = x0 * s + x1 * c;

    const size_t cache_offset = (size_t)pos * num_kv_heads * head_dim + base;
    k_cache[cache_offset + idx0 - base] = rotated0;
    k_cache[cache_offset + idx1 - base] = rotated1;

    v_cache[cache_offset + idx0 - base] = v[idx0];
    v_cache[cache_offset + idx1 - base] = v[idx1];
}

hipError_t gpu_rope_heads_state_from_sin_cos(
    float* d_x,
    const int* d_pos,
    const float* d_sin,
    const float* d_cos,
    const int num_heads,
    const int head_dim,
    const int neox,
    hipStream_t stream
) {
    if (d_pos == nullptr || d_sin == nullptr || d_cos == nullptr ||
        num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int total_pairs = num_heads * (head_dim / 2);
    const int block_size = 256;
    const int grid_size = (total_pairs + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        rope_heads_state_from_sin_cos_kernel,
        grid_size, block_size, 0, stream,
        d_x, d_pos, d_sin, d_cos, num_heads, head_dim, neox
    );

    return hipGetLastError();
}

hipError_t gpu_kv_write_rope_from_sin_cos(
    float* d_k_cache,
    float* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int* d_pos,
    const float* d_sin,
    const float* d_cos,
    const int num_kv_heads,
    const int head_dim,
    const int neox,
    const int max_seq_len,
    hipStream_t stream
) {
    if (d_pos == nullptr || d_sin == nullptr || d_cos == nullptr ||
        num_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        return hipErrorInvalidValue;
    }

    const int total_pairs = num_kv_heads * (head_dim / 2);
    const int block_size = 256;
    const int grid_size = (total_pairs + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        kv_write_rope_from_sin_cos_kernel,
        grid_size, block_size, 0, stream,
        d_k_cache, d_v_cache, d_k, d_v, d_pos, d_sin, d_cos,
        num_kv_heads, head_dim, neox, max_seq_len
    );

    return hipGetLastError();
}

// Partial RoPE: only the first `rotated_dims` dimensions of each head are
// rotated.  The tail dimensions are left untouched.  This matches Qwen3.5's
// partial_rotary_factor configuration and keeps the GPU reference path
// numerically identical to the CPU reference.
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void rope_heads_state_from_sin_cos_partial_kernel(
    float* __restrict__ x,
    const int* __restrict__ pos_ptr,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    const int num_heads,
    const int head_dim,
    const int rotated_dims,
    const int neox
) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= num_heads) return;

    const int half = rotated_dims / 2;
    const int base = head * head_dim;

    for (int i = 0; i < half; ++i) {
        const float s = sin_table[i];
        const float c = cos_table[i];

        int idx0;
        int idx1;
        if (neox) {
            idx0 = base + i;
            idx1 = base + i + half;
        } else {
            idx0 = base + 2 * i;
            idx1 = base + 2 * i + 1;
        }

        const float x0 = x[idx0];
        const float x1 = x[idx1];

        x[idx0] = x0 * c - x1 * s;
        x[idx1] = x0 * s + x1 * c;
    }
}

// Partial RoPE KV write: rotate the first `rotated_dims` dims of each KV head
// and copy the remaining dims unchanged.  All dims are written to the cache.
__attribute__((amdgpu_wave_size(32)))
__launch_bounds__(256, 1)
__global__ void kv_write_rope_from_sin_cos_partial_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    const int num_kv_heads,
    const int head_dim,
    const int rotated_dims,
    const int neox,
    const int max_seq_len
) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= num_kv_heads) return;

    const int pos = *pos_ptr;
    const int half = rotated_dims / 2;
    const int head_half = head_dim / 2;
    const int base = head * head_dim;
    const size_t cache_offset = (size_t)pos * num_kv_heads * head_dim + base;

    for (int i = 0; i < half; ++i) {
        const float s = sin_table[i];
        const float c = cos_table[i];

        int idx0;
        int idx1;
        if (neox) {
            idx0 = base + i;
            idx1 = base + i + half;
        } else {
            idx0 = base + 2 * i;
            idx1 = base + 2 * i + 1;
        }

        const float x0 = k[idx0];
        const float x1 = k[idx1];

        k_cache[cache_offset + idx0 - base] = x0 * c - x1 * s;
        k_cache[cache_offset + idx1 - base] = x0 * s + x1 * c;

        v_cache[cache_offset + idx0 - base] = v[idx0];
        v_cache[cache_offset + idx1 - base] = v[idx1];
    }

    if (neox) {
        // Copy the unrotated tail of each split half.
        for (int i = half; i < head_half; ++i) {
            const int idx0 = base + i;
            const int idx1 = base + i + head_half;
            k_cache[cache_offset + idx0 - base] = k[idx0];
            k_cache[cache_offset + idx1 - base] = k[idx1];
            v_cache[cache_offset + idx0 - base] = v[idx0];
            v_cache[cache_offset + idx1 - base] = v[idx1];
        }
    } else {
        for (int idx = base + rotated_dims; idx < base + head_dim; ++idx) {
            k_cache[cache_offset + idx - base] = k[idx];
            v_cache[cache_offset + idx - base] = v[idx];
        }
    }
}

hipError_t gpu_rope_heads_state_from_sin_cos_partial(
    float* d_x,
    const int* d_pos,
    const float* d_sin,
    const float* d_cos,
    const int num_heads,
    const int head_dim,
    const int rotated_dims,
    const int neox,
    hipStream_t stream
) {
    if (d_pos == nullptr || d_sin == nullptr || d_cos == nullptr ||
        num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
        rotated_dims <= 0 || rotated_dims % 2 != 0 || rotated_dims > head_dim) {
        return hipErrorInvalidValue;
    }

    const int block_size = 256;
    const int grid_size = (num_heads + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        rope_heads_state_from_sin_cos_partial_kernel,
        grid_size, block_size, 0, stream,
        d_x, d_pos, d_sin, d_cos, num_heads, head_dim, rotated_dims, neox
    );

    return hipGetLastError();
}

hipError_t gpu_kv_write_rope_from_sin_cos_partial(
    float* d_k_cache,
    float* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int* d_pos,
    const float* d_sin,
    const float* d_cos,
    const int num_kv_heads,
    const int head_dim,
    const int rotated_dims,
    const int neox,
    const int max_seq_len,
    hipStream_t stream
) {
    if (d_pos == nullptr || d_sin == nullptr || d_cos == nullptr ||
        num_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
        rotated_dims <= 0 || rotated_dims % 2 != 0 || rotated_dims > head_dim) {
        return hipErrorInvalidValue;
    }

    const int block_size = 256;
    const int grid_size = (num_kv_heads + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        kv_write_rope_from_sin_cos_partial_kernel,
        grid_size, block_size, 0, stream,
        d_k_cache, d_v_cache, d_k, d_v, d_pos, d_sin, d_cos,
        num_kv_heads, head_dim, rotated_dims, neox, max_seq_len
    );

    return hipGetLastError();
}

} // extern "C"
