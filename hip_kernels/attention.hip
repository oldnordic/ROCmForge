#include "common.hip"

// ── KV Cache Write ───────────────────────────────────────────────────────────────

/// Write K/V to cache at a single position.
///
/// # Arguments
/// * k_cache - Key cache [max_seq * kv_size]
/// * v_cache - Value cache [max_seq * kv_size]
/// * k - Key to write [kv_size]
/// * v - Value to write [kv_size]
/// * pos - Position to write at
/// * kv_size - Size per position
__global__ void kv_write_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int pos,
    const int kv_size
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < kv_size) {
        const size_t cache_idx = (size_t)pos * kv_size + i;
        k_cache[cache_idx] = k[i];
        v_cache[cache_idx] = v[i];
    }
}

/// Batched KV write for prefill.
///
/// # Arguments
/// * k_cache - Key cache [max_seq * kv_size]
/// * v_cache - Value cache [max_seq * kv_size]
/// * k - Keys to write [seq_len * kv_size]
/// * v - Values to write [seq_len * kv_size]
/// * start_pos - Starting position
/// * kv_size - Size per position
/// * seq_len - Number of positions to write
__global__ void kv_write_batched_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int start_pos,
    const int kv_size,
    const int seq_len
) {
    const int s = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;

    if (s < seq_len && i < kv_size) {
        const int pos = start_pos + s;
        const size_t cache_idx = (size_t)pos * kv_size + i;
        const size_t input_idx = (size_t)s * kv_size + i;
        k_cache[cache_idx] = k[input_idx];
        v_cache[cache_idx] = v[input_idx];
    }
}

// ── Flash Attention (Single Token Decode) ────────────────────────────────────────

/// Flash attention kernel for single-token decode.
///
/// Computes attention for a single query token against all cached K/V.
/// Uses online softmax for numerical stability.
///
/// # Arguments
/// * out - Output [head_dim] (single head's output)
/// * q - Query [head_dim]
/// * k_cache - Key cache [seq_len * kv_size] (all cached positions)
/// * v_cache - Value cache [seq_len * kv_size]
/// * seq_len - Number of cached positions
/// * head_dim - Head dimension
/// * scale - Attention scale (1 / sqrt(head_dim))
__global__ void flash_attn_decode_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Shared memory for reduction
    extern __shared__ float smem[];

    const int tid = threadIdx.x;

    // Initialize accumulator (output values)
    if (tid < head_dim) {
        out[tid] = 0.0f;
        smem[tid] = 0.0f;  // Placeholder
    }

    // Initialize max and sum for online softmax
    float m = -3.402823466e+38f;  // -inf
    float l = 0.0f;

    // Process each cached position
    for (int pos = 0; pos < seq_len; ++pos) {
        // Compute dot(q, k[pos]) * scale
        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += q[i] * k_cache[(size_t)pos * head_dim + i];
        }

        // Reduce score across threads
        smem[tid] = score;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncthreads();
        }
        score = smem[0] * scale;

        // Online softmax update (only thread 0 updates m, l)
        if (tid == 0) {
            const float m_new = m > score ? m : score;
            const float exp_diff = exp(m - m_new);
            const float exp_score = exp(score - m_new);
            m = m_new;
            l = l * exp_diff + exp_score;
        }
        __syncthreads();

        // Update accumulator
        const float alpha = tid == 0 ? exp(score - m) / l : 0.0f;
        // Broadcast alpha to all threads
        // (simplified: assume single thread handles accumulation)
        if (tid == 0) {
            for (int i = 0; i < head_dim; ++i) {
                out[i] = out[i] * (l - exp(score - m)) / l + v_cache[(size_t)pos * head_dim + i] * alpha;
            }
        }
        __syncthreads();
    }
}

// ── Flash Attention (Batch Prefill) ───────────────────────────────────────────────

/// Causal flash attention for prefill.
///
/// Each token in the batch attends to all previous tokens (including itself).
///
/// # Arguments
/// * out - Output [seq_len * head_dim]
/// * q - Queries [seq_len * head_dim]
/// * k - Keys [seq_len * head_dim]
/// * v - Values [seq_len * head_dim]
/// * seq_len - Number of tokens
/// * head_dim - Head dimension
/// * scale - Attention scale
__global__ void flash_attn_prefill_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    extern __shared__ float smem[];

    const int query_pos = blockIdx.x;  // Which query we're computing
    const int tid = threadIdx.x;

    if (query_pos >= seq_len) return;

    // Get pointer to this query
    const float* q_ptr = q + (size_t)query_pos * head_dim;
    float* out_ptr = out + (size_t)query_pos * head_dim;

    // Initialize output to zero
    if (tid < head_dim) {
        out_ptr[tid] = 0.0f;
    }

    float m = -3.402823466e+38f;  // -inf
    float l = 0.0f;

    // Causal: query_pos attends to 0..=query_pos
    for (int key_pos = 0; key_pos <= query_pos; ++key_pos) {
        // Compute dot(q[query_pos], k[key_pos]) * scale
        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += q_ptr[i] * k[(size_t)key_pos * head_dim + i];
        }

        // Reduce score
        smem[tid] = score;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncthreads();
        }
        score = smem[0] * scale;

        // Online softmax update
        if (tid == 0) {
            const float m_new = m > score ? m : score;
            const float exp_diff = exp(m - m_new);
            const float exp_score = exp(score - m_new);
            m = m_new;
            l = l * exp_diff + exp_score;
        }
        __syncthreads();

        // Update accumulator (simplified - single thread)
        if (tid == 0) {
            const float alpha = exp(score - m);
            for (int i = 0; i < head_dim; ++i) {
                out_ptr[i] = out_ptr[i] * (l - alpha) / l + v[(size_t)key_pos * head_dim + i] * alpha / l;
            }
        }
        __syncthreads();
    }
}

// C wrapper functions for dynamic linking
extern "C" {

hipError_t gpu_kv_write(
    float* d_k_cache,
    float* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int pos,
    const int kv_size,
    const int max_seq
) {
    const int block_size = 256;
    const int grid_size = (kv_size + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        kv_write_kernel,
        grid_size, block_size, 0, 0,
        d_k_cache, d_v_cache, d_k, d_v, pos, kv_size
    );

    return hipGetLastError();
}

hipError_t gpu_kv_write_batched(
    float* d_k_cache,
    float* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int start_pos,
    const int kv_size,
    const int max_seq,
    const int seq_len
) {
    const int block_size = 256;
    const int grid_x = (kv_size + block_size - 1) / block_size;

    dim3 grid(grid_x, seq_len);

    hipLaunchKernelGGL(
        kv_write_batched_kernel,
        grid, block_size, 0, 0,
        d_k_cache, d_v_cache, d_k, d_v, start_pos, kv_size, seq_len
    );

    return hipGetLastError();
}

hipError_t gpu_flash_attn_decode(
    float* d_out,
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int block_size = 256;
    const int smem_size = block_size * sizeof(float);

    hipLaunchKernelGGL(
        flash_attn_decode_kernel,
        1, block_size, smem_size, 0,
        d_out, d_q, d_k_cache, d_v_cache, seq_len, head_dim, scale
    );

    return hipGetLastError();
}

hipError_t gpu_flash_attn_prefill(
    float* d_out,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int block_size = 256;
    const int smem_size = block_size * sizeof(float);

    hipLaunchKernelGGL(
        flash_attn_prefill_kernel,
        seq_len, block_size, smem_size, 0,
        d_out, d_q, d_k, d_v, seq_len, head_dim, scale
    );

    return hipGetLastError();
}

} // extern "C"
