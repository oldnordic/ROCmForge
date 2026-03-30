#include "common.hip"

namespace {

constexpr float kNegInf = -3.402823466e+38f;
constexpr int kDecodeScoreCapacity = 2048;
constexpr int kDecodeReductionScratch = MAX_BLOCK_SIZE / 32;

__device__ inline float warp_reduce_sum(float value) {
    #pragma unroll
    for (int delta = warpSize / 2; delta > 0; delta >>= 1) {
        value += __shfl_down(value, delta);
    }
    return value;
}

__device__ inline float warp_reduce_max(float value) {
    #pragma unroll
    for (int delta = warpSize / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down(value, delta));
    }
    return value;
}

__device__ inline float block_reduce_sum(float value, float* scratch) {
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        scratch[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        value = (lane < warp_count) ? scratch[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ inline float block_reduce_max(float value, float* scratch) {
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    value = warp_reduce_max(value);
    if (lane == 0) {
        scratch[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        value = (lane < warp_count) ? scratch[lane] : kNegInf;
        value = warp_reduce_max(value);
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

} // namespace

// ── Multi-Head Flash Attention (Single Token Decode) ──────────────────────────────

// One block handles ONE head.
// The grid size is (num_heads).
// This uses all CUs effectively.

__global__ void flash_attn_decode_strided_multi_head_kernel(
    float* __restrict__ out,            // [num_heads * head_dim]
    const float* __restrict__ q,        // [num_heads * head_dim]
    const float* __restrict__ k_cache,  // [max_seq * kv_size]
    const float* __restrict__ v_cache,  // [max_seq * kv_size]
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    extern __shared__ float smem[]; // Size: blockDim.x * sizeof(float)

    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    // Output pointer for this head
    float* head_out = out + head_idx * head_dim;
    const float* head_q = q + head_idx * head_dim;

    // Local registers for online softmax
    float m = kNegInf;
    float l = 0.0f;

    // Buffer for results in shared memory (one per thread)
    // We reuse smem for reduction scores
    
    // Process all positions
    for (int pos = 0; pos < seq_len; ++pos) {
        const size_t cache_base = (size_t)pos * kv_size + head_offset;
        
        // Dot product Q * K[pos]
        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += head_q[i] * k_cache[cache_base + i];
        }

        // Parallel reduction within block
        smem[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        score = smem[0] * scale;

        // Softmax update
        const float m_prev = m;
        const float l_prev = l;
        m = max(m_prev, score);
        const float exp_score = expf(score - m);
        const float exp_diff = expf(m_prev - m);
        l = l_prev * exp_diff + exp_score;

        // Update output (incremental)
        // This part is memory-intensive. We can optimize it by doing it at the end
        // but online softmax usually does it per-step.
        // For better performance, we'll store the partial results.
    }

    // Wait! A better way for small models:
    // Compute all scores first, store in LDS if possible, then do ONE pass over V.
}

// ── Optimized Multi-Head Attention (One Pass over V) ──────────────────────────────

__global__ void flash_attn_decode_strided_multi_head_v2_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    const float* head_q = q + head_idx * head_dim;
    
    // 1. Compute scores for all positions
    // We use shared memory to store scores if seq_len is small.
    extern __shared__ float s_scores[];
    float* s_reduce = s_scores + kDecodeScoreCapacity;
    
    // Each thread handles some positions
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const size_t cache_base = (size_t)pos * kv_size + head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += head_q[i] * k_cache[cache_base + i];
        }
        score *= scale;
        s_scores[pos] = score;
    }
    __syncthreads();

    float local_max = kNegInf;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[pos]);
    }
    const float m = block_reduce_max(local_max, s_reduce);

    // 3. Compute exp and sum
    float local_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(s_scores[pos] - m);
        s_scores[pos] = weight;
        local_sum += weight;
    }
    const float s_sum = block_reduce_sum(local_sum, s_reduce);

    // 4. Compute final output values
    const float inv_sum = 1.0f / s_sum;
    float* head_out = out + head_idx * head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            const size_t cache_base = (size_t)pos * kv_size + head_offset;
            val += s_scores[pos] * v_cache[cache_base + i];
        }
        head_out[i] = val * inv_sum;
    }
}

__global__ void flash_attn_decode_strided_multi_head_state_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int* __restrict__ seq_len_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int seq_len = *seq_len_ptr;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    const float* head_q = q + head_idx * head_dim;
    extern __shared__ float s_scores[];

    float* s_reduce = s_scores + kDecodeScoreCapacity;

    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const size_t cache_base = (size_t)pos * kv_size + head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += head_q[i] * k_cache[cache_base + i];
        }
        score *= scale;
        s_scores[pos] = score;
    }
    __syncthreads();

    float local_max = kNegInf;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[pos]);
    }
    const float m = block_reduce_max(local_max, s_reduce);

    float local_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(s_scores[pos] - m);
        s_scores[pos] = weight;
        local_sum += weight;
    }
    const float s_sum = block_reduce_sum(local_sum, s_reduce);

    const float inv_sum = 1.0f / s_sum;
    float* head_out = out + head_idx * head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            const size_t cache_base = (size_t)pos * kv_size + head_offset;
            val += s_scores[pos] * v_cache[cache_base + i];
        }
        head_out[i] = val * inv_sum;
    }
}

extern "C" hipError_t gpu_flash_attn_decode_strided_multi_head(
    float* d_out,
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    hipStream_t stream
) {
    const int threads_per_block = 256;
    // We need shared memory for scores. Assume max seq_len = 2048 for now.
    const size_t smem_size = (kDecodeScoreCapacity + kDecodeReductionScratch) * sizeof(float);

    flash_attn_decode_strided_multi_head_v2_kernel<<<num_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    return hipGetLastError();
}

extern "C" hipError_t gpu_flash_attn_decode_strided_multi_head_state(
    float* d_out,
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    const int* d_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    hipStream_t stream
) {
    if (d_seq_len == nullptr) {
        return hipErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const size_t smem_size = (kDecodeScoreCapacity + kDecodeReductionScratch) * sizeof(float);

    flash_attn_decode_strided_multi_head_state_kernel<<<num_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, d_seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    return hipGetLastError();
}

// ── KV Cache Write ───────────────────────────────────────────────────────────────

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

__global__ void kv_write_state_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const int kv_size
) {
    const int pos = *pos_ptr;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_size) {
        const size_t cache_idx = (size_t)pos * kv_size + i;
        k_cache[cache_idx] = k[i];
        v_cache[cache_idx] = v[i];
    }
}

__global__ void kv_write_rope_state_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox
) {
    const int pos = *pos_ptr;
    const int kv_size = num_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= kv_size) {
        return;
    }

    const size_t cache_idx = (size_t)pos * kv_size + i;
    v_cache[cache_idx] = v[i];

    const int head = i / head_dim;
    const int idx_in_head = i - head * head_dim;
    const int head_base = head * head_dim;
    const int half = head_dim / 2;

    int pair_idx;
    int idx0;
    int idx1;

    if (neox) {
        if (idx_in_head < half) {
            pair_idx = idx_in_head;
            idx0 = head_base + idx_in_head;
            idx1 = idx0 + half;
        } else {
            pair_idx = idx_in_head - half;
            idx0 = head_base + pair_idx;
            idx1 = idx0 + half;
        }
    } else {
        pair_idx = idx_in_head / 2;
        idx0 = head_base + 2 * pair_idx;
        idx1 = idx0 + 1;
    }

    const float exponent = (float)(2 * pair_idx) / (float)head_dim;
    const float theta = 1.0f / pow(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cos(angle);
    const float sin_val = sin(angle);
    const float x0 = k[idx0];
    const float x1 = k[idx1];

    const bool first = neox ? (idx_in_head < half) : ((idx_in_head & 1) == 0);
    k_cache[cache_idx] = first
        ? (x0 * cos_val - x1 * sin_val)
        : (x0 * sin_val + x1 * cos_val);
}

__global__ void kv_write_batched_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int start_pos,
    const int kv_size,
    const int seq_len
) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < seq_len && i < kv_size) {
        const int pos = start_pos + s;
        const size_t cache_idx = (size_t)pos * kv_size + i;
        const size_t input_idx = (size_t)s * kv_size + i;
        k_cache[cache_idx] = k[input_idx];
        v_cache[cache_idx] = v[input_idx];
    }
}

// ── Flash Attention (Batch Prefill) ───────────────────────────────────────────────

__global__ void flash_attn_prefill_strided_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int seq_len,
    const int head_dim,
    const int out_stride,
    const int q_stride,
    const int kv_stride,
    const int out_head_offset,
    const int q_head_offset,
    const int kv_head_offset,
    const float scale
) {
    extern __shared__ float smem[];
    const int query_pos = blockIdx.x;
    const int tid = threadIdx.x;
    if (query_pos >= seq_len) return;

    const float* q_ptr = q + (size_t)query_pos * q_stride + q_head_offset;
    float* out_ptr = out + (size_t)query_pos * out_stride + out_head_offset;

    if (tid < head_dim) out_ptr[tid] = 0.0f;
    __syncthreads();

    float m = -3.402823466e+38f;
    float l = 0.0f;

    for (int key_pos = 0; key_pos <= query_pos; ++key_pos) {
        const float* k_ptr = k + (size_t)key_pos * kv_stride + kv_head_offset;
        const float* v_ptr = v + (size_t)key_pos * kv_stride + kv_head_offset;

        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) score += q_ptr[i] * k_ptr[i];

        smem[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        score = smem[0] * scale;

        if (tid == 0) {
            const float prev_m = m;
            const float prev_l = l;
            m = max(prev_m, score);
            const float exp_diff = expf(prev_m - m);
            const float exp_score = expf(score - m);
            l = prev_l * exp_diff + exp_score;
            const float old_scale = (l > 0.0f) ? (prev_l * exp_diff / l) : 0.0f;
            const float new_scale = (l > 0.0f) ? (exp_score / l) : 0.0f;

            for (int i = 0; i < head_dim; ++i) {
                out_ptr[i] = out_ptr[i] * old_scale + v_ptr[i] * new_scale;
            }
        }
        __syncthreads();
    }
}

extern "C" {

hipError_t gpu_kv_write(float* d_k_cache, float* d_v_cache, const float* d_k, const float* d_v, const int pos, const int kv_size, const int max_seq, hipStream_t stream) {
    hipLaunchKernelGGL(kv_write_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, pos, kv_size);
    return hipGetLastError();
}

hipError_t gpu_kv_write_state(float* d_k_cache, float* d_v_cache, const float* d_k, const float* d_v, const int* d_pos, const int kv_size, const int max_seq, hipStream_t stream) {
    (void)max_seq;
    if (d_pos == nullptr) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(kv_write_state_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, d_pos, kv_size);
    return hipGetLastError();
}

hipError_t gpu_kv_write_rope_state(float* d_k_cache, float* d_v_cache, const float* d_k, const float* d_v, const int* d_pos, const int num_kv_heads, const int head_dim, const float theta_base, const int neox, hipStream_t stream) {
    if (d_pos == nullptr) {
        return hipErrorInvalidValue;
    }
    const int kv_size = num_kv_heads * head_dim;
    hipLaunchKernelGGL(kv_write_rope_state_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, d_pos, num_kv_heads, head_dim, theta_base, neox);
    return hipGetLastError();
}

hipError_t gpu_kv_write_batched(float* d_k_cache, float* d_v_cache, const float* d_k, const float* d_v, const int start_pos, const int kv_size, const int max_seq, const int seq_len) {
    hipLaunchKernelGGL(kv_write_batched_kernel, dim3((kv_size + 255) / 256, seq_len), 256, 0, 0, d_k_cache, d_v_cache, d_k, d_v, start_pos, kv_size, seq_len);
    return hipGetLastError();
}

hipError_t gpu_flash_attn_prefill_strided(float* d_out, const float* d_q, const float* d_k, const float* d_v, const int seq_len, const int head_dim, const int out_stride, const int q_stride, const int kv_stride, const int out_head_offset, const int q_head_offset, const int kv_head_offset, const float scale) {
    hipLaunchKernelGGL(flash_attn_prefill_strided_kernel, seq_len, 256, 256 * sizeof(float), 0, d_out, d_q, d_k, d_v, seq_len, head_dim, out_stride, q_stride, kv_stride, out_head_offset, q_head_offset, kv_head_offset, scale);
    return hipGetLastError();
}

} // extern "C"
