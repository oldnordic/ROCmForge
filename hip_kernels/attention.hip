#include "common.hip"

namespace {

constexpr float kNegInf = -3.402823466e+38f;
constexpr int kDecodeScoreCapacity = 2048;
// Scratch buffer for cross-wave reduction
// Decode kernels use block_size=256: 256 / 32 = 8 warps for wave32
constexpr int kDecodeReductionScratch = 256 / 32;

__device__ inline float warp_reduce_sum(float value) {
    #pragma unroll
    for (int delta = WARP_SIZE_COMPTIME / 2; delta > 0; delta >>= 1) {
        value += __shfl_down(value, delta, WARP_SIZE_COMPTIME);
    }
    return value;
}

__device__ inline float warp_reduce_max(float value) {
    #pragma unroll
    for (int delta = WARP_SIZE_COMPTIME / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down(value, delta, WARP_SIZE_COMPTIME));
    }
    return value;
}

__device__ inline float block_reduce_sum(float value, float* scratch) {
    const int lane = threadIdx.x % WARP_SIZE_COMPTIME;
    const int warp_id = threadIdx.x / WARP_SIZE_COMPTIME;
    const int warp_count = (blockDim.x + WARP_SIZE_COMPTIME - 1) / WARP_SIZE_COMPTIME;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        scratch[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        // Only lanes 0 to warp_count-1 have valid data in scratch
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
    const int lane = threadIdx.x % WARP_SIZE_COMPTIME;
    const int warp_id = threadIdx.x / WARP_SIZE_COMPTIME;
    const int warp_count = (blockDim.x + WARP_SIZE_COMPTIME - 1) / WARP_SIZE_COMPTIME;

    value = warp_reduce_max(value);
    if (lane == 0) {
        scratch[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        // Only lanes 0 to warp_count-1 have valid data in scratch
        value = (lane < warp_count) ? scratch[lane] : kNegInf;
        value = warp_reduce_max(value);
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ inline float get_rope_k(
    int i,
    const float* k,
    int pos,
    int num_kv_heads,
    int head_dim,
    float theta_base,
    int neox
) {
    const int head = i / head_dim;
    const int idx_in_head = i % head_dim;
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
    const float theta = 1.0f / powf(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float x0 = k[idx0];
    const float x1 = k[idx1];

    const bool first = neox ? (idx_in_head < half) : ((idx_in_head & 1) == 0);
    return first
        ? (x0 * cos_val - x1 * sin_val)
        : (x0 * sin_val + x1 * cos_val);
}

__device__ inline float get_reconstructed_k(
    int pos,
    int i, // Index within kv_size
    const float* k_cache,
    int kv_lora_dim,
    const float* w_up_k
) {
    if (w_up_k != nullptr) {
        float val = 0.0f;
        for (int j = 0; j < kv_lora_dim; ++j) {
            val += k_cache[(size_t)pos * kv_lora_dim + j] * w_up_k[i * kv_lora_dim + j];
        }
        return val;
    } else {
        return (i < kv_lora_dim) ? k_cache[(size_t)pos * kv_lora_dim + i] : 0.0f;
    }
}

__device__ inline float get_reconstructed_v(
    int pos,
    int i, // Index within kv_size
    const float* v_cache,
    int kv_lora_dim,
    const float* w_up_v
) {
    if (w_up_v != nullptr) {
        float val = 0.0f;
        for (int j = 0; j < kv_lora_dim; ++j) {
            val += v_cache[(size_t)pos * kv_lora_dim + j] * w_up_v[i * kv_lora_dim + j];
        }
        return val;
    } else {
        return (i < kv_lora_dim) ? v_cache[(size_t)pos * kv_lora_dim + i] : 0.0f;
    }
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
    const float scale,
    const int kv_lora_dim,
    const int adastate_anchors_enabled,
    const float* __restrict__ w_up_k,
    const float* __restrict__ w_up_v
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int ratio = num_heads / num_kv_heads;
    const int kv_head_idx = (ratio > 0) ? (head_idx / ratio) : 0;
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    const float* head_q = q + head_idx * head_dim;
    
    // 1. Compute scores for all positions
    // We use shared memory to store scores if seq_len is small.
    extern __shared__ float s_scores[];
    float* s_reduce = s_scores + kDecodeScoreCapacity;
    
    float adastate_scale = scale;
    if (adastate_anchors_enabled && seq_len > 0) {
        float q_norm = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            q_norm += head_q[i] * head_q[i];
        }
        q_norm = sqrtf(q_norm);
        if (q_norm > 0.0f) {
            adastate_scale *= (1.0f / (q_norm * 0.1f + 0.9f));
        }
    }

    // Each thread handles some positions
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        float score = 0.0f;
        if (kv_lora_dim > 0) {
            for (int i = 0; i < head_dim; ++i) {
                float k_val = get_reconstructed_k(pos, head_offset + i, k_cache, kv_lora_dim, w_up_k);
                score += head_q[i] * k_val;
            }
        } else {
            const size_t cache_base = (size_t)pos * kv_size + head_offset;
            for (int i = 0; i < head_dim; ++i) {
                score += head_q[i] * k_cache[cache_base + i];
            }
        }
        score *= adastate_scale;
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
            float v_val = (kv_lora_dim > 0)
                ? get_reconstructed_v(pos, head_offset + i, v_cache, kv_lora_dim, w_up_v)
                : v_cache[(size_t)pos * kv_size + head_offset + i];
            val += s_scores[pos] * v_val;
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
    const float scale,
    const int kv_lora_dim,
    const int adastate_anchors_enabled,
    const float* __restrict__ w_up_k,
    const float* __restrict__ w_up_v
) {
    const int seq_len = *seq_len_ptr;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int ratio = num_heads / num_kv_heads;
    const int kv_head_idx = (ratio > 0) ? (head_idx / ratio) : 0;
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    const float* head_q = q + head_idx * head_dim;
    extern __shared__ float s_scores[];

    float* s_reduce = s_scores + kDecodeScoreCapacity;

    float adastate_scale = scale;
    if (adastate_anchors_enabled && seq_len > 0) {
        float q_norm = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            q_norm += head_q[i] * head_q[i];
        }
        q_norm = sqrtf(q_norm);
        if (q_norm > 0.0f) {
            adastate_scale *= (1.0f / (q_norm * 0.1f + 0.9f));
        }
    }

    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        float score = 0.0f;
        if (kv_lora_dim > 0) {
            for (int i = 0; i < head_dim; ++i) {
                float k_val = get_reconstructed_k(pos, head_offset + i, k_cache, kv_lora_dim, w_up_k);
                score += head_q[i] * k_val;
            }
        } else {
            const size_t cache_base = (size_t)pos * kv_size + head_offset;
            for (int i = 0; i < head_dim; ++i) {
                score += head_q[i] * k_cache[cache_base + i];
            }
        }
        score *= adastate_scale;
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
            float v_val = (kv_lora_dim > 0)
                ? get_reconstructed_v(pos, head_offset + i, v_cache, kv_lora_dim, w_up_v)
                : v_cache[(size_t)pos * kv_size + head_offset + i];
            val += s_scores[pos] * v_val;
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
    const int kv_lora_dim,
    const int adastate_anchors_enabled,
    const float* w_up_k,
    const float* w_up_v,
    hipStream_t stream
) {
    if (seq_len <= 0 || !d_out || !d_q || !d_k_cache || !d_v_cache) {
        return hipErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const size_t smem_size = (kDecodeScoreCapacity + kDecodeReductionScratch) * sizeof(float);

    flash_attn_decode_strided_multi_head_v2_kernel<<<num_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, seq_len, num_heads, num_kv_heads, head_dim, scale,
        kv_lora_dim, adastate_anchors_enabled, w_up_k, w_up_v
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
    const int kv_lora_dim,
    const int adastate_anchors_enabled,
    const float* w_up_k,
    const float* w_up_v,
    hipStream_t stream
) {
    if (d_seq_len == nullptr) {
        return hipErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const size_t smem_size = (kDecodeScoreCapacity + kDecodeReductionScratch) * sizeof(float);

    flash_attn_decode_strided_multi_head_state_kernel<<<num_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, d_seq_len, num_heads, num_kv_heads, head_dim, scale,
        kv_lora_dim, adastate_anchors_enabled, w_up_k, w_up_v
    );

    return hipGetLastError();
}

__device__ inline float project_k_no_rope(
    const float* k,
    const float* w_down_k,
    int j,
    int kv_size
) {
    float val = 0.0f;
    if (w_down_k != nullptr) {
        for (int i = 0; i < kv_size; ++i) {
            val += k[i] * w_down_k[j * kv_size + i];
        }
    } else {
        if (j < kv_size) {
            val = k[j];
        }
    }
    return val;
}

__device__ inline float project_k_with_rope(
    const float* k,
    const float* w_down_k,
    int j,
    int kv_size,
    int pos,
    int num_kv_heads,
    int head_dim,
    float theta_base,
    int neox
) {
    if (theta_base <= 0.0f) {
        return project_k_no_rope(k, w_down_k, j, kv_size);
    }
    float val = 0.0f;
    if (w_down_k != nullptr) {
        for (int i = 0; i < kv_size; ++i) {
            float k_rope = get_rope_k(i, k, pos, num_kv_heads, head_dim, theta_base, neox);
            val += k_rope * w_down_k[j * kv_size + i];
        }
    } else {
        if (j < kv_size) {
            val = get_rope_k(j, k, pos, num_kv_heads, head_dim, theta_base, neox);
        }
    }
    return val;
}

__device__ inline float project_v(
    const float* v,
    const float* w_down_v,
    int j,
    int kv_size
) {
    float val = 0.0f;
    if (w_down_v != nullptr) {
        for (int i = 0; i < kv_size; ++i) {
            val += v[i] * w_down_v[j * kv_size + i];
        }
    } else {
        if (j < kv_size) {
            val = v[j];
        }
    }
    return val;
}

__global__ void kv_write_compressed_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox,
    const int kv_lora_dim,
    const int kv_frame_codec_enabled,
    const float* __restrict__ w_down_k,
    const float* __restrict__ w_down_v
) {
    const int pos = pos_ptr ? *pos_ptr : 0;
    const int kv_size = num_kv_heads * head_dim;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= kv_lora_dim) return;

    float val_k = project_k_with_rope(k, w_down_k, j, kv_size, pos, num_kv_heads, head_dim, theta_base, neox);
    float val_v = project_v(v, w_down_v, j, kv_size);

    if (kv_frame_codec_enabled && pos > 0) {
        float delta_k = val_k - k_cache[(size_t)(pos - 1) * kv_lora_dim + j];
        float delta_v = val_v - v_cache[(size_t)(pos - 1) * kv_lora_dim + j];
        
        k_cache[(size_t)pos * kv_lora_dim + j] = delta_k;
        v_cache[(size_t)pos * kv_lora_dim + j] = delta_v;

        // Reconstruct immediately in place for decoding
        k_cache[(size_t)pos * kv_lora_dim + j] = k_cache[(size_t)(pos - 1) * kv_lora_dim + j] + delta_k;
        v_cache[(size_t)pos * kv_lora_dim + j] = v_cache[(size_t)(pos - 1) * kv_lora_dim + j] + delta_v;
    } else {
        k_cache[(size_t)pos * kv_lora_dim + j] = val_k;
        v_cache[(size_t)pos * kv_lora_dim + j] = val_v;
    }
}

__global__ void kv_write_batched_compressed_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int start_pos,
    const int num_kv_heads,
    const int head_dim,
    const int seq_len,
    const int kv_lora_dim,
    const int kv_frame_codec_enabled,
    const float* __restrict__ w_down_k,
    const float* __restrict__ w_down_v
) {
    const int s = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int kv_size = num_kv_heads * head_dim;

    if (s < seq_len && j < kv_lora_dim) {
        const int pos = start_pos + s;
        
        const float* step_k = k + (size_t)s * kv_size;
        const float* step_v = v + (size_t)s * kv_size;

        float val_k = project_k_no_rope(step_k, w_down_k, j, kv_size);
        float val_v = project_v(step_v, w_down_v, j, kv_size);

        if (kv_frame_codec_enabled && pos > 0) {
            float prev_k = 0.0f;
            float prev_v = 0.0f;
            if (s > 0) {
                const float* prev_step_k = k + (size_t)(s - 1) * kv_size;
                const float* prev_step_v = v + (size_t)(s - 1) * kv_size;
                prev_k = project_k_no_rope(prev_step_k, w_down_k, j, kv_size);
                prev_v = project_v(prev_step_v, w_down_v, j, kv_size);
            } else {
                prev_k = k_cache[(size_t)(pos - 1) * kv_lora_dim + j];
                prev_v = v_cache[(size_t)(pos - 1) * kv_lora_dim + j];
            }

            k_cache[(size_t)pos * kv_lora_dim + j] = val_k - prev_k;
            v_cache[(size_t)pos * kv_lora_dim + j] = val_v - prev_v;
        } else {
            k_cache[(size_t)pos * kv_lora_dim + j] = val_k;
            v_cache[(size_t)pos * kv_lora_dim + j] = val_v;
        }
    }
}

__global__ void reconstruct_kv_cache_prefix_sum_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int start_pos,
    const int seq_len,
    const int kv_lora_dim
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= kv_lora_dim) return;

    float accum_k = (start_pos > 0) ? k_cache[(size_t)(start_pos - 1) * kv_lora_dim + j] : 0.0f;
    float accum_v = (start_pos > 0) ? v_cache[(size_t)(start_pos - 1) * kv_lora_dim + j] : 0.0f;

    for (int s = 0; s < seq_len; ++s) {
        const int pos = start_pos + s;
        const size_t idx = (size_t)pos * kv_lora_dim + j;
        
        accum_k += k_cache[idx];
        accum_v += v_cache[idx];
        
        k_cache[idx] = accum_k;
        v_cache[idx] = accum_v;
    }
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
    const float scale,
    const int kv_lora_dim,
    const float* __restrict__ w_up_k,
    const float* __restrict__ w_up_v
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
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float k_val = (kv_lora_dim > 0)
                ? get_reconstructed_k(key_pos, kv_head_offset + i, k, kv_lora_dim, w_up_k)
                : k_ptr[i];
            score += q_ptr[i] * k_val;
        }

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
                float v_val = (kv_lora_dim > 0)
                    ? get_reconstructed_v(key_pos, kv_head_offset + i, v, kv_lora_dim, w_up_v)
                    : v_ptr[i];
                out_ptr[i] = out_ptr[i] * old_scale + v_val * new_scale;
            }
        }
        __syncthreads();
    }
}

__global__ void kv_write_rope_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int pos,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox
) {
    const int kv_size = num_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= kv_size) return;

    const size_t cache_idx = (size_t)pos * kv_size + i;
    v_cache[cache_idx] = v[i];

    const int head = i / head_dim;
    const int idx_in_head = i % head_dim;
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
    const float theta = 1.0f / powf(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float x0 = k[idx0];
    const float x1 = k[idx1];

    const bool first = neox ? (idx_in_head < half) : ((idx_in_head & 1) == 0);
    k_cache[cache_idx] = first
        ? (x0 * cos_val - x1 * sin_val)
        : (x0 * sin_val + x1 * cos_val);
}

extern "C" {

hipError_t gpu_kv_write_rope(float* d_k_cache, float* d_v_cache, const float* d_k, const float* d_v, const int pos, const int num_kv_heads, const int head_dim, const float theta_base, const int neox, hipStream_t stream) {
    const int kv_size = num_kv_heads * head_dim;
    hipLaunchKernelGGL(kv_write_rope_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, pos, num_kv_heads, head_dim, theta_base, neox);
    return hipGetLastError();
}

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

hipError_t gpu_flash_attn_prefill_strided(
    float* d_out,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    const int seq_len,
    const int head_dim,
    const int out_stride,
    const int q_stride,
    const int kv_stride,
    const int out_head_offset,
    const int q_head_offset,
    const int kv_head_offset,
    const float scale,
    const int kv_lora_dim,
    const float* w_up_k,
    const float* w_up_v
) {
    hipLaunchKernelGGL(flash_attn_prefill_strided_kernel, seq_len, 256, 256 * sizeof(float), 0, d_out, d_q, d_k, d_v, seq_len, head_dim, out_stride, q_stride, kv_stride, out_head_offset, q_head_offset, kv_head_offset, scale, kv_lora_dim, w_up_k, w_up_v);
    return hipGetLastError();
}

hipError_t gpu_kv_write_compressed(
    float* d_k_cache,
    float* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int* d_pos,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox,
    const int kv_lora_dim,
    const int kv_frame_codec_enabled,
    const float* w_down_k,
    const float* w_down_v,
    hipStream_t stream
) {
    if (kv_lora_dim <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(kv_write_compressed_kernel, (kv_lora_dim + 255) / 256, 256, 0, stream,
                       d_k_cache, d_v_cache, d_k, d_v, d_pos,
                       num_kv_heads, head_dim, theta_base, neox,
                       kv_lora_dim, kv_frame_codec_enabled, w_down_k, w_down_v);
    return hipGetLastError();
}

hipError_t gpu_kv_write_batched_compressed(
    float* d_k_cache,
    float* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int start_pos,
    const int num_kv_heads,
    const int head_dim,
    const int seq_len,
    const int kv_lora_dim,
    const int kv_frame_codec_enabled,
    const float* w_down_k,
    const float* w_down_v,
    hipStream_t stream
) {
    if (kv_lora_dim <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(kv_write_batched_compressed_kernel, dim3((kv_lora_dim + 255) / 256, seq_len), 256, 0, stream,
                       d_k_cache, d_v_cache, d_k, d_v, start_pos,
                       num_kv_heads, head_dim, seq_len,
                       kv_lora_dim, kv_frame_codec_enabled, w_down_k, w_down_v);
    return hipGetLastError();
}

hipError_t gpu_reconstruct_kv_cache_prefix_sum(
    float* d_k_cache,
    float* d_v_cache,
    const int start_pos,
    const int seq_len,
    const int kv_lora_dim,
    hipStream_t stream
) {
    if (kv_lora_dim <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(reconstruct_kv_cache_prefix_sum_kernel, (kv_lora_dim + 255) / 256, 256, 0, stream,
                       d_k_cache, d_v_cache, start_pos, seq_len, kv_lora_dim);
    return hipGetLastError();
}

} // extern "C"

// ── TurboQuant Helpers ─────────────────────────────────────────────────────────────

__device__ inline float get_random_projection_sign(int i, int j) {
    unsigned int seed = i * 1664525 + j + 1013904223;
    seed = seed * 1103515245 + 12345;
    return (seed & 1) ? 1.0f : -1.0f;
}

__device__ inline void pack_3bit(uint8_t* dest, const int* src_indices, int d) {
    for (int i = 0; i + 7 < d; i += 8) {
        int x0 = src_indices[i];
        int x1 = src_indices[i+1];
        int x2 = src_indices[i+2];
        int x3 = src_indices[i+3];
        int x4 = src_indices[i+4];
        int x5 = src_indices[i+5];
        int x6 = src_indices[i+6];
        int x7 = src_indices[i+7];

        int byte_idx = (i / 8) * 3;
        dest[byte_idx]   = (x0 & 7) | ((x1 & 7) << 3) | ((x2 & 3) << 6);
        dest[byte_idx+1] = ((x2 >> 2) & 1) | ((x3 & 7) << 1) | ((x4 & 7) << 4) | ((x5 & 1) << 7);
        dest[byte_idx+2] = ((x5 >> 1) & 3) | ((x6 & 7) << 2) | ((x7 & 7) << 5);
    }
}

__device__ inline int get_unpacked_3bit(const uint8_t* src, int j) {
    int group = j / 8;
    int offset = j % 8;
    int byte_idx = group * 3;
    uint8_t b0 = src[byte_idx];
    uint8_t b1 = src[byte_idx+1];
    uint8_t b2 = src[byte_idx+2];
    if (offset == 0) return b0 & 7;
    if (offset == 1) return (b0 >> 3) & 7;
    if (offset == 2) return (b0 >> 6) | ((b1 & 1) << 2);
    if (offset == 3) return (b1 >> 1) & 7;
    if (offset == 4) return (b1 >> 4) & 7;
    if (offset == 5) return (b1 >> 7) | ((b2 & 3) << 1);
    if (offset == 6) return (b2 >> 2) & 7;
    if (offset == 7) return (b2 >> 5) & 7;
    return 0;
}

__device__ inline int get_unpacked_1bit(const uint8_t* src, int i) {
    int byte_idx = i / 8;
    int bit_offset = i % 8;
    return (src[byte_idx] >> bit_offset) & 1;
}

__device__ inline void pack_1bit(uint8_t* dest, const int* src_indices, int d) {
    for (int i = 0; i + 7 < d; i += 8) {
        uint8_t byte = 0;
        for (int b = 0; b < 8; ++b) {
            byte |= (src_indices[i + b] & 1) << b;
        }
        dest[i / 8] = byte;
    }
}

__device__ inline void pack_2bit(uint8_t* dest, const int* src_indices, int d) {
    for (int i = 0; i + 3 < d; i += 4) {
        int x0 = src_indices[i];
        int x1 = src_indices[i+1];
        int x2 = src_indices[i+2];
        int x3 = src_indices[i+3];
        dest[i / 4] = (x0 & 3) | ((x1 & 3) << 2) | ((x2 & 3) << 4) | ((x3 & 3) << 6);
    }
}

__device__ inline int get_unpacked_2bit(const uint8_t* src, int j) {
    return (src[j / 4] >> ((j % 4) * 2)) & 3;
}

__device__ inline void pack_4bit(uint8_t* dest, const int* src_indices, int d) {
    for (int i = 0; i + 1 < d; i += 2) {
        int x0 = src_indices[i];
        int x1 = src_indices[i+1];
        dest[i / 2] = (x0 & 0xF) | ((x1 & 0xF) << 4);
    }
}

__device__ inline int get_unpacked_4bit(const uint8_t* src, int j) {
    uint8_t byte = src[j / 2];
    return (j % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
}

// Supported TurboQuant bit widths: 1, 2, 3, 4.
// kv_lora_dim is always a power of two >= 8, so divisibility by
// packing granularity (8 for 1/3-bit, 4 for 2-bit, 2 for 4-bit) is guaranteed.
__device__ inline void pack_nbit(uint8_t* dest, const int* src_indices, int d, int bits) {
    if (bits == 1) {
        pack_1bit(dest, src_indices, d);
    } else if (bits == 2) {
        pack_2bit(dest, src_indices, d);
    } else if (bits == 4) {
        pack_4bit(dest, src_indices, d);
    } else {
        // bits == 3 (default / legacy) and any unsupported fallback
        pack_3bit(dest, src_indices, d);
    }
}

__device__ inline int get_unpacked_nbit(const uint8_t* src, int j, int bits) {
    if (bits == 1) {
        return get_unpacked_1bit(src, j);
    } else if (bits == 2) {
        return get_unpacked_2bit(src, j);
    } else if (bits == 4) {
        return get_unpacked_4bit(src, j);
    } else {
        // bits == 3 (default / legacy) and any unsupported fallback
        return get_unpacked_3bit(src, j);
    }
}

__device__ void parallel_fwht(float* s_data, int N, int tid, int num_threads) {
    for (int len = 1; len < N; len <<= 1) {
        __syncthreads();
        for (int idx = tid; idx < N / 2; idx += num_threads) {
            int base = (idx / len) * (len * 2);
            int offset = idx % len;
            int idx0 = base + offset;
            int idx1 = idx0 + len;
            float u = s_data[idx0];
            float v = s_data[idx1];
            s_data[idx0] = u + v;
            s_data[idx1] = u - v;
        }
    }
    __syncthreads();
}

// ── TurboQuant Kernels ─────────────────────────────────────────────────────────────

__global__ void kv_write_turboquant_kernel(
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox,
    const int kv_lora_dim,
    const int bits,
    const int num_centroids,
    const float* __restrict__ centroids,
    const float qjl_scale,
    const float* __restrict__ w_down_k,
    const float* __restrict__ w_down_v
) {
    const int pos = pos_ptr ? *pos_ptr : 0;
    const int kv_size = num_kv_heads * head_dim;
    const int j = threadIdx.x;

    if (j >= kv_lora_dim) return;

    float val_k = project_k_with_rope(k, w_down_k, j, kv_size, pos, num_kv_heads, head_dim, theta_base, neox);
    float val_v = project_v(v, w_down_v, j, kv_size);

    extern __shared__ float s_mem[];
    float* s_k = s_mem;
    float* s_v = s_mem + kv_lora_dim;
    int* s_k_idx = (int*)(s_mem + kv_lora_dim * 2);
    int* s_v_idx = s_k_idx + kv_lora_dim;
    int* s_k_sign = s_v_idx + kv_lora_dim;
    int* s_v_sign = s_k_sign + kv_lora_dim;

    s_k[j] = val_k;
    s_v[j] = val_v;
    __syncthreads();

    // Compute dynamic RMS scaling parameters
    __shared__ float s_rms_k;
    __shared__ float s_rms_v;
    if (j == 0) {
        float sum_sq_k = 0.0f;
        float sum_sq_v = 0.0f;
        for (int c = 0; c < kv_lora_dim; ++c) {
            sum_sq_k += s_k[c] * s_k[c];
            sum_sq_v += s_v[c] * s_v[c];
        }
        s_rms_k = sqrtf(sum_sq_k / (float)kv_lora_dim);
        s_rms_v = sqrtf(sum_sq_v / (float)kv_lora_dim);
    }
    __syncthreads();

    float rms_k = s_rms_k != 0.0f ? s_rms_k : 1.0f;
    float rms_v = s_rms_v != 0.0f ? s_rms_v : 1.0f;

    // Scale coordinates into optimal standard normal distribution range
    float scale_k = rms_k * sqrtf((float)kv_lora_dim);
    float scale_v = rms_v * sqrtf((float)kv_lora_dim);

    s_k[j] = val_k / scale_k;
    s_v[j] = val_v / scale_v;
    __syncthreads();

    parallel_fwht(s_k, kv_lora_dim, j, blockDim.x);
    parallel_fwht(s_v, kv_lora_dim, j, blockDim.x);

    float y_k = s_k[j];
    float y_v = s_v[j];

    int idx_k = 0;
    float min_dist_k = fabsf(y_k - centroids[0]);
    for (int c = 1; c < num_centroids; ++c) {
        float dist = fabsf(y_k - centroids[c]);
        if (dist < min_dist_k) {
            min_dist_k = dist;
            idx_k = c;
        }
    }

    int idx_v = 0;
    float min_dist_v = fabsf(y_v - centroids[0]);
    for (int c = 1; c < num_centroids; ++c) {
        float dist = fabsf(y_v - centroids[c]);
        if (dist < min_dist_v) {
            min_dist_v = dist;
            idx_v = c;
        }
    }

    s_k_idx[j] = idx_k;
    s_v_idx[j] = idx_v;

    s_k[j] = y_k - centroids[idx_k];
    s_v[j] = y_v - centroids[idx_v];
    __syncthreads();

    float sum_k = 0.0f;
    float sum_v = 0.0f;
    for (int col = 0; col < kv_lora_dim; ++col) {
        float R_jc = get_random_projection_sign(j, col);
        sum_k += R_jc * s_k[col];
        sum_v += R_jc * s_v[col];
    }

    s_k_sign[j] = (sum_k >= 0.0f) ? 1 : 0;
    s_v_sign[j] = (sum_v >= 0.0f) ? 1 : 0;
    __syncthreads();

    if (j == 0) {
        size_t pack_bytes = (kv_lora_dim * bits + 7) / 8;
        size_t qjl_bytes = (kv_lora_dim + 7) / 8;
        // V cache stores RMS scales (not QJL signs) at pos_v_base + pack_bytes,
        // so the per-position stride must accommodate the larger of:
        //   K: pack_bytes + qjl_bytes  (indices + signs)
        //   V: pack_bytes + TURBOQUANT_RMS_SCALE_BYTES  (indices + scales)
        size_t content_bytes = pack_bytes + ((qjl_bytes > TURBOQUANT_RMS_SCALE_BYTES) ? qjl_bytes : TURBOQUANT_RMS_SCALE_BYTES);
        size_t aligned_pos_bytes = (content_bytes + TURBOQUANT_POS_ALIGN_MASK) & ~static_cast<size_t>(TURBOQUANT_POS_ALIGN_MASK);

        uint8_t* pos_k_base = k_cache + (size_t)pos * aligned_pos_bytes;
        uint8_t* pos_v_base = v_cache + (size_t)pos * aligned_pos_bytes;

        pack_nbit(pos_k_base, s_k_idx, kv_lora_dim, bits);
        pack_nbit(pos_v_base, s_v_idx, kv_lora_dim, bits);

        uint8_t* qjl_k_base = pos_k_base + pack_bytes;
        uint8_t* qjl_v_base = pos_v_base + pack_bytes;

        for (int i = 0; i < kv_lora_dim; i += 8) {
            uint8_t b_k = 0;
            for (int bit = 0; bit < 8; ++bit) {
                b_k |= (s_k_sign[i + bit] << bit);
            }
            qjl_k_base[i / 8] = b_k;
        }

        // Store dynamic RMS values in the unused QJL V cache space (takes 8 bytes)
        float* scales_dest = (float*)qjl_v_base;
        scales_dest[0] = rms_k;
        scales_dest[1] = rms_v;
    }
}

__global__ void flash_attn_decode_turboquant_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    const int kv_lora_dim,
    const int bits,
    const int num_centroids,
    const float* __restrict__ centroids,
    const float qjl_scale,
    const float* __restrict__ w_up_k,
    const float* __restrict__ w_up_v
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int ratio = num_heads / num_kv_heads;
    const int kv_head_idx = (ratio > 0) ? (head_idx / ratio) : 0;
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    const float* head_q = q + head_idx * head_dim;

    extern __shared__ float s_scores[];
    float* s_reduce = s_scores + kDecodeScoreCapacity;
    float* s_q_latent = s_scores + kDecodeScoreCapacity + kDecodeReductionScratch;
    int* s_q_sign = (int*)(s_q_latent + kv_lora_dim);
    float* s_v = s_q_latent + kv_lora_dim * 2;

    if (tid < kv_lora_dim) {
        float val = 0.0f;
        if (w_up_k != nullptr) {
            for (int i = 0; i < head_dim; ++i) {
                val += head_q[i] * w_up_k[(head_offset + i) * kv_lora_dim + tid];
            }
        } else {
            val = (tid >= head_offset && tid < head_offset + head_dim) ? head_q[tid - head_offset] : 0.0f;
        }
        s_q_latent[tid] = val;
    }
    __syncthreads();

    parallel_fwht(s_q_latent, kv_lora_dim, tid, blockDim.x);

    if (tid < kv_lora_dim) {
        float sum_q = 0.0f;
        for (int col = 0; col < kv_lora_dim; ++col) {
            float R_tc = get_random_projection_sign(tid, col);
            sum_q += R_tc * s_q_latent[col];
        }
        s_q_sign[tid] = (sum_q >= 0.0f) ? 1 : 0;
    }
    __syncthreads();

    size_t pack_bytes = (kv_lora_dim * bits + 7) / 8;
    size_t qjl_bytes = (kv_lora_dim + 7) / 8;
    // V cache stores RMS scales (not QJL signs) at pos_v_base + pack_bytes,
    // so the per-position stride must accommodate the larger of:
    //   K: pack_bytes + qjl_bytes  (indices + signs)
    //   V: pack_bytes + TURBOQUANT_RMS_SCALE_BYTES  (indices + scales)
    size_t content_bytes = pack_bytes + ((qjl_bytes > TURBOQUANT_RMS_SCALE_BYTES) ? qjl_bytes : TURBOQUANT_RMS_SCALE_BYTES);
    size_t aligned_pos_bytes = (content_bytes + TURBOQUANT_POS_ALIGN_MASK) & ~static_cast<size_t>(TURBOQUANT_POS_ALIGN_MASK);

    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const uint8_t* pos_k_base = k_cache + (size_t)pos * aligned_pos_bytes;
        const uint8_t* qjl_k_base = pos_k_base + pack_bytes;
        const uint8_t* pos_v_base = v_cache + (size_t)pos * aligned_pos_bytes;
        const uint8_t* qjl_v_base = pos_v_base + pack_bytes;

        const float* scales_src = (const float*)qjl_v_base;
        float rms_k = scales_src[0];

        float score_base = 0.0f;
        for (int j = 0; j < kv_lora_dim; ++j) {
            int idx_k = get_unpacked_nbit(pos_k_base, j, bits);
            score_base += s_q_latent[j] * centroids[idx_k];
        }
        score_base = (score_base / sqrtf((float)kv_lora_dim)) * rms_k;

        float sum_corr = 0.0f;
        for (int i = 0; i < kv_lora_dim; ++i) {
            float q_s = s_q_sign[i] ? 1.0f : -1.0f;
            float k_s = get_unpacked_1bit(qjl_k_base, i) ? 1.0f : -1.0f;
            sum_corr += q_s * k_s;
        }
        float correction = (rms_k / sqrtf((float)kv_lora_dim)) * sum_corr;

        s_scores[pos] = (score_base + correction) * scale;
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

    float thread_val = 0.0f;
    for (int pos = 0; pos < seq_len; ++pos) {
        const uint8_t* pos_v_base = v_cache + (size_t)pos * aligned_pos_bytes;
        const uint8_t* qjl_v_base = pos_v_base + pack_bytes;

        const float* scales_src = (const float*)qjl_v_base;
        float rms_v = scales_src[1];

        if (tid < kv_lora_dim) {
            int idx_v = get_unpacked_nbit(pos_v_base, tid, bits);
            s_v[tid] = centroids[idx_v] * (rms_v / sqrtf((float)kv_lora_dim));
        }
        __syncthreads();

        parallel_fwht(s_v, kv_lora_dim, tid, blockDim.x);

        float v_val = 0.0f;
        if (tid < head_dim) {
            if (w_up_v != nullptr) {
                for (int j = 0; j < kv_lora_dim; ++j) {
                    v_val += s_v[j] * w_up_v[(head_offset + tid) * kv_lora_dim + j];
                }
            } else {
                v_val = (head_offset + tid < kv_lora_dim) ? s_v[head_offset + tid] : 0.0f;
            }
            thread_val += s_scores[pos] * v_val;
        }
        __syncthreads();
    }

    if (tid < head_dim) {
        head_out[tid] = thread_val * inv_sum;
    }
}

// ── TurboQuant FFI Launchers ───────────────────────────────────────────────────────

extern "C" {

hipError_t gpu_kv_write_turboquant(
    uint8_t* d_k_cache,
    uint8_t* d_v_cache,
    const float* d_k,
    const float* d_v,
    const int* d_pos,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox,
    const int kv_lora_dim,
    const int bits,
    const int num_centroids,
    const float* d_centroids,
    const float qjl_scale,
    const float* w_down_k,
    const float* w_down_v,
    hipStream_t stream
) {
    if (kv_lora_dim <= 0 || bits <= 0 || num_centroids <= 0) return hipErrorInvalidValue;

    // Grid: 1 block to process all kv_lora_dim coordinates
    // Block: kv_lora_dim threads
    size_t shared_mem_bytes = (kv_lora_dim * 2 * sizeof(float)) + (kv_lora_dim * 4 * sizeof(int));

    hipLaunchKernelGGL(kv_write_turboquant_kernel, 1, kv_lora_dim, shared_mem_bytes, stream,
                       d_k_cache, d_v_cache, d_k, d_v, d_pos,
                       num_kv_heads, head_dim, theta_base, neox,
                       kv_lora_dim, bits, num_centroids, d_centroids, qjl_scale, w_down_k, w_down_v);

    return hipGetLastError();
}

hipError_t gpu_flash_attn_decode_turboquant(
    float* d_out,
    const float* d_q,
    const uint8_t* d_k_cache,
    const uint8_t* d_v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    const int kv_lora_dim,
    const int bits,
    const int num_centroids,
    const float* d_centroids,
    const float qjl_scale,
    const float* w_up_k,
    const float* w_up_v,
    hipStream_t stream
) {
    if (kv_lora_dim <= 0 || bits <= 0 || num_centroids <= 0) return hipErrorInvalidValue;

    // Grid: num_heads blocks
    // Block: kv_lora_dim threads (or 256 fallback) to cover all latent dimension loading
    int threads_per_block = (kv_lora_dim > 256) ? kv_lora_dim : 256;
    size_t smem_size = (kDecodeScoreCapacity + kDecodeReductionScratch) * sizeof(float)
                     + (kv_lora_dim * 2 * sizeof(float)) + (kv_lora_dim * sizeof(int));

    hipLaunchKernelGGL(flash_attn_decode_turboquant_kernel, num_heads, threads_per_block, smem_size, stream,
                       d_out, d_q, d_k_cache, d_v_cache, seq_len,
                       num_heads, num_kv_heads, head_dim, scale,
                       kv_lora_dim, bits, num_centroids, d_centroids, qjl_scale, w_up_k, w_up_v);

    return hipGetLastError();
}

} // extern "C"

