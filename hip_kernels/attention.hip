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

    float val_k = 0.0f;
    float val_v = 0.0f;

    if (w_down_k != nullptr) {
        for (int i = 0; i < kv_size; ++i) {
            float k_rope = get_rope_k(i, k, pos, num_kv_heads, head_dim, theta_base, neox);
            val_k += k_rope * w_down_k[j * kv_size + i];
        }
    } else {
        if (j < kv_size) {
            val_k = get_rope_k(j, k, pos, num_kv_heads, head_dim, theta_base, neox);
        }
    }

    if (w_down_v != nullptr) {
        for (int i = 0; i < kv_size; ++i) {
            val_v += v[i] * w_down_v[j * kv_size + i];
        }
    } else {
        if (j < kv_size) {
            val_v = v[j];
        }
    }

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
        
        float val_k = 0.0f;
        float val_v = 0.0f;
        
        const float* step_k = k + (size_t)s * kv_size;
        const float* step_v = v + (size_t)s * kv_size;

        if (w_down_k != nullptr) {
            for (int i = 0; i < kv_size; ++i) {
                val_k += step_k[i] * w_down_k[j * kv_size + i];
            }
        } else {
            if (j < kv_size) val_k = step_k[j];
        }

        if (w_down_v != nullptr) {
            for (int i = 0; i < kv_size; ++i) {
                val_v += step_v[i] * w_down_v[j * kv_size + i];
            }
        } else {
            if (j < kv_size) val_v = step_v[j];
        }

        if (kv_frame_codec_enabled && pos > 0) {
            float prev_k = 0.0f;
            float prev_v = 0.0f;
            if (s > 0) {
                const float* prev_step_k = k + (size_t)(s - 1) * kv_size;
                const float* prev_step_v = v + (size_t)(s - 1) * kv_size;
                if (w_down_k != nullptr) {
                    for (int i = 0; i < kv_size; ++i) prev_k += prev_step_k[i] * w_down_k[j * kv_size + i];
                } else {
                    if (j < kv_size) prev_k = prev_step_k[j];
                }
                if (w_down_v != nullptr) {
                    for (int i = 0; i < kv_size; ++i) prev_v += prev_step_v[i] * w_down_v[j * kv_size + i];
                } else {
                    if (j < kv_size) prev_v = prev_step_v[j];
                }
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
