// hip_kernels/quant/q4_0_fused_norm_qkv_rope_dp4a.hip
#include "common.hip"

// Q4_0 constants
#define QK4_0 32

// Q4_0 block structure
struct Q4_0_block_nqr {
    half d;              // scale (half precision)
    int8_t qs[16];       // 4-bit quantized weights
};

// DP4A-optimized Q4_0 dot product using __builtin_amdgcn_sdot4
// Computes dot product of a single row/weight vector with the shared activations
__device__ float vec_dot_q4_0_dp4a(
    const Q4_0_block_nqr* __restrict__ row_blocks,
    const float* __restrict__ s_hidden,
    int ncols
) {
    const int n_blocks = ncols / QK4_0;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const Q4_0_block_nqr* block = &row_blocks[block_idx];
        const float d = __half2float(block->d);

        // Load 32 float activations from s_hidden
        const int base_idx = block_idx * QK4_0;

        // Find absolute max for quantization (on-the-fly activation quantization)
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_0; ++i) {
            amax = fmaxf(amax, fabsf(s_hidden[base_idx + i]));
        }

        const float x_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        const float inv_x_scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;

        // Pack quantized activations into 8 int32 values (little-endian)
        int xq[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int base = base_idx + i * 4;
            const int xq0 = __float2int_rn(s_hidden[base + 0] * x_scale);
            const int xq1 = __float2int_rn(s_hidden[base + 1] * x_scale);
            const int xq2 = __float2int_rn(s_hidden[base + 2] * x_scale);
            const int xq3 = __float2int_rn(s_hidden[base + 3] * x_scale);

            // Pack 4 signed int8 values into int32
            xq[i] = (xq0 & 0xFF) | ((xq1 & 0xFF) << 8) |
                    ((xq2 & 0xFF) << 16) | ((xq3 & 0xFF) << 24);
        }

        // Pack signed weights with offset (-8)
        int nib_even[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int base = i * 4;
            const int pk0_s = static_cast<int>(block->qs[base + 0] & 0xF) - 8;
            const int pk1_s = static_cast<int>(block->qs[base + 1] & 0xF) - 8;
            const int pk2_s = static_cast<int>(block->qs[base + 2] & 0xF) - 8;
            const int pk3_s = static_cast<int>(block->qs[base + 3] & 0xF) - 8;

            nib_even[i] = (pk0_s & 0xFF) | ((pk1_s & 0xFF) << 8) |
                          ((pk2_s & 0xFF) << 16) | ((pk3_s & 0xFF) << 24);
        }

        int nib_odd[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int base = i * 4;
            const int pk0_s = static_cast<int>(block->qs[base + 0] >> 4) - 8;
            const int pk1_s = static_cast<int>(block->qs[base + 1] >> 4) - 8;
            const int pk2_s = static_cast<int>(block->qs[base + 2] >> 4) - 8;
            const int pk3_s = static_cast<int>(block->qs[base + 3] >> 4) - 8;

            nib_odd[i] = (pk0_s & 0xFF) | ((pk1_s & 0xFF) << 8) |
                         ((pk2_s & 0xFF) << 16) | ((pk3_s & 0xFF) << 24);
        }

        // Perform DP4A
        int dot_sum = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dot_sum = DOT4(nib_even[i], xq[i * 2 + 0], dot_sum);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dot_sum = DOT4(nib_odd[i], xq[i * 2 + 1], dot_sum);
        }

        // Apply scales
        sum += d * inv_x_scale * static_cast<float>(dot_sum);
    }

    return sum;
}

// DP4A-optimized fused kernel for RDNA2+ (gfx1030+)
// Combines RMSNorm + QKV projection + RoPE + KV cache write
__launch_bounds__(256, 1)
__global__ void gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_kernel(
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    const void* __restrict__ w_q,
    const void* __restrict__ w_k,
    const void* __restrict__ w_v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ out_q,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int n_rows,
    const int n_q,
    const int n_kv,
    const int* __restrict__ pos_ptr,
    const int head_dim,
    const float theta_base,
    const bool neox,
    const int hidden_size
) {
    const int kv_head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (kv_head_idx >= n_kv) return;

    const int heads_per_kv = n_q / n_kv;
    const int q_head_start = kv_head_idx * heads_per_kv;

    // Shared memory for normalized hidden state (supports hidden_size up to 4096)
    __shared__ float s_hidden[4096];

    // 1. Parallel RMSNorm: Compute sum of squares using all 256 threads
    __shared__ float s_sum_sq[8]; // Reduction buffer for 8 warps

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += 256) {
        sum_sq += raw_hidden[i] * raw_hidden[i];
    }

    // Warp reduction (32 threads per warp)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, 32);
    }

    // Write warp sum to shared memory
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    if (lane_id == 0) {
        s_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    // Reduce warp sums in thread 0
    float block_sum_sq = 0.0f;
    if (tid < 8) {
        block_sum_sq = s_sum_sq[tid];
    }
    if (tid < 32) {
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_sum_sq += __shfl_down(block_sum_sq, offset, 32);
        }
    }

    // Broadcast block_sum_sq to all threads via shared memory
    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(block_sum_sq / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();

    const float rms = s_rms;

    // Normalize and store to s_hidden using all threads
    for (int i = tid; i < hidden_size; i += 256) {
        s_hidden[i] = raw_hidden[i] * rms * norm_weight[i];
    }
    __syncthreads();

    const int n_blocks_per_row = hidden_size / QK4_0;
    const int half_dim = head_dim / 2;

    const Q4_0_block_nqr* blocks_k = reinterpret_cast<const Q4_0_block_nqr*>(w_k);
    const Q4_0_block_nqr* blocks_v = reinterpret_cast<const Q4_0_block_nqr*>(w_v);
    const Q4_0_block_nqr* blocks_q = reinterpret_cast<const Q4_0_block_nqr*>(w_q);

    const int pos = (pos_ptr != nullptr) ? pos_ptr[0] : 0;

    // 2. K projection + RoPE + Cache Write (Warp-free, thread-level pairing)
    if (tid < half_dim) {
        const int row_idx_x = kv_head_idx * head_dim + tid;
        const int row_idx_y = kv_head_idx * head_dim + tid + half_dim;

        float k_x = vec_dot_q4_0_dp4a(blocks_k + row_idx_x * n_blocks_per_row, s_hidden, hidden_size);
        if (bias_k != nullptr) {
            k_x += bias_k[row_idx_x];
        }

        float k_y = vec_dot_q4_0_dp4a(blocks_k + row_idx_y * n_blocks_per_row, s_hidden, hidden_size);
        if (bias_k != nullptr) {
            k_y += bias_k[row_idx_y];
        }

        // Apply RoPE rotation
        const float freq = theta_base * __int2float_rn(tid);
        const float cos_val = cosf(pos * freq);
        const float sin_val = sinf(pos * freq);

        const float rotated_k_x = k_x * cos_val - k_y * sin_val;
        const float rotated_k_y = k_x * sin_val + k_y * cos_val;

        // Write to global cache with pos-based offset
        const size_t cache_idx_x = (size_t)pos * (n_kv * head_dim) + row_idx_x;
        const size_t cache_idx_y = (size_t)pos * (n_kv * head_dim) + row_idx_y;
        k_cache[cache_idx_x] = rotated_k_x;
        k_cache[cache_idx_y] = rotated_k_y;
    }

    // 3. V projection + Cache Write
    if (tid < head_dim) {
        const int row_idx = kv_head_idx * head_dim + tid;

        float v_val = vec_dot_q4_0_dp4a(blocks_v + row_idx * n_blocks_per_row, s_hidden, hidden_size);
        if (bias_v != nullptr) {
            v_val += bias_v[row_idx];
        }

        const size_t cache_idx = (size_t)pos * (n_kv * head_dim) + row_idx;
        v_cache[cache_idx] = v_val;
    }

    // 4. Q projection + RoPE + Output Write
    if (tid < half_dim) {
        for (int head_offset = 0; head_offset < heads_per_kv; ++head_offset) {
            const int q_head_idx = q_head_start + head_offset;
            const int row_idx_x = q_head_idx * head_dim + tid;
            const int row_idx_y = q_head_idx * head_dim + tid + half_dim;

            float q_x = vec_dot_q4_0_dp4a(blocks_q + row_idx_x * n_blocks_per_row, s_hidden, hidden_size);
            if (bias_q != nullptr) {
                q_x += bias_q[row_idx_x];
            }

            float q_y = vec_dot_q4_0_dp4a(blocks_q + row_idx_y * n_blocks_per_row, s_hidden, hidden_size);
            if (bias_q != nullptr) {
                q_y += bias_q[row_idx_y];
            }

            // Apply RoPE rotation
            const float freq = theta_base * __int2float_rn(tid);
            const float cos_val = cosf(pos * freq);
            const float sin_val = sinf(pos * freq);

            const float rotated_q_x = q_x * cos_val - q_y * sin_val;
            const float rotated_q_y = q_x * sin_val + q_y * cos_val;

            out_q[row_idx_x] = rotated_q_x;
            out_q[row_idx_y] = rotated_q_y;
        }
    }
}

// C launch wrapper
extern "C" hipError_t gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_launch(
    const float* raw_hidden,
    const float* norm_weight,
    float eps,
    const void* w_q,
    const void* w_k,
    const void* w_v,
    const float* bias_q,
    const float* bias_k,
    const float* bias_v,
    float* out_q,
    float* k_cache,
    float* v_cache,
    int n_rows,
    int n_q,
    int n_kv,
    const int* pos_ptr,
    int head_dim,
    float theta_base,
    int neox,
    hipStream_t stream
) {
    const int n_kv_launch = std::max(1, n_kv);
    const int threads_per_block = 256;
    const int hidden_size = n_q * head_dim;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_kernel),
        dim3(n_kv_launch),
        dim3(threads_per_block),
        0,
        stream,
        raw_hidden,
        norm_weight,
        eps,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        out_q,
        k_cache,
        v_cache,
        n_rows,
        n_q,
        n_kv,
        pos_ptr,
        head_dim,
        theta_base,
        static_cast<bool>(neox),
        hidden_size
    );

    return hipGetLastError();
}
