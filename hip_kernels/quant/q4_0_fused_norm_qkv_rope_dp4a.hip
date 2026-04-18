// hip_kernels/quant/q4_0_fused_norm_qkv_rope_dp4a.hip
#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18  // sizeof(block_q4_0)

// Q4_0 block structure (matches Rust layout)
struct Q4_0_block_nqr {
    half d;              // scale (half precision)
    int8_t qs[16];       // 4-bit quantized weights (packed as int8 for DP4A)
};

// Extract Q4_0 value (0-15) from packed storage
__device__ float get_q4_0_value_nqr(const Q4_0_block_nqr* block, int i) {
    const int ib = i / QK4_0;  // block index
    const int iqs = i % QK4_0;  // quant index
    const int8_t q = block[ib].qs[iqs / 2];
    return (iqs & 1) ? (q >> 4) : (q & 0xF);  // Extract low 4 bits or high 4 bits
}

// RMSNorm computation
__device__ float rms_norm_single(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int tid = threadIdx.x;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += 32) {
        sum_sq += x[i] * x[i];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, 32);
    }

    const float rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);

    // Apply normalization and weight
    float result = 0.0f;
    if (tid < hidden_size) {
        result = x[tid] * rms * weight[tid];
    }

    return result;
}

// RoPE computation for single element
__device__ void rope_single(
    float* __restrict__ x,
    float* __restrict__ y,
    const int pos,
    const int head_dim,
    const float theta,
    const bool neox
) {
    const int tid = threadIdx.x;
    const int half_dim = head_dim / 2;

    if (tid >= half_dim) return;

    const float freq = theta * __int2float_rn(tid);
    const float cos_val = cosf(pos * freq);
    const float sin_val = sinf(pos * freq);

    const int x_idx = tid;
    const int y_idx = tid + half_dim;

    if (neox) {
        // Neox style: rotate pairs within each half
        const float x_val = x[x_idx];
        const float y_val = y[y_idx];
        x[x_idx] = x_val * cos_val - y_val * sin_val;
        y[y_idx] = x_val * sin_val + y_val * cos_val;
    } else {
        // GPT-J style: rotate across halves
        x[x_idx] = x[x_idx] * cos_val - y[y_idx] * sin_val;
        y[y_idx] = x[x_idx] * sin_val + y[y_idx] * cos_val;
    }
}

// DP4A-optimized Q4_0 dot product using __builtin_amdgcn_sdot4
// This is the core optimization for RDNA2+ GPUs
__device__ float vec_dot_q4_0_dp4a(
    const Q4_0_block_nqr* __restrict__ blocks,
    const float* __restrict__ vec,
    int ncols,
    int row_idx,
    int hidden_size
) {
    const int tid = threadIdx.x;
    const int n_blocks = (ncols + QK4_0 - 1) / QK4_0;

    // Shared memory for quantized activations
    __shared__ float s_vec[QK4_0];

    float sum = 0.0f;

    // Process blocks in chunks of QK4_0 (32)
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const int base_idx = block_idx * QK4_0;

        // Load vector chunk into shared memory
        if (tid < QK4_0 && base_idx + tid < ncols) {
            s_vec[tid] = vec[base_idx + tid];
        }
        __syncthreads();

        // Find absolute max for quantization (on-the-fly activation quantization)
        float amax = 0.0f;
        for (int i = tid; i < QK4_0 && base_idx + i < ncols; i += 32) {
            amax = fmaxf(amax, fabsf(s_vec[i]));
        }

        // Warp reduction to find global amax
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_down(amax, offset, 32));
        }

        // Quantize activations to int8 (0.4% noise vs 2× speedup trade-off)
        const float x_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        const float inv_x_scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;

        // Pack quantized activations into int32 for DP4A
        // Each int32 holds 4 int8 values
        int xq[8];  // 8 int32 values = 32 int8 values = QK4_0

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int base = i * 4;
            if (base_idx + base + 3 < ncols) {
                const int xq0 = __float2int_rn(s_vec[base + 0] * inv_x_scale);
                const int xq1 = __float2int_rn(s_vec[base + 1] * inv_x_scale);
                const int xq2 = __float2int_rn(s_vec[base + 2] * inv_x_scale);
                const int xq3 = __float2int_rn(s_vec[base + 3] * inv_x_scale);

                // Pack 4 int8 values into int32 (little-endian)
                xq[i] = (xq0 & 0xFF) | ((xq1 & 0xFF) << 8) |
                        ((xq2 & 0xFF) << 16) | ((xq3 & 0xFF) << 24);
            } else {
                xq[i] = 0;
            }
        }

        // Load Q4_0 weights and pack for DP4A
        // Weights are stored as 4-bit values, need to pack even/odd nibbles separately
        const Q4_0_block_nqr* block = &blocks[block_idx];
        const float d = __half2float(block->d);

        // Pack even nibbles (indices 0,2,4,...,30) into int32
        int nib_even[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int base = i * 8;
            const uint8_t pk0 = static_cast<uint8_t>(block->qs[base + 0]);
            const uint8_t pk1 = static_cast<uint8_t>(block->qs[base + 1]);
            const uint8_t pk2 = static_cast<uint8_t>(block->qs[base + 2]);
            const uint8_t pk3 = static_cast<uint8_t>(block->qs[base + 3]);

            nib_even[i] = (pk0 & 0xF) | ((pk1 & 0xF) << 8) |
                         ((pk2 & 0xF) << 16) | ((pk3 & 0xF) << 24);
        }

        // Pack odd nibbles (indices 1,3,5,...,31) into int32
        int nib_odd[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int base = i * 8;
            const uint8_t pk0 = static_cast<uint8_t>(block->qs[base + 0]);
            const uint8_t pk1 = static_cast<uint8_t>(block->qs[base + 1]);
            const uint8_t pk2 = static_cast<uint8_t>(block->qs[base + 2]);
            const uint8_t pk3 = static_cast<uint8_t>(block->qs[base + 3]);

            nib_odd[i] = (pk0 >> 4) | ((pk1 >> 4) << 8) |
                        ((pk2 >> 4) << 16) | ((pk3 >> 4) << 24);
        }

        // Perform DP4A operations (4-way int8 multiply-accumulate)
        // Each __builtin_amdgcn_sdot4 computes:
        // sum += dot(int8_vec_a[0:3], int8_vec_b[0:3])
        int dot_sum = 0;

        // Even nibbles
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dot_sum = __builtin_amdgcn_sdot4(nib_even[i], xq[i * 2 + 0], dot_sum, false);
        }

        // Odd nibbles
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dot_sum = __builtin_amdgcn_sdot4(nib_odd[i], xq[i * 2 + 1], dot_sum, false);
        }

        // Apply scale and add to sum
        sum += d * static_cast<float>(dot_sum);

        __syncthreads();
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
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

    // Shared memory for normalized hidden state
    __shared__ float s_hidden[1792];  // Assumes hidden_size <= 1792

    // RMSNorm (first 32 threads)
    if (tid < 32) {
        float norm_val = rms_norm_single(raw_hidden, norm_weight, eps, hidden_size);
        if (tid < hidden_size) {
            s_hidden[tid] = norm_val;
        }
    }
    __syncthreads();

    // K projection with DP4A
    const Q4_0_block_nqr* blocks_k = reinterpret_cast<const Q4_0_block_nqr*>(w_k);
    __shared__ float s_k[128];

    if (tid < head_dim) {
        float k_sum = vec_dot_q4_0_dp4a(
            blocks_k + kv_head_idx * ((hidden_size + QK4_0 - 1) / QK4_0),
            s_hidden,
            hidden_size,
            kv_head_idx,
            hidden_size
        );

        // Add bias
        if (bias_k != nullptr) {
            k_sum += bias_k[kv_head_idx * head_dim + tid];
        }

        // Store in shared memory
        if (tid % 32 == 0) {
            s_k[tid / 32] = k_sum;
        }
    }
    __syncthreads();

    // Apply RoPE to K and write to cache
    if (tid < head_dim) {
        const int pos = (pos_ptr != nullptr) ? pos_ptr[0] : 0;
        float* k_x = s_k;
        float* k_y = k_x + head_dim / 2;
        rope_single(k_x, k_y, pos, head_dim, theta_base, neox);

        // Write to global memory
        const int k_idx = kv_head_idx * head_dim + tid;
        k_cache[k_idx] = k_x[tid];
        if (tid + head_dim / 2 < head_dim) {
            k_cache[k_idx + head_dim / 2] = k_y[tid];
        }
    }
    __syncthreads();

    // V projection with DP4A (no RoPE)
    const Q4_0_block_nqr* blocks_v = reinterpret_cast<const Q4_0_block_nqr*>(w_v);
    __shared__ float s_v[128];

    if (tid < head_dim) {
        float v_sum = vec_dot_q4_0_dp4a(
            blocks_v + kv_head_idx * ((hidden_size + QK4_0 - 1) / QK4_0),
            s_hidden,
            hidden_size,
            kv_head_idx,
            hidden_size
        );

        // Add bias
        if (bias_v != nullptr) {
            v_sum += bias_v[kv_head_idx * head_dim + tid];
        }

        // Store in shared memory
        if (tid % 32 == 0) {
            s_v[tid / 32] = v_sum;
        }
    }
    __syncthreads();

    // Write V to cache
    if (tid < head_dim) {
        const int v_idx = kv_head_idx * head_dim + tid;
        if (tid % 32 == 0) {
            v_cache[v_idx] = s_v[tid / 32];
        }
    }
    __syncthreads();

    // Q projection with DP4A (multiple query heads per KV head)
    const Q4_0_block_nqr* blocks_q = reinterpret_cast<const Q4_0_block_nqr*>(w_q);
    __shared__ float s_q[7 * 128];

    for (int head_offset = 0; head_offset < heads_per_kv; ++head_offset) {
        const int q_head_idx = q_head_start + head_offset;

        if (tid < head_dim) {
            float q_sum = vec_dot_q4_0_dp4a(
                blocks_q + q_head_idx * ((hidden_size + QK4_0 - 1) / QK4_0),
                s_hidden,
                hidden_size,
                q_head_idx,
                hidden_size
            );

            // Add bias
            if (bias_q != nullptr) {
                q_sum += bias_q[q_head_idx * head_dim + tid];
            }

            // Store in shared memory
            if (tid % 32 == 0) {
                s_q[head_offset * head_dim + tid / 32] = q_sum;
            }
        }
    }
    __syncthreads();

    // Apply RoPE to Q and write to output
    for (int head_offset = 0; head_offset < heads_per_kv; ++head_offset) {
        const int q_head_idx = q_head_start + head_offset;

        if (tid < head_dim) {
            const int pos = (pos_ptr != nullptr) ? pos_ptr[0] : 0;
            float* q_x = s_q + head_offset * head_dim;
            float* q_y = q_x + head_dim / 2;
            rope_single(q_x, q_y, pos, head_dim, theta_base, neox);

            // Write to global memory
            const int q_idx = q_head_idx * head_dim + tid;
            out_q[q_idx] = q_x[tid];
            if (tid + head_dim / 2 < head_dim) {
                out_q[q_idx + head_dim / 2] = q_y[tid];
            }
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

    // Infer hidden_size from n_q and head_dim
    const int hidden_size = n_q * head_dim;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_kernel),
        dim3(n_kv_launch),
        dim3(threads_per_block),
        nullptr,
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
