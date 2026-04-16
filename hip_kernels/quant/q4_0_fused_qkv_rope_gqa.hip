// hip_kernels/quant/q4_0_fused_qkv_rope_gqa.hip
#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define BLOCK_SIZE_Q4_0 18  // sizeof(block_q4_0)

// Q4_0 block structure
struct block_q4_0 {
    float d;      // scale (f32, not f16 for simplicity)
    uint8_t qs[QK4_0 / 2];  // 4-bit quantized weights
};

// Extract Q4_0 value (0-15) from quantized storage
__device__ float get_q4_0_value(const block_q4_0* block, int i) {
    const int ib = i / QK4_0;  // block index
    const int iqs = i % QK4_0;  // quant index
    const uint8_t q = block[ib].qs[iqs / 2];
    return (iqs & 1) ? (q >> 4) : (q & 0xF);  // Extract low 4 bits or high 4 bits
}

// Q4_0 dot product for one output element
__device__ float vec_dot_q4_0(const block_q4_0* __restrict__ blocks, const float* vec, int ncols) {
    const int tid = threadIdx.x;
    const int n_blocks = (ncols + QK4_0 - 1) / QK4_0;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const float q_val = get_q4_0_value(blocks, block_idx * QK4_0 + tid);
        sum += blocks[block_idx].d * q_val * vec[block_idx * QK4_0 + tid];
    }

    // Warp reduction with explicit warpSize=32
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    return sum;
}

// Basic GQA QKV kernel (without RoPE)
__launch_bounds__(256, 1)
__global__ void fused_qkv_q4_0_gqa_kernel(
    const void* __restrict__ w_q,
    const void* __restrict__ w_k,
    const void* __restrict__ w_v,
    const float* __restrict__ input,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    const int n_heads,
    const int n_kv_heads,
    const int head_dim,
    const int hidden_size
) {
    const int kv_head_idx = blockIdx.x;  // Which KV head this block processes
    const int tid = threadIdx.x;

    if (kv_head_idx >= n_kv_heads) return;

    // Each KV head serves n_heads / n_kv_heads query heads
    const int heads_per_kv = n_heads / n_kv_heads;  // 7 for Qwen2.5-0.5B
    const int q_head_start = kv_head_idx * heads_per_kv;

    // Shared memory for input (broadcast to all threads)
    __shared__ float s_input[1792];  // Assumes hidden_size <= 1792 (14 * 128)
    if (tid < hidden_size) {
        s_input[tid] = input[tid];
    }
    __syncthreads();

    // Process K and V projections for this KV head
    const block_q4_0* blocks_k = reinterpret_cast<const block_q4_0*>(w_k);
    const block_q4_0* blocks_v = reinterpret_cast<const block_q4_0*>(w_v);

    // K projection (n_kv_heads * head_dim outputs)
    const int k_idx = kv_head_idx * head_dim;
    if (tid < head_dim) {
        float k_sum = 0.0f;
        for (int block_idx = 0; block_idx < (hidden_size + QK4_0 - 1) / QK4_0; ++block_idx) {
            k_sum += blocks_k[block_idx * n_kv_heads + kv_head_idx].d *
                     get_q4_0_value(blocks_k, block_idx * QK4_0 * n_kv_heads + kv_head_idx * QK4_0 + tid) *
                     s_input[block_idx * QK4_0 + tid];
        }
        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            k_sum += __shfl_down(k_sum, offset, 32);
        }
        if (tid % 32 == 0) {
            out_k[k_idx + tid / 32] = k_sum;
        }
    }

    // V projection (n_kv_heads * head_dim outputs)
    if (tid < head_dim) {
        float v_sum = 0.0f;
        for (int block_idx = 0; block_idx < (hidden_size + QK4_0 - 1) / QK4_0; ++block_idx) {
            v_sum += blocks_v[block_idx * n_kv_heads + kv_head_idx].d *
                     get_q4_0_value(blocks_v, block_idx * QK4_0 * n_kv_heads + kv_head_idx * QK4_0 + tid) *
                     s_input[block_idx * QK4_0 + tid];
        }
        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            v_sum += __shfl_down(v_sum, offset, 32);
        }
        if (tid % 32 == 0) {
            out_v[k_idx + tid / 32] = v_sum;
        }
    }

    __syncthreads();

    // Process Q projections for heads_per_kv query heads (7 heads)
    const block_q4_0* blocks_q = reinterpret_cast<const block_q4_0*>(w_q);

    for (int head_offset = 0; head_offset < heads_per_kv; ++head_offset) {
        const int q_head_idx = q_head_start + head_offset;
        const int q_idx = q_head_idx * head_dim;

        if (tid < head_dim) {
            float q_sum = 0.0f;
            for (int block_idx = 0; block_idx < (hidden_size + QK4_0 - 1) / QK4_0; ++block_idx) {
                q_sum += blocks_q[block_idx * n_heads + q_head_idx].d *
                         get_q4_0_value(blocks_q, block_idx * QK4_0 * n_heads + q_head_idx * QK4_0 + tid) *
                         s_input[block_idx * QK4_0 + tid];
            }
            // Warp reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_sum += __shfl_down(q_sum, offset, 32);
            }
            if (tid % 32 == 0) {
                out_q[q_idx + tid / 32] = q_sum;
            }
        }
    }
}

// Kernel launch wrapper
extern "C" hipError_t fused_qkv_q4_0_gqa_launch(
    const void* w_q,
    const void* w_k,
    const void* w_v,
    const float* input,
    float* out_q,
    float* out_k,
    float* out_v,
    const int n_heads,
    const int n_kv_heads,
    const int head_dim,
    const int hidden_size,
    hipStream_t stream
) {
    const int n_kv_heads_launch = std::max(1, n_kv_heads);
    const int threads_per_block = 256;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(fused_qkv_q4_0_gqa_kernel),
        dim3(n_kv_heads_launch),  // grid dims
        dim3(threads_per_block),  // block dims
        nullptr,  // shared memory
        stream,   // HIP stream
        w_q, w_k, w_v, input, out_q, out_k, out_v,
        n_heads, n_kv_heads, head_dim, hidden_size
    );

    return hipGetLastError();
}
