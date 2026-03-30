#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block {
    half d;
    int8_t qs[16];
};

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ── Wavefront-Parallel Fused QKV Kernel (Fast Path) ───────────────────────────────

template<int N_WAVES>
__global__ void gemv_qkv_q4_0_f32_wave_parallel_kernel(
    const void* __restrict__ w_q,
    const void* __restrict__ w_k,
    const void* __restrict__ w_v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    const float* __restrict__ input,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int n_rows,
    int n_q,
    int n_kv
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col = blockIdx.x * N_WAVES + wave_id;
    const int n_blocks_total = n_rows / QK4_0;
    const int total_cols = n_q + 2 * n_kv;

    if (col >= total_cols) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const void* weights;
    int out_col;
    if (col < n_q) {
        weights = w_q; out_col = col;
    } else if (col < n_q + n_kv) {
        weights = w_k; out_col = col - n_q;
    } else {
        weights = w_v; out_col = col - n_q - n_kv;
    }

    const Q4_0_block* w_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(weights) + (out_col * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    float sum = 0.0f;
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q4_0_block* b = &w_col[block_idx];
        const float d = __half2float(b->d);
        const int row_offset = block_idx * QK4_0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            const uint8_t q = static_cast<uint8_t>(b->qs[l]);
            sum += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input[row_offset + l];
            sum += d * (static_cast<float>(q >> 4) - 8.0f) * s_input[row_offset + l + 16];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);

    if (lane_id == 0) {
        if (col < n_q) {
            out_q[col] = sum + (bias_q ? bias_q[col] : 0.0f);
        } else if (col < n_q + n_kv) {
            const int k_col = col - n_q;
            out_k[k_col] = sum + (bias_k ? bias_k[k_col] : 0.0f);
        } else {
            const int v_col = col - n_q - n_kv;
            out_v[v_col] = sum + (bias_v ? bias_v[v_col] : 0.0f);
        }
    }
}

// ── Wavefront-Parallel Fused Gate/Up + SwiGLU Kernel (Fast Path) ────────────────────

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_kernel(
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    const float* __restrict__ input,
    float* __restrict__ out_swiglu,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx = blockIdx.x * N_WAVES + wave_id;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx >= n_ff) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // Gate
    const Q4_0_block* w_gate_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    float sum_gate = 0.0f;
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q4_0_block* b = &w_gate_col[block_idx];
        const float d = __half2float(b->d);
        const int row_offset = block_idx * QK4_0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            const uint8_t q = static_cast<uint8_t>(b->qs[l]);
            sum_gate += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input[row_offset + l];
            sum_gate += d * (static_cast<float>(q >> 4) - 8.0f) * s_input[row_offset + l + 16];
        }
    }

    // Up
    const Q4_0_block* w_up_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    float sum_up = 0.0f;
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q4_0_block* b = &w_up_col[block_idx];
        const float d = __half2float(b->d);
        const int row_offset = block_idx * QK4_0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            const uint8_t q = static_cast<uint8_t>(b->qs[l]);
            sum_up += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input[row_offset + l];
            sum_up += d * (static_cast<float>(q >> 4) - 8.0f) * s_input[row_offset + l + 16];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate += __shfl_down(sum_gate, offset);
        sum_up += __shfl_down(sum_up, offset);
    }

    if (lane_id == 0) {
        out_swiglu[ff_idx] = silu(sum_gate) * sum_up;
    }
}

// ── Chunked Fallback Kernels ────────────────────────────────────────────────────────

template<int N_WAVES>
__global__ void gemv_qkv_q4_0_f32_chunked_kernel(
    const void* __restrict__ w_q, const void* __restrict__ w_k, const void* __restrict__ w_v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    const float* __restrict__ input,
    float* __restrict__ out_q, float* __restrict__ out_k, float* __restrict__ out_v,
    int n_rows, int n_q, int n_kv
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col = blockIdx.x * N_WAVES + wave_id;
    const int total_cols = n_q + 2 * n_kv;
    if (col >= total_cols) return;

    const void* weights;
    int out_col;
    if (col < n_q) { weights = w_q; out_col = col; }
    else if (col < n_q + n_kv) { weights = w_k; out_col = col - n_q; }
    else { weights = w_v; out_col = col - n_q - n_kv; }

    const int n_blocks_total = n_rows / QK4_0;
    const Q4_0_block* w_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(weights) + (out_col * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    __shared__ float s_input_chunk[4096];
    float sum = 0.0f;

    for (int chunk_start = 0; chunk_start < n_rows; chunk_start += 4096) {
        const int chunk_end = min(chunk_start + 4096, n_rows);
        for (int i = tid; i < (chunk_end - chunk_start); i += blockDim.x)
            s_input_chunk[i] = input[chunk_start + i];
        __syncthreads();

        for (int block_idx = (chunk_start/QK4_0) + lane_id; block_idx < (chunk_end/QK4_0); block_idx += 32) {
            const Q4_0_block* b = &w_col[block_idx];
            const float d = __half2float(b->d);
            const int lds_off = (block_idx - (chunk_start/QK4_0)) * QK4_0;
            #pragma unroll
            for (int l = 0; l < 16; ++l) {
                const uint8_t q = static_cast<uint8_t>(b->qs[l]);
                sum += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input_chunk[lds_off + l];
                sum += d * (static_cast<float>(q >> 4) - 8.0f) * s_input_chunk[lds_off + l + 16];
            }
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
    if (lane_id == 0) {
        if (col < n_q) {
            out_q[col] = sum + (bias_q ? bias_q[col] : 0.0f);
        } else if (col < n_q + n_kv) {
            const int k_col = col - n_q;
            out_k[k_col] = sum + (bias_k ? bias_k[k_col] : 0.0f);
        } else {
            const int v_col = col - n_q - n_kv;
            out_v[v_col] = sum + (bias_v ? bias_v[v_col] : 0.0f);
        }
    }
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_chunked_kernel(
    const void* __restrict__ w_gate, const void* __restrict__ w_up,
    const float* __restrict__ input,
    float* __restrict__ out_swiglu,
    int n_rows, int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx = blockIdx.x * N_WAVES + wave_id;
    if (ff_idx >= n_ff) return;

    const int n_blocks_total = n_rows / QK4_0;
    const Q4_0_block* w_gate_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    const Q4_0_block* w_up_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    __shared__ float s_input_chunk[4096];
    float sum_gate = 0.0f, sum_up = 0.0f;

    for (int chunk_start = 0; chunk_start < n_rows; chunk_start += 4096) {
        const int chunk_end = min(chunk_start + 4096, n_rows);
        for (int i = tid; i < (chunk_end - chunk_start); i += blockDim.x)
            s_input_chunk[i] = input[chunk_start + i];
        __syncthreads();

        for (int block_idx = (chunk_start/QK4_0) + lane_id; block_idx < (chunk_end/QK4_0); block_idx += 32) {
            const int lds_off = (block_idx - (chunk_start/QK4_0)) * QK4_0;
            
            const Q4_0_block* b_g = &w_gate_col[block_idx];
            float d_g = __half2float(b_g->d);
            
            const Q4_0_block* b_u = &w_up_col[block_idx];
            float d_u = __half2float(b_u->d);

            #pragma unroll
            for (int l = 0; l < 16; ++l) {
                const uint8_t q_g = static_cast<uint8_t>(b_g->qs[l]);
                sum_gate += d_g * (static_cast<float>(q_g & 0x0F) - 8.0f) * s_input_chunk[lds_off + l];
                sum_gate += d_g * (static_cast<float>(q_g >> 4) - 8.0f) * s_input_chunk[lds_off + l + 16];

                const uint8_t q_u = static_cast<uint8_t>(b_u->qs[l]);
                sum_up += d_u * (static_cast<float>(q_u & 0x0F) - 8.0f) * s_input_chunk[lds_off + l];
                sum_up += d_u * (static_cast<float>(q_u >> 4) - 8.0f) * s_input_chunk[lds_off + l + 16];
            }
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate += __shfl_down(sum_gate, offset);
        sum_up += __shfl_down(sum_up, offset);
    }
    if (lane_id == 0) out_swiglu[ff_idx] = silu(sum_gate) * sum_up;
}

// ── Dispatch Functions ─────────────────────────────────────────────────────────────

extern "C" hipError_t gemv_qkv_q4_0_f32_launch(
    const void* w_q, const void* w_k, const void* w_v,
    const float* bias_q, const float* bias_k, const float* bias_v,
    const float* input,
    float* out_q, float* out_k, float* out_v,
    int n_rows, int n_q, int n_kv,
    hipStream_t stream
) {
    const int N_WAVES = 8;
    int total_cols = n_q + 2 * n_kv;
    int n_blocks_x = (total_cols + N_WAVES - 1) / N_WAVES;

    if (n_rows * sizeof(float) <= 32768) {
        size_t shared_mem = n_rows * sizeof(float);
        gemv_qkv_q4_0_f32_wave_parallel_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
            w_q, w_k, w_v, bias_q, bias_k, bias_v, input, out_q, out_k, out_v, n_rows, n_q, n_kv
        );
    } else {
        gemv_qkv_q4_0_f32_chunked_kernel<N_WAVES><<<n_blocks_x, 256, 0, stream>>>(
            w_q, w_k, w_v, bias_q, bias_k, bias_v, input, out_q, out_k, out_v, n_rows, n_q, n_kv
        );
    }
    return hipGetLastError();
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_vulkan_style_kernel(
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    const float* __restrict__ input,
    float* __restrict__ out_swiglu,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    // v2 style: 2 FF indices per wave
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 2;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const Q4_0_block* w_gate_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    const Q4_0_block* w_up_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    
    bool has_ff1 = (ff_idx_base + 1 < n_ff);
    const Q4_0_block* w_gate_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;
    const Q4_0_block* w_up_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;

    float sum_gate0 = 0.0f, sum_up0 = 0.0f;
    float sum_gate1 = 0.0f, sum_up1 = 0.0f;

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK4_0;
        
        const float d_gate0 = __half2float(w_gate_col0[block_idx].d);
        const float d_up0   = __half2float(w_up_col0[block_idx].d);
        
        float d_gate1 = 0.0f, d_up1 = 0.0f;
        if (has_ff1) {
            d_gate1 = __half2float(w_gate_col1[block_idx].d);
            d_up1   = __half2float(w_up_col1[block_idx].d);
        }

        const uint32_t* qs_gate0 = reinterpret_cast<const uint32_t*>(w_gate_col0[block_idx].qs);
        const uint32_t* qs_up0   = reinterpret_cast<const uint32_t*>(w_up_col0[block_idx].qs);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4*i])[0];
            float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4*i + 16])[0];

            auto process_nibbles = [&](uint32_t q, float d, float4 il, float4 ih, float& sum) {
                sum += d * (static_cast<float>( q        & 0x0F) - 8.0f) * il.x;
                sum += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * il.y;
                sum += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * il.z;
                sum += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * il.w;
                sum += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * ih.x;
                sum += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * ih.y;
                sum += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * ih.z;
                sum += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * ih.w;
            };

            process_nibbles(qs_gate0[i], d_gate0, in_l, in_h, sum_gate0);
            process_nibbles(qs_up0[i],   d_up0,   in_l, in_h, sum_up0);
            if (has_ff1) {
                process_nibbles(reinterpret_cast<const uint32_t*>(w_gate_col1[block_idx].qs)[i], d_gate1, in_l, in_h, sum_gate1);
                process_nibbles(reinterpret_cast<const uint32_t*>(w_up_col1[block_idx].qs)[i],   d_up1,   in_l, in_h, sum_up1);
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate0 += __shfl_down(sum_gate0, offset);
        sum_up0   += __shfl_down(sum_up0, offset);
        if (has_ff1) {
            sum_gate1 += __shfl_down(sum_gate1, offset);
            sum_up1   += __shfl_down(sum_up1, offset);
        }
    }

    if (lane_id == 0) {
        out_swiglu[ff_idx_base] = (sum_gate0 / (1.0f + expf(-sum_gate0))) * sum_up0;
        if (has_ff1) {
            out_swiglu[ff_idx_base + 1] = (sum_gate1 / (1.0f + expf(-sum_gate1))) * sum_up1;
        }
    }
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_0_f32_launch(
    const void* w_gate, const void* w_up,
    const float* input,
    float* out_swiglu,
    int n_rows, int n_ff,
    hipStream_t stream
) {
    if (n_rows * sizeof(float) <= 32768) {
        size_t shared_mem = n_rows * sizeof(float);
        if (n_rows <= 1024) {
            const int N_WAVES = 4;
            int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
            gemv_gate_up_swiglu_q4_0_f32_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        } else {
            const int N_WAVES = 8;
            int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
            gemv_gate_up_swiglu_q4_0_f32_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        }
    } else {
        const int N_WAVES = 8;
        int n_blocks_x = (n_ff + N_WAVES - 1) / N_WAVES;
        gemv_gate_up_swiglu_q4_0_f32_chunked_kernel<N_WAVES><<<n_blocks_x, 256, 0, stream>>>(
            w_gate, w_up, input, out_swiglu, n_rows, n_ff
        );
    }
    return hipGetLastError();
}
