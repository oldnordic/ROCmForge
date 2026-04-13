#include "common.hip"

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18
#define Q8_0_BLOCK_SIZE 34

struct Q4_0_block {
    half d;
    int8_t qs[16];
};

struct Q8_0_block {
    half d;
    int8_t qs[QK4_0];
};

__device__ inline float silu_q8(float x) {
    return x / (1.0f + expf(-x));
}

// Compute dot product for one packed Q4_0 byte against two Q8_0 values.
// Keeps exact math: ((q&0x0F)-8)*x_lo + ((q>>4)-8)*x_hi
__device__ inline int dot_q4_0_q8_0_byte(uint8_t q, int8_t x_lo, int8_t x_hi) {
    const int lo = static_cast<int>(q & 0x0F) - 8;
    const int hi = static_cast<int>(q >> 4) - 8;
    return lo * static_cast<int>(x_lo) + hi * static_cast<int>(x_hi);
}

template<int N_WAVES>
__global__ void gemv_gate_up_q4_0_q8_0_vulkan_style_kernel(
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    const void* __restrict__ input_q8_0,
    float* __restrict__ out_gate,
    float* __restrict__ out_up,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 2;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ Q8_0_block s_input_q8[];
    uint8_t* s_input_bytes = reinterpret_cast<uint8_t*>(s_input_q8);
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input_q8_0);
    const int total_input_bytes = n_blocks_total * Q8_0_BLOCK_SIZE;

    for (int i = tid; i < total_input_bytes; i += blockDim.x) {
        s_input_bytes[i] = input_bytes[i];
    }
    __syncthreads();

    const Q4_0_block* w_gate_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    const Q4_0_block* w_up_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    const bool has_ff1 = (ff_idx_base + 1 < n_ff);
    const Q4_0_block* w_gate_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;
    const Q4_0_block* w_up_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;

    float sum_gate0 = 0.0f, sum_up0 = 0.0f;
    float sum_gate1 = 0.0f, sum_up1 = 0.0f;

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);
        const int8_t* x_qs = x_block->qs;

        const Q4_0_block* gate0 = &w_gate_col0[block_idx];
        const Q4_0_block* up0 = &w_up_col0[block_idx];

        int gate0_block_sum = 0;
        int up0_block_sum = 0;
        int gate1_block_sum = 0;
        int up1_block_sum = 0;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t q_gate0 = static_cast<uint8_t>(gate0->qs[i]);
            gate0_block_sum += (static_cast<int>(q_gate0 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
            gate0_block_sum += (static_cast<int>(q_gate0 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

            const uint8_t q_up0 = static_cast<uint8_t>(up0->qs[i]);
            up0_block_sum += (static_cast<int>(q_up0 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
            up0_block_sum += (static_cast<int>(q_up0 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

            if (has_ff1) {
                const uint8_t q_gate1 = static_cast<uint8_t>(w_gate_col1[block_idx].qs[i]);
                gate1_block_sum += (static_cast<int>(q_gate1 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
                gate1_block_sum += (static_cast<int>(q_gate1 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

                const uint8_t q_up1 = static_cast<uint8_t>(w_up_col1[block_idx].qs[i]);
                up1_block_sum += (static_cast<int>(q_up1 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
                up1_block_sum += (static_cast<int>(q_up1 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);
            }
        }

        sum_gate0 += __half2float(gate0->d) * x_scale * static_cast<float>(gate0_block_sum);
        sum_up0 += __half2float(up0->d) * x_scale * static_cast<float>(up0_block_sum);
        if (has_ff1) {
            sum_gate1 += __half2float(w_gate_col1[block_idx].d) * x_scale * static_cast<float>(gate1_block_sum);
            sum_up1 += __half2float(w_up_col1[block_idx].d) * x_scale * static_cast<float>(up1_block_sum);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate0 += __shfl_down(sum_gate0, offset, 32);
        sum_up0 += __shfl_down(sum_up0, offset, 32);
        if (has_ff1) {
            sum_gate1 += __shfl_down(sum_gate1, offset, 32);
            sum_up1 += __shfl_down(sum_up1, offset, 32);
        }
    }

    if (lane_id == 0) {
        out_gate[ff_idx_base] = sum_gate0;
        out_up[ff_idx_base] = sum_up0;
        if (has_ff1) {
            out_gate[ff_idx_base + 1] = sum_gate1;
            out_up[ff_idx_base + 1] = sum_up1;
        }
    }
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_q8_0_vulkan_style_kernel(
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    const void* __restrict__ input_q8_0,
    float* __restrict__ out_swiglu,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 2;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ Q8_0_block s_input_q8[];
    uint8_t* s_input_bytes = reinterpret_cast<uint8_t*>(s_input_q8);
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input_q8_0);
    const int total_input_bytes = n_blocks_total * Q8_0_BLOCK_SIZE;

    for (int i = tid; i < total_input_bytes; i += blockDim.x) {
        s_input_bytes[i] = input_bytes[i];
    }
    __syncthreads();

    const Q4_0_block* w_gate_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    const Q4_0_block* w_up_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    const bool has_ff1 = (ff_idx_base + 1 < n_ff);
    const Q4_0_block* w_gate_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;
    const Q4_0_block* w_up_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;

    float sum_gate0 = 0.0f, sum_up0 = 0.0f;
    float sum_gate1 = 0.0f, sum_up1 = 0.0f;

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);
        const int8_t* x_qs = x_block->qs;

        const Q4_0_block* gate0 = &w_gate_col0[block_idx];
        const Q4_0_block* up0 = &w_up_col0[block_idx];

        int gate0_block_sum = 0;
        int up0_block_sum = 0;
        int gate1_block_sum = 0;
        int up1_block_sum = 0;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t q_gate0 = static_cast<uint8_t>(gate0->qs[i]);
            gate0_block_sum += (static_cast<int>(q_gate0 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
            gate0_block_sum += (static_cast<int>(q_gate0 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

            const uint8_t q_up0 = static_cast<uint8_t>(up0->qs[i]);
            up0_block_sum += (static_cast<int>(q_up0 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
            up0_block_sum += (static_cast<int>(q_up0 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

            if (has_ff1) {
                const uint8_t q_gate1 = static_cast<uint8_t>(w_gate_col1[block_idx].qs[i]);
                gate1_block_sum += (static_cast<int>(q_gate1 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
                gate1_block_sum += (static_cast<int>(q_gate1 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

                const uint8_t q_up1 = static_cast<uint8_t>(w_up_col1[block_idx].qs[i]);
                up1_block_sum += (static_cast<int>(q_up1 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
                up1_block_sum += (static_cast<int>(q_up1 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);
            }
        }

        sum_gate0 += __half2float(gate0->d) * x_scale * static_cast<float>(gate0_block_sum);
        sum_up0 += __half2float(up0->d) * x_scale * static_cast<float>(up0_block_sum);
        if (has_ff1) {
            sum_gate1 += __half2float(w_gate_col1[block_idx].d) * x_scale * static_cast<float>(gate1_block_sum);
            sum_up1 += __half2float(w_up_col1[block_idx].d) * x_scale * static_cast<float>(up1_block_sum);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate0 += __shfl_down(sum_gate0, offset, 32);
        sum_up0 += __shfl_down(sum_up0, offset, 32);
        if (has_ff1) {
            sum_gate1 += __shfl_down(sum_gate1, offset, 32);
            sum_up1 += __shfl_down(sum_up1, offset, 32);
        }
    }

    if (lane_id == 0) {
        out_swiglu[ff_idx_base] = silu_q8(sum_gate0) * sum_up0;
        if (has_ff1) {
            out_swiglu[ff_idx_base + 1] = silu_q8(sum_gate1) * sum_up1;
        }
    }
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_kernel(
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
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 2;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ Q8_0_block s_input_q8[];
    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = input[row_offset + lane_id];
        float max_val = fabsf(val);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, 32));
        }

        const float scale = fmaxf(__shfl(max_val, 0, 32) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);
        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
        }

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        s_input_q8[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
    }
    __syncthreads();

    const Q4_0_block* w_gate_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    const Q4_0_block* w_up_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    const bool has_ff1 = (ff_idx_base + 1 < n_ff);
    const Q4_0_block* w_gate_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;
    const Q4_0_block* w_up_col1 = has_ff1 ? reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + ((ff_idx_base + 1) * n_blocks_total) * Q4_0_BLOCK_SIZE
    ) : nullptr;

    float sum_gate0 = 0.0f, sum_up0 = 0.0f;
    float sum_gate1 = 0.0f, sum_up1 = 0.0f;

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);
        const int8_t* x_qs = x_block->qs;

        const Q4_0_block* gate0 = &w_gate_col0[block_idx];
        const Q4_0_block* up0 = &w_up_col0[block_idx];

        int gate0_block_sum = 0;
        int up0_block_sum = 0;
        int gate1_block_sum = 0;
        int up1_block_sum = 0;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t q_gate0 = static_cast<uint8_t>(gate0->qs[i]);
            gate0_block_sum += (static_cast<int>(q_gate0 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
            gate0_block_sum += (static_cast<int>(q_gate0 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

            const uint8_t q_up0 = static_cast<uint8_t>(up0->qs[i]);
            up0_block_sum += (static_cast<int>(q_up0 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
            up0_block_sum += (static_cast<int>(q_up0 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

            if (has_ff1) {
                const uint8_t q_gate1 = static_cast<uint8_t>(w_gate_col1[block_idx].qs[i]);
                gate1_block_sum += (static_cast<int>(q_gate1 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
                gate1_block_sum += (static_cast<int>(q_gate1 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);

                const uint8_t q_up1 = static_cast<uint8_t>(w_up_col1[block_idx].qs[i]);
                up1_block_sum += (static_cast<int>(q_up1 & 0x0F) - 8) * static_cast<int>(x_qs[i]);
                up1_block_sum += (static_cast<int>(q_up1 >> 4) - 8) * static_cast<int>(x_qs[i + 16]);
            }
        }

        sum_gate0 += __half2float(gate0->d) * x_scale * static_cast<float>(gate0_block_sum);
        sum_up0 += __half2float(up0->d) * x_scale * static_cast<float>(up0_block_sum);
        if (has_ff1) {
            sum_gate1 += __half2float(w_gate_col1[block_idx].d) * x_scale * static_cast<float>(gate1_block_sum);
            sum_up1 += __half2float(w_up_col1[block_idx].d) * x_scale * static_cast<float>(up1_block_sum);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate0 += __shfl_down(sum_gate0, offset, 32);
        sum_up0 += __shfl_down(sum_up0, offset, 32);
        if (has_ff1) {
            sum_gate1 += __shfl_down(sum_gate1, offset, 32);
            sum_up1 += __shfl_down(sum_up1, offset, 32);
        }
    }

    if (lane_id == 0) {
        out_swiglu[ff_idx_base] = silu_q8(sum_gate0) * sum_up0;
        if (has_ff1) {
            out_swiglu[ff_idx_base + 1] = silu_q8(sum_gate1) * sum_up1;
        }
    }
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_v2_kernel(
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
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ Q8_0_block s_input_q8[];
    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = input[row_offset + lane_id];
        float max_val = fabsf(val);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, 32));
        }

        const float scale = fmaxf(__shfl(max_val, 0, 32) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);
        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
        }

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        s_input_q8[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
    }
    __syncthreads();

    const Q4_0_block* w_gate_cols[4];
    const Q4_0_block* w_up_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int ff_idx = ff_idx_base + c;
        w_gate_cols[c] = reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(w_gate) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
        w_up_cols[c] = reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(w_up) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
    }

    float sum_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);
        const int8_t* x_qs = x_block->qs;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const Q4_0_block* gate_block = &w_gate_cols[c][block_idx];
            const Q4_0_block* up_block = &w_up_cols[c][block_idx];

            int gate_block_sum = 0;
            int up_block_sum = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                gate_block_sum += dot_q4_0_q8_0_byte(
                    static_cast<uint8_t>(gate_block->qs[i]), x_qs[i], x_qs[i + 16]);
                up_block_sum += dot_q4_0_q8_0_byte(
                    static_cast<uint8_t>(up_block->qs[i]), x_qs[i], x_qs[i + 16]);
            }

            sum_gate[c] += __half2float(gate_block->d) * x_scale * static_cast<float>(gate_block_sum);
            sum_up[c] += __half2float(up_block->d) * x_scale * static_cast<float>(up_block_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_gate[c] += __shfl_down(sum_gate[c], offset, 32);
            sum_up[c] += __shfl_down(sum_up[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            out_swiglu[ff_idx_base + c] = silu_q8(sum_gate[c]) * sum_up[c];
        }
    }
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_vulkan_style_v2_kernel(
    const void* __restrict__ w_gate_up_interleaved,
    const float* __restrict__ input,
    float* __restrict__ out_swiglu,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ Q8_0_block s_input_q8[];
    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = input[row_offset + lane_id];
        float max_val = fabsf(val);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, 32));
        }

        const float scale = fmaxf(__shfl(max_val, 0, 32) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);
        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
        }

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        s_input_q8[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
    }
    __syncthreads();

    const uint8_t* base = static_cast<const uint8_t*>(w_gate_up_interleaved);
    const size_t pair_stride = static_cast<size_t>(n_blocks_total) * (2 * Q4_0_BLOCK_SIZE);

    float sum_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);
        const int8_t* x_qs = x_block->qs;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int ff_idx = ff_idx_base + c;
            if (ff_idx >= n_ff) {
                continue;
            }

            const uint8_t* pair_ptr = base + static_cast<size_t>(ff_idx) * pair_stride
                + static_cast<size_t>(block_idx) * (2 * Q4_0_BLOCK_SIZE);
            const Q4_0_block* gate_block = reinterpret_cast<const Q4_0_block*>(pair_ptr);
            const Q4_0_block* up_block =
                reinterpret_cast<const Q4_0_block*>(pair_ptr + Q4_0_BLOCK_SIZE);

            int gate_block_sum = 0;
            int up_block_sum = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                gate_block_sum += dot_q4_0_q8_0_byte(
                    static_cast<uint8_t>(gate_block->qs[i]), x_qs[i], x_qs[i + 16]);
                up_block_sum += dot_q4_0_q8_0_byte(
                    static_cast<uint8_t>(up_block->qs[i]), x_qs[i], x_qs[i + 16]);
            }

            sum_gate[c] += __half2float(gate_block->d) * x_scale * static_cast<float>(gate_block_sum);
            sum_up[c] += __half2float(up_block->d) * x_scale * static_cast<float>(up_block_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_gate[c] += __shfl_down(sum_gate[c], offset, 32);
            sum_up[c] += __shfl_down(sum_up[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int ff_idx = ff_idx_base + c;
            if (ff_idx < n_ff) {
                out_swiglu[ff_idx] = silu_q8(sum_gate[c]) * sum_up[c];
            }
        }
    }
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_vulkan_style_v2_kernel(
    const void* __restrict__ w_gate_up_interleaved_tile4,
    const float* __restrict__ input,
    float* __restrict__ out_swiglu,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ Q8_0_block s_input_q8[];
    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = input[row_offset + lane_id];
        float max_val = fabsf(val);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, 32));
        }

        const float scale = fmaxf(__shfl(max_val, 0, 32) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);
        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
        }

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        s_input_q8[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
    }
    __syncthreads();

    const uint8_t* base = static_cast<const uint8_t*>(w_gate_up_interleaved_tile4);
    const size_t tile_stride = static_cast<size_t>(n_blocks_total) * (4 * 2 * Q4_0_BLOCK_SIZE);
    const uint8_t* tile_base = base + static_cast<size_t>(ff_idx_base / 4) * tile_stride;

    float sum_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);
        const int8_t* x_qs = x_block->qs;
        const uint8_t* block_base =
            tile_base + static_cast<size_t>(block_idx) * (4 * 2 * Q4_0_BLOCK_SIZE);

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint8_t* pair_ptr = block_base + c * (2 * Q4_0_BLOCK_SIZE);
            const Q4_0_block* gate_block = reinterpret_cast<const Q4_0_block*>(pair_ptr);
            const Q4_0_block* up_block =
                reinterpret_cast<const Q4_0_block*>(pair_ptr + Q4_0_BLOCK_SIZE);

            int gate_block_sum = 0;
            int up_block_sum = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                gate_block_sum += dot_q4_0_q8_0_byte(
                    static_cast<uint8_t>(gate_block->qs[i]), x_qs[i], x_qs[i + 16]);
                up_block_sum += dot_q4_0_q8_0_byte(
                    static_cast<uint8_t>(up_block->qs[i]), x_qs[i], x_qs[i + 16]);
            }

            sum_gate[c] += __half2float(gate_block->d) * x_scale * static_cast<float>(gate_block_sum);
            sum_up[c] += __half2float(up_block->d) * x_scale * static_cast<float>(up_block_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_gate[c] += __shfl_down(sum_gate[c], offset, 32);
            sum_up[c] += __shfl_down(sum_up[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            out_swiglu[ff_idx_base + c] = silu_q8(sum_gate[c]) * sum_up[c];
        }
    }
}

extern "C" hipError_t gemv_gate_up_q4_0_q8_0_launch(
    const void* w_gate, const void* w_up,
    const void* input_q8_0,
    float* out_gate, float* out_up,
    int n_rows, int n_ff,
    hipStream_t stream
) {
    CHECK_NULL(w_gate);
    CHECK_NULL(w_up);
    CHECK_NULL(input_q8_0);
    CHECK_NULL(out_gate);
    CHECK_NULL(out_up);
    if (n_rows <= 0 || n_ff <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) {
        return hipErrorInvalidValue;
    }

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > 32768) {
        return hipErrorInvalidValue;
    }

    if (n_rows <= 1024) {
        const int N_WAVES = 4;
        int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
        gemv_gate_up_q4_0_q8_0_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
            w_gate, w_up, input_q8_0, out_gate, out_up, n_rows, n_ff
        );
    } else {
        const int N_WAVES = 8;
        int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
        gemv_gate_up_q4_0_q8_0_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
            w_gate, w_up, input_q8_0, out_gate, out_up, n_rows, n_ff
        );
    }
    return hipGetLastError();
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_0_q8_0_launch(
    const void* w_gate, const void* w_up,
    const void* input_q8_0,
    float* out_swiglu,
    int n_rows, int n_ff,
    hipStream_t stream
) {
    CHECK_NULL(w_gate);
    CHECK_NULL(w_up);
    CHECK_NULL(input_q8_0);
    CHECK_NULL(out_swiglu);
    if (n_rows <= 0 || n_ff <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) {
        return hipErrorInvalidValue;
    }

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > 32768) {
        return hipErrorInvalidValue;
    }

    if (n_rows <= 1024) {
        const int N_WAVES = 4;
        int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
        gemv_gate_up_swiglu_q4_0_q8_0_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
            w_gate, w_up, input_q8_0, out_swiglu, n_rows, n_ff
        );
    } else {
        const int N_WAVES = 8;
        int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
        gemv_gate_up_swiglu_q4_0_q8_0_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
            w_gate, w_up, input_q8_0, out_swiglu, n_rows, n_ff
        );
    }
    return hipGetLastError();
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant_launch(
    const void* w_gate, const void* w_up,
    const float* input,
    float* out_swiglu,
    int n_rows, int n_ff,
    int variant,
    hipStream_t stream
) {
    CHECK_NULL(w_gate);
    CHECK_NULL(w_up);
    CHECK_NULL(input);
    CHECK_NULL(out_swiglu);
    if (n_rows <= 0 || n_ff <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) {
        return hipErrorInvalidValue;
    }

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > 32768) {
        return hipErrorInvalidValue;
    }

    if (variant != 0 && variant != 1 && variant != 2) {
        return hipErrorInvalidValue;
    }

    if (n_rows <= 4096) {
        if ((n_ff % 4) == 0) {
            // variant 0 = current heuristic, variant 1 = force 4-wave v2 launch,
            // variant 2 = force 6-wave v2 launch to probe the occupancy/LDS tradeoff.
            if (variant == 1) {
                const int N_WAVES = 4;
                const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
                gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_v2_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
                    w_gate, w_up, input, out_swiglu, n_rows, n_ff
                );
            } else if (variant == 2) {
                const int N_WAVES = 6;
                const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
                gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_v2_kernel<N_WAVES><<<n_blocks_x, 192, shared_mem, stream>>>(
                    w_gate, w_up, input, out_swiglu, n_rows, n_ff
                );
            } else if (n_ff >= 4096) {
                // Baseline heuristic keeps 8-subwave launch for large FF widths.
                const int N_WAVES = 8;
                const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
                gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_v2_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
                    w_gate, w_up, input, out_swiglu, n_rows, n_ff
                );
            } else {
                const int N_WAVES = 4;
                const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
                gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_v2_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
                    w_gate, w_up, input, out_swiglu, n_rows, n_ff
                );
            }
        } else {
            const int N_WAVES = 4;
            int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
            gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        }
    } else {
        if (variant == 1) {
            const int N_WAVES = 4;
            int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
            gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        } else if (variant == 2) {
            const int N_WAVES = 6;
            int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
            gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 192, shared_mem, stream>>>(
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        } else {
            const int N_WAVES = 8;
            int n_blocks_x = (n_ff + (N_WAVES * 2) - 1) / (N_WAVES * 2);
            gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        }
    }
    return hipGetLastError();
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_launch(
    const void* w_gate_up_interleaved,
    const float* input,
    float* out_swiglu,
    int n_rows,
    int n_ff,
    hipStream_t stream
) {
    CHECK_NULL(w_gate_up_interleaved);
    CHECK_NULL(input);
    CHECK_NULL(out_swiglu);
    if (n_rows <= 0 || n_ff <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) {
        return hipErrorInvalidValue;
    }

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > 32768) {
        return hipErrorInvalidValue;
    }

    if ((n_ff % 4) == 0 && n_rows <= 4096) {
        const int N_WAVES = 8;
        const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
        gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_vulkan_style_v2_kernel<N_WAVES><<<
            n_blocks_x, 256, shared_mem, stream>>>(
            w_gate_up_interleaved, input, out_swiglu, n_rows, n_ff
        );
    } else {
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_launch(
    const void* w_gate_up_interleaved_tile4,
    const float* input,
    float* out_swiglu,
    int n_rows,
    int n_ff,
    hipStream_t stream
) {
    CHECK_NULL(w_gate_up_interleaved_tile4);
    CHECK_NULL(input);
    CHECK_NULL(out_swiglu);
    if (n_rows <= 0 || n_ff <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0 || (n_ff % 4) != 0) {
        return hipErrorInvalidValue;
    }

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > 32768) {
        return hipErrorInvalidValue;
    }

    if (n_rows <= 4096) {
        const int N_WAVES = 8;
        const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
        gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_vulkan_style_v2_kernel<N_WAVES><<<
            n_blocks_x, 256, shared_mem, stream>>>(
            w_gate_up_interleaved_tile4, input, out_swiglu, n_rows, n_ff
        );
    } else {
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_0_f32_q8_inline_launch(
    const void* w_gate, const void* w_up,
    const float* input,
    float* out_swiglu,
    int n_rows, int n_ff,
    hipStream_t stream
) {
    return gemv_gate_up_swiglu_q4_0_f32_q8_inline_variant_launch(
        w_gate, w_up, input, out_swiglu, n_rows, n_ff, 0, stream
    );
}
