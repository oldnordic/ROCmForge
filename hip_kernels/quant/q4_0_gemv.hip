#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18
#define Q8_0_BLOCK_SIZE 36  // Changed from 34 to 36 (4 bytes float + 32 bytes qs)
constexpr int Q4_0_MULTI_ROW_COLS = 4;
constexpr int Q4_0_MULTI_ROW_SHARED_MEM_LIMIT = 32 * 1024;
constexpr int Q4_0_SMALL_WAVES = 4;
constexpr int Q4_0_FIXED_WAVES = 8;
constexpr int Q4_0_HIGH_WAVES = 16;
constexpr int Q4_0_THREADS_PER_BLOCK = Q4_0_FIXED_WAVES * WARP_SIZE;
constexpr float LAUNCH_MARKER_VALUE = 999999.0f;  // OBVIOUS marker for launch function

// Global counter to track launch function calls
__device__ int launch_call_counter = 0;

struct Q4_0_block {
    half d;
    uint8_t qs[16];
};

struct Q8_0_block {
    float d;  // Changed from half to float to fix conversion bug
    int8_t qs[QK4_0];
};

__device__ void debug_q4_0_block(const Q4_0_block* block, int block_idx, int thread_id) {
    if (block_idx == 0 && thread_id == 0) {
        // Read the scale as __half first, then convert
        const __half d_half = block->d;
        const float d = __half2float(d_half);

        printf("[GPU KERNEL] Block 0 scale (__half raw): 0x%04x\n", *(reinterpret_cast<const uint16_t*>(&d_half)));
        printf("[GPU KERNEL] Block 0 scale (converted): %f\n", d);

        // Print first 8 bytes of qs array
        printf("[GPU KERNEL] First 8 QS bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n",
               block->qs[0], block->qs[1], block->qs[2], block->qs[3],
               block->qs[4], block->qs[5], block->qs[6], block->qs[7]);

        // Dequantize first value manually
        const uint8_t q0 = block->qs[0];
        const float val0_lo = d * (static_cast<float>(q0 & 0x0F) - 8.0f);
        const float val0_hi = d * (static_cast<float>((q0 >> 4) & 0x0F) - 8.0f);

        printf("[GPU KERNEL] First Q[0] (lo): quant=%d, dequant=%f\n", q0 & 0x0F, val0_lo);
        printf("[GPU KERNEL] First Q[1] (hi): quant=%d, dequant=%f\n", (q0 >> 4) & 0x0F, val0_hi);
    }
}

static inline int select_multi_row_waves_q4_0(int /* ncols_dst */) {
    // The last documented fast decode baseline on this tree used the
    // 8-subwave launcher for Q4_0 GEMV and residual paths. A later 4-subwave
    // heuristic regressed the FFN-down hotspot on the 0.5B decode workload,
    // so keep the stable 8-subwave geometry here.
    return Q4_0_FIXED_WAVES;
}

static inline int select_multi_row_waves_q4_0_q8(int n_blocks_total) {
    // Keep the 4-wave path only for very small rows. For decode rows around
    // 896 (28 blocks), 8 waves improve shared-input reuse across output columns.
    if (n_blocks_total <= 24) {
        return Q4_0_SMALL_WAVES;
    }
    return Q4_0_FIXED_WAVES;
}

__device__ __forceinline__ int q4_0_q8_0_block_dot(
    const Q4_0_block* __restrict__ w_block,
    const int8_t* __restrict__ x_qs
) {
    int block_sum = 0;
    const uint32_t* packed_words = reinterpret_cast<const uint32_t*>(w_block->qs);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t packed_word = packed_words[i];
        const int base = i * 4;

        const uint8_t b0 = static_cast<uint8_t>(packed_word & 0xFF);
        const uint8_t b1 = static_cast<uint8_t>((packed_word >> 8) & 0xFF);
        const uint8_t b2 = static_cast<uint8_t>((packed_word >> 16) & 0xFF);
        const uint8_t b3 = static_cast<uint8_t>((packed_word >> 24) & 0xFF);

        block_sum += (static_cast<int>(b0 & 0x0F) - 8) * static_cast<int>(x_qs[base + 0]);
        block_sum += (static_cast<int>(b1 & 0x0F) - 8) * static_cast<int>(x_qs[base + 1]);
        block_sum += (static_cast<int>(b2 & 0x0F) - 8) * static_cast<int>(x_qs[base + 2]);
        block_sum += (static_cast<int>(b3 & 0x0F) - 8) * static_cast<int>(x_qs[base + 3]);

        block_sum += (static_cast<int>(b0 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 0]);
        block_sum += (static_cast<int>(b1 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 1]);
        block_sum += (static_cast<int>(b2 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 2]);
        block_sum += (static_cast<int>(b3 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 3]);
    }

    return block_sum;
}


// Loads input in 4096-element (16KB) chunks to stay within LDS limits.
template<int N_WAVES>
__launch_bounds__(Q4_0_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_0_f32_chunked_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col = blockIdx.x * N_WAVES + wave_id;
    const int n_blocks_total = n_rows / QK4_0;

    if (col >= ncols_dst) return;

    // DIAGNOSTIC: Write marker to output to confirm kernel execution
    if (col == 0 && lane_id == 0) {
        output[col] = -88888.0f;  // Unique marker for chunked kernel
    }

    const Q4_0_block* w_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    __shared__ float s_input_chunk[4096];
    float sum = 0.0f;

    for (int chunk_start = 0; chunk_start < n_rows; chunk_start += 4096) {
        const int chunk_end = min(chunk_start + 4096, n_rows);
        const int chunk_size = chunk_end - chunk_start;

        // Load chunk into LDS
        for (int i = tid; i < chunk_size; i += blockDim.x) {
            s_input_chunk[i] = input[chunk_start + i];
        }
        __syncthreads();

        // Process blocks in this chunk
        const int block_start = chunk_start / QK4_0;
        const int block_end = chunk_end / QK4_0;

        for (int block_idx = block_start + lane_id; block_idx < block_end; block_idx += WARP_SIZE) {
            const Q4_0_block* b = &w_col[block_idx];
            const float d = __half2float(b->d);
            const int lds_offset = (block_idx - block_start) * QK4_0;
            
            #pragma unroll
            for (int l = 0; l < 16; ++l) {
                const uint8_t q = static_cast<uint8_t>(b->qs[l]);
                sum += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input_chunk[lds_offset + l];
                sum += d * (static_cast<float>(q >> 4) - 8.0f) * s_input_chunk[lds_offset + l + 16];
            }
        }
        __syncthreads();
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane_id == 0) {
        // DIAGNOSTIC: Add marker to confirm chunked kernel execution
        if (col == 0) {
            sum += 200000.0f;  // Unique marker
        }
        output[col] = sum;
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_0_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_0_f32_residual_chunked_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col = blockIdx.x * N_WAVES + wave_id;
    const int n_blocks_total = n_rows / QK4_0;

    if (col >= ncols_dst) return;

    const Q4_0_block* w_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    __shared__ float s_input_chunk[4096];
    float sum = 0.0f;

    for (int chunk_start = 0; chunk_start < n_rows; chunk_start += 4096) {
        const int chunk_end = min(chunk_start + 4096, n_rows);
        const int chunk_size = chunk_end - chunk_start;

        for (int i = tid; i < chunk_size; i += blockDim.x) {
            s_input_chunk[i] = input[chunk_start + i];
        }
        __syncthreads();

        const int block_start = chunk_start / QK4_0;
        const int block_end = chunk_end / QK4_0;

        for (int block_idx = block_start + lane_id; block_idx < block_end; block_idx += WARP_SIZE) {
            const Q4_0_block* b = &w_col[block_idx];
            const float d = __half2float(b->d);
            const int lds_offset = (block_idx - block_start) * QK4_0;

            #pragma unroll
            for (int l = 0; l < 16; ++l) {
                const uint8_t q = static_cast<uint8_t>(b->qs[l]);
                sum += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input_chunk[lds_offset + l];
                sum += d * (static_cast<float>(q >> 4) - 8.0f) * s_input_chunk[lds_offset + l + 16];
            }
        }
        __syncthreads();
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane_id == 0) {
        output[col] = sum + residual[col];
    }
}

// ── Vulkan-style Multi-Row GEMV (Optimized Path) ──────────────────────────────────

template<int N_WAVES>
__launch_bounds__(Q4_0_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_0_f32_multi_row_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Each wave processes 4 output columns
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    // CRITICAL TEST: Write OBVIOUS pattern to ENTIRE output buffer
    // ALL threads participate to ensure complete overwrite
    if (blockIdx.x == 0 && wave_id == 0 && lane_id == 0) {
        // First thread writes entire buffer
        for (int i = 0; i < ncols_dst && i < 10; ++i) {
            output[i] = 77777.0f + i;  // OBVIOUS pattern: 77777, 77778, 77779, ...
        }
    }
    __syncthreads();  // Ensure marker is written
    if (blockIdx.x == 0) return;  // ALL threads in first block return after marker
    if (col_base >= ncols_dst) return;

    // DIAGNOSTIC: Write marker to output to confirm kernel execution
    if (blockIdx.x == 0 && wave_id == 0 && lane_id == 0) {
        if (col_base < ncols_dst) {
            output[col_base] = -99999.0f;  // Unique marker for multi-row kernel
        }
    }

    if (col_base >= ncols_dst) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const Q4_0_block* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const int row_offset = block_idx * QK4_0;
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4*i])[0];
            float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4*i + 16])[0];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (w_cols[c]) {
                    const Q4_0_block* b = &w_cols[c][block_idx];
                    const float d = __half2float(b->d);
                    const uint32_t q = reinterpret_cast<const uint32_t*>(b->qs)[i];

                    sums[c] += d * (static_cast<float>( q        & 0x0F) - 8.0f) * in_l.x;
                    sums[c] += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * in_l.y;
                    sums[c] += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * in_l.z;
                    sums[c] += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * in_l.w;
                    sums[c] += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * in_h.x;
                    sums[c] += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * in_h.y;
                    sums[c] += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * in_h.z;
                    sums[c] += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * in_h.w;
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (col_base + c < ncols_dst) {
                // DIAGNOSTIC: Add marker to confirm multi-row kernel execution
                if (col_base + c == 0) {
                    sums[c] += 100000.0f;  // Unique marker
                }
                output[col_base + c] = sums[c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_0_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_0_f32_residual_multi_row_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const Q4_0_block* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const int row_offset = block_idx * QK4_0;
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4*i])[0];
            float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4*i + 16])[0];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (w_cols[c]) {
                    const Q4_0_block* b = &w_cols[c][block_idx];
                    const float d = __half2float(b->d);
                    const uint32_t q = reinterpret_cast<const uint32_t*>(b->qs)[i];

                    sums[c] += d * (static_cast<float>( q        & 0x0F) - 8.0f) * in_l.x;
                    sums[c] += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * in_l.y;
                    sums[c] += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * in_l.z;
                    sums[c] += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * in_l.w;
                    sums[c] += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * in_h.x;
                    sums[c] += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * in_h.y;
                    sums[c] += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * in_h.z;
                    sums[c] += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * in_h.w;
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c] + residual[col_base + c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_0_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_0_q8_0_residual_multi_row_kernel(
    const void* __restrict__ weights_q4_0,
    const void* __restrict__ input_q8_0,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    extern __shared__ Q8_0_block s_input_q8[];
    uint8_t* s_input_bytes = reinterpret_cast<uint8_t*>(s_input_q8);
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input_q8_0);
    const int total_input_bytes = n_blocks_total * Q8_0_BLOCK_SIZE;

    for (int i = tid; i < total_input_bytes; i += blockDim.x) {
        s_input_bytes[i] = input_bytes[i];
    }
    __syncthreads();

    const Q4_0_block* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = x_block->d;  // Now float32, no conversion needed
        const int8_t* x_qs = x_block->qs;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (!w_cols[c]) {
                continue;
            }

            const Q4_0_block* w_block = &w_cols[c][block_idx];
            const float combined_scale = __half2float(w_block->d) * x_scale;
            const int block_sum = q4_0_q8_0_block_dot(w_block, x_qs);

            sums[c] += combined_scale * static_cast<float>(block_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c] + residual[col_base + c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_0_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_0_f32_q8_inline_residual_multi_row_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    extern __shared__ Q8_0_block s_input_q8[];
    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = input[row_offset + lane_id];
        float max_val = fabsf(val);

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, 32));
        }

        const float scale = fmaxf(__shfl(max_val, 0, WARP_SIZE) / 127.0f, 1e-30f);
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

    const Q4_0_block* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = x_block->d;  // Now float32, no conversion needed
        const int8_t* x_qs = x_block->qs;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (!w_cols[c]) {
                continue;
            }

            const Q4_0_block* w_block = &w_cols[c][block_idx];
            const float combined_scale = __half2float(w_block->d) * x_scale;
            const int block_sum = q4_0_q8_0_block_dot(w_block, x_qs);

            sums[c] += combined_scale * static_cast<float>(block_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c] + residual[col_base + c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_0_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_0_q8_0_multi_row_kernel(
    const void* __restrict__ weights_q4_0,
    const void* __restrict__ input_q8_0,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    extern __shared__ Q8_0_block s_input_q8[];
    uint8_t* s_input_bytes = reinterpret_cast<uint8_t*>(s_input_q8);
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input_q8_0);
    const int total_input_bytes = n_blocks_total * Q8_0_BLOCK_SIZE;

    for (int i = tid; i < total_input_bytes; i += blockDim.x) {
        s_input_bytes[i] = input_bytes[i];
    }
    __syncthreads();

    const Q4_0_block* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const Q8_0_block* x_block = &s_input_q8[block_idx];
        const float x_scale = x_block->d;  // Now float32, no conversion needed
        const int8_t* x_qs = x_block->qs;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (!w_cols[c]) {
                continue;
            }

            const Q4_0_block* w_block = &w_cols[c][block_idx];
            const float combined_scale = __half2float(w_block->d) * x_scale;
            const int block_sum = q4_0_q8_0_block_dot(w_block, x_qs);

            sums[c] += combined_scale * static_cast<float>(block_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

extern "C" hipError_t gemv_q4_0_f32_launch(
    const void* weights_q4_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    // CRITICAL: Write OBVIOUS pattern to ALL output values
    if (ncols_dst >= 10) {
        float markers[10] = {11111.0f, 22222.0f, 33333.0f, 44444.0f, 55555.0f, 66666.0f, 77777.0f, 88888.0f, 99999.0f, 101010.0f};
        hipMemcpyAsync(output, markers, 10 * sizeof(float), hipMemcpyHostToDevice, stream);
    }

    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;

    const int n_waves = select_multi_row_waves_q4_0(ncols_dst);
    const int threads_per_block = n_waves * WARP_SIZE;
    const int n_blocks_x =
        (ncols_dst + (n_waves * Q4_0_MULTI_ROW_COLS) - 1) / (n_waves * Q4_0_MULTI_ROW_COLS);

    if (n_rows * sizeof(float) <= Q4_0_MULTI_ROW_SHARED_MEM_LIMIT) {
        size_t shared_mem = n_rows * sizeof(float);
        gemv_q4_0_f32_multi_row_kernel<Q4_0_FIXED_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_q4_0, input, output, n_rows, ncols_dst
        );
    } else {
        // Fallback to old chunked path (TODO: optimize chunked path too)
        const int old_blocks_x = (ncols_dst + n_waves - 1) / n_waves;
        gemv_q4_0_f32_chunked_kernel<Q4_0_FIXED_WAVES><<<old_blocks_x, threads_per_block, 0, stream>>>(
            weights_q4_0, input, output, n_rows, ncols_dst
        );
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_f32_residual_launch(
    const void* weights_q4_0,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || residual == nullptr) return hipErrorInvalidValue;

    const int n_waves = select_multi_row_waves_q4_0(ncols_dst);
    const int threads_per_block = n_waves * WARP_SIZE;
    const int n_blocks_x =
        (ncols_dst + (n_waves * Q4_0_MULTI_ROW_COLS) - 1) / (n_waves * Q4_0_MULTI_ROW_COLS);

    if (n_rows * sizeof(float) <= Q4_0_MULTI_ROW_SHARED_MEM_LIMIT) {
        size_t shared_mem = n_rows * sizeof(float);
        gemv_q4_0_f32_residual_multi_row_kernel<Q4_0_FIXED_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_q4_0, input, residual, output, n_rows, ncols_dst
        );
    } else {
        const int old_blocks_x = (ncols_dst + n_waves - 1) / n_waves;
        gemv_q4_0_f32_residual_chunked_kernel<Q4_0_FIXED_WAVES><<<old_blocks_x, threads_per_block, 0, stream>>>(
            weights_q4_0, input, residual, output, n_rows, ncols_dst
        );
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_q8_0_launch(
    const void* weights_q4_0,
    const void* input_q8_0,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK4_0 != 0 || input_q8_0 == nullptr) return hipErrorInvalidValue;

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > Q4_0_MULTI_ROW_SHARED_MEM_LIMIT) return hipErrorInvalidValue;

    const int n_waves = select_multi_row_waves_q4_0(ncols_dst);
    const int threads_per_block = n_waves * WARP_SIZE;
    const int n_blocks_x =
        (ncols_dst + (n_waves * Q4_0_MULTI_ROW_COLS) - 1) / (n_waves * Q4_0_MULTI_ROW_COLS);

    gemv_q4_0_q8_0_multi_row_kernel<Q4_0_FIXED_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
        weights_q4_0, input_q8_0, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_q8_0_residual_launch(
    const void* weights_q4_0,
    const void* input_q8_0,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || residual == nullptr) return hipErrorInvalidValue;
    if (n_rows % QK4_0 != 0 || input_q8_0 == nullptr) return hipErrorInvalidValue;

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > Q4_0_MULTI_ROW_SHARED_MEM_LIMIT) return hipErrorInvalidValue;

    const int n_waves = select_multi_row_waves_q4_0(ncols_dst);
    const int threads_per_block = n_waves * WARP_SIZE;
    const int n_blocks_x =
        (ncols_dst + (n_waves * Q4_0_MULTI_ROW_COLS) - 1) / (n_waves * Q4_0_MULTI_ROW_COLS);

    gemv_q4_0_q8_0_residual_multi_row_kernel<Q4_0_FIXED_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
        weights_q4_0, input_q8_0, residual, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_f32_q8_inline_residual_variant_launch(
    const void* weights_q4_0,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    int variant,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || input == nullptr || residual == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) {
        return hipErrorInvalidValue;
    }

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_BLOCK_SIZE;
    if (shared_mem > Q4_0_MULTI_ROW_SHARED_MEM_LIMIT) {
        return hipErrorInvalidValue;
    }

    // variant 0 = heuristic select, variant 1 = force Q4_0_SMALL_WAVES, variant 2 = force Q4_0_HIGH_WAVES
    if (variant != 0 && variant != 1 && variant != 2) {
        return hipErrorInvalidValue;
    }
    const int n_waves = (variant == 1) ? Q4_0_SMALL_WAVES :
                        (variant == 2) ? Q4_0_HIGH_WAVES :
                        select_multi_row_waves_q4_0_q8(n_blocks_total);
    const int threads_per_block = n_waves * WARP_SIZE;
    const int n_blocks_x =
        (ncols_dst + (n_waves * Q4_0_MULTI_ROW_COLS) - 1) / (n_waves * Q4_0_MULTI_ROW_COLS);

    if (n_waves == Q4_0_SMALL_WAVES) {
        gemv_q4_0_f32_q8_inline_residual_multi_row_kernel<Q4_0_SMALL_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_q4_0, input, residual, output, n_rows, ncols_dst
        );
    } else if (n_waves == Q4_0_HIGH_WAVES) {
        gemv_q4_0_f32_q8_inline_residual_multi_row_kernel<Q4_0_HIGH_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_q4_0, input, residual, output, n_rows, ncols_dst
        );
    } else {
        gemv_q4_0_f32_q8_inline_residual_multi_row_kernel<Q4_0_FIXED_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_q4_0, input, residual, output, n_rows, ncols_dst
        );
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_f32_q8_inline_residual_launch(
    const void* weights_q4_0,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    // Delegate to variant launcher with variant=0 (heuristic selection)
    return gemv_q4_0_f32_q8_inline_residual_variant_launch(
        weights_q4_0, input, residual, output, n_rows, ncols_dst, 0, stream
    );
}
