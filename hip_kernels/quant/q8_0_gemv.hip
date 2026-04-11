#include "common.hip"

// Q8_0 constants
#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34
#define Q8_0_LM_HEAD_COLS 4
#define Q8_0_LM_HEAD_COLS_V2 8

constexpr int Q8_0_LM_HEAD_SHARED_MEM_LIMIT = 32 * 1024;

struct Q8_0_block {
    half d;
    int8_t qs[32];
};

__device__ inline float vec_dot_q8_0(const Q8_0_block* block, const float* vec, int offset) {
    float d = __half2float(block->d);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        sum += static_cast<float>(block->qs[i]) * vec[offset + i];
    }
    return d * sum;
}

__global__ void gemv_q8_0_f32_kernel(
    const void* __restrict__ weights_q8_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK8_0;

    if (col >= ncols_dst) return;

    const Q8_0_block* w_col = reinterpret_cast<const Q8_0_block*>(
        static_cast<const uint8_t*>(weights_q8_0) + col * n_blocks * Q8_0_BLOCK_SIZE
    );

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        sum += vec_dot_q8_0(&w_col[block_idx], input, block_idx * QK8_0);
    }

    // CRITICAL FIX: Use dynamic shared memory based on actual block size
    // The kernel was launched with variable block sizes (64, 128, 256) but
    // hardcoded shared memory array of 256, causing out-of-bounds access.
    extern __shared__ float partial_sums[];
    partial_sums[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[col] = partial_sums[0];
    }
}

static inline int select_lm_head_subwaves(int ncols_dst) {
    if (ncols_dst <= 4096) {
        return 4;
    }
    return 8;
}

static inline int select_lm_head_block_size(int n_rows) {
    const int n_blocks = n_rows / QK8_0;

    if (n_blocks <= 64) {
        return 64;
    }
    if (n_blocks <= 128) {
        return 128;
    }
    return 256;
}

template<int N_SUBWAVES>
__global__ void gemv_q8_0_f32_lm_head_multi_row_kernel(
    const void* __restrict__ weights_q8_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int subwave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col_base = (blockIdx.x * N_SUBWAVES + subwave_id) * Q8_0_LM_HEAD_COLS;
    const int n_blocks_total = n_rows / QK8_0;

    if (col_base >= ncols_dst) {
        return;
    }

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const Q8_0_block* w_cols[Q8_0_LM_HEAD_COLS];
    #pragma unroll
    for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? reinterpret_cast<const Q8_0_block*>(
                static_cast<const uint8_t*>(weights_q8_0) + col * n_blocks_total * Q8_0_BLOCK_SIZE
            )
            : nullptr;
    }

    float sums[Q8_0_LM_HEAD_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK8_0;
        const float4* input_vec = reinterpret_cast<const float4*>(&s_input[row_offset]);

        #pragma unroll
        for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
            if (!w_cols[c]) {
                continue;
            }

            const Q8_0_block* block = &w_cols[c][block_idx];
            const float d = __half2float(block->d);
            float dot = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 in = input_vec[i];
                const int q_offset = i * 4;

                dot += static_cast<float>(block->qs[q_offset + 0]) * in.x;
                dot += static_cast<float>(block->qs[q_offset + 1]) * in.y;
                dot += static_cast<float>(block->qs[q_offset + 2]) * in.z;
                dot += static_cast<float>(block->qs[q_offset + 3]) * in.w;
            }

            sums[c] += d * dot;
        }
    }

    #pragma unroll
    for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q8_0_LM_HEAD_COLS; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                output[col] = sums[c];
            }
        }
    }
}

template<int N_SUBWAVES>
__global__ void gemv_q8_0_f32_lm_head_multi_row_v2_kernel(
    const void* __restrict__ weights_q8_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int subwave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col_base = (blockIdx.x * N_SUBWAVES + subwave_id) * Q8_0_LM_HEAD_COLS_V2;
    const int n_blocks_total = n_rows / QK8_0;

    if (col_base >= ncols_dst) {
        return;
    }

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const Q8_0_block* w_cols[Q8_0_LM_HEAD_COLS_V2];
    #pragma unroll
    for (int c = 0; c < Q8_0_LM_HEAD_COLS_V2; ++c) {
        const int col = col_base + c;
        w_cols[c] = reinterpret_cast<const Q8_0_block*>(
            static_cast<const uint8_t*>(weights_q8_0) + col * n_blocks_total * Q8_0_BLOCK_SIZE
        );
    }

    float sums[Q8_0_LM_HEAD_COLS_V2] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK8_0;
        const float4* input_vec = reinterpret_cast<const float4*>(&s_input[row_offset]);

        #pragma unroll
        for (int c = 0; c < Q8_0_LM_HEAD_COLS_V2; ++c) {
            const Q8_0_block* block = &w_cols[c][block_idx];
            const float d = __half2float(block->d);
            float dot = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 in = input_vec[i];
                const int q_offset = i * 4;

                dot += static_cast<float>(block->qs[q_offset + 0]) * in.x;
                dot += static_cast<float>(block->qs[q_offset + 1]) * in.y;
                dot += static_cast<float>(block->qs[q_offset + 2]) * in.z;
                dot += static_cast<float>(block->qs[q_offset + 3]) * in.w;
            }

            sums[c] += d * dot;
        }
    }

    #pragma unroll
    for (int c = 0; c < Q8_0_LM_HEAD_COLS_V2; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q8_0_LM_HEAD_COLS_V2; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                output[col] = sums[c];
            }
        }
    }
}

extern "C" hipError_t gemv_q8_0_f32_launch(
    const void* weights_q8_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK8_0 != 0) return hipErrorInvalidValue;

    const int block_size = 256;
    const size_t dynamic_shared_mem = block_size * sizeof(float);

    gemv_q8_0_f32_kernel<<<ncols_dst, block_size, dynamic_shared_mem, stream>>>(
        weights_q8_0, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}

extern "C" hipError_t gemv_q8_0_f32_lm_head_launch(
    const void* weights_q8_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK8_0 != 0) return hipErrorInvalidValue;
    CHECK_NULL(weights_q8_0);
    CHECK_NULL(input);
    CHECK_NULL(output);

    const size_t input_shared_mem = static_cast<size_t>(n_rows) * sizeof(float);
    if (input_shared_mem <= Q8_0_LM_HEAD_SHARED_MEM_LIMIT) {
        const int n_subwaves = select_lm_head_subwaves(ncols_dst);
        const int threads_per_block = n_subwaves * 32;
        // The v2 path was originally restricted to <=1024 rows (0.5B-focused
        // tuning). Reuse it for larger decode rows when the input staging
        // still fits in LDS and the vocab layout is v2-aligned.
        const bool use_v2 =
            (n_subwaves == 8) &&
            ((ncols_dst % Q8_0_LM_HEAD_COLS_V2) == 0);

        const int blocks_x =
            (ncols_dst + (n_subwaves * Q8_0_LM_HEAD_COLS) - 1) / (n_subwaves * Q8_0_LM_HEAD_COLS);
        const int blocks_x_v2 =
            (ncols_dst + (n_subwaves * Q8_0_LM_HEAD_COLS_V2) - 1) / (n_subwaves * Q8_0_LM_HEAD_COLS_V2);

        if (use_v2) {
            gemv_q8_0_f32_lm_head_multi_row_v2_kernel<8>
                <<<blocks_x_v2, threads_per_block, input_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
        } else if (n_subwaves == 4) {
            gemv_q8_0_f32_lm_head_multi_row_kernel<4>
                <<<blocks_x, threads_per_block, input_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
        } else {
            gemv_q8_0_f32_lm_head_multi_row_kernel<8>
                <<<blocks_x, threads_per_block, input_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
        }
    } else {
        switch (select_lm_head_block_size(n_rows)) {
            case 64: {
                const size_t dynamic_shared_mem = 64 * sizeof(float);
                gemv_q8_0_f32_kernel<<<ncols_dst, 64, dynamic_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
                break;
            }
            case 128: {
                const size_t dynamic_shared_mem = 128 * sizeof(float);
                gemv_q8_0_f32_kernel<<<ncols_dst, 128, dynamic_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
                break;
            }
            default: {
                const size_t dynamic_shared_mem = 256 * sizeof(float);
                gemv_q8_0_f32_kernel<<<ncols_dst, 256, dynamic_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
                break;
            }
        }
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_q8_0_f32_lm_head_variant_launch(
    const void* weights_q8_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int variant,
    hipStream_t stream
) {
    if (variant != 0 && variant != 1) return hipErrorInvalidValue;
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK8_0 != 0) return hipErrorInvalidValue;
    CHECK_NULL(weights_q8_0);
    CHECK_NULL(input);
    CHECK_NULL(output);

    const size_t input_shared_mem = static_cast<size_t>(n_rows) * sizeof(float);
    if (input_shared_mem <= Q8_0_LM_HEAD_SHARED_MEM_LIMIT) {
        const int baseline_subwaves = select_lm_head_subwaves(ncols_dst);
        const int n_subwaves = (variant == 1) ? 4 : baseline_subwaves;
        const int threads_per_block = n_subwaves * 32;
        const bool use_v2 =
            (n_subwaves == 8) &&
            ((ncols_dst % Q8_0_LM_HEAD_COLS_V2) == 0);

        const int blocks_x =
            (ncols_dst + (n_subwaves * Q8_0_LM_HEAD_COLS) - 1) / (n_subwaves * Q8_0_LM_HEAD_COLS);
        const int blocks_x_v2 =
            (ncols_dst + (n_subwaves * Q8_0_LM_HEAD_COLS_V2) - 1) / (n_subwaves * Q8_0_LM_HEAD_COLS_V2);

        if (use_v2) {
            gemv_q8_0_f32_lm_head_multi_row_v2_kernel<8>
                <<<blocks_x_v2, threads_per_block, input_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
        } else if (n_subwaves == 4) {
            gemv_q8_0_f32_lm_head_multi_row_kernel<4>
                <<<blocks_x, threads_per_block, input_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
        } else {
            gemv_q8_0_f32_lm_head_multi_row_kernel<8>
                <<<blocks_x, threads_per_block, input_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
        }
    } else {
        switch (select_lm_head_block_size(n_rows)) {
            case 64: {
                const size_t dynamic_shared_mem = 64 * sizeof(float);
                gemv_q8_0_f32_kernel<<<ncols_dst, 64, dynamic_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
                break;
            }
            case 128: {
                const size_t dynamic_shared_mem = 128 * sizeof(float);
                gemv_q8_0_f32_kernel<<<ncols_dst, 128, dynamic_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
                break;
            }
            default: {
                const size_t dynamic_shared_mem = 256 * sizeof(float);
                gemv_q8_0_f32_kernel<<<ncols_dst, 256, dynamic_shared_mem, stream>>>(
                    weights_q8_0, input, output, n_rows, ncols_dst
                );
                break;
            }
        }
    }

    return hipGetLastError();
}
