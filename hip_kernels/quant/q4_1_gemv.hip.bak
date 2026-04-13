#include "common.hip"

// Q4_1 constants
#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

struct Q4_1_block {
    half d;
    half m;
    int8_t qs[16];
};

constexpr int Q4_1_MULTI_ROW_COLS = 4;
constexpr int Q4_1_MULTI_ROW_SHARED_MEM_LIMIT = 32 * 1024;

static inline int select_multi_row_waves_q4_1(int /* ncols_dst */) {
    // Keep Q4_1 decode on the 8-wave geometry. The 4-wave branch regressed
    // the real-model decode hot path (FFN down / residual GEMV) on this tree.
    return 8;
}

__device__ inline float vec_dot_q4_1(const Q4_1_block* block, const float* vec, int offset) {
    const float d = __half2float(block->d);
    const float m = __half2float(block->m);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = static_cast<uint8_t>(block->qs[i]);
        sum += (static_cast<float>(q & 0x0F) * d + m) * vec[offset + i];
        sum += (static_cast<float>(q >> 4) * d + m) * vec[offset + i + 16];
    }
    return sum;
}

template<bool ADD_RESIDUAL>
__global__ void gemv_q4_1_f32_legacy_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK4_1;

    if (col >= ncols_dst) return;

    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * n_blocks * Q4_1_BLOCK_SIZE
    );

    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        sum += vec_dot_q4_1(&w_col[block_idx], input, block_idx * QK4_1);
    }

    __shared__ float partial_sums[256];
    partial_sums[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float result = partial_sums[0];
        if constexpr (ADD_RESIDUAL) {
            result += residual[col];
        }
        output[col] = result;
    }
}

template<int N_WAVES, bool ADD_RESIDUAL>
__global__ void gemv_q4_1_f32_multi_row_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_1_MULTI_ROW_COLS;
    const int n_blocks_total = n_rows / QK4_1;

    if (col_base >= ncols_dst) {
        return;
    }

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const Q4_1_block* w_cols[Q4_1_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_1_MULTI_ROW_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? reinterpret_cast<const Q4_1_block*>(
                static_cast<const uint8_t*>(weights_q4_1) + col * n_blocks_total * Q4_1_BLOCK_SIZE
            )
            : nullptr;
    }

    float sums[Q4_1_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const int row_offset = block_idx * QK4_1;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i])[0];
            const float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i + 16])[0];

            #pragma unroll
            for (int c = 0; c < Q4_1_MULTI_ROW_COLS; ++c) {
                if (!w_cols[c]) {
                    continue;
                }

                const Q4_1_block* block = &w_cols[c][block_idx];
                const float d = __half2float(block->d);
                const float m = __half2float(block->m);
                const uint32_t q = reinterpret_cast<const uint32_t*>(block->qs)[i];

                sums[c] += (static_cast<float>( q        & 0x0F) * d + m) * in_l.x;
                sums[c] += (static_cast<float>((q >>  8) & 0x0F) * d + m) * in_l.y;
                sums[c] += (static_cast<float>((q >> 16) & 0x0F) * d + m) * in_l.z;
                sums[c] += (static_cast<float>((q >> 24) & 0x0F) * d + m) * in_l.w;
                sums[c] += (static_cast<float>((q >>  4) & 0x0F) * d + m) * in_h.x;
                sums[c] += (static_cast<float>((q >> 12) & 0x0F) * d + m) * in_h.y;
                sums[c] += (static_cast<float>((q >> 20) & 0x0F) * d + m) * in_h.z;
                sums[c] += (static_cast<float>((q >> 28) & 0x0F) * d + m) * in_h.w;
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_1_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_1_MULTI_ROW_COLS; ++c) {
            const int col = col_base + c;
            if (col >= ncols_dst) {
                continue;
            }

            float result = sums[c];
            if constexpr (ADD_RESIDUAL) {
                result += residual[col];
            }
            output[col] = result;
        }
    }
}

extern "C" hipError_t gemv_q4_1_f32_launch(
    const void* weights_q4_1,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK4_1 != 0) return hipErrorInvalidValue;
    CHECK_NULL(weights_q4_1);
    CHECK_NULL(input);
    CHECK_NULL(output);

    const int n_waves = select_multi_row_waves_q4_1(ncols_dst);
    const int threads_per_block = n_waves * WARP_SIZE;
    const size_t input_shared_mem = static_cast<size_t>(n_rows) * sizeof(float);

    if (input_shared_mem <= Q4_1_MULTI_ROW_SHARED_MEM_LIMIT) {
        const int blocks_x =
            (ncols_dst + (n_waves * Q4_1_MULTI_ROW_COLS) - 1) / (n_waves * Q4_1_MULTI_ROW_COLS);
        gemv_q4_1_f32_multi_row_kernel<8, false>
            <<<blocks_x, threads_per_block, input_shared_mem, stream>>>(
                weights_q4_1,
                input,
                nullptr,
                output,
                n_rows,
                ncols_dst
            );
    } else {
        gemv_q4_1_f32_legacy_kernel<false><<<ncols_dst, 256, 0, stream>>>(
            weights_q4_1,
            input,
            nullptr,
            output,
            n_rows,
            ncols_dst
        );
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_1_f32_residual_variant_launch(
    const void* weights_q4_1,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    int variant,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK4_1 != 0) return hipErrorInvalidValue;
    CHECK_NULL(weights_q4_1);
    CHECK_NULL(input);
    CHECK_NULL(residual);
    CHECK_NULL(output);

    const int n_waves = select_multi_row_waves_q4_1(ncols_dst);
    const int threads_per_block = n_waves * WARP_SIZE;
    const size_t input_shared_mem = static_cast<size_t>(n_rows) * sizeof(float);

    if (input_shared_mem <= Q4_1_MULTI_ROW_SHARED_MEM_LIMIT) {
        // Multi-row path stays unchanged regardless of variant
        const int blocks_x =
            (ncols_dst + (n_waves * Q4_1_MULTI_ROW_COLS) - 1) / (n_waves * Q4_1_MULTI_ROW_COLS);
        gemv_q4_1_f32_multi_row_kernel<8, true>
            <<<blocks_x, threads_per_block, input_shared_mem, stream>>>(
                weights_q4_1,
                input,
                residual,
                output,
                n_rows,
                ncols_dst
            );
    } else {
        // Legacy path: variant 0 uses 256 threads, variant 1 uses 128 threads
        const int legacy_threads = (variant == 1) ? 128 : 256;
        gemv_q4_1_f32_legacy_kernel<true><<<ncols_dst, legacy_threads, 0, stream>>>(
            weights_q4_1,
            input,
            residual,
            output,
            n_rows,
            ncols_dst
        );
    }

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_1_f32_residual_launch(
    const void* weights_q4_1,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    // Delegate to variant launch with baseline variant (0)
    return gemv_q4_1_f32_residual_variant_launch(
        weights_q4_1, input, residual, output, n_rows, ncols_dst, 0, stream
    );
}
