#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block {
    half d;
    int8_t qs[16];
};

/**
 * FFN-Down prototype using Multi-Output Reuse (NUM_ROWS > 1).
 * Each wavefront processes 4 output elements (columns of the weight matrix).
 * Reuses the input vector from LDS 4 times.
 */
template<int N_WAVES>
__global__ void gemv_q4_0_f32_multi_row_vulkan_style_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Each wave processes 4 output columns
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // Setup pointers for 4 output columns
    const Q4_0_block* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK4_0;
        
        // Pre-load input chunks (vectorized)
        // We use 4 iterations of float4 to cover 32 elements (QK4_0)
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

                    // Low nibbles
                    sums[c] += d * (static_cast<float>( q        & 0x0F) - 8.0f) * in_l.x;
                    sums[c] += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * in_l.y;
                    sums[c] += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * in_l.z;
                    sums[c] += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * in_l.w;
                    // High nibbles
                    sums[c] += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * in_h.x;
                    sums[c] += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * in_h.y;
                    sums[c] += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * in_h.z;
                    sums[c] += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * in_h.w;
                }
            }
        }
    }

    // Wave-level reduction for 4 sums
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
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

extern "C" {
    hipError_t gemv_q4_0_f32_vulkan_style_launch(
        const void* weights_q4_0,
        const float* input,
        float* output,
        int n_rows,
        int ncols_dst,
        int n_waves,
        hipStream_t stream
    ) {
        CHECK_NULL(weights_q4_0);
        CHECK_NULL(input);
        CHECK_NULL(output);

        if (n_rows % QK4_0 != 0) return hipErrorInvalidValue;
        if (n_rows * sizeof(float) > MAX_SHARED_MEM) return hipErrorInvalidValue;

        // Simple heuristic selection for prototypes
        if (n_waves == 4) {
            int n_blocks = (ncols_dst + (4 * 4) - 1) / (4 * 4);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_q4_0_f32_multi_row_vulkan_style_kernel<4>),
                dim3(n_blocks), dim3(128), n_rows * sizeof(float), stream,
                weights_q4_0, input, output, n_rows, ncols_dst
            );
        } else {
            int n_blocks = (ncols_dst + (8 * 4) - 1) / (8 * 4);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_q4_0_f32_multi_row_vulkan_style_kernel<8>),
                dim3(n_blocks), dim3(256), n_rows * sizeof(float), stream,
                weights_q4_0, input, output, n_rows, ncols_dst
            );
        }
        CHECK_LAST();
        return hipSuccess;
    }
}
