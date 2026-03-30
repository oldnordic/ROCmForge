#include "common.hip"

#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

struct Q4_1_block {
    half d;
    half m;
    int8_t qs[16];
};

__device__ inline float silu_ffn_exp(float x) {
    return x / (1.0f + expf(-x));
}

__device__ inline float vec_dot_q4_1_swiglu(
    const Q4_1_block* block,
    const float* gate,
    const float* up,
    int offset
) {
    const float d = __half2float(block->d);
    const float m = __half2float(block->m);
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = static_cast<uint8_t>(block->qs[i]);

        const float swiglu0 = silu_ffn_exp(gate[offset + i]) * up[offset + i];
        const float swiglu1 = silu_ffn_exp(gate[offset + i + 16]) * up[offset + i + 16];

        sum += (static_cast<float>(q & 0x0F) * d + m) * swiglu0;
        sum += (static_cast<float>(q >> 4) * d + m) * swiglu1;
    }

    return sum;
}

// Experimental FFN-down kernel:
// - consumes gate/up vectors directly
// - fuses SwiGLU with the Q4_1 down projection
// - uses one Wave32 per output column instead of the generic 256-thread reduction
__global__ void gemv_ffn_down_swiglu_q4_1_f32_experimental_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int lane_id = threadIdx.x;
    const int col = blockIdx.x;
    const int n_blocks = n_rows / QK4_1;

    if (col >= ncols_dst) return;

    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * n_blocks * Q4_1_BLOCK_SIZE
    );

    float sum = 0.0f;
    for (int block_idx = lane_id; block_idx < n_blocks; block_idx += 32) {
        sum += vec_dot_q4_1_swiglu(&w_col[block_idx], gate, up, block_idx * QK4_1);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (lane_id == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_ffn_down_swiglu_q4_1_f32_experimental_launch(
    const void* weights_q4_1,
    const float* gate,
    const float* up,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (weights_q4_1 == nullptr || gate == nullptr || up == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (n_rows <= 0 || ncols_dst <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_1 != 0) {
        return hipErrorInvalidValue;
    }

    gemv_ffn_down_swiglu_q4_1_f32_experimental_kernel<<<ncols_dst, 32, 0, stream>>>(
        weights_q4_1, gate, up, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
