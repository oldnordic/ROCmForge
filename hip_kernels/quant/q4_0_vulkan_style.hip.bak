#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block {
    half d;
    int8_t qs[16];
};

__device__ inline float silu_vulkan(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * Prototyping a "Vulkan-style" FFN kernel for Q4_0.
 * 
 * Key differences from the production fused kernel:
 * 1. Interleaved Gate/Up loops: Reads s_input once and multiplies by both gate/up weights.
 * 2. Vectorized Weight Access: Uses uint4 to load chunks of weights if possible.
 * 3. Minimal Register Pressure: Tries to keep the footprint low for higher occupancy.
 */
template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_vulkan_q4_0_f32_kernel(
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

    // Gate/Up columns
    const Q4_0_block* w_gate_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    const Q4_0_block* w_up_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    // Interleaved loop: process both in one pass through rows
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q4_0_block* b_gate = &w_gate_col[block_idx];
        const Q4_0_block* b_up = &w_up_col[block_idx];
        
        const float d_gate = __half2float(b_gate->d);
        const float d_up = __half2float(b_up->d);
        const int row_offset = block_idx * QK4_0;

        // Vectorized load of qs (16 bytes = 4 x uint32)
        const uint32_t* qs_gate_u32 = reinterpret_cast<const uint32_t*>(b_gate->qs);
        const uint32_t* qs_up_u32   = reinterpret_cast<const uint32_t*>(b_up->qs);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t q_gate = qs_gate_u32[i];
            uint32_t q_up   = qs_up_u32[i];
            
            // Each uint32 has 4 bytes = 8 nibbles.
            // These nibbles correspond to indices:
            // low nibbles: 4*i + 0, 4*i + 1, 4*i + 2, 4*i + 3
            // high nibbles: 4*i + 16, 4*i + 17, 4*i + 18, 4*i + 19
            
            float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4*i])[0];
            float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4*i + 16])[0];

            // Extract and multiply
            auto process_nibbles = [&](uint32_t q, float d, float4 il, float4 ih, float& sum) {
                // Low nibbles
                sum += d * (static_cast<float>( q        & 0x0F) - 8.0f) * il.x;
                sum += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * il.y;
                sum += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * il.z;
                sum += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * il.w;
                // High nibbles
                sum += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * ih.x;
                sum += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * ih.y;
                sum += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * ih.z;
                sum += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * ih.w;
            };

            process_nibbles(q_gate, d_gate, in_l, in_h, sum_gate);
            process_nibbles(q_up,   d_up,   in_l, in_h, sum_up);
        }
    }

    // Wave-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate += __shfl_down(sum_gate, offset);
        sum_up   += __shfl_down(sum_up, offset);
    }

    if (lane_id == 0) {
        out_swiglu[ff_idx] = silu_vulkan(sum_gate) * sum_up;
    }
}

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_vulkan_q4_0_f32_v2_kernel(
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
    // Each wave now processes 2 FF indices
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 2;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx_base >= n_ff) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // Weights for both FF indices
    const Q4_0_block* w_gate_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    const Q4_0_block* w_up_col0 = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx_base * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    
    // Check if we have a second FF index
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
        const Q4_0_block* b_gate0 = &w_gate_col0[block_idx];
        const Q4_0_block* b_up0   = &w_up_col0[block_idx];
        
        const float d_gate0 = __half2float(b_gate0->d);
        const float d_up0   = __half2float(b_up0->d);
        const int row_offset = block_idx * QK4_0;

        const uint32_t* qs_gate0_u32 = reinterpret_cast<const uint32_t*>(b_gate0->qs);
        const uint32_t* qs_up0_u32   = reinterpret_cast<const uint32_t*>(b_up0->qs);

        // Optional second index weights
        float d_gate1 = 0.0f, d_up1 = 0.0f;
        const uint32_t* qs_gate1_u32 = nullptr;
        const uint32_t* qs_up1_u32   = nullptr;
        if (has_ff1) {
            const Q4_0_block* b_gate1 = &w_gate_col1[block_idx];
            const Q4_0_block* b_up1   = &w_up_col1[block_idx];
            d_gate1 = __half2float(b_gate1->d);
            d_up1   = __half2float(b_up1->d);
            qs_gate1_u32 = reinterpret_cast<const uint32_t*>(b_gate1->qs);
            qs_up1_u32   = reinterpret_cast<const uint32_t*>(b_up1->qs);
        }

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

            process_nibbles(qs_gate0_u32[i], d_gate0, in_l, in_h, sum_gate0);
            process_nibbles(qs_up0_u32[i],   d_up0,   in_l, in_h, sum_up0);
            if (has_ff1) {
                process_nibbles(qs_gate1_u32[i], d_gate1, in_l, in_h, sum_gate1);
                process_nibbles(qs_up1_u32[i],   d_up1,   in_l, in_h, sum_up1);
            }
        }
    }

    // Wave-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate0 += __shfl_down(sum_gate0, offset);
        sum_up0   += __shfl_down(sum_up0, offset);
        if (has_ff1) {
            sum_gate1 += __shfl_down(sum_gate1, offset);
            sum_up1   += __shfl_down(sum_up1, offset);
        }
    }

    if (lane_id == 0) {
        out_swiglu[ff_idx_base] = silu_vulkan(sum_gate0) * sum_up0;
        if (has_ff1) {
            out_swiglu[ff_idx_base + 1] = silu_vulkan(sum_gate1) * sum_up1;
        }
    }
}

extern "C" {
    void gemv_gate_up_swiglu_vulkan_q4_0_f32_launch(
        const void* w_gate,
        const void* w_up,
        const float* input,
        float* out_swiglu,
        int n_rows,
        int n_ff,
        int n_waves,
        hipStream_t stream
    ) {
        // v2 uses 2 FF indices per wave.
        if (n_waves == 4) {
            // 4 waves per block = 8 FF indices per block
            int n_blocks = (n_ff + 7) / 8;
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_gate_up_swiglu_vulkan_q4_0_f32_v2_kernel<4>),
                dim3(n_blocks), dim3(128), n_rows * sizeof(float), stream,
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        } else {
            // 8 waves per block = 16 FF indices per block
            int n_blocks = (n_ff + 15) / 16;
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_gate_up_swiglu_vulkan_q4_0_f32_v2_kernel<8>),
                dim3(n_blocks), dim3(256), n_rows * sizeof(float), stream,
                w_gate, w_up, input, out_swiglu, n_rows, n_ff
            );
        }
    }
}
