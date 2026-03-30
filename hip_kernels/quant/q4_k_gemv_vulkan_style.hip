#include "common.hip"

#define QK_K 256
#define K_SCALE_SIZE 12
#define Q4_K_BLOCK_SIZE 144

// Must match the Rust struct
struct Q4_K_block {
    half2 dm; // [0] = d, [1] = dmin
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

/**
 * Accurate and vectorized Q4_K dot product.
 * Ported from llama.cpp's Vulkan dequantization logic.
 */
__device__ inline float vec_dot_q4_k_vulkan(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const Q4_K_block* block = reinterpret_cast<const Q4_K_block*>(block_ptr);
    
    // Unpack base scales
    float2 loadd = __half22float2(block->dm);
    
    float sum = 0.0f;
    
    // Each thread processes 8 elements (4 bytes)
    // We split QK_K (256) among 32 threads in a wave
    // iqs goes from 0 to 255 in steps of 2, so iqs/2 goes from 0 to 127
    
    // Let's do 4 pairs (8 elements) per thread
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        int iqs_div2 = tid * 4 + l; // 0..127
        
        uint n = iqs_div2 / 32;                   // 0,1,2,3
        uint b = (iqs_div2 % 32) / 16;            // 0,1
        uint is = 2 * n + b;                      // 0..7
        uint qsi = n * 32 + (iqs_div2 % 16) * 2;  // 0,2,4..126
        
        uint scidx0 = (is < 4) ? is : (is + 4);
        uint scidx1 = (is < 4) ? is : (is - 4);
        uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
        uint scidxshift1 = (is < 4) ? 0 : 2;
        
        uint mbidx0 = is + 4;
        uint mbidx1 = (is < 4) ? is + 4 : is;
        uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
        uint mbidxshift0 = (is < 4) ? 0 : 4;
        uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
        uint mbidxshift1 = (is < 4) ? 0 : 2;
        
        uint8_t sc = ((block->scales[scidx0] & 0xF) | ((block->scales[scidx1] & scidxmask1) >> scidxshift1));
        uint8_t mbyte = (((block->scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((block->scales[mbidx1] & mbidxmask1) >> mbidxshift1));
        
        float d = loadd.x * sc;
        float m = -loadd.y * mbyte;
        
        uint8_t q0 = (block->qs[qsi]     >> (b * 4)) & 0xF;
        uint8_t q1 = (block->qs[qsi + 1] >> (b * 4)) & 0xF;
        
        float val0 = d * q0 + m;
        float val1 = d * q1 + m;
        
        int vec_idx = offset + iqs_div2 * 2;
        sum += val0 * vec[vec_idx] + val1 * vec[vec_idx + 1];
    }
    
    return sum;
}

template<int N_WAVES>
__global__ void gemv_q4_k_f32_multi_row_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Each wave processes 4 output columns (NUM_ROWS = 4)
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK_K;

    if (col_base >= ncols_dst) return;

    // Load input chunk into LDS
    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const Q4_K_block* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_K_block*>(
            static_cast<const uint8_t*>(weights_q4_k) + col * n_blocks_total * Q4_K_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = 0; block_idx < n_blocks_total; ++block_idx) {
        const int row_offset = block_idx * QK_K;
        
        // This time we pass the shared memory pointer to avoid global memory hits
        // We need a modified vec_dot that takes the lane_id and handles 4 columns
        
        // Unpack base scales for all 4 columns
        float2 loadd[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (w_cols[c]) loadd[c] = __half22float2(w_cols[c][block_idx].dm);
        }
        
        // Process 8 elements per lane
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int iqs_div2 = lane_id * 4 + l; // 0..127
            
            uint n = iqs_div2 / 32;
            uint b = (iqs_div2 % 32) / 16;
            uint is = 2 * n + b;
            uint qsi = n * 32 + (iqs_div2 % 16) * 2;
            
            uint scidx0 = (is < 4) ? is : (is + 4);
            uint scidx1 = (is < 4) ? is : (is - 4);
            uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
            uint scidxshift1 = (is < 4) ? 0 : 2;
            
            uint mbidx0 = is + 4;
            uint mbidx1 = (is < 4) ? is + 4 : is;
            uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
            uint mbidxshift0 = (is < 4) ? 0 : 4;
            uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
            uint mbidxshift1 = (is < 4) ? 0 : 2;
            
            int vec_idx = row_offset + iqs_div2 * 2;
            float in0 = s_input[vec_idx];
            float in1 = s_input[vec_idx + 1];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (w_cols[c]) {
                    const Q4_K_block* block = &w_cols[c][block_idx];
                    
                    uint8_t sc = ((block->scales[scidx0] & 0xF) | ((block->scales[scidx1] & scidxmask1) >> scidxshift1));
                    uint8_t mbyte = (((block->scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((block->scales[mbidx1] & mbidxmask1) >> mbidxshift1));
                    
                    float d = loadd[c].x * sc;
                    float m = -loadd[c].y * mbyte;
                    
                    uint8_t q0 = (block->qs[qsi]     >> (b * 4)) & 0xF;
                    uint8_t q1 = (block->qs[qsi + 1] >> (b * 4)) & 0xF;
                    
                    float val0 = d * q0 + m;
                    float val1 = d * q1 + m;
                    
                    sums[c] += val0 * in0 + val1 * in1;
                }
            }
        }
    }

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
    hipError_t gemv_q4_k_f32_vulkan_style_launch(
        const void* weights_q4_k,
        const float* input,
        float* output,
        int n_rows,
        int ncols_dst,
        int n_waves,
        hipStream_t stream
    ) {
        if (n_rows % QK_K != 0) return hipErrorInvalidValue;

        if (n_waves == 4) {
            int n_blocks = (ncols_dst + (4 * 4) - 1) / (4 * 4);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_q4_k_f32_multi_row_kernel<4>),
                dim3(n_blocks), dim3(128), n_rows * sizeof(float), stream,
                weights_q4_k, input, output, n_rows, ncols_dst
            );
        } else {
            int n_blocks = (ncols_dst + (8 * 4) - 1) / (8 * 4);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_q4_k_f32_multi_row_kernel<8>),
                dim3(n_blocks), dim3(256), n_rows * sizeof(float), stream,
                weights_q4_k, input, output, n_rows, ncols_dst
            );
        }
        return hipGetLastError();
    }
}
