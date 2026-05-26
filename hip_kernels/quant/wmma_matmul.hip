#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

// GGUF Q4_0 block layout (matches Rust types)
struct Q4_0_block {
    half d;          // scale (2 bytes)
    int8_t qs[16];   // quantized values packed as 2x4-bit per byte (16 bytes)
};

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

/// Native RDNA3 WMMA Wave32 prefill kernel for Q4_0 weights.
/// Computes output[batch][row] = sum_k(weights[row][k] * input[batch][k]).
///
/// Thread block executes as a single 32-thread wavefront:
/// - Process a 16x16 output tile (row_start to row_start+15, batch_start to batch_start+15)
/// - Row mappings duplicated between threads 0..15 and 16..31
/// - Batch columns split: threads 0..15 handle cols 0..7, threads 16..31 handle cols 8..15
__launch_bounds__(32, 2)
__global__ void wmma_matmul_q4_0_kernel(
    const void* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_dim,
    int out_dim,
    int batch_size
) {
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= out_dim) return;
    if (batch_start >= batch_size) return;

    // Safely clamp thread rows and batches within dimension bounds
    const int safe_row = (row_start + (tid & 15) < out_dim) ? (row_start + (tid & 15)) : (out_dim - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int n_blocks = in_dim / QK4_0;
    const Q4_0_block* row_base = (const Q4_0_block*)weights + (long long)safe_row * n_blocks;
    const float* x_base = input + (long long)safe_batch * in_dim;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Stride through K in chunks of 32 (split into two 16-element WMMA steps)
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const Q4_0_block* block = &row_base[block_idx];
        
        // Load scale d and convert to half
        _Float16 sc_h = block->d;
        _Float16 zp_h = -8.0f * sc_h;

        // Load 16 bytes of qs (coalesced 32-bit reads directly into registers)
        const uint32_t* qs_u32 = (const uint32_t*)(block->qs);
        uint32_t pk0 = qs_u32[0];
        uint32_t pk1 = qs_u32[1];
        uint32_t pk2 = qs_u32[2];
        uint32_t pk3 = qs_u32[3];

        const int base_k = block_idx * QK4_0;

        // ─── Step 0: Even nibbles (elements 0 to 15) ───
        {
            half16_t a_reg;
            #define DQ_LOW(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0x0Fu) + zp_h
            DQ_LOW(0,  pk0,  0); DQ_LOW(1,  pk0,  8); DQ_LOW(2,  pk0, 16); DQ_LOW(3,  pk0, 24);
            DQ_LOW(4,  pk1,  0); DQ_LOW(5,  pk1,  8); DQ_LOW(6,  pk1, 16); DQ_LOW(7,  pk1, 24);
            DQ_LOW(8,  pk2,  0); DQ_LOW(9,  pk2,  8); DQ_LOW(10, pk2, 16); DQ_LOW(11, pk2, 24);
            DQ_LOW(12, pk3,  0); DQ_LOW(13, pk3,  8); DQ_LOW(14, pk3, 16); DQ_LOW(15, pk3, 24);
            #undef DQ_LOW

            half16_t b_reg;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_reg[i] = (_Float16)x_base[base_k + i];
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }

        // ─── Step 1: Odd nibbles (elements 16 to 31) ───
        {
            half16_t a_reg;
            #define DQ_HIGH(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> ((sh) + 4)) & 0x0Fu) + zp_h
            DQ_HIGH(0,  pk0,  0); DQ_HIGH(1,  pk0,  8); DQ_HIGH(2,  pk0, 16); DQ_HIGH(3,  pk0, 24);
            DQ_HIGH(4,  pk1,  0); DQ_HIGH(5,  pk1,  8); DQ_HIGH(6,  pk1, 16); DQ_HIGH(7,  pk1, 24);
            DQ_HIGH(8,  pk2,  0); DQ_HIGH(9,  pk2,  8); DQ_HIGH(10, pk2, 16); DQ_HIGH(11, pk2, 24);
            DQ_HIGH(12, pk3,  0); DQ_HIGH(13, pk3,  8); DQ_HIGH(14, pk3, 16); DQ_HIGH(15, pk3, 24);
            #undef DQ_HIGH

            half16_t b_reg;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_reg[i] = (_Float16)x_base[base_k + 16 + i];
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    const int out_row = row_start + (tid & 15);
    const int out_batch_base = batch_start + (tid >> 4) * 8;

    if (out_row < out_dim) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int bidx = out_batch_base + j;
            if (bidx < batch_size) {
                output[(long long)bidx * out_dim + out_row] = acc[j];
            }
        }
    }
}

extern "C" hipError_t gpu_wmma_matmul_q4_0(
    const void* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_dim,
    int out_dim,
    int batch_size,
    hipStream_t stream
) {
    if (in_dim == 0 || out_dim == 0 || batch_size == 0) {
        return hipErrorInvalidValue;
    }

    // Wave32: 32 threads per block, each block computes a 16x16 output tile
    dim3 block(32, 1, 1);
    dim3 grid((out_dim + 15) / 16, (batch_size + 15) / 16, 1);

    hipLaunchKernelGGL(
        wmma_matmul_q4_0_kernel,
        grid,
        block,
        0,
        stream,
        weights,
        input,
        output,
        in_dim,
        out_dim,
        batch_size
    );

    return hipGetLastError();
}
