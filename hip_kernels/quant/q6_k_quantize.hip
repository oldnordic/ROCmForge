#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__global__ void quantize_q6_k_kernel(
    const float* __restrict__ input,
    void* __restrict__ output,
    int n
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = (n + QK_K - 1) / QK_K;

    if (block_idx >= n_blocks) return;

    // Shared memory for accumulating block statistics
    __shared__ float s_scale;
    __shared__ unsigned int s_ql[32];
    __shared__ unsigned int s_qh[16];
    __shared__ int8_t s_scales[16];

    const int elem_start = block_idx * QK_K;
    const int elem_end = min(elem_start + QK_K, n);
    const int block_n = elem_end - elem_start;

    // Initialize shared memory
    if (tid == 0) {
        s_scale = 0.0f;
    }
    if (tid < 32) s_ql[tid] = 0;
    if (tid < 16) s_qh[tid] = 0;
    if (tid < 16) s_scales[tid] = 0;

    __syncthreads();

    // Find max absolute value in this block for scale computation
    __shared__ float s_max_abs;
    if (tid == 0) {
        s_max_abs = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < block_n; i += blockDim.x) {
        float val = fabsf(input[elem_start + i]);
        if (val > s_max_abs) {
            atomicMax(&s_max_abs, val);
        }
    }

    __syncthreads();

    // Compute scale: d = max_abs / 32.0 (Q6_K range is -32 to +31)
    if (tid == 0) {
        s_scale = s_max_abs / 32.0f;
        if (s_scale < 0.0001f) s_scale = 1.0f; // Avoid division by zero
    }
    __syncthreads();

    // Quantize 256 values: first pass quantizes low 4 bits to ql
    for (int i = tid; i < block_n; i += blockDim.x) {
        float val = input[elem_start + i];
        int q = static_cast<int>(roundf(val / s_scale)) + 32; // Shift to 0..63 range

        // Clamp to 6-bit range (0..63)
        q = max(0, min(63, q));

        // Pack: low 4 bits to ql (32 x unsigned int), high 2 bits to qh (16 x unsigned int)
        int ql_word_idx = i / 8;
        int ql_shift = (i % 8) * 4;
        int qh_word_idx = i / 16;
        int qh_shift = (i % 16) * 2;

        atomicOr(&s_ql[ql_word_idx], ((q & 0xF) << ql_shift));
        atomicOr(&s_qh[qh_word_idx], ((q >> 4) << qh_shift));
    }

    __syncthreads();

    // Compute scales for each group of 16 elements
    // Simplified: use uniform scale for now (llama.cpp uses adaptive scales)
    if (tid < 16) {
        s_scales[tid] = static_cast<int8_t>(1.0f); // Identity scale
    }
    __syncthreads();

    // Write block to global memory in Q6_K format
    uint8_t* block_bytes = static_cast<uint8_t*>(output) + block_idx * Q6_K_BLOCK_SIZE;

    const uint8_t* s_ql_bytes = reinterpret_cast<const uint8_t*>(s_ql);
    const uint8_t* s_qh_bytes = reinterpret_cast<const uint8_t*>(s_qh);

    if (tid < 128) {
        block_bytes[tid] = s_ql_bytes[tid];
    }
    if (tid < 64) {
        block_bytes[128 + tid] = s_qh_bytes[tid];
    }
    if (tid < 16) {
        reinterpret_cast<int8_t*>(block_bytes + 192)[tid] = s_scales[tid];
    }
    if (tid == 0) {
        half d_half = __float2half(s_scale);
        memcpy(&block_bytes[208], &d_half, sizeof(half));
    }
}

extern "C" hipError_t quantize_q6_k_launch(
    const float* input,
    void* output,
    int n,
    hipStream_t stream
) {
    if (n <= 0) return hipErrorInvalidValue;

    const int threads_per_block = 256;
    const int n_blocks = (n + QK_K - 1) / QK_K;

    hipLaunchKernelGGL(
        quantize_q6_k_kernel,
        dim3(n_blocks),
        dim3(threads_per_block),
        0,
        stream,
        input,
        output,
        n
    );

    return hipGetLastError();
}
