#include "common.hip"
#include <hip/hip_fp16.h>

// Q6_K constants (must match CPU quant.rs)
#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__global__ void dequantize_q6_k_kernel(
    const void* __restrict__ input,
    void* __restrict__ output,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const uint8_t* block_bytes = static_cast<const uint8_t*>(input);
    float* output_float = static_cast<float*>(output);

    // Find which block this thread is dequantizing
    const int block_idx = tid / QK_K;
    const int elem_idx = tid % QK_K;

    // Q6_K block layout: ql[0..128], qh[128..192], scales[192..208], d[208..210]
    const uint8_t* block = &block_bytes[block_idx * Q6_K_BLOCK_SIZE];
    const uint8_t* ql = &block[0];      // 128 bytes
    const uint8_t* qh = &block[128];    // 64 bytes
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]); // 16 bytes

    half d_half;
    memcpy(&d_half, &block[208], sizeof(half));
    const float d = __half2float(d_half);

    // Dequantize following llama.cpp dequantize_row_q6_k pattern
    // for (int n = 0; n < QK_K; n += 128) {
    //     for (int l = 0; l < 32; ++l) {
    //         const int is = l / 16;
    //         const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    //         const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    //         const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    //         const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    //         y[l + 0] = d * sc[is + 0] * q1;
    //         y[l + 32] = d * sc[is + 2] * q2;
    //         y[l + 64] = d * sc[is + 4] * q3;
    //         y[l + 96] = d * sc[is + 6] * q4;
    //     }
    //     y  += 128;  ql += 64;  qh += 32;  sc += 8;
    // }

    int n = elem_idx / 128; // 0 or 1
    int l = (elem_idx % 128) / 4; // 0..31
    int lane = elem_idx % 4; // 0..3

    int is = l / 16;
    int ql_offset = n * 64;
    int qh_offset = n * 32;
    int sc_offset = n * 8;

    int8_t q;
    if (lane == 0) {
        uint8_t ql_part = ql[ql_offset + l] & 0xF;
        uint8_t qh_part = ((qh[qh_offset + l] >> 0) & 3) << 4;
        q = static_cast<int8_t>(ql_part | qh_part) - 32;
    } else if (lane == 1) {
        uint8_t ql_part = ql[ql_offset + l + 32] & 0xF;
        uint8_t qh_part = ((qh[qh_offset + l] >> 2) & 3) << 4;
        q = static_cast<int8_t>(ql_part | qh_part) - 32;
    } else if (lane == 2) {
        uint8_t ql_part = (ql[ql_offset + l] >> 4) & 0xF;
        uint8_t qh_part = ((qh[qh_offset + l] >> 4) & 3) << 4;
        q = static_cast<int8_t>(ql_part | qh_part) - 32;
    } else { // lane == 3
        uint8_t ql_part = (ql[ql_offset + l + 32] >> 4) & 0xF;
        uint8_t qh_part = ((qh[qh_offset + l] >> 6) & 3) << 4;
        q = static_cast<int8_t>(ql_part | qh_part) - 32;
    }

    int scale_idx;
    if (lane == 0) scale_idx = sc_offset + is + 0;
    else if (lane == 1) scale_idx = sc_offset + is + 2;
    else if (lane == 2) scale_idx = sc_offset + is + 4;
    else scale_idx = sc_offset + is + 6;

    float scale = d * static_cast<float>(scales[scale_idx]);
    output_float[tid] = scale * static_cast<float>(q);
}

extern "C" hipError_t dequantize_q6_k_launch(
    const void* input,
    void* output,
    int n,
    hipStream_t stream
) {
    if (n <= 0) return hipErrorInvalidValue;

    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        dequantize_q6_k_kernel,
        dim3(num_blocks),
        dim3(threads_per_block),
        0,
        stream,
        input,
        output,
        n
    );

    return hipGetLastError();
}
