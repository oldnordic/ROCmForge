#include "common.hip"
#include <hip/hip_fp16.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__global__ void verify_q6_k_kernel(
    const float* __restrict__ original,
    const void* __restrict__ quantized,
    float* __restrict__ errors,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const uint8_t* block_bytes = static_cast<const uint8_t*>(quantized);
    const float* orig = original;

    const int block_idx = tid / QK_K;
    const int elem_idx = tid % QK_K;

    const uint8_t* block = &block_bytes[block_idx * Q6_K_BLOCK_SIZE];
    const uint8_t* ql = &block[0];
    const uint8_t* qh = &block[128];
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);

    half d_half;
    memcpy(&d_half, &block[208], sizeof(half));
    const float d = __half2float(d_half);

    // Dequantize (same logic as dequantize kernel)
    int n_block = elem_idx / 128;
    int l = (elem_idx % 128) / 4;
    int lane = elem_idx % 4;

    int is = l / 16;
    int ql_offset = n_block * 64;
    int qh_offset = n_block * 32;
    int sc_offset = n_block * 8;

    int8_t q;
    if (lane == 0) {
        q = static_cast<int8_t>((ql[ql_offset + l] & 0xF) | (((qh[qh_offset + l] >> 0) & 3) << 4)) - 32;
    } else if (lane == 1) {
        q = static_cast<int8_t>((ql[ql_offset + l + 32] & 0xF) | (((qh[qh_offset + l] >> 2) & 3) << 4)) - 32;
    } else if (lane == 2) {
        q = static_cast<int8_t>(((ql[ql_offset + l] >> 4) & 0xF) | (((qh[qh_offset + l] >> 4) & 3) << 4)) - 32;
    } else {
        q = static_cast<int8_t>(((ql[ql_offset + l + 32] >> 4) & 0xF) | (((qh[qh_offset + l] >> 6) & 3) << 4)) - 32;
    }

    int scale_idx = sc_offset + is + (lane * 2);
    float scale = d * static_cast<float>(scales[scale_idx]);
    float dequant = scale * static_cast<float>(q);

    // Compute error
    float orig_val = orig[tid];
    float abs_error = fabsf(orig_val - dequant);

    // Atomic reduction for max error and MSE
    atomicMax(&errors[0], abs_error); // max_error
    atomicAdd(&errors[1], abs_error * abs_error); // mse

    // Total magnitude for relative error
    atomicAdd(&errors[2], orig_val * orig_val);
    atomicAdd(&errors[3], dequant * dequant);
}

extern "C" hipError_t verify_q6_k_launch(
    const float* original,
    const void* quantized,
    float* errors,
    int n,
    hipStream_t stream
) {
    if (n <= 0) return hipErrorInvalidValue;

    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Initialize errors to 0
    hipMemsetAsync(errors, 0, 4 * sizeof(float), stream);

    hipLaunchKernelGGL(
        verify_q6_k_kernel,
        dim3(num_blocks),
        dim3(threads_per_block),
        0,
        stream,
        original,
        quantized,
        errors,
        n
    );

    return hipGetLastError();
}
