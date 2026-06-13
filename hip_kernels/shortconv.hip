#include <hip/hip_runtime.h>

extern "C" __global__ void shortconv_f32_kernel(
    float* __restrict__ output,        // [h]
    const float* __restrict__ input,   // [3*h] (B, C, x interleaved or concatenated)
    const float* __restrict__ weight,  // [L*h] (causal conv weights)
    float* __restrict__ state,         // [(L-1)*h]
    const int h,
    const int L
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;

    float b = input[i];
    float c = input[h + i];
    float x = input[2 * h + i];
    float bx = b * x;

    float acc = weight[(L - 1) * h + i] * bx;
    for (int k = 0; k < L - 1; ++k) {
        acc += weight[k * h + i] * state[k * h + i];
    }

    output[i] = c * acc;

    for (int k = 0; k < L - 2; ++k) {
        state[k * h + i] = state[(k + 1) * h + i];
    }
    if (L > 1) {
        state[(L - 2) * h + i] = acc;
    }
}

extern "C" __global__ void shortconv_sequence_f32_kernel(
    float* __restrict__ output,        // [seq_len * h]
    const float* __restrict__ input,   // [seq_len * 3 * h]
    const float* __restrict__ weight,  // [L * h]
    float* __restrict__ state,         // [(L-1) * h]
    const int h,
    const int L,
    const int seq_len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;

    // Use a small fixed-size buffer for local state to avoid division
    // Support L up to 8 (LFM 2.5 uses 3)
    float local_state[8];
    for (int k = 0; k < L - 1; ++k) {
        local_state[k] = state[k * h + i];
    }

    for (int t = 0; t < seq_len; ++t) {
        float b = input[t * 3 * h + i];
        float c = input[t * 3 * h + h + i];
        float x = input[t * 3 * h + 2 * h + i];
        float bx_t = b * x;

        float acc = weight[(L - 1) * h + i] * bx_t;
        for (int k = 0; k < L - 1; ++k) {
            acc += weight[k * h + i] * local_state[k];
        }
        output[t * h + i] = c * acc;

        // Update local state
        for (int k = 0; k < L - 2; ++k) {
            local_state[k] = local_state[k + 1];
        }
        if (L > 1) {
            local_state[L - 2] = acc;
        }
    }

    // Update global state with final local state
    for (int k = 0; k < L - 1; ++k) {
        state[k * h + i] = local_state[k];
    }
}

extern "C" hipError_t gpu_shortconv_f32(
    float* output,
    const float* input,
    const float* weight,
    float* state,
    int h,
    int L,
    hipStream_t stream
) {
    const int block = 256;
    const int grid = (h + block - 1) / block;

    hipLaunchKernelGGL(
        shortconv_f32_kernel,
        dim3(grid, 1, 1), dim3(block, 1, 1), 0, stream,
        output, input, weight, state, h, L
    );

    return hipGetLastError();
}

extern "C" hipError_t gpu_shortconv_sequence_f32(
    float* output,
    const float* input,
    const float* weight,
    float* state,
    int h,
    int L,
    int seq_len,
    hipStream_t stream
) {
    const int block = 256;
    const int grid = (h + block - 1) / block;

    hipLaunchKernelGGL(
        shortconv_sequence_f32_kernel,
        dim3(grid, 1, 1), dim3(block, 1, 1), 0, stream,
        output, input, weight, state, h, L, seq_len
    );

    return hipGetLastError();
}
