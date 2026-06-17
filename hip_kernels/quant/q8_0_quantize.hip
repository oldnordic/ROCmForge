#include "common.hip"

// Q8_0 constants (must match Rust types.rs)
#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34

/// Quantize f32 data to Q8_0 format (device kernel)
/// Each thread block processes one 32-element block
/// Quantize f32 data to standard Q8_0 format (34 bytes per block, half scale)
__global__ void quantize_q8_0_device(
    const float* __restrict__ input,   // [n] f32 input data
    void* __restrict__ output,         // [n/32 * 34] Q8_0 quantized output
    int n                              // Total number of elements
) {
    int block_idx = blockIdx.x;
    const float* x = &input[block_idx * QK8_0];

    // Each thread loads one element (QK8_0 = 32 threads)
    float val = (threadIdx.x < QK8_0) ? x[threadIdx.x] : 0.0f;
    float abs_val = fabsf(val);

    // Warp reduction to find max (RDNA warp = 32 threads)
    float max_val = abs_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down(max_val, offset));
    }

    // Compute scale (thread 0 only)
    __shared__ float s_scale_f32;
    if (threadIdx.x == 0) {
        s_scale_f32 = fmaxf(max_val / 127.0f, 1e-30f);
    }
    __syncthreads();

    // Quantize values (standard GGUF is 34 bytes per block)
    uint8_t* out = static_cast<uint8_t*>(output) + block_idx * 34;

    // Thread 0 writes scale as half (2 bytes)
    if (threadIdx.x == 0) {
        half* d_ptr = reinterpret_cast<half*>(out);
        *d_ptr = __float2half(s_scale_f32);
    }

    // All threads write their quantized value (starting at byte 2)
    if (threadIdx.x < QK8_0) {
        // Round to nearest int, then clamp and convert to int8_t
        const float x0 = val / s_scale_f32;
        const float x0_rounded = roundf(x0);
        const float x0_clamped = fminf(fmaxf(x0_rounded, -127.0f), 127.0f);
        int8_t q = static_cast<int8_t>(x0_clamped);
        out[2 + threadIdx.x] = reinterpret_cast<uint8_t&>(q);
    }
}

/// Quantize f32 data to decode-fastpath Q8_0 format.
///
/// This must match the layout consumed by the Q4_0 x Q8_0 decode kernels:
/// 2-byte f16 scale followed by 32 signed bytes.
__global__ void quantize_q8_0_act_device(
    const float* __restrict__ input,   // [n] f32 input data
    void* __restrict__ output,         // [n/32 * 34] Q8_0 quantized output
    int n                              // Total number of elements
) {
    int block_idx = blockIdx.x;
    const float* x = &input[block_idx * QK8_0];

    // Each thread loads one element (QK8_0 = 32 threads)
    float val = (threadIdx.x < QK8_0) ? x[threadIdx.x] : 0.0f;
    float abs_val = fabsf(val);

    // Warp reduction to find max (RDNA warp = 32 threads)
    float max_val = abs_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down(max_val, offset));
    }

    // Compute scale (thread 0 only)
    __shared__ float s_scale_f32;
    if (threadIdx.x == 0) {
        s_scale_f32 = fmaxf(max_val / 127.0f, 1e-30f);
    }
    __syncthreads();

    uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q8_0_BLOCK_SIZE;

    // Thread 0 writes scale as f16 so the workspace matches the decode kernels.
    if (threadIdx.x == 0) {
        half* d_ptr = reinterpret_cast<half*>(out);
        *d_ptr = __float2half(s_scale_f32);
    }

    // All threads write their quantized value (starting at byte 2)
    if (threadIdx.x < QK8_0) {
        // Round to nearest int, then clamp and convert to int8_t
        const float x0 = val / s_scale_f32;
        const float x0_rounded = roundf(x0);
        const float x0_clamped = fminf(fmaxf(x0_rounded, -127.0f), 127.0f);
        int8_t q = static_cast<int8_t>(x0_clamped);
        out[2 + threadIdx.x] = reinterpret_cast<uint8_t&>(q);
    }
}

// ── Dispatch Functions (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t quantize_q8_0_kernel(
    const float* input,
    void* output,
    int n
) {
    if (n <= 0) return hipErrorInvalidValue;
    int num_blocks = (n + QK8_0 - 1) / QK8_0;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize_q8_0_device),
        dim3(num_blocks),
        dim3(QK8_0),
        0,
        0,
        input,
        output,
        n
    );
    return hipGetLastError();
}

extern "C" hipError_t quantize_q8_0_on_stream_kernel(
    const float* input,
    void* output,
    int n,
    hipStream_t stream
) {
    if (n <= 0) return hipErrorInvalidValue;
    int num_blocks = (n + QK8_0 - 1) / QK8_0;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize_q8_0_act_device),
        dim3(num_blocks),
        dim3(QK8_0),
        0,
        stream,
        input,
        output,
        n
    );
    return hipGetLastError();
}
