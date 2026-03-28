#include "common.hip"

// Q8_0 constants (must match Rust types.rs)
#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34

/// Quantize f32 data to Q8_0 format (device kernel)
/// Each thread block processes one 32-element block
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
    __shared__ half s_scale;
    if (threadIdx.x == 0) {
        s_scale = __float2half(fmaxf(max_val / 127.0f, 1e-30f));
    }
    __syncthreads();

    // Quantize values
    uint8_t* out = static_cast<uint8_t*>(output) + block_idx * Q8_0_BLOCK_SIZE;

    // Thread 0 writes scale (f16 = 2 bytes)
    if (threadIdx.x == 0) {
        half* d_ptr = reinterpret_cast<half*>(out);
        *d_ptr = s_scale;
    }

    // All threads write their quantized value
    if (threadIdx.x < QK8_0) {
        int8_t q = static_cast<int8_t>(
            fminf(fmaxf(val / __half2float(s_scale), -127.0f), 127.0f)
        );
        out[2 + threadIdx.x] = reinterpret_cast<uint8_t&>(q);
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t quantize_q8_0_kernel(
    const float* input,
    void* output,
    int n
) {
    // Validate dimensions
    if (n <= 0) {
        return hipErrorInvalidValue;
    }

    // Calculate number of blocks
    int num_blocks = (n + QK8_0 - 1) / QK8_0;

    // Launch kernel with 32 threads per block (QK8_0)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize_q8_0_device),
        dim3(num_blocks),
        dim3(QK8_0),
        0,  // shared memory
        0,  // stream
        input,
        output,
        n
    );

    return hipGetLastError();
}
