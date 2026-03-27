//! Q4_K quantization kernel
//!
//! Quantizes f32 weights to Q4_K format (4-bit quantization with 256-element blocks).
//! Reference-inspired from llama.cpp ggml_quants.c quantize_row_q4_K_impl.
//!
//! Q4_K format (144 bytes for 256 f32 values):
//! - d: f16 scale (2 bytes)
//! - dmin: f16 minimum scale (2 bytes)
//! - scales[12]: quantized scales (12 bytes)
//! - qs[128]: 4-bit quantized values packed as bytes (128 bytes)
//!
//! Algorithm:
//! 1. Find min/max in block of 256 elements
//! 2. Compute scale (d) and minimum (dmin)
//! 3. Quantize to 4-bit values using non-uniform scales
//! 4. Pack scales and quantized values

#include "common.hip"
#include <hip/hip_fp16.h>

// ── Constants ─────────────────────────────────────────────────────────────────

#define QK_K 256
#define K_SCALE_SIZE 12

// ── Device Functions ─────────────────────────────────────────────────────────────

//! Find min/max in array (reduction across threads)
static __device__ void find_min_max(const float* x, int n, float& min_val, float& max_val) {
    min_val = 1e38f;
    max_val = -1e38f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = x[i];
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_min = __shfl_down(min_val, offset);
        float other_max = __shfl_down(max_val, offset);
        min_val = fminf(min_val, other_min);
        max_val = fmaxf(max_val, other_max);
    }

    // Final reduction across warps (only first thread in warp)
    __shared__ float shared_min[32];
    __shared__ float shared_max[32];
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane_id == 0) {
        shared_min[warp_id] = min_val;
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        min_val = shared_min[0];
        max_val = shared_max[0];
        for (int i = 1; i < blockDim.x / 32; i++) {
            min_val = fminf(min_val, shared_min[i]);
            max_val = fmaxf(max_val, shared_max[i]);
        }
    }
    __syncthreads();

    // Broadcast result to all threads
    min_val = __shfl(min_val, 0);
    max_val = __shfl(max_val, 0);
}

//! Quantize value to 4-bit using non-uniform scales
static __device__ __inline__ int make_qk_quants(int offset, float min_val, float max_val, float scale) {
    // Compute quantized value (0-15)
    float q = fmaf(scale, max_val - min_val, 0.5f);
    int iq = (int)q;
    if (iq > 15) iq = 15;
    if (iq < 0) iq = 0;
    return iq;
}

// ── Main Kernel ─────────────────────────────────────────────────────────────────

extern "C" __global__ void quantize_q4_k_kernel(
    const float* __restrict__ input,   // [n] f32 input data
    void* __restrict__ output,         // [n/256 * 144] Q4_K quantized output
    int n                              // Total number of elements
) {
    // Each thread block processes one 256-element block
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;

    // Shared memory for quantized data
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ half s_d;
    __shared__ half s_dmin;
    __shared__ uint8_t s_scales[K_SCALE_SIZE];
    __shared__ uint8_t s_qs[QK_K / 2]; // 128 bytes for 256 4-bit values

    const float* x = &input[block_start];

    // Step 1: Find min/max
    float min_val, max_val;
    find_min_max(x, QK_K, min_val, max_val);

    if (threadIdx.x == 0) {
        s_min = min_val;
        s_max = max_val;
    }
    __syncthreads();

    // Step 2: Compute scale and dmin
    if (threadIdx.x == 0) {
        float range = s_max - s_min;
        if (range < 1e-30f) {
            // All values are the same
            s_d = __float2half(1.0f);
            s_dmin = __float2half(s_min);
        } else {
            // Scale factor (maps to 4-bit range)
            s_d = __float2half(15.0f / range);
            s_dmin = __float2half(s_min);
        }
    }
    __syncthreads();

    // Step 3: Quantize to 4-bit values
    // Use simplified quantization for now (uniform quantization)
    // TODO: Implement proper Q4_K non-uniform scale quantization

    for (int i = threadIdx.x; i < QK_K; i += blockDim.x) {
        float val = x[i];
        float q = __half2float(s_d) * (val - __half2float(s_dmin));
        int iq = (int)(q + 0.5f);
        if (iq > 15) iq = 15;
        if (iq < 0) iq = 0;

        // Pack two 4-bit values per byte
        if (i % 2 == 0) {
            s_qs[i / 2] = iq;
        } else {
            s_qs[i / 2] |= (iq << 4);
        }
    }
    __syncthreads();

    // Step 4: Write output (only first 144 bytes)
    if (threadIdx.x == 0) {
        uint8_t* out = static_cast<uint8_t*>(output) + block_idx * 144;

        // Write d (f16 scale)
        half* d_ptr = reinterpret_cast<half*>(out);
        *d_ptr = s_d;

        // Write dmin (f16 minimum)
        half* dmin_ptr = reinterpret_cast<half*>(out + 2);
        *dmin_ptr = s_dmin;

        // Write scales (all zeros for uniform quantization)
        // TODO: Implement proper scale quantization
        for (int i = 0; i < K_SCALE_SIZE; i++) {
            out[4 + i] = 0;
        }

        // Write quantized values
        for (int i = 0; i < QK_K / 2; i++) {
            out[4 + K_SCALE_SIZE + i] = s_qs[i];
        }
    }
}
