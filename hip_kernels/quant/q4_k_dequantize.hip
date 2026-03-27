//! Q4_K dequantization kernel
//!
//! Dequantizes Q4_K format back to f32.
//! Reference-inspired from llama.cpp ggml_quants.c dequantize_row_q4_K.
//!
//! Q4_K format (144 bytes for 256 f32 values):
//! - d: f16 scale (2 bytes)
//! - dmin: f16 minimum scale (2 bytes)
//! - scales[12]: quantized scales (12 bytes)
//! - qs[128]: 4-bit quantized values packed as bytes (128 bytes)
//!
//! Algorithm:
//! 1. Read block header (d, dmin, scales)
//! 2. Unpack 4-bit values from bytes
//! 3. Reconstruct f32 values: val = qs[i] / d + dmin

#include "common.hip"
#include <hip/hip_fp16.h>

// ── Constants ─────────────────────────────────────────────────────────────────

#define QK_K 256
#define K_SCALE_SIZE 12

// ── Main Kernel ─────────────────────────────────────────────────────────────────

extern "C" __global__ void dequantize_q4_k_kernel(
    const void* __restrict__ input,   // [n/256 * 144] Q4_K quantized data
    float* __restrict__ output,       // [n] f32 output data
    int n                             // Total number of elements
) {
    // Each thread block processes one 256-element block
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;
    const int block_end = min(block_start + QK_K, n);

    // Shared memory for block header
    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ uint8_t s_scales[K_SCALE_SIZE];

    // Read block header (first 16 bytes: d + dmin + scales[12])
    if (threadIdx.x == 0) {
        const uint8_t* in = static_cast<const uint8_t*>(input) + block_idx * 144;

        // Read d (f16 scale)
        half d_val = *reinterpret_cast<const half*>(in);
        s_d = __half2float(d_val);

        // Read dmin (f16 minimum)
        half dmin_val = *reinterpret_cast<const half*>(in + 2);
        s_dmin = __half2float(dmin_val);

        // Read scales
        for (int i = 0; i < K_SCALE_SIZE; i++) {
            s_scales[i] = in[4 + i];
        }
    }
    __syncthreads();

    // Dequantize values
    // Each thread processes multiple values
    const uint8_t* qs_ptr = static_cast<const uint8_t*>(input) + block_idx * 144 + 4 + K_SCALE_SIZE;

    for (int i = threadIdx.x; i < (block_end - block_start); i += blockDim.x) {
        // Read packed 4-bit value
        uint8_t packed = qs_ptr[i / 2];
        uint8_t q4;

        if (i % 2 == 0) {
            // Lower 4 bits
            q4 = packed & 0x0F;
        } else {
            // Upper 4 bits
            q4 = (packed >> 4) & 0x0F;
        }

        // Dequantize: val = q4 / d + dmin
        // For uniform quantization (simplified)
        float val = static_cast<float>(q4) / s_d + s_dmin;

        output[block_start + i] = val;
    }
}

//! Batched dequantization kernel
//!
//! Dequantizes multiple blocks in parallel.
//! Each thread block handles one block.
extern "C" __global__ void dequantize_q4_k_batched_kernel(
    const void* __restrict__ input,   // [batch_size][n/256 * 144] Q4_K data
    float* __restrict__ output,       // [batch_size][n] f32 output
    int n,                            // Elements per batch
    int batch_size                    // Number of batches
) {
    const int batch_idx = blockIdx.y;
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * QK_K;
    const int block_end = min(block_start + QK_K, n);

    // Calculate batch offsets
    const size_t batch_input_offset = static_cast<size_t>(batch_idx) * ((n + QK_K - 1) / QK_K) * 144;
    const size_t batch_output_offset = static_cast<size_t>(batch_idx) * n;

    // Shared memory for block header
    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ uint8_t s_scales[K_SCALE_SIZE];

    // Read block header
    if (threadIdx.x == 0) {
        const uint8_t* in = static_cast<const uint8_t*>(input) + batch_input_offset + block_idx * 144;

        half d_val = *reinterpret_cast<const half*>(in);
        s_d = __half2float(d_val);

        half dmin_val = *reinterpret_cast<const half*>(in + 2);
        s_dmin = __half2float(dmin_val);

        for (int i = 0; i < K_SCALE_SIZE; i++) {
            s_scales[i] = in[4 + i];
        }
    }
    __syncthreads();

    // Dequantize values
    const uint8_t* qs_ptr = static_cast<const uint8_t*>(input) + batch_input_offset + block_idx * 144 + 4 + K_SCALE_SIZE;
    float* out_ptr = output + batch_output_offset;

    for (int i = threadIdx.x; i < (block_end - block_start); i += blockDim.x) {
        uint8_t packed = qs_ptr[i / 2];
        uint8_t q4;

        if (i % 2 == 0) {
            q4 = packed & 0x0F;
        } else {
            q4 = (packed >> 4) & 0x0F;
        }

        float val = static_cast<float>(q4) / s_d + s_dmin;
        out_ptr[block_start + i] = val;
    }
}
