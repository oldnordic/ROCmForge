/**
 * q4_0_dequant.hip - Q4_0 Dequantization Kernels
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32 threads)
 *
 * Implements dequantization of Q4_0 format from GGUF specification.
 * Q4_0 is the most common quantization format for LLM weights.
 *
 * Q4_0 Format:
 * - Block size: 32 elements
 * - Per block: scale (f32, 4 bytes) + 16 bytes 4-bit packed values = 20 bytes
 * - Dequantization: value = scale * ((packed & 0x0F) - 8)
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

// Q4_0 block size in bytes
constexpr int Q4_0_BLOCK_SIZE = 20;  // 4 bytes scale + 16 bytes packed data

// Elements per Q4_0 block
constexpr int Q4_0_ELEMENTS_PER_BLOCK = 32;

/**
 * Unpack 4-bit Q4_0 value
 *
 * Q4_0 stores 32 values in 16 bytes (2 values per byte).
 * Values are unsigned 0-15, interpreted as signed -8 to +7.
 *
 * @param data    Packed data array
 * @param element Index within block (0-31)
 * @return Unpacked 4-bit value as signed float (-8 to +7)
 */
__device__ __forceinline__ float unpack_q4_0_value(const uint8_t* data, int element) {
    int byte_idx = element / 2;
    int nibble_idx = element % 2;

    uint8_t packed = data[byte_idx];
    uint8_t quant;

    if (nibble_idx == 0) {
        // Low nibble
        quant = packed & 0x0F;
    } else {
        // High nibble
        quant = (packed >> 4) & 0x0F;
    }

    // Convert to signed range: 0-15 -> -8 to +7
    return __int2float_rn(static_cast<int>(quant) - 8);
}

/**
 * Q4_0 dequantization kernel
 *
 * Dequantizes Q4_0 blocks to FP32 (full precision float)
 * Each block: 32 elements packed into 20 bytes (4 scale + 16 data)
 *
 * Grid: (num_blocks, 1, 1) - one block per Q4_0 block
 * Block: BLOCK_SIZE threads (256 threads, only 32 active per block)
 *
 * @param input      Packed Q4_0 data
 * @param output     Output FP32 data
 * @param num_blocks Number of Q4_0 blocks to process
 */
extern "C" __global__ void q4_0_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_blocks
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Boundary check
    if (block_idx >= num_blocks) {
        return;
    }

    // Each block processes 32 elements, but we have 256 threads
    // Only first 32 threads do work
    if (tid >= Q4_0_ELEMENTS_PER_BLOCK) {
        return;
    }

    // Calculate element index
    const int element_idx = block_idx * Q4_0_ELEMENTS_PER_BLOCK + tid;

    // Read block scale
    const int block_offset = block_idx * Q4_0_BLOCK_SIZE;
    const float scale = *reinterpret_cast<const float*>(input + block_offset);

    // Read packed 4-bit data (starts after 4-byte scale)
    const uint8_t* quant_data = input + block_offset + 4;

    // Unpack and dequantize
    float quant_value = unpack_q4_0_value(quant_data, tid);
    float val = scale * quant_value;

    // Store as FP32
    output[element_idx] = val;
}

/**
 * Batched Q4_0 dequantization kernel
 *
 * Optimized version for processing multiple blocks.
 * Uses element-based grid for better load balancing.
 *
 * Grid: (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE blocks
 * Block: BLOCK_SIZE threads
 *
 * @param input        Packed Q4_0 data
 * @param output       Output FP32 data
 * @param num_elements Total number of output elements
 */
extern "C" __global__ void q4_0_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (element_idx >= num_elements) {
        return;
    }

    // Calculate block and element positions
    const int block_idx = element_idx / Q4_0_ELEMENTS_PER_BLOCK;
    const int element_in_block = element_idx % Q4_0_ELEMENTS_PER_BLOCK;

    // Read block scale
    const int block_offset = block_idx * Q4_0_BLOCK_SIZE;
    const float scale = *reinterpret_cast<const float*>(input + block_offset);

    // Read packed 4-bit data
    const uint8_t* quant_data = input + block_offset + 4;

    // Unpack and dequantize
    float quant_value = unpack_q4_0_value(quant_data, element_in_block);
    float val = scale * quant_value;

    // Store as FP32
    output[element_idx] = val;
}
