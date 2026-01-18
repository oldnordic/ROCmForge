/**
 * q6_k_dequant.hip - Q6_K Dequantization Kernels
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Implements dequantization of GGUF Q6_K format.
 * Q6_K is a "K-quant" format with higher precision than Q4_K.
 *
 * Q6_K format specification:
 * - Block size: 256 elements
 * - Per block (256 bytes):
 *   - 32 bytes: 16 half-precision scales (2 bytes each)
 *   - 192 bytes: 6-bit packed quantized values (256 * 6 / 8 = 192)
 *   - 32 bytes: padding (for alignment)
 * - 16 elements share one scale
 * - Dequantization:
 *   1. Extract 6-bit value (packed across byte boundaries)
 *   2. Convert to signed: if >= 32, subtract 64 (range: [-32, 31])
 *   3. Apply: result = signed_val * scale
 *
 * Reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

// Q6_K block size
constexpr int Q6_K_BLOCK_SIZE = 256;      // Total bytes per block
constexpr int Q6_K_ELEMENTS_PER_BLOCK = 256;  // Elements per block
constexpr int Q6_K_SCALES_COUNT = 16;     // Number of scales per block
constexpr int Q6_K_ELEMENTS_PER_SCALE = 16;  // Elements sharing one scale

// Offsets within block
constexpr int Q6_K_SCALES_OFFSET = 0;      // Start of scales (32 bytes)
constexpr int Q6_K_QUANTS_OFFSET = 32;     // Start of quants (after scales)

/**
 * Convert half-precision float (FP16) to single-precision float (FP32)
 * Using direct bit manipulation for speed
 */
__device__ __forceinline__ float f16_to_f32(uint16_t f16_bits) {
    // FP16 format: [sign(1) | exp(5) | mant(10)]
    // FP32 format: [sign(1) | exp(8) | mant(23)]

    uint32_t f32_bits;

    if ((f16_bits & 0x7FFF) == 0) {
        // Zero or denormal - just return zero
        f32_bits = (f16_bits & 0x8000) << 16;
    } else {
        // Normal number
        uint32_t sign = (f16_bits & 0x8000) << 16;
        uint32_t mant = (f16_bits & 0x03FF) << 13;
        int32_t exp = (f16_bits & 0x7C00) >> 10;

        // Adjust exponent bias (15 for FP16, 127 for FP32)
        exp = exp - 15 + 127;

        f32_bits = sign | (exp << 23) | mant;
    }

    return *reinterpret_cast<float*>(&f32_bits);
}

/**
 * Q6_K dequantization kernel
 *
 * Dequantizes Q6_K blocks to FP32 (full precision float)
 * Each block: 256 elements packed into 256 bytes
 *
 * Grid: (num_blocks, 1, 1) - one block per Q6_K block
 * Block: BLOCK_SIZE threads (one per element in block)
 *
 * @param input      Packed Q6_K data
 * @param output     Output FP32 data
 * @param num_blocks Number of Q6_K blocks to process
 */
extern "C" __global__ void q6_k_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_blocks
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    // Each block contains 256 elements
    const int element_idx = block_idx * Q6_K_ELEMENTS_PER_BLOCK + tid;

    if (tid >= Q6_K_ELEMENTS_PER_BLOCK) {
        return;
    }

    // Calculate which scale to use (every 16 elements share a scale)
    const int scale_idx = tid / Q6_K_ELEMENTS_PER_SCALE;

    // Base offset for this block
    const int block_offset = block_idx * Q6_K_BLOCK_SIZE;

    // Read scale (half-precision) for this group
    const int scale_offset = block_offset + Q6_K_SCALES_OFFSET + scale_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(input + scale_offset);
    const float scale = f16_to_f32(scale_bits);

    // Extract 6-bit quantized value
    // Similar to MXFP6 unpacking (6 bits per element)
    const int quants_base = block_offset + Q6_K_QUANTS_OFFSET;
    const int bit_offset = (tid * 6) % 8;
    const int byte_idx = (tid * 6) / 8;

    // Read the two bytes that contain our 6-bit value
    const uint16_t combined = (static_cast<uint16_t>(input[quants_base + byte_idx + 1]) << 8) |
                              static_cast<uint16_t>(input[quants_base + byte_idx]);

    // Extract 6-bit value (position depends on bit_offset)
    // When bit_offset is 0-2: value spans two bytes
    const uint8_t quant_val = (combined >> (10 - bit_offset)) & 0x3F;

    // Convert to signed range: [0, 63] -> [-32, 31]
    const float signed_val = (quant_val >= 32) ?
                              (static_cast<float>(quant_val) - 64.0f) :
                              static_cast<float>(quant_val);

    // Dequantize: result = signed_val * scale
    const float val = signed_val * scale;

    // Store as FP32
    output[element_idx] = val;
}

/**
 * Batched Q6_K dequantization kernel
 *
 * Optimized version for processing multiple blocks.
 * Each thread processes one element directly using element index calculation.
 *
 * Grid: ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed Q6_K data
 * @param output     Output FP32 data
 * @param num_elements Total number of output elements
 */
extern "C" __global__ void q6_k_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx >= num_elements) {
        return;
    }

    // Calculate which block and element position
    const int block_idx = element_idx / Q6_K_ELEMENTS_PER_BLOCK;
    const int element_in_block = element_idx % Q6_K_ELEMENTS_PER_BLOCK;

    // Calculate which scale to use (every 16 elements share a scale)
    const int scale_idx = element_in_block / Q6_K_ELEMENTS_PER_SCALE;

    // Base offset for this block
    const int block_offset = block_idx * Q6_K_BLOCK_SIZE;

    // Read scale (half-precision) for this group
    const int scale_offset = block_offset + Q6_K_SCALES_OFFSET + scale_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(input + scale_offset);
    const float scale = f16_to_f32(scale_bits);

    // Extract 6-bit quantized value
    const int quants_base = block_offset + Q6_K_QUANTS_OFFSET;
    const int bit_offset = (element_in_block * 6) % 8;
    const int byte_idx = (element_in_block * 6) / 8;

    // Read the two bytes that contain our 6-bit value
    const uint16_t combined = (static_cast<uint16_t>(input[quants_base + byte_idx + 1]) << 8) |
                              static_cast<uint16_t>(input[quants_base + byte_idx]);

    // Extract 6-bit value
    const uint8_t quant_val = (combined >> (10 - bit_offset)) & 0x3F;

    // Convert to signed range: [0, 63] -> [-32, 31]
    const float signed_val = (quant_val >= 32) ?
                              (static_cast<float>(quant_val) - 64.0f) :
                              static_cast<float>(quant_val);

    // Dequantize: result = signed_val * scale
    const float val = signed_val * scale;

    // Store as FP32
    output[element_idx] = val;
}
