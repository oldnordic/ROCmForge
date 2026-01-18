/**
 * q4_k_dequant.hip - Q4_K Dequantization Kernels
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Implements dequantization of GGUF Q4_K format.
 * Q4_K is a "K-quant" format optimized for quality/size trade-off.
 *
 * Q4_K format specification:
 * - Super-block size: 256 elements (8 sub-blocks of 32 elements each)
 * - Per super-block (256 bytes):
 *   - 16 bytes: 8 half-precision scales (2 bytes each) for 8 sub-blocks
 *   - 16 bytes: 8 int8 mins (1 byte each) for 8 sub-blocks
 *   - 160 bytes: 8 sub-blocks of 4-bit quantized values (20 bytes each)
 *   - 64 bytes: additional data (for QK format compatibility)
 * - Each sub-block (32 elements): scale (f16) + min (int8) + 20 bytes packed 4-bit values
 * - Dequantization: value = min + (quant * scale)
 *
 * Reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

// Q4_K super-block size
constexpr int Q4_K_SUPER_BLOCK_SIZE = 256;  // Total bytes per super-block
constexpr int Q4_K_ELEMENTS_PER_BLOCK = 256;  // Elements per super-block
constexpr int Q4_K_SUB_BLOCKS = 8;  // Sub-blocks per super-block
constexpr int Q4_K_ELEMENTS_PER_SUB_BLOCK = 32;  // Elements per sub-block

// Offsets within super-block
constexpr int Q4_K_SCALES_OFFSET = 0;      // Start of scales
constexpr int Q4_K_MINS_OFFSET = 16;       // Start of mins (after 16 bytes of scales)
constexpr int Q4_K_QUANTS_OFFSET = 32;     // Start of quants (after 32 bytes of scales+mins)

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
 * Q4_K dequantization kernel
 *
 * Dequantizes Q4_K super-blocks to FP32 (full precision float)
 * Each super-block: 256 elements packed into 256 bytes
 *
 * Grid: (num_super_blocks, 1, 1) - one block per Q4_K super-block
 * Block: BLOCK_SIZE threads (one per element in super-block)
 *
 * @param input      Packed Q4_K data
 * @param output     Output FP32 data
 * @param num_super_blocks Number of Q4_K super-blocks to process
 */
extern "C" __global__ void q4_k_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_super_blocks
) {
    const int super_block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (super_block_idx >= num_super_blocks) {
        return;
    }

    // Each super-block contains 256 elements
    const int element_idx = super_block_idx * Q4_K_ELEMENTS_PER_BLOCK + tid;

    if (tid >= Q4_K_ELEMENTS_PER_BLOCK) {
        return;
    }

    // Calculate which sub-block and position within sub-block
    const int sub_block_idx = tid / Q4_K_ELEMENTS_PER_SUB_BLOCK;  // 0-7
    const int element_in_sub_block = tid % Q4_K_ELEMENTS_PER_SUB_BLOCK;  // 0-31

    // Base offset for this super-block
    const int super_block_offset = super_block_idx * Q4_K_SUPER_BLOCK_SIZE;

    // Read scale (half-precision) for this sub-block
    const int scale_offset = super_block_offset + Q4_K_SCALES_OFFSET + sub_block_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(input + scale_offset);
    const float scale = f16_to_f32(scale_bits);

    // Read min (int8) for this sub-block
    const int min_offset = super_block_offset + Q4_K_MINS_OFFSET + sub_block_idx;
    const int8_t min_i8 = reinterpret_cast<const int8_t*>(input + min_offset)[0];
    const float min = static_cast<float>(min_i8);

    // Extract 4-bit quantized value for this element
    const int quants_base = super_block_offset + Q4_K_QUANTS_OFFSET + sub_block_idx * 20;
    const int bit_pos = element_in_sub_block * 4;
    const int byte_idx = bit_pos / 8;
    const int bit_offset = bit_pos % 8;

    // Read the two bytes that contain our 4-bit value
    const uint16_t combined = (static_cast<uint16_t>(input[quants_base + byte_idx + 1]) << 8) |
                              static_cast<uint16_t>(input[quants_base + byte_idx]);
    const uint8_t quant = (combined >> bit_offset) & 0x0F;

    // Dequantize: value = min + (quant * scale)
    const float val = min + (static_cast<float>(quant) * scale);

    // Store as FP32
    output[element_idx] = val;
}

/**
 * Batched Q4_K dequantization kernel
 *
 * Optimized version for processing multiple super-blocks.
 * Each thread processes one element directly using element index calculation.
 *
 * Grid: ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed Q4_K data
 * @param output     Output FP32 data
 * @param num_elements Total number of output elements
 */
extern "C" __global__ void q4_k_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx >= num_elements) {
        return;
    }

    // Calculate which super-block, sub-block, and element position
    const int super_block_idx = element_idx / Q4_K_ELEMENTS_PER_BLOCK;
    const int element_in_super_block = element_idx % Q4_K_ELEMENTS_PER_BLOCK;
    const int sub_block_idx = element_in_super_block / Q4_K_ELEMENTS_PER_SUB_BLOCK;
    const int element_in_sub_block = element_in_super_block % Q4_K_ELEMENTS_PER_SUB_BLOCK;

    // Base offset for this super-block
    const int super_block_offset = super_block_idx * Q4_K_SUPER_BLOCK_SIZE;

    // Read scale (half-precision) for this sub-block
    const int scale_offset = super_block_offset + Q4_K_SCALES_OFFSET + sub_block_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(input + scale_offset);
    const float scale = f16_to_f32(scale_bits);

    // Read min (int8) for this sub-block
    const int min_offset = super_block_offset + Q4_K_MINS_OFFSET + sub_block_idx;
    const int8_t min_i8 = reinterpret_cast<const int8_t*>(input + min_offset)[0];
    const float min = static_cast<float>(min_i8);

    // Extract 4-bit quantized value for this element
    const int quants_base = super_block_offset + Q4_K_QUANTS_OFFSET + sub_block_idx * 20;
    const int bit_pos = element_in_sub_block * 4;
    const int byte_idx = bit_pos / 8;
    const int bit_offset = bit_pos % 8;

    // Read the two bytes that contain our 4-bit value
    const uint16_t combined = (static_cast<uint16_t>(input[quants_base + byte_idx + 1]) << 8) |
                              static_cast<uint16_t>(input[quants_base + byte_idx]);
    const uint8_t quant = (combined >> bit_offset) & 0x0F;

    // Dequantize: value = min + (quant * scale)
    const float val = min + (static_cast<float>(quant) * scale);

    // Store as FP32
    output[element_idx] = val;
}
