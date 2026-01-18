/**
 * q2_k_dequant.hip - Q2_K Dequantization Kernels
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Implements dequantization of GGUF Q2_K format.
 * Q2_K is a "K-quant" format with 2-bit precision (most complex).
 *
 * Q2_K format specification:
 * - Super-block size: 256 elements (16 sub-blocks of 16 elements each)
 * - Per super-block (256 bytes):
 *   - 32 bytes: 16 half-precision scales (2 bytes each) for 16 sub-blocks
 *   - 4 bytes: qh (high bits for 2-bit quants - 1 bit per pair of elements)
 *   - 64 bytes: 2-bit quantized values (16 * 2 / 8 * 16 = 64 bytes)
 *   - 156 bytes: additional data
 * - Each sub-block (16 elements): scale (f16) + 2-bit quants with high bits
 * - Dequantization: value = (quant - 2) * scale (signed 2-bit: -2 to +1)
 *
 * Reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Q2_K super-block size
constexpr int Q2_K_SUPER_BLOCK_SIZE = 256;
constexpr int Q2_K_ELEMENTS_PER_BLOCK = 256;
constexpr int Q2_K_SUB_BLOCKS = 16;
constexpr int Q2_K_ELEMENTS_PER_SUB_BLOCK = 16;

// Offsets within super-block
constexpr int Q2_K_SCALES_OFFSET = 0;
constexpr int Q2_K_QH_OFFSET = 32;
constexpr int Q2_K_QUANTS_OFFSET = 36;

/**
 * Convert half-precision float (FP16) to single-precision float (FP32)
 */
__device__ __forceinline__ float f16_to_f32(uint16_t f16_bits) {
    uint32_t f32_bits;

    if ((f16_bits & 0x7FFF) == 0) {
        f32_bits = (f16_bits & 0x8000) << 16;
    } else {
        uint32_t sign = (f16_bits & 0x8000) << 16;
        uint32_t mant = (f16_bits & 0x03FF) << 13;
        int32_t exp = (f16_bits & 0x7C00) >> 10;
        exp = exp - 15 + 127;
        f32_bits = sign | (exp << 23) | mant;
    }

    return *reinterpret_cast<float*>(&f32_bits);
}

/**
 * Q2_K dequantization kernel
 *
 * Dequantizes Q2_K super-blocks to FP32
 */
extern "C" __global__ void q2_k_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_super_blocks
) {
    const int super_block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (super_block_idx >= num_super_blocks) {
        return;
    }

    const int element_idx = super_block_idx * Q2_K_ELEMENTS_PER_BLOCK + tid;

    if (tid >= Q2_K_ELEMENTS_PER_BLOCK) {
        return;
    }

    const int sub_block_idx = tid / Q2_K_ELEMENTS_PER_SUB_BLOCK;
    const int sub_block_elem = tid % Q2_K_ELEMENTS_PER_SUB_BLOCK;

    const uint8_t* super_block = &input[super_block_idx * Q2_K_SUPER_BLOCK_SIZE];

    // Read scale for this sub-block
    const int scale_offset = Q2_K_SCALES_OFFSET + sub_block_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(&super_block[scale_offset]);
    const float scale = f16_to_f32(scale_bits);

    // Read high bits from qh (1 bit per pair of elements, so 8 bits per sub-block)
    const int qh_offset = Q2_K_QH_OFFSET + sub_block_idx;
    const uint8_t qh = super_block[qh_offset];

    // Extract 2-bit quantized value
    // 16 values * 2 bits = 32 bits = 4 bytes per sub-block
    const int quants_offset = Q2_K_QUANTS_OFFSET + sub_block_idx * 4;
    const int bit_pos = sub_block_elem * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_offset = bit_pos % 8;

    const uint8_t low_bits = (super_block[quants_offset + byte_idx] >> bit_offset) & 0x03;

    // Combine with high bit from qh (qh has 1 bit per 2 elements)
    const uint8_t high_bit = (qh >> sub_block_elem) & 1;
    const int8_t quant = (low_bits | (high_bit << 2)) - 2;

    output[element_idx] = quant * scale;
}

/**
 * Q2_K batch dequantization kernel
 *
 * Element-based grid for better load balancing
 */
extern "C" __global__ void q2_k_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx >= num_elements) {
        return;
    }

    const int super_block_idx = element_idx / Q2_K_ELEMENTS_PER_BLOCK;
    const int tid = element_idx % Q2_K_ELEMENTS_PER_BLOCK;

    const int sub_block_idx = tid / Q2_K_ELEMENTS_PER_SUB_BLOCK;
    const int sub_block_elem = tid % Q2_K_ELEMENTS_PER_SUB_BLOCK;

    const uint8_t* super_block = &input[super_block_idx * Q2_K_SUPER_BLOCK_SIZE];

    const int scale_offset = Q2_K_SCALES_OFFSET + sub_block_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(&super_block[scale_offset]);
    const float scale = f16_to_f32(scale_bits);

    const int qh_offset = Q2_K_QH_OFFSET + sub_block_idx;
    const uint8_t qh = super_block[qh_offset];

    const int quants_offset = Q2_K_QUANTS_OFFSET + sub_block_idx * 4;
    const int bit_pos = sub_block_elem * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_offset = bit_pos % 8;

    const uint8_t low_bits = (super_block[quants_offset + byte_idx] >> bit_offset) & 0x03;

    const uint8_t high_bit = (qh >> sub_block_elem) & 1;
    const int8_t quant = (low_bits | (high_bit << 2)) - 2;

    output[element_idx] = quant * scale;
}
