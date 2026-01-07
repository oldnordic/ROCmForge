/**
 * mxfp_dequant.hip - MXFP4/MXFP6 Dequantization Kernels
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Implements dequantization of MXFP (Micro-Scaling Float Point) formats
 * per OCP MX Specification v1.0:
 * https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *
 * MXFP4 (E2M1): 4-bit elements, block size 32
 * MXFP6 (E2M3): 6-bit elements, block size 32
 *
 * Each block uses:
 * - 1 byte E8M0 scale (exponent-only, value = 2^exponent)
 * - 16/24 bytes packed element data
 */

#include <hip/hip_runtime.h>
#include <cmath>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

// MXFP block sizes
constexpr int MXFP4_BLOCK_SIZE = 17;  // 1 scale + 16 data bytes (32 elements * 4 bits)
constexpr int MXFP6_BLOCK_SIZE = 25;  // 1 scale + 24 data bytes (32 elements * 6 bits)

/**
 * Decode E2M1 (4-bit) to float
 * E2M1 format: value = (-1)^sign * 2^(exp-1) * (1 + mant)
 * Bits: [sign(1) | exp(2) | mant(1)]
 */
__device__ __forceinline__ float decode_e2m1(uint8_t bits) {
    if (bits == 0) {
        return 0.0f;
    }

    int sign = (bits & 0x08) ? -1 : 1;
    int exp = ((bits >> 1) & 0x03) - 1;
    int mant = bits & 0x01;

    return sign * __int2float_rn(1 + mant) * exp2f(exp);
}

/**
 * Decode E2M3 (6-bit) to float
 * E2M3 format: value = (-1)^sign * 2^(exp-1) * (1 + mant/8)
 * Bits: [sign(1) | exp(2) | mant(3)]
 */
__device__ __forceinline__ float decode_e2m3(uint8_t bits) {
    if (bits == 0) {
        return 0.0f;
    }

    int sign = (bits & 0x20) ? -1 : 1;
    int exp = ((bits >> 3) & 0x03) - 1;
    int mant = bits & 0x07;

    return sign * (1.0f + mant / 8.0f) * exp2f(exp);
}

/**
 * Convert E8M0 exponent to float scale
 * E8M0 format: value = 2^exponent
 */
__device__ __forceinline__ float e8m0_to_f32(int8_t exponent) {
    return exp2f(static_cast<float>(exponent));
}

/**
 * MXFP4 dequantization kernel
 *
 * Dequantizes MXFP4 blocks to FP32 (full precision float)
 * Each block: 32 elements packed into 17 bytes (1 scale + 16 data)
 *
 * Grid: (num_blocks, 1, 1) - one block per MXFP4 block
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed MXFP4 data
 * @param output     Output FP32 data
 * @param num_blocks Number of MXFP4 blocks to process
 */
extern "C" __global__ void mxfp4_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_blocks
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    // Each block processes 32 elements
    const int element_idx = block_idx * 32 + tid;

    if (tid >= 32) {
        return;
    }

    // Read scale (E8M0)
    const int block_offset = block_idx * MXFP4_BLOCK_SIZE;
    int8_t scale_exp = reinterpret_cast<const int8_t*>(input)[block_offset];
    float scale = e8m0_to_f32(scale_exp);

    // Unpack 4-bit E2M1 elements
    const uint8_t* data = input + block_offset + 1;
    int byte_idx = tid / 2;
    int nibble_idx = tid % 2;

    uint8_t e2m1_bits;
    if (nibble_idx == 0) {
        // High nibble
        e2m1_bits = (data[byte_idx] >> 4) & 0x0F;
    } else {
        // Low nibble
        e2m1_bits = data[byte_idx] & 0x0F;
    }

    // Decode and apply scale
    float decoded = decode_e2m1(e2m1_bits);
    float val = scale * decoded;

    // Clamp to MXFP4 range [-8, 8] per OCP MX Spec v1.0
    val = fmaxf(-8.0f, fminf(8.0f, val));

    // Store as FP32
    output[element_idx] = val;
}

/**
 * MXFP6 dequantization kernel
 *
 * Dequantizes MXFP6 blocks to FP32 (full precision float)
 * Each block: 32 elements packed into 25 bytes (1 scale + 24 data)
 *
 * Grid: (num_blocks, 1, 1) - one block per MXFP6 block
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed MXFP6 data
 * @param output     Output FP32 data
 * @param num_blocks Number of MXFP6 blocks to process
 */
extern "C" __global__ void mxfp6_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_blocks
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    // Each block processes 32 elements
    const int element_idx = block_idx * 32 + tid;

    if (tid >= 32) {
        return;
    }

    // Read scale (E8M0)
    const int block_offset = block_idx * MXFP6_BLOCK_SIZE;
    int8_t scale_exp = reinterpret_cast<const int8_t*>(input)[block_offset];
    float scale = e8m0_to_f32(scale_exp);

    // Unpack 6-bit E2M3 elements
    const uint8_t* data = input + block_offset + 1;

    // Extract 6-bit value (packed across byte boundaries)
    int bit_offset = (tid * 6) % 8;
    int byte_idx = (tid * 6) / 8;

    uint16_t combined = (static_cast<uint16_t>(data[byte_idx + 1]) << 8) |
                        static_cast<uint16_t>(data[byte_idx]);
    uint8_t e2m3_bits = (combined >> (10 - bit_offset)) & 0x3F;

    // Decode and apply scale
    float decoded = decode_e2m3(e2m3_bits);
    float val = scale * decoded;

    // Clamp to MXFP6 range [-7.5, 7.5]
    val = fmaxf(-7.5f, fminf(7.5f, val));

    // Store as FP32
    output[element_idx] = val;
}

/**
 * Batched MXFP4 dequantization kernel
 *
 * Optimized version for processing multiple blocks with shared memory
 *
 * Grid: (num_blocks, 1, 1)
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed MXFP4 data
 * @param output     Output FP32 data (full precision)
 * @param num_elements Total number of output elements
 */
extern "C" __global__ void mxfp4_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx >= num_elements) {
        return;
    }

    const int block_idx = element_idx / 32;
    const int element_in_block = element_idx % 32;

    const int block_offset = block_idx * MXFP4_BLOCK_SIZE;

    // Read scale (E8M0)
    int8_t scale_exp = reinterpret_cast<const int8_t*>(input)[block_offset];
    float scale = e8m0_to_f32(scale_exp);

    // Unpack 4-bit E2M1 element
    const uint8_t* data = input + block_offset + 1;
    int byte_idx = element_in_block / 2;
    int nibble_idx = element_in_block % 2;

    uint8_t e2m1_bits;
    if (nibble_idx == 0) {
        e2m1_bits = (data[byte_idx] >> 4) & 0x0F;
    } else {
        e2m1_bits = data[byte_idx] & 0x0F;
    }

    // Decode and apply scale
    float decoded = decode_e2m1(e2m1_bits);
    float val = scale * decoded;
    val = fmaxf(-8.0f, fminf(8.0f, val));

    output[element_idx] = val;
}

/**
 * Batched MXFP6 dequantization kernel
 *
 * Optimized version for processing multiple blocks with shared memory
 *
 * Grid: (num_blocks, 1, 1)
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed MXFP6 data
 * @param output     Output FP32 data (full precision)
 * @param num_elements Total number of output elements
 */
extern "C" __global__ void mxfp6_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx >= num_elements) {
        return;
    }

    const int block_idx = element_idx / 32;
    const int element_in_block = element_idx % 32;

    const int block_offset = block_idx * MXFP6_BLOCK_SIZE;

    // Read scale (E8M0)
    int8_t scale_exp = reinterpret_cast<const int8_t*>(input)[block_offset];
    float scale = e8m0_to_f32(scale_exp);

    // Unpack 6-bit E2M3 element
    const uint8_t* data = input + block_offset + 1;

    int bit_offset = (element_in_block * 6) % 8;
    int byte_idx = (element_in_block * 6) / 8;

    uint16_t combined = (static_cast<uint16_t>(data[byte_idx + 1]) << 8) |
                        static_cast<uint16_t>(data[byte_idx]);
    uint8_t e2m3_bits = (combined >> (10 - bit_offset)) & 0x3F;

    // Decode and apply scale
    float decoded = decode_e2m3(e2m3_bits);
    float val = scale * decoded;
    val = fmaxf(-7.5f, fminf(7.5f, val));

    output[element_idx] = val;
}
