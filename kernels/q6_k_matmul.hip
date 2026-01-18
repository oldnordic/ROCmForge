/**
 * q6_k_matmul.hip - Fused Q6_K Dequantization + MatMul Kernel
 *
 * Tunable kernel optimized for AMD GPU architectures:
 * - RDNA3 (gfx1100): Wave32, 256 threads/block
 * - RDNA2 (gfx1030): Wave32, 256 threads/block
 * - CDNA2 (gfx90a): Wave64, 512 threads/block
 *
 * Tuning Parameters (configurable via preprocessor defines):
 * - BLOCK_SIZE: Threads per block (default: 256)
 * - WARP_SIZE: Wavefront size (default: 32 for RDNA, 64 for CDNA)
 * - TILE_SIZE_K: K dimension tile size (default: 32)
 * - TILE_SIZE_N: N dimension tile size (default: 32)
 * - USE_LDS: Enable Local Data Share optimization (default: 1)
 *
 * Implements fused dequantization + matrix multiplication for Q6_K format.
 * Q6_K is a "K-quant" format with higher precision than Q4_K.
 *
 * Q6_K Format Specification:
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
 * Kernel Operation:
 * - Input: Activations [M x K] (FP32), Weights [K x N] (Q6_K quantized)
 * - Output: [M x N] (FP32)
 * - For each output element, dequantize the weight column on-the-fly and compute dot product
 *
 * Memory Bandwidth Savings:
 * - Traditional: Read Q6_K (~1.125*K*N) + Write FP32 (4*K*N) + Read FP32 (4*K*N) = ~9.125*K*N bytes
 * - Fused: Read Q6_K (~1.125*K*N) + Read Q6_K again (~1.125*K*N) = ~2.25*K*N bytes
 * - ~4x reduction in memory bandwidth for the weight matrix
 */

#include <hip/hip_runtime.h>

// Default tuning constants (can be overridden at compile time)
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 256  // Default: 8 waves of 32 threads (RDNA3)
#endif

#ifndef WARP_SIZE
    #define WARP_SIZE 32    // Default: RDNA3 wavefront size
#endif

#ifndef TILE_SIZE_K
    #define TILE_SIZE_K 32  // K dimension tile size
#endif

#ifndef TILE_SIZE_N
    #define TILE_SIZE_N 32  // N dimension tile size
#endif

#ifndef USE_LDS
    #define USE_LDS 1       // Enable LDS optimization by default
#endif

// Q6_K block size constants
constexpr int Q6_K_BLOCK_SIZE = 256;           // Total bytes per block
constexpr int Q6_K_ELEMENTS_PER_BLOCK = 256;   // Elements per block
constexpr int Q6_K_SCALES_COUNT = 16;          // Number of scales per block
constexpr int Q6_K_ELEMENTS_PER_SCALE = 16;    // Elements sharing one scale

// Offsets within block
constexpr int Q6_K_SCALES_OFFSET = 0;   // Start of scales (32 bytes)
constexpr int Q6_K_QUANTS_OFFSET = 32;  // Start of quants (after scales)

// LDS tile size (when USE_LDS is enabled)
#if USE_LDS
    constexpr int LDS_TILE_SIZE = TILE_SIZE_K * TILE_SIZE_N;
#endif

/**
 * Convert half-precision float (FP16) to single-precision float (FP32)
 * Using direct bit manipulation for speed
 */
__device__ __forceinline__ float f16_to_f32(uint16_t f16_bits) {
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
 * Unpack 6-bit Q6_K value from packed data
 *
 * Q6_K stores 256 values in 192 bytes using 6-bit packing.
 * Values are stored as: [0:5][6:11][12:17]... across byte boundaries.
 *
 * @param data    Packed data array
 * @param element Index within block (0-255)
 * @return Unpacked 6-bit value as signed float (-32 to +31)
 */
__device__ __forceinline__ float unpack_q6_k_value(const uint8_t* data, int element) {
    // Calculate bit position: 6 bits per element
    const int bit_pos = element * 6;
    const int byte_idx = bit_pos / 8;
    const int bit_offset = bit_pos % 8;

    // Read up to 2 bytes to get the 6-bit value
    uint16_t combined = static_cast<uint16_t>(data[byte_idx]);
    if (byte_idx + 1 < Q6_K_BLOCK_SIZE - Q6_K_QUANTS_OFFSET) {
        combined |= static_cast<uint16_t>(data[byte_idx + 1]) << 8;
    }

    // Extract the 6-bit value
    const uint8_t quant = (combined >> bit_offset) & 0x3F;

    // Convert to signed: 0-31 -> 0 to 31, 32-63 -> -32 to -1
    int32_t signed_val = static_cast<int32_t>(quant);
    if (signed_val >= 32) {
        signed_val -= 64;
    }

    return __int2float_rn(signed_val);
}

/**
 * Dequantize a single Q6_K value to FP32
 *
 * @param block   Pointer to block data
 * @param element Element index within block (0-255)
 * @return Dequantized FP32 value
 */
__device__ __forceinline__ float dequantize_q6_k_element(const uint8_t* block, int element) {
    // Calculate which scale group
    const int scale_idx = element / Q6_K_ELEMENTS_PER_SCALE;  // 0-15

    // Read scale (half-precision) for this group
    const int scale_offset = Q6_K_SCALES_OFFSET + scale_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(block + scale_offset);
    const float scale = f16_to_f32(scale_bits);

    // Extract and dequantize the 6-bit value
    const float signed_val = unpack_q6_k_value(block + Q6_K_QUANTS_OFFSET, element);

    // Dequantize: value = signed_val * scale
    return signed_val * scale;
}

/**
 * Wave32 reduction for RDNA architectures
 *
 * Uses wave-level reduction for efficient partial sum accumulation.
 * RDNA3/RDNA2 use wave32 (32 threads per wave).
 */
__device__ __forceinline__ float wave_reduce_sum(float partial) {
#if WARP_SIZE == 32
    // RDNA3/RDNA2: Use native wave32 reduction
    #if __HIP_DEVICE_COMPILE__
        partial = __builtin_amdgcn_wave_reduce_fadd(partial);
    #else
        // Fallback: manual reduction
        for (int stride = 16; stride > 0; stride >>= 1) {
            partial += __shfl_down_f32(partial, stride, WARP_SIZE);
        }
    #endif
#else
    // CDNA: Use wave64 reduction
    #if __HIP_DEVICE_COMPILE__
        partial = __builtin_amdgcn_wave_reduce_fadd(partial);
    #else
        for (int stride = 32; stride > 0; stride >>= 1) {
            partial += __shfl_down_f32(partial, stride, WARP_SIZE);
        }
    #endif
#endif
    return partial;
}

/**
 * Fused Q6_K dequantization + matrix multiplication kernel with LDS optimization
 *
 * Computes output[m, n] = sum(activation[m, k] * dequantized_weight[k, n])
 * where dequantized_weight is computed on-the-fly from Q6_K format.
 *
 * Grid strategy: One block per output row (M dimension)
 * - Each block computes one full row of the output [1 x N]
 * - Threads within block collaborate on dot products via wave-level reduction
 *
 * Block strategy: Tunable threads for efficient reduction
 * - Threads iterate over K dimension in tiles of TILE_SIZE_K
 * - Each iteration: loads activation, dequantizes weight, multiplies, reduces
 * - Wave-level reduction combines partials from WARP_SIZE threads
 *
 * LDS Optimization (when USE_LDS=1):
 * - Loads frequently accessed weight tiles into LDS
 * - Reduces global memory access pressure
 * - Improves performance for larger matrices
 *
 * Memory access pattern:
 * - Activations: Coalesced reads (contiguous threads read contiguous memory)
 * - Weights: Strided reads based on output column index
 *
 * @param activations  Input tensor [M x K] in row-major (FP32)
 * @param weights_q6_k Quantized weights [K x N] in row-major (Q6_K format)
 * @param output       Output tensor [M x N] in row-major (FP32)
 * @param m            Batch/sequence dimension (rows in activation/output)
 * @param n            Output dimension (columns in output/weight)
 * @param k            Inner dimension (columns in activation, rows in weight)
 */
extern "C" __global__ void q6_k_matmul_kernel(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ weights_q6_k,
    float* __restrict__ output,
    const int m,
    const int n,
    const int k
) {
    // Block indexing: each block handles one row of output
    const int m_idx = blockIdx.x;  // Row index (0 to M-1)
    const int tid = threadIdx.x;   // Thread index in block

    // Bounds check
    if (m_idx >= m) {
        return;
    }

#if USE_LDS
    // LDS cache for weight tiles
    __shared__ float s_weight_tile[TILE_SIZE_K * TILE_SIZE_N];
    __shared__ float s_activation_tile[TILE_SIZE_K];
#endif

    // Activation row for this m_idx (start of row)
    const int activation_row_offset = m_idx * k;

    // Process n output columns in chunks
    // Each thread handles multiple output columns for efficiency
    const int columns_per_thread = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int n_start = tid * columns_per_thread;
    const int n_end = min(n_start + columns_per_thread, n);

    // Accumulators for each output column this thread handles
    float accumulators[16];  // Max 16 columns per thread (supports N up to 4096)
    for (int i = 0; i < columns_per_thread && (n_start + i) < n; ++i) {
        accumulators[i] = 0.0f;
    }

    // Iterate over K dimension (the dot product dimension)
    // Process TILE_SIZE_K elements at a time for efficient memory access
    for (int k_base = 0; k_base < k; k_base += TILE_SIZE_K) {
        // Each thread processes one element of K chunk
        const int k_tile_size = min(TILE_SIZE_K, k - k_base);

        for (int k_tile = 0; k_tile < k_tile_size; ++k_tile) {
            const int k_idx = k_base + k_tile;

            // Load activation value (coalesced read)
            const float activation = activations[activation_row_offset + k_idx];

#if USE_LDS
            // Load weight tile into LDS (cooperative load by all threads)
            // Each thread loads one weight value into LDS
            for (int col = n_start; col < n_end; ++col) {
                const int weight_linear_idx = k_idx * n + col;
                const int block_idx = weight_linear_idx / Q6_K_ELEMENTS_PER_BLOCK;
                const int element_in_block = weight_linear_idx % Q6_K_ELEMENTS_PER_BLOCK;

                const uint8_t* block = weights_q6_k + block_idx * Q6_K_BLOCK_SIZE;
                const float weight = dequantize_q6_k_element(block, element_in_block);

                // Store in LDS tile (indexed by tile position)
                const int lds_idx = k_tile * TILE_SIZE_N + (col - n_start);
                if (lds_idx < LDS_TILE_SIZE) {
                    s_weight_tile[lds_idx] = weight;
                }
            }

            // Store activation in LDS
            if (tid < TILE_SIZE_K) {
                s_activation_tile[k_tile] = activation;
            }
            __syncthreads();

            // Compute matmul using cached values in LDS
            for (int col = n_start; col < n_end; ++col) {
                const int lds_idx = k_tile * TILE_SIZE_N + (col - n_start);
                if (lds_idx < LDS_TILE_SIZE) {
                    accumulators[col - n_start] += s_activation_tile[k_tile] * s_weight_tile[lds_idx];
                }
            }
            __syncthreads();
#else
            // Direct computation without LDS (for small matrices or low-LDS GPUs)
            for (int col = n_start; col < n_end; ++col) {
                const int weight_linear_idx = k_idx * n + col;
                const int block_idx = weight_linear_idx / Q6_K_ELEMENTS_PER_BLOCK;
                const int element_in_block = weight_linear_idx % Q6_K_ELEMENTS_PER_BLOCK;

                const uint8_t* block = weights_q6_k + block_idx * Q6_K_BLOCK_SIZE;
                const float weight = dequantize_q6_k_element(block, element_in_block);

                accumulators[col - n_start] += activation * weight;
            }
#endif
        }
    }

    // Write results to output
    for (int col = n_start; col < n_end; ++col) {
        const int output_idx = m_idx * n + col;
        output[output_idx] = accumulators[col - n_start];
    }
}

/**
 * Fused Q6_K dequantization + matrix multiplication kernel (single output element)
 *
 * Simplified version: Each block computes exactly ONE output element.
 * Grid: (N, M, 1) - one block per output element
 * Block: BLOCK_SIZE threads for wave-level reduction
 *
 * This is simpler but launches more blocks. Use for smaller matrices.
 * Optimized for wave32/wave64 reduction.
 *
 * @param activations  Input tensor [M x K] in row-major (FP32)
 * @param weights_q6_k Quantized weights [K x N] in row-major (Q6_K format)
 * @param output       Output tensor [M x N] in row-major (FP32)
 * @param m            Batch/sequence dimension
 * @param n            Output dimension
 * @param k            Inner dimension
 */
extern "C" __global__ void q6_k_matmul_element_kernel(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ weights_q6_k,
    float* __restrict__ output,
    const int m,
    const int n,
    const int k
) {
    // Block indexing: each block handles one output element [m, n]
    const int n_idx = blockIdx.x;  // Column index (0 to N-1)
    const int m_idx = blockIdx.y;  // Row index (0 to M-1)
    const int tid = threadIdx.x;   // Thread index in block

    // Bounds check
    if (m_idx >= m || n_idx >= n) {
        return;
    }

    // Shared memory for wave-level reduction
    __shared__ float s_partial[WARP_SIZE];

    // Activation row for this m_idx
    const int activation_row_offset = m_idx * k;

    // Each thread computes partial dot product
    float partial_sum = 0.0f;

    // Iterate over K dimension
    // Process TILE_SIZE_K elements per iteration
    for (int k_base = 0; k_base < k; k_base += TILE_SIZE_K) {
        const int k_tile_size = min(TILE_SIZE_K, k - k_base);

        for (int k_tile = 0; k_tile < k_tile_size; ++k_tile) {
            const int k_idx = k_base + k_tile;

            // Check if this thread should process this k element
            // Distribute work across threads in the block
            if ((k_idx % BLOCK_SIZE) == tid) {
                // Load activation
                const float activation = activations[activation_row_offset + k_idx];

                // Dequantize weight[k_idx, n_idx] on-the-fly
                const int weight_linear_idx = k_idx * n + n_idx;
                const int block_idx = weight_linear_idx / Q6_K_ELEMENTS_PER_BLOCK;
                const int element_in_block = weight_linear_idx % Q6_K_ELEMENTS_PER_BLOCK;

                const uint8_t* block = weights_q6_k + block_idx * Q6_K_BLOCK_SIZE;
                const float weight = dequantize_q6_k_element(block, element_in_block);

                partial_sum += activation * weight;
            }
        }
    }

    // Wave-level reduction using wave32 or wave64
    const int lane_id = tid & (WARP_SIZE - 1);  // tid % WARP_SIZE

    // Store partial for this wave lane
    s_partial[lane_id] = partial_sum;
    __syncthreads();

    // Tree reduction optimized for WARP_SIZE
    if (WARP_SIZE == 32) {
        // Wave32 reduction (RDNA2/RDNA3)
        if (lane_id < 16) {
            s_partial[lane_id] += s_partial[lane_id + 16];
        }
        __syncthreads();

        if (lane_id < 8) {
            s_partial[lane_id] += s_partial[lane_id + 8];
        }
        __syncthreads();

        if (lane_id < 4) {
            s_partial[lane_id] += s_partial[lane_id + 4];
        }
        __syncthreads();

        if (lane_id < 2) {
            s_partial[lane_id] += s_partial[lane_id + 2];
        }
        __syncthreads();

        if (lane_id < 1) {
            s_partial[lane_id] += s_partial[lane_id + 1];
        }
        __syncthreads();
    } else {
        // Wave64 reduction (CDNA2/CDNA3)
        if (lane_id < 32) {
            s_partial[lane_id] += s_partial[lane_id + 32];
        }
        __syncthreads();

        if (lane_id < 16) {
            s_partial[lane_id] += s_partial[lane_id + 16];
        }
        __syncthreads();

        if (lane_id < 8) {
            s_partial[lane_id] += s_partial[lane_id + 8];
        }
        __syncthreads();

        if (lane_id < 4) {
            s_partial[lane_id] += s_partial[lane_id + 4];
        }
        __syncthreads();

        if (lane_id < 2) {
            s_partial[lane_id] += s_partial[lane_id + 2];
        }
        __syncthreads();

        if (lane_id < 1) {
            s_partial[lane_id] += s_partial[lane_id + 1];
        }
        __syncthreads();
    }

    // Thread 0 writes the result
    if (tid == 0) {
        const int output_idx = m_idx * n + n_idx;
        output[output_idx] = s_partial[0];
    }
}
