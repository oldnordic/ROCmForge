/**
 * q4_0_matmul.hip - Fused Q4_0 Dequantization + MatMul Kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32 threads)
 *
 * Implements fused dequantization + matrix multiplication for Q4_0 format.
 * This is the key innovation for inference efficiency - we don't materialize
 * the full FP32 weight matrix before matmul, eliminating the intermediate buffer.
 *
 * Q4_0 Format:
 * - Block size: 32 elements
 * - Per block: scale (f32, 4 bytes) + 16 bytes 4-bit packed values = 20 bytes
 * - Dequantization: value = scale * ((packed & 0x0F) - 8)
 *
 * Kernel Operation:
 * - Input: Activations [M x K] (FP32), Weights [K x N] (Q4_0 quantized)
 * - Output: [M x N] (FP32)
 * - For each output element, dequantize the weight column on-the-fly and compute dot product
 *
 * Memory Bandwidth Savings:
 * - Traditional: Read Q4_0 (K*N/4) + Write FP32 (K*N*4) + Read FP32 (K*N*4) = ~8.5*K*N bytes
 * - Fused: Read Q4_0 (K*N/4) + Read Q4_0 again (K*N/4) = ~0.5*K*N bytes
 * - ~17x reduction in memory bandwidth for the weight matrix
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
 * Fused Q4_0 dequantization + matrix multiplication kernel
 *
 * Computes output[m, n] = sum(activation[m, k] * dequantized_weight[k, n])
 * where dequantized_weight is computed on-the-fly from Q4_0 format.
 *
 * Grid strategy: One block per output row (M dimension)
 * - Each block computes one full row of the output [1 x N]
 * - Threads within block collaborate on dot products via wave32 reduction
 *
 * Block strategy: 256 threads for efficient reduction
 * - Threads iterate over K dimension in chunks of 256
 * - Each iteration: loads activation, dequantizes weight, multiplies, reduces
 * - Wave-level reduction combines partials from 32 threads
 *
 * Memory access pattern:
 * - Activations: Coalesced reads (contiguous threads read contiguous memory)
 * - Weights: Strided reads based on output column index
 *
 * @param activations  Input tensor [M x K] in row-major (FP32)
 * @param weights_q4_0 Quantized weights [K x N] in row-major (Q4_0 format)
 *                     Note: stored column-major for efficient access
 * @param output       Output tensor [M x N] in row-major (FP32)
 * @param m            Batch/sequence dimension (rows in activation/output)
 * @param n            Output dimension (columns in output/weight)
 * @param k            Inner dimension (columns in activation, rows in weight)
 */
extern "C" __global__ void q4_0_matmul_kernel(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ weights_q4_0,
    float* __restrict__ output,
    const int m,
    const int n,
    const int k
) {
    // Block indexing: each block handles one row of output
    const int m_idx = blockIdx.x;  // Row index (0 to M-1)
    const int tid = threadIdx.x;   // Thread index in block (0 to 255)

    // Bounds check
    if (m_idx >= m) {
        return;
    }

    // Shared memory for wave32 reduction
    __shared__ float s_partial[BLOCK_SIZE];

    // Activation row for this m_idx (start of row)
    const int activation_row_offset = m_idx * k;

    // Process n output columns in chunks (256 threads can handle multiple columns)
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
    // Process 256 elements at a time for efficient memory access
    for (int k_base = 0; k_base < k; k_base += BLOCK_SIZE) {
        // Each thread processes one element of K chunk
        const int k_idx = k_base + tid;
        if (k_idx >= k) {
            s_partial[tid] = 0.0f;
        } else {
            // Load activation value (coalesced read - all threads read contiguous)
            const float activation = activations[activation_row_offset + k_idx];

            // For each output column, dequantize weight and accumulate
            for (int col = n_start; col < n_end; ++col) {
                // Calculate weight position in Q4_0 format
                // Weights stored as [K x N] in Q4_0 blocks
                // For weight[k_idx, col], we need:
                // - Block index: (k_idx * N + col) / 32
                // - Element in block: (k_idx * N + col) % 32

                const int weight_linear_idx = k_idx * n + col;
                const int block_idx = weight_linear_idx / Q4_0_ELEMENTS_PER_BLOCK;
                const int element_in_block = weight_linear_idx % Q4_0_ELEMENTS_PER_BLOCK;

                // Read Q4_0 block
                const int block_offset = block_idx * Q4_0_BLOCK_SIZE;
                const float scale = *reinterpret_cast<const float*>(weights_q4_0 + block_offset);
                const uint8_t* quant_data = weights_q4_0 + block_offset + 4;

                // Dequantize weight on-the-fly
                const float quant_value = unpack_q4_0_value(quant_data, element_in_block);
                const float weight = scale * quant_value;

                // Accumulate
                accumulators[col - n_start] += activation * weight;
            }

            // Store partial for synchronization
            s_partial[tid] = 1.0f;  // Signal that we processed this k_idx
        }
        __syncthreads();
    }

    // Write results to output
    for (int col = n_start; col < n_end; ++col) {
        const int output_idx = m_idx * n + col;
        output[output_idx] = accumulators[col - n_start];
    }
}

/**
 * Fused Q4_0 dequantization + matrix multiplication kernel (single output element)
 *
 * Simplified version: Each block computes exactly ONE output element.
 * Grid: (N, M, 1) - one block per output element
 * Block: 256 threads for wave-level reduction
 *
 * This is simpler but launches more blocks. Use for smaller matrices.
 *
 * @param activations  Input tensor [M x K] in row-major (FP32)
 * @param weights_q4_0 Quantized weights [K x N] in row-major (Q4_0 format)
 * @param output       Output tensor [M x N] in row-major (FP32)
 * @param m            Batch/sequence dimension
 * @param n            Output dimension
 * @param k            Inner dimension
 */
extern "C" __global__ void q4_0_matmul_element_kernel(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ weights_q4_0,
    float* __restrict__ output,
    const int m,
    const int n,
    const int k
) {
    // Block indexing: each block handles one output element [m, n]
    const int n_idx = blockIdx.x;  // Column index (0 to N-1)
    const int m_idx = blockIdx.y;  // Row index (0 to M-1)
    const int tid = threadIdx.x;   // Thread index in block (0 to 255)

    // Bounds check
    if (m_idx >= m || n_idx >= n) {
        return;
    }

    // Shared memory for wave32 reduction
    __shared__ float s_partial[WARP_SIZE];

    // Activation row for this m_idx
    const int activation_row_offset = m_idx * k;

    // Each thread computes partial dot product
    float partial_sum = 0.0f;

    // Iterate over K dimension (256 elements per iteration)
    for (int k_base = 0; k_base < k; k_base += BLOCK_SIZE) {
        const int k_idx = k_base + tid;
        if (k_idx < k) {
            // Load activation
            const float activation = activations[activation_row_offset + k_idx];

            // Dequantize weight[k_idx, n_idx] on-the-fly
            const int weight_linear_idx = k_idx * n + n_idx;
            const int block_idx = weight_linear_idx / Q4_0_ELEMENTS_PER_BLOCK;
            const int element_in_block = weight_linear_idx % Q4_0_ELEMENTS_PER_BLOCK;

            const int block_offset = block_idx * Q4_0_BLOCK_SIZE;
            const float scale = *reinterpret_cast<const float*>(weights_q4_0 + block_offset);
            const uint8_t* quant_data = weights_q4_0 + block_offset + 4;

            const float quant_value = unpack_q4_0_value(quant_data, element_in_block);
            const float weight = scale * quant_value;

            partial_sum += activation * weight;
        }
    }

    // Wave32 reduction
    // Each group of 32 threads reduces their partials
    const int lane_id = tid & (WARP_SIZE - 1);  // tid % 32
    const int warp_id = tid / WARP_SIZE;

    // Store partial for this warp lane
    s_partial[lane_id] = partial_sum;
    __syncthreads();

    // Only first 32 threads participate in reduction
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

    // Thread 0 writes the result
    if (tid == 0) {
        const int output_idx = m_idx * n + n_idx;
        output[output_idx] = s_partial[0];
    }
}
