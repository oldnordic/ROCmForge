/**
 * flash_attention.hip - Fused Attention kernel with tunable parameters
 *
 * Tunable kernel optimized for AMD GPU architectures:
 * - RDNA3 (gfx1100): Wave32, 256 threads/block
 * - RDNA2 (gfx1030): Wave32, 256 threads/block
 * - CDNA2 (gfx90a): Wave64, 512 threads/block
 *
 * Tuning Parameters (configurable via preprocessor defines):
 * - BLOCK_SIZE: Threads per block (default: 256)
 * - WARP_SIZE: Wavefront size (default: 32 for RDNA, 64 for CDNA)
 * - MAX_HEAD_DIM: Maximum head dimension supported (default: 128)
 * - USE_LDS: Enable Local Data Share optimization (default: 1)
 *
 * Fuses: QK^T matmul, scale, mask, softmax, softmax*V into single kernel
 * Eliminates CPU-GPU round-trips for better performance
 */

#include <hip/hip_runtime.h>

// Default tuning constants (can be overridden at compile time)
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 256  // Default: 8 waves of 32 threads (RDNA3)
#endif

#ifndef WARP_SIZE
    #define WARP_SIZE 32    // Default: RDNA3 wavefront size
#endif

#ifndef MAX_HEAD_DIM
    #define MAX_HEAD_DIM 128  // Maximum head dimension we support
#endif

#ifndef USE_LDS
    #define USE_LDS 1       // Enable LDS optimization by default
#endif

/**
 * FlashAttention kernel - computes attention in a single fused pass
 *
 * Algorithm per block (one query position):
 *   1. Compute QK^T for one query position vs all key positions
 *   2. Apply scaling factor (1.0 / sqrt(head_dim))
 *   3. Apply causal mask if enabled
 *   4. Compute softmax (row-wise)
 *   5. Compute output: softmax x V
 *
 * Tensor layouts (row-major):
 *   Q: [batch_size, seq_len, num_heads, head_dim]
 *   K: [batch_size, seq_len, num_heads, head_dim]
 *   V: [batch_size, seq_len, num_heads, head_dim]
 *   output: [batch_size, seq_len, num_heads, head_dim]
 *   mask: [batch_size, seq_len, seq_len] (optional, row-major)
 *
 * @param Q         Query tensor
 * @param K         Key tensor
 * @param V         Value tensor
 * @param output    Output tensor (pre-allocated)
 * @param mask      Optional causal mask (use nullptr for no mask)
 * @param scale     Scaling factor (typically 1.0 / sqrt(head_dim))
 * @param batch_size Number of batches
 * @param seq_len   Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim  Dimension per head
 */
extern "C" __global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const float* __restrict__ mask,
    const float scale,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    // Each thread block processes one (batch, head, query_pos) triple
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_pos = blockIdx.x;

    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || query_pos >= seq_len) {
        return;
    }

#if USE_LDS
    // Shared memory for intermediate results with LDS optimization
    // Size is configurable based on BLOCK_SIZE and seq_len
    __shared__ float s_partial[BLOCK_SIZE];  // Partial reduction results
    __shared__ float s_scores[256];          // Final attention scores
    __shared__ float s_max;                  // Max for numerical stability
    __shared__ float s_sum;                  // Sum for normalization

    // LDS cache for Q and V rows to reduce global memory access
    __shared__ float s_q_row[MAX_HEAD_DIM];  // Cached Q row
    __shared__ float s_v_row[256 * MAX_HEAD_DIM];  // Cached V rows (seq_len x head_dim)
#else
    // Fallback: Use standard shared memory without LDS caching
    __shared__ float s_partial[BLOCK_SIZE];
    __shared__ float s_scores[256];
    __shared__ float s_max;
    __shared__ float s_sum;
#endif

    // Initialize shared memory to avoid garbage values
    s_partial[tid] = 0.0f;
    if (tid < seq_len) {
        s_scores[tid] = 0.0f;
    }
    if (tid == 0) {
        s_max = 0.0f;
        s_sum = 0.0f;
    }
    __syncthreads();

    // Base pointers for this batch
    // Tensor layout: [batch_size, seq_len, head_dim] (3D, not 4D)
    const int batch_offset = batch_idx * seq_len * head_dim;
    const float* Q_batch = Q + batch_offset;
    const float* K_batch = K + batch_offset;
    const float* V_batch = V + batch_offset;
    float* output_batch = output + batch_offset;

    // Load Q row for this query position
    float q_row[MAX_HEAD_DIM];
    const int q_row_offset = query_pos * head_dim;

    // Load Q row (each thread loads multiple elements if head_dim > BLOCK_SIZE)
    for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
        if (i < MAX_HEAD_DIM) {
            q_row[i] = Q_batch[q_row_offset + i];
        }
    }

#if USE_LDS
    // Load all V rows into LDS cache (seq_len x head_dim)
    // This reduces global memory access during the weighted matmul
    for (int i = tid; i < seq_len * head_dim; i += BLOCK_SIZE) {
        if (i < 256 * MAX_HEAD_DIM) {
            s_v_row[i] = V_batch[i];
        }
    }
    __syncthreads();
#endif

    // Compute QK^T scores for this query position vs all key positions
    for (int key_pos = 0; key_pos < seq_len; key_pos++) {
        // Clear partial reduction memory
        s_partial[tid] = 0.0f;
        __syncthreads();

        // Each thread computes a partial dot product
        float partial_score = 0.0f;

        for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
            if (i < MAX_HEAD_DIM) {
                // QK^T computation: Q[query_pos, i] * K[i, key_pos]
                partial_score += q_row[i] * K_batch[i * seq_len + key_pos];
            }
        }

        // Store partial result in s_partial
        s_partial[tid] = partial_score;
        __syncthreads();

        // Reduce partial scores (wave32/wave64 optimized)
        for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial[tid] += s_partial[tid + stride];
            }
            __syncthreads();
        }

        // First thread handles the score
        if (tid == 0) {
            float score = s_partial[0] * scale;

            // Apply mask if provided
            if (mask != nullptr) {
                const int mask_idx = batch_idx * seq_len * seq_len + query_pos * seq_len + key_pos;
                float mask_val = mask[mask_idx];
                if (mask_val < -1e30f) {
                    score = mask_val;
                }
            }

            s_scores[key_pos] = score;
        }
        __syncthreads();
    }

    // Compute softmax (reduce to find max, then sum of exp)
    // First pass: find max
    float max_val = -1e30f;
    for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
        float val = s_scores[j];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Reduce max across threads using s_partial
    s_partial[tid] = max_val;
    __syncthreads();
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = s_partial[tid];
            float b = s_partial[tid + stride];
            s_partial[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    max_val = s_partial[0];
    s_max = max_val;
    __syncthreads();

    // Second pass: compute exp and sum
    float sum = 0.0f;
    for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
        float val = expf(s_scores[j] - max_val);
        sum += val;
    }

    // Reduce sum across threads using s_partial
    s_partial[tid] = sum;
    __syncthreads();
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    sum = s_partial[0];
    s_sum = sum;
    __syncthreads();

    // Normalize scores: compute softmax and store back in s_scores
    float inv_sum = 1.0f / (sum + 1e-6f);  // Avoid div by zero
    for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
        float val = expf(s_scores[j] - max_val) * inv_sum;
        s_scores[j] = val;  // Now s_scores contains softmax weights
    }
    __syncthreads();

    // Compute output: softmax x V
    const int out_row_offset = query_pos * head_dim;

    for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {
        s_partial[tid] = 0.0f;
        __syncthreads();

        float partial_sum = 0.0f;

#if USE_LDS
        // Use cached V rows from LDS
        for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
            const int v_offset = j * MAX_HEAD_DIM + dim_idx;
            if (v_offset < 256 * MAX_HEAD_DIM) {
                partial_sum += s_scores[j] * s_v_row[v_offset];
            }
        }
#else
        // Direct access to global memory
        for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
            const int v_offset = j * head_dim + dim_idx;
            partial_sum += s_scores[j] * V_batch[v_offset];
        }
#endif

        s_partial[tid] = partial_sum;
        __syncthreads();

        // Reduce partial sums using s_partial
        for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial[tid] += s_partial[tid + stride];
            }
            __syncthreads();
        }

        // Write output
        if (tid == 0 && dim_idx < MAX_HEAD_DIM) {
            output_batch[out_row_offset + dim_idx] = s_partial[0];
        }
    }
}

/**
 * Wave-level reduction helper for flash attention
 *
 * Provides architecture-optimized reduction for wave32 and wave64
 */
__device__ __forceinline__ float wave_reduce_max(float val) {
#if WARP_SIZE == 32
    // Wave32 reduction (RDNA2/RDNA3)
    for (int stride = 16; stride > 0; stride >>= 1) {
        float other = __shfl_down(val, stride, WARP_SIZE);
        val = (val > other) ? val : other;
    }
#else
    // Wave64 reduction (CDNA2/CDNA3)
    for (int stride = 32; stride > 0; stride >>= 1) {
        float other = __shfl_down(val, stride, WARP_SIZE);
        val = (val > other) ? val : other;
    }
#endif
    return val;
}

__device__ __forceinline__ float wave_reduce_add(float val) {
#if WARP_SIZE == 32
    // Wave32 reduction
    for (int stride = 16; stride > 0; stride >>= 1) {
        val += __shfl_down(val, stride, WARP_SIZE);
    }
#else
    // Wave64 reduction
    for (int stride = 32; stride > 0; stride >>= 1) {
        val += __shfl_down(val, stride, WARP_SIZE);
    }
#endif
    return val;
}
