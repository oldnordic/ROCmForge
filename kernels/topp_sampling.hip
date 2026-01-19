/**
 * topp_sampling.hip - Top-p (nucleus) sampling kernel (multi-kernel pipeline)
 *
 * GPU: AMD Radeon (RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * DESIGN: Multi-kernel pipeline to avoid GPU watchdog timeout
 * - Kernel 1: topp_prefix_sum_kernel - Compute CDF using parallel scan
 * - Kernel 2: topp_threshold_kernel - Find top-p cutoff index via binary search
 * - Kernel 3: topp_sample_kernel - Sample token using binary search
 *
 * This approach avoids the watchdog timeout issues that plagued previous
 * single-kernel implementations with large vocab_size (151936 tokens).
 *
 * Based on FlashInfer's sorting-free rejection sampling approach:
 * https://flashinfer.ai/2025/03/10/sampling.html
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // Threads per block
constexpr int WARP_SIZE = 32;    // RDNA3 wavefront size

/**
 * Kernel 1: Compute inclusive prefix sum (CDF) per row
 *
 * Grid: (batch_size) blocks
 * Block: BLOCK_SIZE threads
 *
 * Algorithm: Two-pass parallel scan
 * - Phase 1: Each thread computes sum of its stride elements
 * - Phase 2: Thread 0 computes exclusive prefix sum of thread sums
 * - Phase 3: Each thread adds its offset to compute inclusive prefix sums
 *
 * All threads participate - no single-threaded loops over vocab_size.
 *
 * @param probs          Input probabilities [batch * vocab] (already softmax'd)
 * @param prefix_sum_out  Output CDF [batch * vocab]
 * @param batch          Number of batch elements
 * @param vocab          Vocabulary size
 */
extern "C" __global__ void topp_prefix_sum_kernel(
    const float* __restrict__ probs,
    float* __restrict__ prefix_sum_out,
    const int batch,
    const int vocab
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch) {
        return;
    }

    const int row_offset = batch_idx * vocab;

    // Shared memory for thread offsets
    __shared__ float s_thread_offsets[BLOCK_SIZE];

    // Phase 1: Each thread computes sum of its stride
    float thread_sum = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK_SIZE) {
        thread_sum += probs[row_offset + i];
    }

    s_thread_offsets[tid] = thread_sum;
    __syncthreads();

    // Phase 2: Compute exclusive prefix sum of thread sums
    // Thread 0 does this (only BLOCK_SIZE elements, very fast)
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            const float val = s_thread_offsets[i];
            s_thread_offsets[i] = acc;
            acc += val;
        }
    }
    __syncthreads();

    // Phase 3: Each thread computes inclusive prefix sums for its elements
    const float offset_for_thread = s_thread_offsets[tid];
    float running_sum = offset_for_thread;

    for (int i = tid; i < vocab; i += BLOCK_SIZE) {
        running_sum += probs[row_offset + i];
        prefix_sum_out[row_offset + i] = running_sum;
    }
}

/**
 * Kernel 2: Find threshold index for top-p sampling
 *
 * Grid: (batch_size) blocks
 * Block: BLOCK_SIZE threads (only thread 0 does work - O(log v) binary search)
 *
 * Algorithm: Binary search on CDF
 * - Finds smallest index where CDF >= top_p
 * - Single thread per row is acceptable (binary search is O(log v))
 * - Output: threshold index for topp_sample_kernel
 *
 * @param prefix_sum    CDF from prefix_sum_kernel [batch * vocab]
 * @param threshold_out Output cutoff index per row [batch]
 * @param top_p         Top-p threshold (e.g., 0.9 for top-90%)
 * @param batch         Number of batch elements
 * @param vocab         Vocabulary size
 */
extern "C" __global__ void topp_threshold_kernel(
    const float* __restrict__ prefix_sum,
    int* __restrict__ threshold_out,
    const float top_p,
    const int batch,
    const int vocab
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch) {
        return;
    }

    // Only thread 0 performs binary search (O(log vocab) operations)
    if (tid == 0) {
        const int row_offset = batch_idx * vocab;

        // Edge case: if top_p >= 1.0, include all tokens
        if (top_p >= 1.0f) {
            threshold_out[batch_idx] = vocab - 1;
            return;
        }

        // Edge case: if top_p <= 0.0, include only first token
        if (top_p <= 0.0f) {
            threshold_out[batch_idx] = 0;
            return;
        }

        // Binary search for smallest index where CDF >= top_p
        int low = 0;
        int high = vocab - 1;
        int threshold_idx = vocab - 1;  // Default: include all

        while (low <= high) {
            const int mid = low + (high - low) / 2;
            const float cdf_val = prefix_sum[row_offset + mid];

            if (cdf_val >= top_p) {
                threshold_idx = mid;
                high = mid - 1;  // Try to find a smaller index
            } else {
                low = mid + 1;
            }
        }

        // Ensure we have at least one token
        if (threshold_idx < 0) {
            threshold_idx = 0;
        }

        threshold_out[batch_idx] = threshold_idx;
    }
}

/**
 * Kernel 3: Sample token using binary search on CDF
 *
 * Grid: (batch_size) blocks
 * Block: BLOCK_SIZE threads (only thread 0 does work - O(log v) binary search)
 *
 * Algorithm: Binary search sampling with threshold limit
 * - Given random value u in [0, 1], find token via binary search on CDF
 * - Only search up to threshold index (from topp_threshold_kernel)
 * - Single thread per row is acceptable (binary search is O(log v))
 *
 * @param prefix_sum     CDF from prefix_sum_kernel [batch * vocab]
 * @param threshold_idx  Cutoff index from topp_threshold_kernel [batch]
 * @param random_values  Random values in [0, 1] per batch element [batch]
 * @param sampled_tokens Output token IDs [batch]
 * @param batch          Number of batch elements
 * @param vocab          Vocabulary size
 */
extern "C" __global__ void topp_sample_kernel(
    const float* __restrict__ prefix_sum,
    const int* __restrict__ threshold_idx,
    const float* __restrict__ random_values,
    int* __restrict__ sampled_tokens,
    const int batch,
    const int vocab
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch) {
        return;
    }

    // Only thread 0 performs binary search (O(log vocab) operations)
    if (tid == 0) {
        const int row_offset = batch_idx * vocab;
        const int max_idx = threshold_idx[batch_idx];

        // Clamp max_idx to valid range
        const int search_limit = (max_idx < 0) ? 0 : ((max_idx >= vocab) ? (vocab - 1) : max_idx);

        // Get random value and scale by CDF at search limit
        const float u = random_values[batch_idx];
        const float cdf_max = prefix_sum[row_offset + search_limit];
        const float target = u * cdf_max;

        // Binary search for the token
        int low = 0;
        int high = search_limit;
        int token_id = 0;

        while (low <= high) {
            const int mid = low + (high - low) / 2;
            const float cdf_val = prefix_sum[row_offset + mid];
            const float cdf_prev = (mid > 0) ? prefix_sum[row_offset + mid - 1] : 0.0f;

            if (target > cdf_prev && target <= cdf_val) {
                token_id = mid;
                break;
            } else if (target <= cdf_prev) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }

        // Fallback: if search exhausted, use the last valid index
        if (low > search_limit) {
            token_id = search_limit;
        }

        sampled_tokens[batch_idx] = token_id;
    }
}
