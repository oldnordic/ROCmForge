/**
 * topk_topp_sampling.hip - Fused top-k + top-p sampling kernel
 *
 * GPU: AMD Radeon (RDNA3, wave32)
 * Algorithm: Dual pivot rejection sampling
 *
 * Combines top-k and top-p filtering in single pass:
 * - Only consider tokens in top-k
 * - From those, select tokens covering cumulative probability >= top-p
 * - Sample from filtered set
 *
 * Based on FlashInfer's approach:
 * https://flashinfer.ai/2025/03/10/sampling.html
 */

#include <hip/hip_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_ITERATIONS = 10;

/**
 * Fused top-k + top-p sampling using rejection sampling
 *
 * Grid: (batch_size) blocks
 * Block: BLOCK_SIZE threads
 *
 * @param probabilities  Input probabilities [batch_size, vocab_size]
 * @param random_values  Random values [batch_size]
 * @param output         Sampled token IDs [batch_size]
 * @param top_k          K value (number of top tokens to consider)
 * @param top_p          Cumulative probability threshold
 * @param batch_size     Number of batch elements
 * @param vocab_size     Vocabulary size
 */
extern "C" __global__ void topk_topp_sampling_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ random_values,
    uint32_t* __restrict__ output,
    const int top_k,
    const float top_p,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int row_offset = batch_idx * vocab_size;
    const float u = random_values[batch_idx];

    if (tid == 0) {
        const int effective_k = (top_k >= vocab_size) ? vocab_size : top_k;
        float pivot = 0.0f;
        bool found = false;

        // Rejection sampling loop
        for (int iter = 0; iter < MAX_ITERATIONS && !found; iter++) {
            // Step 1: Count tokens >= pivot and compute their sum
            int count_above_pivot = 0;
            float sum_above_pivot = 0.0f;

            // TODO: Single-threaded loop over vocab_size is a watchdog timeout risk.
            // Refactor to parallel reduction pattern (see topk_sampling.hip for example).
            for (int i = 0; i < vocab_size; i++) {
                const float p = probabilities[row_offset + i];
                if (p >= pivot) {
                    count_above_pivot++;
                    sum_above_pivot += p;
                }
            }

            // Step 2: Check if both conditions can be met
            // Condition 1: At least k tokens above pivot
            // Condition 2: Cumulative probability >= top_p

            if (count_above_pivot < effective_k || sum_above_pivot < top_p) {
                // Update pivot and continue
                // Find a better pivot (next highest probability below current)
                float next_pivot = 0.0f;
                // TODO: Single-threaded loop over vocab_size is a watchdog timeout risk.
                // Refactor to parallel pattern.
                for (int i = 0; i < vocab_size; i++) {
                    const float p = probabilities[row_offset + i];
                    if (p < pivot && p > next_pivot) {
                        next_pivot = p;
                    }
                }
                pivot = next_pivot;
                continue;
            }

            // Step 3: Sample using inverse transform
            // Scale u to range [0, sum_above_pivot]
            const float target = u * sum_above_pivot;
            float cumulative = 0.0f;
            int sampled_idx = -1;

            // TODO: Single-threaded loop over vocab_size is a watchdog timeout risk.
            // Refactor to parallel prefix sum + binary search pattern.
            for (int i = 0; i < vocab_size; i++) {
                const float p = probabilities[row_offset + i];
                if (p >= pivot) {
                    cumulative += p;
                    if (cumulative >= target) {
                        sampled_idx = i;
                        break;
                    }
                }
            }

            if (sampled_idx < 0) {
                sampled_idx = vocab_size - 1;
            }

            // Step 4: Verify sampled token meets criteria
            const float p_sample = probabilities[row_offset + sampled_idx];

            // Count tokens >= p_sample
            int count_at_or_above = 0;
            float sum_at_or_above = 0.0f;

            // TODO: Single-threaded loop over vocab_size is a watchdog timeout risk.
            // Refactor to parallel reduction pattern.
            for (int i = 0; i < vocab_size; i++) {
                const float p = probabilities[row_offset + i];
                if (p >= p_sample) {
                    count_at_or_above++;
                    sum_at_or_above += p;
                }
            }

            // Check both conditions
            if (count_at_or_above >= effective_k && sum_at_or_above >= top_p) {
                // Accept
                output[batch_idx] = static_cast<uint32_t>(sampled_idx);
                found = true;
            } else {
                // Reject and update pivot
                pivot = p_sample;
            }
        }

        // Fallback: if not found, use argmax
        if (!found) {
            float max_p = -1.0f;
            int max_idx = 0;
            // TODO: Single-threaded loop over vocab_size is a watchdog timeout risk.
            // Refactor to parallel reduction pattern.
            for (int i = 0; i < vocab_size; i++) {
                const float p = probabilities[row_offset + i];
                if (p > max_p) {
                    max_p = p;
                    max_idx = i;
                }
            }
            output[batch_idx] = static_cast<uint32_t>(max_idx);
        }
    }
}

/**
 * Simplified fused top-k + top-p sampling
 *
 * Two-pass approach:
 * 1. Apply top-p to get candidate set
 * 2. Apply top-k to candidate set
 * 3. Sample from final set
 *
 * Easier to verify, less efficient than rejection sampling
 *
 * TODO: This entire kernel uses single-threaded loops (tid == 0) over vocab_size.
 * This is a watchdog timeout risk for large vocabularies. Refactor to parallel
 * pattern similar to topk_sampling.hip before using in production.
 *
 * @param probabilities  Input probabilities [batch_size, vocab_size]
 * @param cdf            Pre-computed CDF [batch_size, vocab_size]
 * @param random_values  Random values [batch_size]
 * @param output         Sampled token IDs [batch_size]
 * @param top_k          K value
 * @param top_p          Cumulative probability threshold
 * @param batch_size     Number of batch elements
 * @param vocab_size     Vocabulary size
 */
extern "C" __global__ void topk_topp_sampling_simple_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ cdf,
    const float* __restrict__ random_values,
    uint32_t* __restrict__ output,
    const int top_k,
    const float top_p,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int row_offset = batch_idx * vocab_size;
    const float u = random_values[batch_idx];

    if (tid == 0) {
        const int effective_k = (top_k >= vocab_size) ? vocab_size : top_k;

        // Step 1: Find top-p cutoff index
        int topp_cutoff = vocab_size - 1;
        for (int i = 0; i < vocab_size; i++) {
            if (cdf[row_offset + i] >= top_p) {
                topp_cutoff = i;
                break;
            }
        }

        // Step 2: Among top-p tokens, find top-k
        // Collect (probability, index) pairs for top-p tokens
        struct TokenInfo {
            float prob;
            int idx;
        };

        // Use simplified approach: scan and track top-k
        float top_k_probs[32];
        int top_k_indices[32];
        int top_k_count = 0;

        for (int i = 0; i <= topp_cutoff && i < vocab_size; i++) {
            const float p = probabilities[row_offset + i];

            // Insert into top-k
            if (top_k_count < effective_k) {
                top_k_probs[top_k_count] = p;
                top_k_indices[top_k_count] = i;
                top_k_count++;
            } else {
                // Find min in top-k
                int min_idx = 0;
                for (int j = 1; j < top_k_count; j++) {
                    if (top_k_probs[j] < top_k_probs[min_idx]) {
                        min_idx = j;
                    }
                }

                if (p > top_k_probs[min_idx]) {
                    top_k_probs[min_idx] = p;
                    top_k_indices[min_idx] = i;
                }
            }
        }

        // Step 3: Sample from top-k tokens
        float sum_top_k = 0.0f;
        for (int i = 0; i < top_k_count; i++) {
            sum_top_k += top_k_probs[i];
        }

        if (sum_top_k < 1e-10f) {
            // Fallback to argmax
            output[batch_idx] = static_cast<uint32_t>(top_k_indices[0]);
            return;
        }

        // Normalize and sample
        const float target = u * sum_top_k;
        float cumulative = 0.0f;

        for (int i = 0; i < top_k_count; i++) {
            cumulative += top_k_probs[i];
            if (cumulative >= target) {
                output[batch_idx] = static_cast<uint32_t>(top_k_indices[i]);
                return;
            }
        }

        // Fallback to last in top-k
        output[batch_idx] = static_cast<uint32_t>(top_k_indices[top_k_count - 1]);
    }
}

/**
 * Compute fused top-k + top-p mask (for testing/verification)
 *
 * TODO: This kernel uses single-threaded loops (tid == 0) over vocab_size.
 * This is a watchdog timeout risk for large vocabularies. Refactor to parallel
 * pattern before using in production.
 *
 * @param probabilities Input probabilities [batch_size, vocab_size]
 * @param cdf           Pre-computed CDF [batch_size, vocab_size]
 * @param mask          Output mask [batch_size, vocab_size]
 * @param top_k         K value
 * @param top_p         Cumulative probability threshold
 * @param batch_size    Number of batch elements
 * @param vocab_size    Vocabulary size
 */
extern "C" __global__ void topk_topp_mask_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ cdf,
    float* __restrict__ mask,
    const int top_k,
    const float top_p,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int row_offset = batch_idx * vocab_size;
    const int effective_k = (top_k >= vocab_size) ? vocab_size : top_k;

    if (tid == 0) {
        // Initialize mask to 0
        for (int i = 0; i < vocab_size; i++) {
            mask[row_offset + i] = 0.0f;
        }

        // Step 1: Find top-p cutoff
        int topp_cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            if (cdf[row_offset + i] >= top_p) {
                topp_cutoff = i + 1;  // Include this token
                break;
            }
        }

        // Step 2: Among top-p tokens, mark top-k
        // Use simplified selection
        float top_k_probs[32];
        int top_k_indices[32];
        int top_k_count = 0;

        for (int i = 0; i < topp_cutoff && i < vocab_size; i++) {
            const float p = probabilities[row_offset + i];

            if (top_k_count < effective_k && top_k_count < 32) {
                top_k_probs[top_k_count] = p;
                top_k_indices[top_k_count] = i;
                top_k_count++;
            } else {
                int min_idx = 0;
                for (int j = 1; j < top_k_count; j++) {
                    if (top_k_probs[j] < top_k_probs[min_idx]) {
                        min_idx = j;
                    }
                }

                if (p > top_k_probs[min_idx]) {
                    top_k_probs[min_idx] = p;
                    top_k_indices[min_idx] = i;
                }
            }
        }

        // Mark top-k tokens in mask
        for (int i = 0; i < top_k_count; i++) {
            if (top_k_indices[i] >= 0) {
                mask[row_offset + top_k_indices[i]] = 1.0f;
            }
        }
    }
}
