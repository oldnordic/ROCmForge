/**
 * flash_attention_causal.hip - Fused causal FlashAttention kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 32 threads (1 wave of 32) - exact wavefront size
 * Shared memory: 64 floats (32 for softmax weights, 32 for reduction)
 * Accumulator precision: FP32
 *
 * Computes causal attention in one kernel: QK^T → scale → causal mask → softmax → softmax × V
 *
 * Layout: [batch, heads, seq, dim] x [batch, heads, seq, dim] → [batch, heads, seq, dim]
 *
 * Causal masking: query at position i can only attend to keys at positions j <= i
 *
 * For each (batch, head, query_pos) triple:
 *   scores[key_pos] = scale * Q[query_pos] @ K[key_pos]^T
 *   Apply causal mask: scores[key_pos] = -inf if key_pos > query_pos
 *   weights = softmax(scores)
 *   output[query_pos] = sum over key_pos of (weights[key_pos] * V[key_pos])
 *
 * Thread safety: Block size == WARP_SIZE (32), so all threads participate.
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * Fused causal FlashAttention kernel
 *
 * Each block computes one output row:
 * - For a given (batch, head, query_pos), compute output[dim]
 * - Applies causal masking: query_pos can only attend to key_pos <= query_pos
 *
 * Grid: (seq_q, num_heads, batch_size) blocks
 * Block: WARP_SIZE threads (exactly one wavefront, no partial waves)
 *
 * @param Q          Query tensor [batch, heads, seq_q, dim]
 * @param K          Key tensor [batch, heads, seq_k, dim]
 * @param V          Value tensor [batch, heads, seq_k, dim]
 * @param output     Output tensor [batch, heads, seq_q, dim]
 * @param scale      Scale factor (typically 1.0 / sqrt(dim))
 * @param batch_size Number of batches
 * @param seq_q      Query sequence length
 * @param seq_k      Key sequence length
 * @param num_heads  Number of attention heads
 * @param head_dim   Dimension per head
 */
extern "C" __global__ void flash_attention_causal_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const float scale,
    const int batch_size,
    const int seq_q,
    const int seq_k,
    const int num_heads,
    const int head_dim
) {
    // Block indexing: each block handles one (query_pos, head, batch) triple
    const int query_pos = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;

    // Bounds check
    if (batch_idx >= batch_size || head_idx >= num_heads || query_pos >= seq_q) {
        return;
    }

// Dynamic shared memory layout:
// - s_all_shared[0..seq_k] = softmax weights (s_scores)
// - s_all_shared[seq_k..seq_k+WARP_SIZE] = reduction buffer (s_partial)
extern __shared__ float s_all_shared[];
#define s_scores (s_all_shared)
#define s_partial (s_all_shared + seq_k)

    // === Layout: [batch, heads, seq, dim] ===
    const int q_head_offset = batch_idx * num_heads * seq_q * head_dim
                           + head_idx * seq_q * head_dim;
    const int k_head_offset = batch_idx * num_heads * seq_k * head_dim
                           + head_idx * seq_k * head_dim;
    const int v_head_offset = batch_idx * num_heads * seq_k * head_dim
                           + head_idx * seq_k * head_dim;
    const int out_head_offset = batch_idx * num_heads * seq_q * head_dim
                             + head_idx * seq_q * head_dim;

    // Q row for this query position
    const int q_row_offset = q_head_offset + query_pos * head_dim;

    // === Step 1: Compute all QK^T scores ===
    // Each thread computes scores for key positions where key_pos % WARP_SIZE == tid
    // Each thread computes the full dot product directly from global memory
    for (int key_pos = tid; key_pos < seq_k; key_pos += WARP_SIZE) {
        const int k_row_offset = k_head_offset + key_pos * head_dim;

        // Dot product Q[query_pos] @ K[key_pos]^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[q_row_offset + d] * K[k_row_offset + d];
        }

        // Apply scale
        s_scores[key_pos] = score * scale;
    }
    __syncthreads();

    // === Step 2: Apply causal mask ===
    // Causal: query_pos can only attend to key_pos <= query_pos
    // All threads participate (all need to see updated scores)
    for (int key_pos = tid; key_pos < seq_k; key_pos += WARP_SIZE) {
        if (key_pos > query_pos) {
            s_scores[key_pos] = -(__builtin_inff());  // -inf
        }
    }
    __syncthreads();

    // === Step 3: Softmax (numerically stable) ===
    // Find max
    float max_score = s_scores[0];
    #pragma unroll
    for (int i = 1; i < seq_k; i++) {
        max_score = fmaxf(max_score, s_scores[i]);
    }

    // Compute exp and sum
    float exp_sum = 0.0f;
    for (int i = 0; i < seq_k; i++) {
        // For -inf values, exp should be 0
        if (s_scores[i] > -1e30f) {
            s_scores[i] = expf(s_scores[i] - max_score);
            exp_sum += s_scores[i];
        } else {
            s_scores[i] = 0.0f;
        }
    }

    // Normalize to get weights
    float inv_sum = 1.0f / (exp_sum + 1e-6f);  // Avoid div by zero
    for (int i = 0; i < seq_k; i++) {
        s_scores[i] *= inv_sum;
    }
    __syncthreads();

    // === Step 4: Weighted × V ===
    // output[d] = sum over key_pos of (weights[key_pos] * V[key_pos, d])
    // IMPORTANT: s_scores contains softmax weights, don't overwrite!
    for (int d = 0; d < head_dim; d++) {
        float partial = 0.0f;
        for (int key_pos = tid; key_pos < seq_k; key_pos += WARP_SIZE) {
            partial += s_scores[key_pos] * V[v_head_offset + key_pos * head_dim + d];
        }

        // Wave32 reduction using s_partial (NOT s_scores!)
        s_partial[tid] = partial;
        __syncthreads();
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial[tid] += s_partial[tid + stride];
            }
            __syncthreads();
        }

        // Write output
        if (tid == 0) {
            int out_idx = out_head_offset + query_pos * head_dim + d;
            output[out_idx] = s_partial[0];
        }
        __syncthreads();
    }
}
