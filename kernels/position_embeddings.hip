/**
 * position_embeddings.hip - Position embedding application kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Applies rotary position embeddings (RoPE) to query and key tensors on GPU.
 * This kernel is used by apply_position_embeddings_device to avoid CPU round-trips.
 *
 * Contract:
 * - Input: q, k tensors [seq_len, num_heads, head_dim], cos/sin [seq_len, head_dim/2]
 * - Output: q, k with RoPE applied (modified in-place)
 * - CPU reference: apply_position_embeddings in src/model/glm_position.rs
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * Position embedding kernel - applies RoPE to query and key tensors
 *
 * Rotates pairs of dimensions in head_dim using precomputed cos/sin:
 * x0' = x0 * cos - x1 * sin
 * x1' = x0 * sin + x1 * cos
 *
 * @param q         Query tensor [seq_len, num_heads, head_dim] (modified in-place)
 * @param k         Key tensor [seq_len, num_heads, head_dim] (modified in-place)
 * @param cos       Precomputed cosine values [seq_len, head_dim/2]
 * @param sin       Precomputed sine values [seq_len, head_dim/2]
 * @param seq_len   Sequence length (number of tokens)
 * @param num_heads Number of attention heads
 * @param head_dim  Dimension per head (must be even)
 */
extern "C" __global__ void position_embeddings_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    // Each thread handles one pair in one head for one token
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_pair = threadIdx.x;

    const int half_dim = head_dim / 2;

    // Boundary checks
    if (token_idx >= seq_len || head_idx >= num_heads || dim_pair >= half_dim) {
        return;
    }

    // === Apply RoPE to Query tensor ===
    // Linear index into Q tensor [seq_len, num_heads, head_dim]
    const int q_base = (token_idx * num_heads + head_idx) * head_dim;

    // Pair indices: (dim_pair, dim_pair + half_dim)
    const int q_i0 = q_base + dim_pair;
    const int q_i1 = q_base + dim_pair + half_dim;

    // cos/sin index: [token_idx, dim_pair]
    const int cos_idx = token_idx * half_dim + dim_pair;

    // Load cos/sin values
    const float c = cos[cos_idx];
    const float s = sin[cos_idx];

    // Load Q values
    const float q0 = q[q_i0];
    const float q1 = q[q_i1];

    // Apply rotation to Q
    q[q_i0] = q0 * c - q1 * s;
    q[q_i1] = q0 * s + q1 * c;

    // === Apply RoPE to Key tensor (same operation) ===
    // Linear index into K tensor [seq_len, num_heads, head_dim]
    const int k_base = (token_idx * num_heads + head_idx) * head_dim;

    // Pair indices: (dim_pair, dim_pair + half_dim)
    const int k_i0 = k_base + dim_pair;
    const int k_i1 = k_base + dim_pair + half_dim;

    // Load K values
    const float k0 = k[k_i0];
    const float k1 = k[k_i1];

    // Apply rotation to K
    k[k_i0] = k0 * c - k1 * s;
    k[k_i1] = k0 * s + k1 * c;
}
