#include "common.hip"

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18
#define Q8_0_BLOCK_SIZE 34

struct Q4_0_block_ngu {
    half d;
    int8_t qs[16];
};

struct Q8_0_block_ngu {
    half d;
    int8_t qs[QK4_0];
};

__device__ inline float silu_ngu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ inline int dot_q4_0_q8_0_byte_ngu(uint8_t q, int8_t x_lo, int8_t x_hi) {
    const int lo = static_cast<int>(q & 0x0F) - 8;
    const int hi = static_cast<int>(q >> 4) - 8;
    return lo * static_cast<int>(x_lo) + hi * static_cast<int>(x_hi);
}

// ── Fused RMSNorm + Gate+Up+SiLU Kernel ──────────────────────────────────────
//
// Replaces 2 separate kernel launches per layer:
//   1. rms_norm (ffn_norm)
//   2. gemv_gate_up_swiglu (Gate+Up projection + SiLU activation)
//
// The kernel:
//   1. Loads raw hidden state, computes RMS norm using shared memory reduction
//   2. Quantizes the normalized values to Q8_0 in shared memory
//   3. Runs Gate+Up GEMV with Q4_0×Q8_0 dot products + SiLU
//
// Shared memory is reused between phases:
//   Phase 1 (norm): [n_rows floats] + [32 floats reduction]
//   Phase 2 (gemv): [n_blocks * Q8_0_block] (overlaps phase 1 region)
//
// Requirements:
//   - n_rows % 32 == 0
//   - n_ff % 4 == 0
//   - max(n_rows*4 + 128, n_blocks*34) <= 32768

template<int N_WAVES>
__global__ void gemv_norm_gate_up_swiglu_q4_0_f32_q8_inline_v2_kernel(
    // RMSNorm parameters
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    // Gate+Up weights
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    // Output
    float* __restrict__ out_swiglu,
    // Dimensions
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    // ── Phase 1: RMS Norm in shared memory ─────────────────────────────

    // Use shared memory as f32 array for norm computation
    extern __shared__ char s_raw[];
    float* s_f32 = reinterpret_cast<float*>(s_raw);
    // Reduction scratch sits after the f32 input data
    float* s_reduction = &s_f32[n_rows];

    // Load raw hidden and accumulate sum_sq
    float local_sum_sq = 0.0f;
    for (int i = tid; i < n_rows; i += blockDim.x) {
        float val = raw_hidden[i];
        s_f32[i] = val;
        local_sum_sq += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down(local_sum_sq, offset, 32);
    }

    if (lane_id == 0) {
        s_reduction[wave_id] = local_sum_sq;
    }
    __syncthreads();

    // Cross-warp reduction
    float final_sum_sq = (tid < (blockDim.x / 32)) ? s_reduction[tid] : 0.0f;
    if (wave_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum_sq += __shfl_down(final_sum_sq, offset, 32);
        }
        if (tid == 0) {
            s_reduction[0] = final_sum_sq;
        }
    }
    __syncthreads();

    float inv_rms = rsqrtf(s_reduction[0] / (float)n_rows + eps);

    // Apply norm in-place: s_f32[i] = raw_hidden[i] * inv_rms * norm_weight[i]
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_f32[i] = s_f32[i] * inv_rms * norm_weight[i];
    }
    __syncthreads();

    // ── Phase 2: Quantize normalized input to Q8_0 ────────────────────

    // Reinterpret shared memory as Q8_0 blocks (overlaps the f32 region,
    // but we read s_f32 and write to s_q8 — the Q8_0 data is smaller so
    // we copy through registers to avoid aliasing issues)
    Q8_0_block_ngu* s_input_q8 = reinterpret_cast<Q8_0_block_ngu*>(s_raw);

    // Each wave quantizes a subset of blocks. We read from s_f32 into registers
    // first, then write the Q8_0 blocks. Since Q8_0 blocks are 34 bytes and
    // f32 blocks are 128 bytes (32 * 4), and we process block by block,
    // we need to be careful about aliasing. Read the f32 values into registers
    // before writing Q8_0.
    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        // Read normalized value from shared memory f32 region
        const float val = s_f32[row_offset + lane_id];

        // Compute block-wise absmax for Q8_0 quantization
        float max_val = fabsf(val);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, 32));
        }

        const float scale = fmaxf(__shfl(max_val, 0, 32) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);
        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
        }

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        s_input_q8[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
    }
    __syncthreads();

    // ── Phase 3: Gate+Up GEMV with Q4_0×Q8_0 + SiLU ──────────────────

    if (ff_idx_base >= n_ff) return;

    const Q4_0_block_ngu* w_gate_cols[4];
    const Q4_0_block_ngu* w_up_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int ff_idx = ff_idx_base + c;
        w_gate_cols[c] = reinterpret_cast<const Q4_0_block_ngu*>(
            static_cast<const uint8_t*>(w_gate) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
        w_up_cols[c] = reinterpret_cast<const Q4_0_block_ngu*>(
            static_cast<const uint8_t*>(w_up) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
    }

    float sum_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block_ngu* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);
        const int8_t* x_qs = x_block->qs;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const Q4_0_block_ngu* gate_block = &w_gate_cols[c][block_idx];
            const Q4_0_block_ngu* up_block = &w_up_cols[c][block_idx];

            int gate_block_sum = 0;
            int up_block_sum = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                gate_block_sum += dot_q4_0_q8_0_byte_ngu(
                    static_cast<uint8_t>(gate_block->qs[i]), x_qs[i], x_qs[i + 16]);
                up_block_sum += dot_q4_0_q8_0_byte_ngu(
                    static_cast<uint8_t>(up_block->qs[i]), x_qs[i], x_qs[i + 16]);
            }

            sum_gate[c] += __half2float(gate_block->d) * x_scale * static_cast<float>(gate_block_sum);
            sum_up[c] += __half2float(up_block->d) * x_scale * static_cast<float>(up_block_sum);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_gate[c] += __shfl_down(sum_gate[c], offset, 32);
            sum_up[c] += __shfl_down(sum_up[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            out_swiglu[ff_idx_base + c] = silu_ngu(sum_gate[c]) * sum_up[c];
        }
    }
}

// ── C Dispatch Function ─────────────────────────────────────────────────────────────

extern "C" hipError_t gemv_norm_gate_up_swiglu_q4_0_f32_q8_inline_launch(
    const float* raw_hidden,
    const float* norm_weight,
    float eps,
    const void* w_gate,
    const void* w_up,
    float* out_swiglu,
    int n_rows,
    int n_ff,
    hipStream_t stream
) {
    if (raw_hidden == nullptr || norm_weight == nullptr) return hipErrorInvalidValue;
    if (w_gate == nullptr || w_up == nullptr || out_swiglu == nullptr) return hipErrorInvalidValue;
    if (n_rows <= 0 || n_ff <= 0) return hipErrorInvalidValue;
    if (n_rows % QK4_0 != 0) return hipErrorInvalidValue;
    if ((n_ff % 4) != 0) return hipErrorInvalidValue;

    const int n_blocks_total = n_rows / QK4_0;
    // Shared memory: max of (f32 norm phase, Q8_0 gemv phase)
    const size_t norm_shared = (n_rows + 32) * sizeof(float);
    const size_t gemv_shared = n_blocks_total * Q8_0_BLOCK_SIZE;
    const size_t shared_mem = (norm_shared > gemv_shared) ? norm_shared : gemv_shared;

    if (shared_mem > 32768) {
        return hipErrorInvalidValue;
    }

    // Use same heuristic as the non-norm variant
    if (n_rows <= 4096) {
        if (n_ff >= 4096) {
            const int N_WAVES = 8;
            const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
            gemv_norm_gate_up_swiglu_q4_0_f32_q8_inline_v2_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
                raw_hidden, norm_weight, eps,
                w_gate, w_up, out_swiglu, n_rows, n_ff
            );
        } else {
            const int N_WAVES = 4;
            const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
            gemv_norm_gate_up_swiglu_q4_0_f32_q8_inline_v2_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
                raw_hidden, norm_weight, eps,
                w_gate, w_up, out_swiglu, n_rows, n_ff
            );
        }
    } else {
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}
