#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block {
    half d;
    int8_t qs[16];
};

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ── Wavefront-Parallel Fused QKV Kernel ───────────────────────────────────────────

template<int N_WAVES>
__global__ void gemv_qkv_q4_0_f32_wave_parallel_kernel(
    const void* __restrict__ w_q,
    const void* __restrict__ w_k,
    const void* __restrict__ w_v,
    const float* __restrict__ input,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int n_rows,
    int n_q,
    int n_kv
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col = blockIdx.x * N_WAVES + wave_id;
    const int n_blocks_total = n_rows / QK4_0;
    const int total_cols = n_q + 2 * n_kv;

    if (col >= total_cols) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const void* weights;
    int out_col;
    if (col < n_q) {
        weights = w_q; out_col = col;
    } else if (col < n_q + n_kv) {
        weights = w_k; out_col = col - n_q;
    } else {
        weights = w_v; out_col = col - n_q - n_kv;
    }

    const Q4_0_block* w_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(weights) + (out_col * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    float sum = 0.0f;
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q4_0_block* b = &w_col[block_idx];
        const float d = __half2float(b->d);
        const int row_offset = block_idx * QK4_0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            const uint8_t q = static_cast<uint8_t>(b->qs[l]);
            sum += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input[row_offset + l];
            sum += d * (static_cast<float>(q >> 4) - 8.0f) * s_input[row_offset + l + 16];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);

    if (lane_id == 0) {
        if (col < n_q) out_q[col] = sum;
        else if (col < n_q + n_kv) out_k[col - n_q] = sum;
        else out_v[col - n_q - n_kv] = sum;
    }
}

extern "C" hipError_t gemv_qkv_q4_0_f32_launch(
    const void* w_q, const void* w_k, const void* w_v,
    const float* input,
    float* out_q, float* out_k, float* out_v,
    int n_rows, int n_q, int n_kv,
    hipStream_t stream
) {
    const int N_WAVES = 8;
    const int threads_per_block = N_WAVES * 32;
    int total_cols = n_q + 2 * n_kv;
    int n_blocks_x = (total_cols + N_WAVES - 1) / N_WAVES;
    size_t shared_mem = n_rows * sizeof(float);

    gemv_qkv_q4_0_f32_wave_parallel_kernel<N_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
        w_q, w_k, w_v, input, out_q, out_k, out_v, n_rows, n_q, n_kv
    );
    return hipGetLastError();
}

// ── Wavefront-Parallel Fused Gate/Up + SwiGLU Kernel ─────────────────────────────────

// Processes pairs of (Gate, Up) columns and computes SwiGLU in-place.
// out_swiglu[i] = silu(gate[i]) * up[i]

template<int N_WAVES>
__global__ void gemv_gate_up_swiglu_q4_0_f32_kernel(
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    const float* __restrict__ input,
    float* __restrict__ out_swiglu,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Each wavefront handles ONE row index of the FFN (which has two weights: gate and up)
    const int ff_idx = blockIdx.x * N_WAVES + wave_id;
    const int n_blocks_total = n_rows / QK4_0;

    if (ff_idx >= n_ff) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // 1. Compute Gate dot product
    const Q4_0_block* w_gate_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    float sum_gate = 0.0f;
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q4_0_block* b = &w_gate_col[block_idx];
        const float d = __half2float(b->d);
        const int row_offset = block_idx * QK4_0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            const uint8_t q = static_cast<uint8_t>(b->qs[l]);
            sum_gate += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input[row_offset + l];
            sum_gate += d * (static_cast<float>(q >> 4) - 8.0f) * s_input[row_offset + l + 16];
        }
    }

    // 2. Compute Up dot product
    const Q4_0_block* w_up_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
    );
    float sum_up = 0.0f;
    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q4_0_block* b = &w_up_col[block_idx];
        const float d = __half2float(b->d);
        const int row_offset = block_idx * QK4_0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            const uint8_t q = static_cast<uint8_t>(b->qs[l]);
            sum_up += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input[row_offset + l];
            sum_up += d * (static_cast<float>(q >> 4) - 8.0f) * s_input[row_offset + l + 16];
        }
    }

    // 3. Reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_gate += __shfl_down(sum_gate, offset);
        sum_up += __shfl_down(sum_up, offset);
    }

    // 4. Lane 0 computes SwiGLU and writes
    if (lane_id == 0) {
        out_swiglu[ff_idx] = silu(sum_gate) * sum_up;
    }
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_0_f32_launch(
    const void* w_gate, const void* w_up,
    const float* input,
    float* out_swiglu,
    int n_rows, int n_ff,
    hipStream_t stream
) {
    const int N_WAVES = 8;
    const int threads_per_block = N_WAVES * 32;
    int n_blocks_x = (n_ff + N_WAVES - 1) / N_WAVES;
    size_t shared_mem = n_rows * sizeof(float);

    gemv_gate_up_swiglu_q4_0_f32_kernel<N_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
        w_gate, w_up, input, out_swiglu, n_rows, n_ff
    );
    return hipGetLastError();
}
