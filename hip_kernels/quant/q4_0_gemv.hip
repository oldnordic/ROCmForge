#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block {
    half d;
    int8_t qs[16];
};

// ── Wavefront-Parallel Multi-Column GEMV ──────────────────────────────────────────

// Optimized for small hidden sizes (like 896, 1024, 2048).
// Each block has 256 threads = 8 wavefronts (on AMD).
// Each wavefront handles ONE column.
// One block processes 8 columns.

template<int N_WAVES>
__global__ void gemv_q4_0_f32_wave_parallel_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col = blockIdx.x * N_WAVES + wave_id;
    const int n_blocks_total = n_rows / QK4_0;

    if (col >= ncols_dst) return;

    // 1. Each wavefront collaboratively loads the input vector into LDS
    // Since all waves in a block use the same input, we only load it once.
    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // 2. Each wavefront processes its own column
    const Q4_0_block* w_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
    );

    float sum = 0.0f;
    // Each lane handles a subset of the blocks in the column
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

    // 3. Reduction within wavefront
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    // 4. Lane 0 of each wavefront writes the result
    if (lane_id == 0) {
        output[col] = sum;
    }
}

extern "C" hipError_t gemv_q4_0_f32_launch(
    const void* weights_q4_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;

    const int N_WAVES = 8;
    const int threads_per_block = N_WAVES * 32;
    int n_blocks_x = (ncols_dst + N_WAVES - 1) / N_WAVES;

    size_t shared_mem = n_rows * sizeof(float);

    gemv_q4_0_f32_wave_parallel_kernel<N_WAVES><<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
        weights_q4_0, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
