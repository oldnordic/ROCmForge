#include "common.hip"

// Q4_1 constants (must match Rust types.rs)
#define QK4_1 32
#define Q4_1_BLOCK_SIZE 20

// ── Type Definitions ───────────────────────────────────────────────────────────────

/// Q4_1 block layout (matches Rust Q4_1Block)
struct Q4_1_block {
    half d;          // scale (2 bytes)
    half m;          // min (2 bytes)
    int8_t qs[16];   // quantized values packed as 2x4-bit per byte (16 bytes)
};

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Load Q4_1 block and compute dot product with f32 vector
__device__ inline float vec_dot_q4_1(const Q4_1_block* block, const float* vec, int offset) {
    // Load scale and min
    float d = __half2float(block->d);
    float m = __half2float(block->m);

    // Sequential layout: qs[i] low nibble → element i, high → element i+16
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = static_cast<uint8_t>(block->qs[i]);
        const int32_t q0 = q & 0x0F;
        const int32_t q1 = (q >> 4) & 0x0F;
        sum += (static_cast<float>(q0) * d + m) * vec[offset + i];
        sum += (static_cast<float>(q1) * d + m) * vec[offset + i + 16];
    }

    return sum;
}

// ── Batched GEMM for Prefill ───────────────────────────────────────────────────────

/// Batched Q4_1 × f32 GEMM for prefill processing.
///
/// Computes: output[seq_len][ncols_dst] = gemm(input[seq_len][n_rows], weights[ncols_dst][n_rows_q4_1])
///
/// This kernel processes multiple input sequences (tokens) in parallel, which is the
/// primary workload during prompt prefill. Each token in the prompt is processed
/// independently through the same weight matrix.
///
/// # Arguments
/// * `weights_q4_1` - Weight matrix in Q4_1 format [ncols_dst][n_rows/32][20]
/// * `input` - Input activations [seq_len][n_rows] (row-major)
/// * `output` - Output buffer [seq_len][ncols_dst] (row-major)
/// * `n_rows` - Number of rows in weight matrix (input dimension)
/// * `ncols_dst` - Number of columns in weight matrix (output dimension)
/// * `seq_len` - Number of tokens in batch (sequence length)
///
/// # Thread Block Organization
/// - BlockIdx.x: Output column (0 to ncols_dst-1)
/// - BlockIdx.y: Sequence position (0 to seq_len-1)
/// - ThreadIdx.x: Thread within block (processes weight blocks in parallel)
__launch_bounds__(256, 2)
__global__ void batched_gemm_q4_1_f32_prefill_kernel(
    const void* __restrict__ weights_q4_1,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int seq_len
) {
    const int col = blockIdx.x;
    const int seq_pos = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || seq_pos >= seq_len) return;

    // Calculate starting pointers for this column in weights
    const Q4_1_block* w_col = reinterpret_cast<const Q4_1_block*>(
        static_cast<const uint8_t*>(weights_q4_1) + col * (n_rows / QK4_1) * Q4_1_BLOCK_SIZE
    );

    // Calculate starting pointer for this sequence position in input
    const float* input_seq = input + seq_pos * n_rows;

    // Number of blocks per column
    const int n_blocks = (n_rows + QK4_1 - 1) / QK4_1;

    // Accumulator for this thread's partial sum
    float sum = 0.0f;

    // Stride across blocks with threads
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK4_1;
        sum += vec_dot_q4_1(&w_col[block_idx], input_seq, offset);
    }

    // Block-wide reduction
    __shared__ float partial_sums[256];
    partial_sums[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the result
    if (tid == 0) {
        output[seq_pos * ncols_dst + col] = partial_sums[0];
    }
}

// ── Host Interface ─────────────────────────────────────────────────────────────────

extern "C" hipError_t batched_gemm_q4_1_f32_prefill(
    const void* weights_q4_1,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int seq_len,
    hipStream_t stream
) {
    if (n_rows == 0 || ncols_dst == 0 || seq_len == 0) {
        return hipErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const dim3 grid_dim(ncols_dst, seq_len);
    const dim3 block_dim(threads_per_block);

    hipLaunchKernelGGL(
        batched_gemm_q4_1_f32_prefill_kernel,
        grid_dim,
        block_dim,
        0,
        stream,
        weights_q4_1,
        input,
        output,
        n_rows,
        ncols_dst,
        seq_len
    );

    return hipGetLastError();
}
