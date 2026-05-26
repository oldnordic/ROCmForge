#include "common.hip"

// Q4_0 constants (must match Rust types.rs)
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

// ── Type Definitions ───────────────────────────────────────────────────────────────

/// Q4_0 block layout (matches Rust Q4_0Block)
struct Q4_0_block {
    half d;          // scale (2 bytes)
    int8_t qs[16];   // quantized values packed as 2x4-bit per byte (16 bytes)
};

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Load Q4_0 block and compute dot product with f32 vector
__device__ inline float vec_dot_q4_0(const Q4_0_block* block, const float* vec, int offset) {
    // Load scale
    float d = __half2float(block->d);

    // Sequential layout: qs[i] low nibble → element i, high → element i+16
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = static_cast<uint8_t>(block->qs[i]);
        const int32_t q0 = q & 0x0F;
        const int32_t q1 = (q >> 4) & 0x0F;
        sum += (static_cast<float>(q0) - 8.0f) * vec[offset + i];
        sum += (static_cast<float>(q1) - 8.0f) * vec[offset + i + 16];
    }

    return d * sum;
}

/// Optimized Q4_0 dot product with packed loads and 4-way unrolling.
/// Processes 4 consecutive Q4_0 blocks and accumulates into separate accumulators.
/// This matches the hipfire style for better instruction-level parallelism.
__device__ inline void vec_dot_q4_0_packed_x4(
    const Q4_0_block* blocks[4],
    const float* vec,
    int offset,
    float& acc0,
    float& acc1,
    float& acc2,
    float& acc3
) {
    // Load scales for all 4 blocks
    float d0 = __half2float(blocks[0]->d);
    float d1 = __half2float(blocks[1]->d);
    float d2 = __half2float(blocks[2]->d);
    float d3 = __half2float(blocks[3]->d);

    // Reset accumulators
    acc0 = 0.0f;
    acc1 = 0.0f;
    acc2 = 0.0f;
    acc3 = 0.0f;

    // Process 4 blocks with packed uint32 loads
    // Each uint32 contains 4 bytes (8 nibbles = 8 Q4_0 values)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Packed load: 4 consecutive bytes as uint32
        const uint32_t pk0 = *reinterpret_cast<const uint32_t*>(&blocks[0]->qs[i * 4]);
        const uint32_t pk1 = *reinterpret_cast<const uint32_t*>(&blocks[1]->qs[i * 4]);
        const uint32_t pk2 = *reinterpret_cast<const uint32_t*>(&blocks[2]->qs[i * 4]);
        const uint32_t pk3 = *reinterpret_cast<const uint32_t*>(&blocks[3]->qs[i * 4]);

        // Extract 8 nibbles from each packed uint32 and accumulate
        // Each nibble is subtracted by 8 (Q4_0 zero point)
        #define PROCESS_NIBBLE(pk, shift, idx, acc, d) \
            (acc) += ((float)(((pk) >> (shift)) & 0xF) - 8.0f) * vec[offset + (idx)]

        // Process 8 nibbles from pk0 (indices i*4*8 to i*4*8+7)
        int base = i * 32;
        PROCESS_NIBBLE(pk0, 0,  base + 0,  acc0, d0);
        PROCESS_NIBBLE(pk0, 4,  base + 1,  acc0, d0);
        PROCESS_NIBBLE(pk0, 8,  base + 2,  acc0, d0);
        PROCESS_NIBBLE(pk0, 12, base + 3,  acc0, d0);
        PROCESS_NIBBLE(pk0, 16, base + 4,  acc0, d0);
        PROCESS_NIBBLE(pk0, 20, base + 5,  acc0, d0);
        PROCESS_NIBBLE(pk0, 24, base + 6,  acc0, d0);
        PROCESS_NIBBLE(pk0, 28, base + 7,  acc0, d0);

        // Process 8 nibbles from pk1
        PROCESS_NIBBLE(pk1, 0,  base + 0,  acc1, d1);
        PROCESS_NIBBLE(pk1, 4,  base + 1,  acc1, d1);
        PROCESS_NIBBLE(pk1, 8,  base + 2,  acc1, d1);
        PROCESS_NIBBLE(pk1, 12, base + 3,  acc1, d1);
        PROCESS_NIBBLE(pk1, 16, base + 4,  acc1, d1);
        PROCESS_NIBBLE(pk1, 20, base + 5,  acc1, d1);
        PROCESS_NIBBLE(pk1, 24, base + 6,  acc1, d1);
        PROCESS_NIBBLE(pk1, 28, base + 7,  acc1, d1);

        // Process 8 nibbles from pk2
        PROCESS_NIBBLE(pk2, 0,  base + 0,  acc2, d2);
        PROCESS_NIBBLE(pk2, 4,  base + 1,  acc2, d2);
        PROCESS_NIBBLE(pk2, 8,  base + 2,  acc2, d2);
        PROCESS_NIBBLE(pk2, 12, base + 3,  acc2, d2);
        PROCESS_NIBBLE(pk2, 16, base + 4,  acc2, d2);
        PROCESS_NIBBLE(pk2, 20, base + 5,  acc2, d2);
        PROCESS_NIBBLE(pk2, 24, base + 6,  acc2, d2);
        PROCESS_NIBBLE(pk2, 28, base + 7,  acc2, d2);

        // Process 8 nibbles from pk3
        PROCESS_NIBBLE(pk3, 0,  base + 0,  acc3, d3);
        PROCESS_NIBBLE(pk3, 4,  base + 1,  acc3, d3);
        PROCESS_NIBBLE(pk3, 8,  base + 2,  acc3, d3);
        PROCESS_NIBBLE(pk3, 12, base + 3,  acc3, d3);
        PROCESS_NIBBLE(pk3, 16, base + 4,  acc3, d3);
        PROCESS_NIBBLE(pk3, 20, base + 5,  acc3, d3);
        PROCESS_NIBBLE(pk3, 24, base + 6,  acc3, d3);
        PROCESS_NIBBLE(pk3, 28, base + 7,  acc3, d3);

        #undef PROCESS_NIBBLE
    }

    // Apply scales
    acc0 *= d0;
    acc1 *= d1;
    acc2 *= d2;
    acc3 *= d3;
}

// ── Batched GEMM for Prefill ───────────────────────────────────────────────────────

/// Batched Q4_0 × f32 GEMM for prefill processing.
///
/// Computes: output[seq_len][ncols_dst] = gemm(input[seq_len][n_rows], weights[ncols_dst][n_rows_q4_0])
///
/// This kernel processes multiple input sequences (tokens) in parallel, which is the
/// primary workload during prompt prefill. Each token in the prompt is processed
/// independently through the same weight matrix.
///
/// # Arguments
/// * `weights_q4_0` - Weight matrix in Q4_0 format [ncols_dst][n_rows/32][18]
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
__global__ void batched_gemm_q4_0_f32_prefill_kernel(
    const void* __restrict__ weights_q4_0,
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
    const Q4_0_block* w_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(weights_q4_0) + col * (n_rows / QK4_0) * Q4_0_BLOCK_SIZE
    );

    // Calculate starting pointer for this sequence position in input
    const float* input_seq = input + seq_pos * n_rows;

    // Number of blocks per column
    const int n_blocks = (n_rows + QK4_0 - 1) / QK4_0;

    // Accumulator for this thread's partial sum
    float sum = 0.0f;

    // Stride across blocks with threads
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK4_0;
        sum += vec_dot_q4_0(&w_col[block_idx], input_seq, offset);
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

extern "C" hipError_t batched_gemm_q4_0_f32_prefill(
    const void* weights_q4_0,
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
        batched_gemm_q4_0_f32_prefill_kernel,
        grid_dim,
        block_dim,
        0,
        stream,
        weights_q4_0,
        input,
        output,
        n_rows,
        ncols_dst,
        seq_len
    );

    return hipGetLastError();
}

// ── Batched Fused Gate-Up + SwiGLU for Prefill ───────────────────────────────────────

/// Batched fused Gate/Up GEMV + SwiGLU for prefill processing.
///
/// Computes for all sequence positions in parallel:
///   output[seq_len][ff_size] = SiLU(gate[seq_len][ff_size]) * up[seq_len][ff_size]
///   where gate = input × W_gate and up = input × W_up
///
/// # Arguments
/// * `w_gate_q4_0` - Gate weights in Q4_0 format [ff_size][n_rows/32][18]
/// * `w_up_q4_0` - Up weights in Q4_0 format [ff_size][n_rows/32][18]
/// * `input` - Input activations [seq_len][n_rows] (row-major)
/// * `output` - Output buffer [seq_len][ff_size] (row-major)
/// * `n_rows` - Input dimension (hidden size)
/// * `ff_size` - Feed-forward dimension (gate/up output size)
/// * `seq_len` - Number of tokens in batch
///
/// # Thread Block Organization
/// - BlockIdx.x: Output column (0 to ff_size-1)
/// - BlockIdx.y: Sequence position (0 to seq_len-1)
/// - ThreadIdx.x: Thread within block (processes weight blocks in parallel)
///
/// # Optimization
/// Uses 4-way unrolled accumulators with packed uint32 loads for better ILP.
__launch_bounds__(256, 2)
__global__ void batched_fused_gate_up_q4_0_f32_prefill_kernel(
    const void* __restrict__ w_gate_q4_0,
    const void* __restrict__ w_up_q4_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ff_size,
    int seq_len
) {
    const int col = blockIdx.x;
    const int seq_pos = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ff_size || seq_pos >= seq_len) return;

    // Calculate starting pointers for this column in gate weights
    const Q4_0_block* w_gate_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_gate_q4_0) + col * (n_rows / QK4_0) * Q4_0_BLOCK_SIZE
    );

    // Calculate starting pointers for this column in up weights
    const Q4_0_block* w_up_col = reinterpret_cast<const Q4_0_block*>(
        static_cast<const uint8_t*>(w_up_q4_0) + col * (n_rows / QK4_0) * Q4_0_BLOCK_SIZE
    );

    // Calculate starting pointer for this sequence position in input
    const float* input_seq = input + seq_pos * n_rows;

    // Number of blocks per column
    const int n_blocks = (n_rows + QK4_0 - 1) / QK4_0;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    // Stride across blocks with threads (safe, correct, parallelized)
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const int offset = block_idx * QK4_0;
        gate_sum += vec_dot_q4_0(&w_gate_col[block_idx], input_seq, offset);
        up_sum += vec_dot_q4_0(&w_up_col[block_idx], input_seq, offset);
    }

    // Block-wide reduction for both gate and up
    __shared__ float partial_sums[512];
    partial_sums[tid] = gate_sum;
    partial_sums[tid + 256] = up_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
            partial_sums[tid + 256] += partial_sums[tid + stride + 256];
        }
        __syncthreads();
    }

    // Thread 0 writes the result with SwiGLU activation
    if (tid == 0) {
        const float gate = partial_sums[0];
        const float up = partial_sums[256];
        // SiLU(x) = x / (1.0f + exp(-x))
        const float silu_gate = gate / (1.0f + expf(-gate));
        output[seq_pos * ff_size + col] = silu_gate * up;
    }
}

// Host interface for batched fused gate-up
extern "C" hipError_t batched_fused_gate_up_q4_0_f32_prefill(
    const void* w_gate_q4_0,
    const void* w_up_q4_0,
    const float* input,
    float* output,
    int n_rows,
    int ff_size,
    int seq_len,
    hipStream_t stream
) {
    if (n_rows == 0 || ff_size == 0 || seq_len == 0) {
        return hipErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const dim3 grid_dim(ff_size, seq_len);
    const dim3 block_dim(threads_per_block);

    hipLaunchKernelGGL(
        batched_fused_gate_up_q4_0_f32_prefill_kernel,
        grid_dim,
        block_dim,
        0,
        stream,
        w_gate_q4_0,
        w_up_q4_0,
        input,
        output,
        n_rows,
        ff_size,
        seq_len
    );

    return hipGetLastError();
}
