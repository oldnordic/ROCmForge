#include "common.hip"
#include <hip/hip_fp16.h>
#include "q6_k_tile.hip"

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

// ── Template Kernel with ncols_dst (compile-time optimization) ─────────

template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemm_q6_k_f32_kernel_templated(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int seq_len
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Bounds check with compile-time constant (compiler can optimize)
    if (col >= ncols_dst || batch_idx >= seq_len) return;

    const uint8_t* weights = static_cast<const uint8_t*>(weights_q6_k);
    const int n_blocks = n_rows / QK_K;

    float sum = 0.0f;
    const uint8_t* col_base = weights + col * n_blocks * Q6_K_BLOCK_SIZE;

    // CRITICAL FIX: Do NOT offset input by batch_idx
    // The input pointer already points to the correct token's data
    // const float* input_batch = input + batch_idx * n_rows;  // WRONG
    const float* input_batch = input;  // CORRECT

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;
        const int offset = block_idx * QK_K;

        // Extract scale d (fp16 at bytes 208-209)
        half d_half;
        memcpy(&d_half, &block[208], sizeof(half));
        const float d = __half2float(d_half);

        // Q4_K safety: check for denormal d
        if (fabsf(d) < 1e-7f) continue;

        // Pointer to scales (int8_t at bytes 192-207)
        const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);

        // Each thread processes ONE l value (0-31) - matches llama.cpp pattern exactly
        const int l = tid;

        // Process group 0 (elements 0-127)
        {
            const uint8_t* ql = &block[0];
            const uint8_t* qh = &block[128];
            const int8_t* sc = &scales[0];

            const int is = l / 16;

            // Extract 4 values exactly like llama.cpp
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            // Use correct output positions (matching llama.cpp)
            sum += input_batch[offset + l + 0] * (d * (float)sc[is + 0] * q1);
            sum += input_batch[offset + l + 32] * (d * (float)sc[is + 2] * q2);
            sum += input_batch[offset + l + 64] * (d * (float)sc[is + 4] * q3);
            sum += input_batch[offset + l + 96] * (d * (float)sc[is + 6] * q4);
        }

        // Process group 1 (elements 128-255) - same pattern with pointer arithmetic
        {
            const uint8_t* ql = &block[64];
            const uint8_t* qh = &block[160];
            const int8_t* sc = &scales[8];

            const int is = l / 16;

            // Extract 4 values exactly like llama.cpp (second iteration)
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            // Use correct output positions (matching llama.cpp)
            sum += input_batch[offset + l + 128] * (d * (float)sc[is + 0] * q1);
            sum += input_batch[offset + l + 160] * (d * (float)sc[is + 2] * q2);
            sum += input_batch[offset + l + 192] * (d * (float)sc[is + 4] * q3);
            sum += input_batch[offset + l + 224] * (d * (float)sc[is + 6] * q4);
        }
    }

    // Warp reduction with unroll
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Generic Kernel (runtime ncols_dst) ─────────────────────────────────────

__launch_bounds__(32, 1)
__global__ void gemm_q6_k_f32_kernel_generic(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int seq_len
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;  // Y dimension = batch/token position
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= seq_len) return;

    const uint8_t* weights = static_cast<const uint8_t*>(weights_q6_k);
    const int n_blocks = n_rows / QK_K;

    float sum = 0.0f;
    const uint8_t* col_base = weights + col * n_blocks * Q6_K_BLOCK_SIZE;

    // CRITICAL FIX: Do NOT offset input by batch_idx
    // The input pointer already points to the correct token's data
    // const float* input_batch = input + batch_idx * n_rows;  // WRONG
    const float* input_batch = input;  // CORRECT

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;
        const int offset = block_idx * QK_K;

        // Extract scale d (fp16 at bytes 208-209)
        half d_half;
        memcpy(&d_half, &block[208], sizeof(half));
        const float d = __half2float(d_half);

        // Q4_K safety: check for denormal d
        if (fabsf(d) < 1e-7f) continue;

        // Pointer to scales (int8_t at bytes 192-207)
        const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);

        // Each thread processes ONE l value (0-31) - matches llama.cpp pattern exactly
        const int l = tid;

        // Process group 0 (elements 0-127)
        {
            const uint8_t* ql = &block[0];
            const uint8_t* qh = &block[128];
            const int8_t* sc = &scales[0];

            const int is = l / 16;

            // Extract 4 values exactly like llama.cpp
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            // Use correct output positions (matching llama.cpp)
            sum += input_batch[offset + l + 0] * (d * (float)sc[is + 0] * q1);
            sum += input_batch[offset + l + 32] * (d * (float)sc[is + 2] * q2);
            sum += input_batch[offset + l + 64] * (d * (float)sc[is + 4] * q3);
            sum += input_batch[offset + l + 96] * (d * (float)sc[is + 6] * q4);
        }

        // Process group 1 (elements 128-255) - same pattern with pointer arithmetic
        {
            const uint8_t* ql = &block[64];
            const uint8_t* qh = &block[160];
            const int8_t* sc = &scales[8];

            const int is = l / 16;

            // Extract 4 values exactly like llama.cpp (second iteration)
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            // Use correct output positions (matching llama.cpp)
            sum += input_batch[offset + l + 128] * (d * (float)sc[is + 0] * q1);
            sum += input_batch[offset + l + 160] * (d * (float)sc[is + 2] * q2);
            sum += input_batch[offset + l + 192] * (d * (float)sc[is + 4] * q3);
            sum += input_batch[offset + l + 224] * (d * (float)sc[is + 6] * q4);
        }
    }

    // Warp reduction (32 threads = 1 wavefront on RDNA)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Dispatch: Tile vs Direct Access ───────────────────────────────────────────────

/// Decide whether to use tile-based kernel based on problem size
/// Tile-based processing is beneficial for larger problems
/// Small problems may not benefit from the shared memory overhead
inline bool should_use_tile_processing(int n_rows, int ncols_dst) {
    // Tile kernel has performance issues due to synchronization overhead
    // Direct access is faster and proven safe
    // Disable tile processing for now
    return false;
}

extern "C" hipError_t gemm_q6_k_f32_launch(
    const void* weights_q6_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int seq_len,
    hipStream_t stream
) {
    // Validate all parameters
    if (n_rows <= 0 || ncols_dst <= 0 || seq_len <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    CHECK_NULL(weights_q6_k);
    CHECK_NULL(input);
    CHECK_NULL(output);

    // Choose kernel based on problem size
    bool use_tiles = should_use_tile_processing(n_rows, ncols_dst);

    const int block_size = 32;
    const dim3 gridDim(ncols_dst, seq_len);
    const dim3 blockDim(block_size, 1);

    if (use_tiles) {
        // Use tile-based kernel with shared memory optimization
        gemm_q6_k_f32_kernel_tiled<1><<<gridDim, blockDim, 0, stream>>>(
            weights_q6_k, input, output, n_rows, seq_len);

        // For now, only support ncols_dst=1 with tile kernel
        // TODO: Implement full tile kernel with template specialization
    } else {
        // Use direct access kernel (current implementation)
        gemm_q6_k_f32_kernel_generic<<<gridDim, blockDim, 0, stream>>>(
            weights_q6_k, input, output, n_rows, ncols_dst, seq_len);
    }

    CHECK_HIP(hipGetLastError());
    return hipSuccess;
}

// ── Dispatch Wrapper for Template Specialization ─────────────────────────────

extern "C" hipError_t gemm_q6_k_f32_launch_dispatch(
    const void* weights_q6_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int seq_len,
    hipStream_t stream
) {
    // Validate parameters
    if (n_rows <= 0 || ncols_dst <= 0 || seq_len <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    CHECK_NULL(weights_q6_k);
    CHECK_NULL(input);
    CHECK_NULL(output);

    dim3 grid(ncols_dst, seq_len);
    dim3 block(32, 1);

    // Dispatch templated kernel for common ncols_dst values
    switch (ncols_dst) {
        case 1:
            gemm_q6_k_f32_kernel_templated<1><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 2:
            gemm_q6_k_f32_kernel_templated<2><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 3:
            gemm_q6_k_f32_kernel_templated<3><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 4:
            gemm_q6_k_f32_kernel_templated<4><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 5:
            gemm_q6_k_f32_kernel_templated<5><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 6:
            gemm_q6_k_f32_kernel_templated<6><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 7:
            gemm_q6_k_f32_kernel_templated<7><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 8:
            gemm_q6_k_f32_kernel_templated<8><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 4096:
            gemm_q6_k_f32_kernel_templated<4096><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        case 8192:
            gemm_q6_k_f32_kernel_templated<8192><<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, seq_len);
            break;
        default:
            // Fall back to generic kernel for uncommon ncols_dst
            gemm_q6_k_f32_kernel_generic<<<grid, block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, ncols_dst, seq_len);
            break;
    }

    CHECK_HIP(hipGetLastError());
    return hipSuccess;
}

