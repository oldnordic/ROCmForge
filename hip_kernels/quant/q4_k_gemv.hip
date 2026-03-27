#include "common.hip"

// Q4_K constants (must match Rust types.rs)
#define QK_K 256
#define Q4_K_BLOCK_SIZE 144

// ── Type Definitions ───────────────────────────────────────────────────────────────

/// Q4_K block layout (matches Rust Q4KBlock)
struct Q4_K_block {
    half d;              // scale (2 bytes)
    half dmin;           // min scale (2 bytes)
    uint8_t scales[12];  // non-uniform scales (12 bytes)
    uint8_t qs[128];     // packed 4-bit values (128 bytes)
};

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Extract scale and min from Q4_K scales array (llama.cpp pattern)
/// Source: llama.cpp ggml/src/ggml-cuda/convert.cu:182-189
__device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

/// Vector dot product: Q4_K block × f32 vector segment
/// Thread layout matches llama.cpp dequantize_block_q4_k (32 threads)
__device__ inline float vec_dot_q4_k(const Q4_K_block* block, const float* vec, int offset) {
    // Thread layout (llama.cpp pattern)
    const int tid = threadIdx.x;
    const int il  = tid / 8;    // 0-3: which group of 8 threads
    const int ir  = tid % 8;    // 0-7: which thread within group
    const int is  = 2 * il;     // 0, 2, 4, 6: scale index
    const int n   = 4;          // values per iteration

    // Load global scales
    const float dall = __half2float(block->d);
    const float dmin = __half2float(block->dmin);

    // Get local scales (2 different scales per thread)
    uint8_t sc, m;
    get_scale_min_k4(is + 0, block->scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;

    get_scale_min_k4(is + 1, block->scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    // Load quantized values
    const uint8_t* q = block->qs + 32 * il + n * ir;

    // Dequantize and accumulate dot product
    float sum = 0.0f;
    #pragma unroll
    for (int l = 0; l < n; ++l) {
        const float v0 = d1 * (q[l] & 0xF) - m1;           // Lower 4 bits
        const float v1 = d2 * (q[l] >>  4) - m2;           // Upper 4 bits
        sum += v0 * vec[offset + 64*il + l];              // Accounts for il offset (llama.cpp pattern)
        sum += v1 * vec[offset + 64*il + l + 32];
    }

    return sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

/// Template kernel for Q4_K × f32 GEMV with ncols_dst as compile-time parameter
/// Follows llama.cpp pattern for optimal performance
template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemv_q4_k_f32_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    // Bounds check
    if (col >= ncols_dst) return;

    // Calculate column pointer (interleaved layout)
    const Q4_K_block* w_col = reinterpret_cast<const Q4_K_block*>(
        static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE
    );

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    // Stride across blocks with threads
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        sum += vec_dot_q4_k(&w_col[block_idx], input, block_idx * QK_K);
    }

    // Warp reduction (RDNA: 32 threads = 1 wavefront)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    // Thread 0 writes result
    if (tid == 0) {
        output[col] = sum;
    }
}

// ── Generic Kernel (non-template) ───────────────────────────────────────────────────

/// Generic kernel for arbitrary ncols_dst
/// Less optimized but handles any size
__global__ void gemv_q4_k_f32_generic_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= ncols_dst) return;

    const Q4_K_block* w_col = reinterpret_cast<const Q4_K_block*>(
        static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE
    );

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        sum += vec_dot_q4_k(&w_col[block_idx], input, block_idx * QK_K);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t gemv_q4_k_f32_launch(
    const void* weights_q4_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    // Input validation
    if (!weights_q4_k || !input || !output) {
        return hipErrorInvalidValue;
    }
    if (n_rows <= 0 || ncols_dst <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK_K != 0) {
        return hipErrorInvalidValue;
    }
    if (ncols_dst > 1024) {
        return hipErrorInvalidValue;
    }

    // Dispatch based on ncols_dst (llama.cpp pattern)
    switch (ncols_dst) {
        case 1:
            gemv_q4_k_f32_kernel<1><<<dim3(1), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        case 2:
            gemv_q4_k_f32_kernel<2><<<dim3(2), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        case 3:
            gemv_q4_k_f32_kernel<3><<<dim3(3), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        case 4:
            gemv_q4_k_f32_kernel<4><<<dim3(4), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        case 5:
            gemv_q4_k_f32_kernel<5><<<dim3(5), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        case 6:
            gemv_q4_k_f32_kernel<6><<<dim3(6), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        case 7:
            gemv_q4_k_f32_kernel<7><<<dim3(7), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        case 8:
            gemv_q4_k_f32_kernel<8><<<dim3(8), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows);
            break;
        default:
            gemv_q4_k_f32_generic_kernel<<<dim3(ncols_dst), dim3(32), 0, stream>>>(
                weights_q4_k, input, output, n_rows, ncols_dst);
            break;
    }

    return hipGetLastError();
}

