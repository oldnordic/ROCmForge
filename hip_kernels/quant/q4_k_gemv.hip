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

/// Vector dot product: Q4_K block × f32 vector segment (uniform quantization)
///
/// Uses uniform dequantization: val = q4 / d + dmin
/// Thread layout: 32 threads, each processes 8 elements
__device__ inline float vec_dot_q4_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;

    // Access block as raw bytes to avoid struct padding issues
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Load d (bytes 0-1)
    half d_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    const float d = __half2float(d_half);

    // Load dmin (bytes 2-3)
    half dmin_half;
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float dmin = __half2float(dmin_half);

    // qs array starts at byte 16 (after d, dmin, and scales[12])
    const uint8_t* qs = block_bytes + 16;

    // Handle edge case where d is near zero
    if (fabsf(d) < 1e-7f) {
        return 0.0f;
    }

    float sum = 0.0f;

    // Each thread processes 8 elements (256 / 32 = 8)
    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;  // Global element index in block
        const int byte_idx = i / 2;  // Which byte in qs array

        // Read packed 4-bit value
        const uint8_t packed = qs[byte_idx];
        uint8_t q4;

        if (i % 2 == 0) {
            // Lower 4 bits
            q4 = packed & 0x0F;
        } else {
            // Upper 4 bits
            q4 = (packed >> 4) & 0x0F;
        }

        // Dequantize: val = q4 / d + dmin (uniform quantization)
        const float val = static_cast<float>(q4) / d + dmin;

        // Accumulate dot product
        sum += val * vec[offset + i];
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

    // Calculate column base pointer (interleaved layout)
    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    // Process each block - all threads collaborate on each block
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q4_K_BLOCK_SIZE;
        sum += vec_dot_q4_k(block_ptr, input, block_idx * QK_K);
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

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    // Process each block - all threads collaborate on each block
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q4_K_BLOCK_SIZE;
        sum += vec_dot_q4_k(block_ptr, input, block_idx * QK_K);
    }

    // Warp reduction (RDNA: 32 threads = 1 wavefront)
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

