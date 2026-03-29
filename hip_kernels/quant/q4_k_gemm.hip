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
__device__ inline float vec_dot_q4_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;

    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half;
    memcpy(&d_half, block_bytes, sizeof(half));
    const float d = __half2float(d_half);

    half dmin_half;
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float dmin = __half2float(dmin_half);

    const uint8_t* qs = block_bytes + 16;

    if (fabsf(d) < 1e-7f) {
        return 0.0f;
    }

    float sum = 0.0f;

    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;
        const int byte_idx = i / 2;

        const uint8_t packed = qs[byte_idx];
        uint8_t q4;

        if (i % 2 == 0) {
            q4 = packed & 0x0F;
        } else {
            q4 = (packed >> 4) & 0x0F;
        }

        const float val = static_cast<float>(q4) / d + dmin;
        sum += val * vec[offset + i];
    }

    return sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemm_q4_k_f32_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q4_K_BLOCK_SIZE;
        sum += vec_dot_q4_k(block_ptr, input_batch, block_idx * QK_K);
    }

    // Warp reduction (RDNA: 32 threads = 1 wavefront)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Generic Kernel (non-template) ───────────────────────────────────────────────────

__global__ void gemm_q4_k_f32_generic_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q4_K_BLOCK_SIZE;
        sum += vec_dot_q4_k(block_ptr, input_batch, block_idx * QK_K);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t gemm_q4_k_f32_launch(
    const void* weights_q4_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int batch_size,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK_K != 0) {
        return hipErrorInvalidValue;
    }

    dim3 grid(ncols_dst, batch_size);
    dim3 block(32);

    switch (ncols_dst) {
        case 1:
            gemm_q4_k_f32_kernel<1><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 2:
            gemm_q4_k_f32_kernel<2><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 3:
            gemm_q4_k_f32_kernel<3><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 4:
            gemm_q4_k_f32_kernel<4><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 5:
            gemm_q4_k_f32_kernel<5><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 6:
            gemm_q4_k_f32_kernel<6><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 7:
            gemm_q4_k_f32_kernel<7><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 8:
            gemm_q4_k_f32_kernel<8><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        default:
            gemm_q4_k_f32_generic_kernel<<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, ncols_dst, batch_size);
            break;
    }

    return hipGetLastError();
}
