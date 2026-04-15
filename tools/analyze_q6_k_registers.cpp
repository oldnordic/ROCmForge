#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <string.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

__device__ inline float vec_dot_q6_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Extract scale d (fp16 at bytes 208-209)
    half d_half;
    memcpy(&d_half, &block_bytes[208], sizeof(half));
    const float d = __half2float(d_half);

    // Q4_K safety: check for denormal d
    if (fabsf(d) < 1e-7f) return 0.0f;

    // Pointer to scales (int8_t at bytes 192-207)
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block_bytes[192]);

    float sum = 0.0f;

    // OPTIMIZED: Declare all variables INSIDE loop to minimize live ranges
    // Process 8 elements: Thread T (0-31) processes T, T+32, T+64, T+96, T+128, T+160, T+192, T+224
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int i = tid * 8 + l;

        // All index calculations use bit manipulation (NO division/modulo)
        const int group = i >> 7;                          // i / 128
        const int l_base = i & 0x1F;                       // i % 32
        const int quadrant = (i >> 5) & 0x3;               // (i / 32) % 4
        const int scale_idx = (group << 3) | (((l_base >> 4) & 0x1) << 1) | quadrant;

        const float scale = d * (float)scales[scale_idx];

        // Memory offsets
        const int ql_offset = (group << 6) | l_base | (quadrant << 5);
        const uint8_t ql_byte = block_bytes[ql_offset];

        const int qh_offset = 128 + (group << 5) + l_base;
        const uint8_t qh_byte = block_bytes[qh_offset];

        // Extract Q6_K value (bit manipulation)
        const int is_low_half = quadrant & 0x2;
        const int shift = (is_low_half ^ 0x2) & 0x4;
        const int qh_shift = ((quadrant & 0x1) << 1) | ((is_low_half >> 1) & 0x4);

        const int8_t q = (int8_t)(((ql_byte >> shift) & 0x0F) | (((qh_byte >> qh_shift) & 0x03) << 4)) - 32;

        sum += vec[offset + i] * (scale * (float)q);
    }

    return sum;
}

__global__ void gemv_q6_k_test_kernel(const void* __restrict__ weights,
                                       const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int n_rows,
                                       int ncols_dst) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights) +
                              col * n_blocks * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q6_k(col_base + block_idx * Q6_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

int main() {
    printf("Q6_K Kernel Resource Usage Analysis\n");
    printf("====================================\n\n");

    // Get device properties
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Registers Per Block: %d\n", prop.regsPerBlock);
    printf("Warps Per Multiprocessor: %d\n\n", prop.warpSize);

    // Get kernel function attributes
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gemv_q6_k_test_kernel);

    printf("Q6_K Kernel Attributes:\n");
    printf("Binary Version: %d\n", attr.binaryVersion);
    printf("ConstSizeBytes: %zu\n", attr.constSizeBytes);
    printf("LocalSizeBytes: %zu\n", attr.localSizeBytes);
    printf("MaxThreadsPerBlock: %d\n", attr.maxThreadsPerBlock);
    printf("NumRegs: %d\n", attr.numRegs);
    printf("SharedSizeBytes: %zu\n", attr.sharedSizeBytes);

    // Calculate occupancy
    int threads_per_block = 32;

    int min_grid_size = 0;
    int max_grid_size = 0;

    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_grid_size,
        gemv_q6_k_test_kernel,
        threads_per_block,
        0
    );

    printf("\nOccupancy Analysis:\n");
    printf("Threads Per Block: %d\n", threads_per_block);
    printf("Min Grid Size: %d\n", min_grid_size);
    printf("Max Grid Size: %d\n", max_grid_size);
    printf("Active Warps Per Block: %d\n", threads_per_block / 32);

    // Calculate theoretical VGPR usage
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int warps_per_block = threads_per_block / 32;
    int max_warps_per_sm = max_threads_per_sm / 32;

    printf("\nTheoretical Analysis:\n");
    printf("Max Warps Per SM: %d\n", max_warps_per_sm);
    printf("Active Warps Per Block: %d\n", warps_per_block);
    printf("Max Concurrent Blocks: %d\n", max_grid_size);

    // Calculate VGPRs used
    int threads_per_block_actual = min_grid_size * threads_per_block;
    if (max_grid_size > 0) {
        int vgprs_used = prop.regsPerBlock / max_grid_size;
        printf("\nEstimated VGPR Usage:\n");
        printf("VGPRs Per Block (limit): %d\n", prop.regsPerBlock);
        printf("Max Concurrent Blocks: %d\n", max_grid_size);
        printf("Estimated VGPRs Used: ~%d\n", vgprs_used);
    }

    return 0;
}
