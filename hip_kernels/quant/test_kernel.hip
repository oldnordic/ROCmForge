#include "common.hip"

/// Simple vector addition kernel for testing CMake build
/// Each thread computes one element: c[i] = a[i] + b[i]
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = get_global_id();
    CHECK_BOUNDS(idx, n);

    c[idx] = a[idx] + b[idx];
}

/// C wrapper for Rust FFI
extern "C" hipError_t test_vector_add(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int n,
    hipStream_t stream
) {
    // Launch parameters
    int block_size = BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;

    // Launch kernel
    vector_add<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);

    // Check for launch errors
    CHECK_LAST();

    return hipSuccess;
}
