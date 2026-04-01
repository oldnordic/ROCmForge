#ifndef ROCMFORGE_QUANT_COMMON_HIP
#define ROCMFORGE_QUANT_COMMON_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stddef.h>

// ── Safety-First Error Handling ─────────────────────────────────────────────────────

/// CHECK_HIP macro for error checking
/// All HIP API calls MUST be wrapped with this macro
/// Returns error code on failure, never continues past errors
#define CHECK_HIP(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            return error; \
        } \
    } while (0)

/// Host-side null check macro for launchers
#define CHECK_NULL(ptr) \
    if ((ptr) == nullptr) { \
        return hipErrorInvalidValue; \
    }

/// Kernel-safe bounds check macro
/// Use this to validate indices before memory access
#define CHECK_BOUNDS(idx, max) \
    if ((idx) >= (max)) { \
        return; \
    }

/// CHECK_LAST macro for kernel launch error checking
/// Must be called immediately after kernel launch
#define CHECK_LAST() \
    do { \
        hipError_t error = hipGetLastError(); \
        if (error != hipSuccess) { \
            return error; \
        } \
    } while (0)

// ── Quantization Constants ──────────────────────────────────────────────────────────

constexpr int QK_K = 256;          // Quantization block size (K formats)
constexpr int BLOCK_SIZE = 256;    // Default CUDA/HIP block size
constexpr int WARP_SIZE = 32;      // Warp size for wavefront programming
constexpr int MAX_SHARED_MEM = 64 * 1024; // Maximum shared memory per block

// ─── Type Aliases ───────────────────────────────────────────────────────────────────

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;

// ─── Device Utility Functions ─────────────────────────────────────────────────────

/// Device-side square root
__device__ inline float rsqrtf_gpu(float x) {
    return rsqrtf(x);
}

/// Device-side reciprocal
__device__ inline float rcp_gpu(float x) {
    return 1.0f / x;
}

/// Device-side minimum
__device__ inline float fmin_gpu(float a, float b) {
    return fminf(a, b);
}

/// Device-side maximum
__device__ inline float fmax_gpu(float a, float b) {
    return fmaxf(a, b);
}

/// Get global thread ID (flattened)
__device__ inline size_t get_global_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/// Unpack a 4-bit nibble from sequential (llama.cpp) packed layout.
/// Layout: qs[i] low nibble → element i, qs[i] high nibble → element i+16.
/// tid: thread index within the 32-thread block (0-31).
__device__ inline uint8_t unpack_nibble_sequential(const uint8_t* qs, int tid) {
    return (tid < 16) ? (qs[tid] & 0x0F) : (qs[tid - 16] >> 4);
}

/// Get lane ID within warp (0-31)
__device__ inline int get_lane_id() {
    return threadIdx.x & 31;
}

/// Get warp ID within block
__device__ inline int get_warp_id() {
    return threadIdx.x >> 5;
}

#endif // ROCMFORGE_QUANT_COMMON_HIP
