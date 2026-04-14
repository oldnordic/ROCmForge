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
constexpr int WARP_SIZE_COMPTIME = WARP_SIZE;  // Compile-time warp size
constexpr int MAX_SHARED_MEM = 64 * 1024; // Maximum shared memory per block

// ── Architecture Detection Macros ────────────────────────────────────────────────

/// Detect AMD GPU architecture from compute capability
/// Returns: 900 (RDNA1), 906 (RDNA2), 1100 (RDNA3), 80x/90x (CDNA)
#if defined(__HIP_PLATFORM_AMD__)
    #define ROCMFORGE_IS_AMD 1

    // Runtime architecture detection (set during kernel launch)
    // Default to RDNA3 for compilation
    #ifndef ROCMFORGE_ARCH
        #define ROCMFORGE_ARCH 1100
    #endif

    #define ROCMFORGE_IS_RDNA1 (ROCMFORGE_ARCH == 900)
    #define ROCMFORGE_IS_RDNA2 (ROCMFORGE_ARCH == 906)
    #define ROCMFORGE_IS_RDNA3 (ROCMFORGE_ARCH == 1100)
    #define ROCMFORGE_IS_CDNA ((ROCMFORGE_ARCH >= 800 && ROCMFORGE_ARCH < 900) || \
                               (ROCMFORGE_ARCH >= 900 && ROCMFORGE_ARCH < 1000))

    // Feature detection
    #if defined(__has_builtin)
        #if __has_builtin(__builtin_amdgcn_sudot4)
            #define ROCMFORGE_HAS_SUDOT4 1
        #endif
        #if __has_builtin(__builtin_amdgcn_sdot4)
            #define ROCMFORGE_HAS_SDOT4 1
        #endif
        #if __has_builtin(__builtin_elementwise_sub_sat)
            #define ROCMFORGE_HAS_SUB_SAT 1
        #endif
    #endif

    // Default to 0 if builtin detection not available
    #ifndef ROCMFORGE_HAS_SUDOT4
        #define ROCMFORGE_HAS_SUDOT4 0
    #endif
    #ifndef ROCMFORGE_HAS_SDOT4
        #define ROCMFORGE_HAS_SDOT4 0
    #endif
    #ifndef ROCMFORGE_HAS_SUB_SAT
        #define ROCMFORGE_HAS_SUB_SAT 0
    #endif

#else
    #define ROCMFORGE_IS_AMD 0
    #define ROCMFORGE_IS_RDNA1 0
    #define ROCMFORGE_IS_RDNA2 0
    #define ROCMFORGE_IS_RDNA3 0
    #define ROCMFORGE_IS_CDNA 0
    #define ROCMFORGE_HAS_SUDOT4 0
    #define ROCMFORGE_HAS_SDOT4 0
    #define ROCMFORGE_HAS_SUB_SAT 0
#endif

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
