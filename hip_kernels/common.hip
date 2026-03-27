#ifndef ROCMFORGE_COMMON_HIP
#define ROCMFORGE_COMMON_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

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

/// Kernel-safe bounds check macro
/// Use this to validate indices before memory access
#define CHECK_BOUNDS(idx, max) \
    if ((idx) >= (max)) { \
        return; \
    }

// ── Common Constants ───────────────────────────────────────────────────────────────

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 64;

// ── Utility Functions ───────────────────────────────────────────────────────────────

/// Device-side square root
__device__ inline float rsqrt(float x) {
    return rsqrtf(x);
}

/// Device-side sine
__device__ inline float sin_val(float x) {
    return sinf(x);
}

/// Device-side cosine
__device__ inline float cos_val(float x) {
    return cosf(x);
}

/// Device-side power
__device__ inline float pow_val(float base, float exp) {
    return powf(base, exp);
}

#endif // ROCMFORGE_COMMON_HIP
