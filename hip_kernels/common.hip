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

// ── Common Constants ───────────────────────────────────────────────────────────────

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCK_SIZE = 1024;

// ── AMD GPU Architecture Wavefront Sizes ─────────────────────────────────────────────
// Wavefront size is the SIMD unit size (similar to CUDA warp size)
// - RDNA1/2/3/4: 32 threads per wavefront (gfx1010+, gfx1030+, gfx1100+, gfx1201+)
// - CDNA1/2/3: 64 threads per wavefront (gfx908+, gfx90a+, gfx942+)
// - Vega: 64 threads per wavefront (gfx900)
//
// build.rs detects the target architecture and sets WARP_SIZE correctly.
// This default is only used if detection fails (conservative RDNA default).
//
// IMPORTANT: Always use runtime warpSize in device code, not this constant!
// The runtime value is correct for the actual device, this is just build-time.
// ────────────────────────────────────────────────────────────────────────────────────

// Wavefront size (compile-time, build.rs sets this correctly)
// RDNA (gfx1100+, gfx1030+, gfx1201+): 32 threads per wavefront
// CDNA/Vega (gfx900+): 64 threads per wavefront
// Default to 32 (RDNA) as conservative fallback
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

constexpr int WARP_SIZE_COMPTIME = WARP_SIZE;

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
