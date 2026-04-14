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
constexpr int Q6_K_BLOCK_SIZE = 210; // Bytes per Q6_K block (256 quantized values + metadata)
constexpr int QI6_K = 4;            // Quantization indices per block for Q6_K

// ── Tile Processing Constants ───────────────────────────────────────────────────

/// Tile size for Q6_K dequantization
/// Values chosen based on llama.cpp research:
/// - RDNA1: 64 threads per tile (limited shared memory)
/// - RDNA2+: 128 threads per tile (more shared memory available)
constexpr int TILE_Y = WARP_SIZE;  // Number of rows in tile (one per thread)

/// Padded tile size to avoid shared memory bank conflicts
/// Padding ensures 4 mod 8 alignment for bank conflict avoidance
constexpr int TILE_SIZE_PADDED = WARP_SIZE + WARP_SIZE/8 + 8;  // 44 = 4 mod 8

/// Static assertion to verify padding is correct (4 mod 8 ensures bank alignment)
static_assert(TILE_SIZE_PADDED % 8 == 4, "Tile size must be 4 mod 8 for bank alignment");

/// Q6_K-specific tile sizes for different data types
/// These match the tile layout used in llama.cpp mmq.cuh
namespace q6_k_tile {
    // Number of quantized values per tile (qs)
    constexpr int TILE_QS = WARP_SIZE * 2 + TILE_Y;

    // Number of d scales per tile (dm)
    constexpr int TILE_DM = WARP_SIZE / QI6_K + TILE_Y / QI6_K;

    // Number of quantization scales per tile (sc)
    constexpr int TILE_SC = WARP_SIZE / 8 + TILE_Y / 8;

    // Total tile size (with padding)
    constexpr int TILE_X_K_Q6_K = 2 * WARP_SIZE + WARP_SIZE / QI6_K + WARP_SIZE / 8 + 8;  // 84 = 4 mod 8

    static_assert(TILE_X_K_Q6_K % 8 == 4, "Q6_K tile size must be 4 mod 8");
}

/// Helper to compute padded index for bank conflict avoidance
/// Ensures threads don't access the same memory bank simultaneously
__device__ inline int compute_padded_index(int thread_id, int tile_base, int tile_stride) {
    // Stride pattern: skip enough elements to avoid bank conflicts
    const int offset = thread_id % (WARP_SIZE / 8);
    return tile_base * TILE_SIZE_PADDED + offset + (thread_id / (WARP_SIZE / 8)) * tile_stride;
}

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
