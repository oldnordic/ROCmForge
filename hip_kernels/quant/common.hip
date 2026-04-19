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

// ── Vector Intrinsics (from llama.cpp) ────────────────────────────────────────────────

/// Read 32-bit integer from byte array
/// Matches llama.cpp's get_int_b2() for optimized memory access
/// Safety: Read-only operation, no side effects
__device__ inline int get_int_b2(const uint8_t* p, const int i) {
    return *(const int*)(p + i * sizeof(int));
}

/// Vector subtract with saturation (4 int8 values at once)
/// Matches llama.cpp's __vsubss4() for Q6_K dequantization
/// Subtracts b from a, saturating at int8 min/max for each of 4 bytes
/// Safety: Read-only operation, pure computation
__device__ inline int __vsubss4_gpu(const int a, const int b) {
    // Portable implementation that works on both AMD and NVIDIA
    // Processes 4 int8 values in parallel using bit operations
    // This should be optimized by the compiler to use vector instructions

    int result = 0;

    // Process each byte separately
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Extract each byte
        const int byte_a = (a >> (i * 8)) & 0xFF;
        const int byte_b = (b >> (i * 8)) & 0xFF;

        // Reinterpret as signed int8
        const int8_t signed_a = static_cast<int8_t>(byte_a);
        const int8_t signed_b = static_cast<int8_t>(byte_b);

        // Subtract with saturation
        int16_t sub_result = static_cast<int16_t>(signed_a) - static_cast<int16_t>(signed_b);

        // Saturate to int8 range
        if (sub_result < -128) sub_result = -128;
        if (sub_result > 127) sub_result = 127;

        // Pack back into result
        result |= (static_cast<int>(static_cast<uint8_t>(sub_result)) << (i * 8));
    }

    return result;
}

// ── WMMA (Wave Matrix Multiply Accumulate) Intrinsics for CDNA ─────

// WMMA matrix fragment types (CDNA architectures only)
// NOTE: MFMA intrinsics are NOT available on RDNA (gfx1030, gfx1100)
// They are only available on CDNA architectures (gfx90a, gfx940, etc.)
// These intrinsics are not currently used and are reserved for future CDNA optimization
//
// TODO: Enable WMMA intrinsics for CDNA architectures when needed
// #if defined(__AMDGPU__) && !defined(__gfx1100__) && !defined(__gfx1030__)
//
// // MFMA (Matrix Fuse Multiply-Accumulate) intrinsics
// __device__ __forceinline__ int mfnacc_i32_4x4_i8(
//     int a, int b, int c, bool negate
// ) {
//     return __builtin_amdgcn_mfma_i32_4x4_i8(a, b, c, negate);
// }
//
// #define MFMA_I32_4X4_I8(a, b, c, neg) \
//     __builtin_amdgcn_mfma_i32_4x4_i8((a), (b), (c), (neg))
//
// #endif // CDNA architectures only

// ── Portable 4-Way Int8 Dot Product ─────────────────────────────────────

/// Software implementation of 4-way int8 dot product
/// Computes: sum += dot(int8_vec_a[0:3], int8_vec_b[0:3])
///
/// Portably extracts 4 int8 values from packed int32,
/// sign-extends to int32, multiplies pairwise, and sums.
///
/// Performance: Slower than hardware DP4A, but works on all architectures.
__device__ inline int dot4_manual(int a_packed, int b_packed, int acc) {
    int dot = 0;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Extract byte i with sign extension using arithmetic shift
        // Shift to MSB position, then arithmetic shift right to sign-extend
        const int signed_a = (a_packed << (24 - i * 8)) >> 24;
        const int signed_b = (b_packed << (24 - i * 8)) >> 24;

        // Multiply-accumulate
        dot += signed_a * signed_b;
    }

    return acc + dot;
}

/// Architecture-aware dot4 dispatcher
/// Uses hardware DP4A on RDNA2 (gfx1030), software fallback elsewhere
#if defined(__AMDGPU__) && defined(__gfx1030__)
    // RDNA2: Use hardware DP4A (dot1-insts feature available)
    #define DOT4(a, b, acc) __builtin_amdgcn_sdot4((a), (b), (acc), false)
#else
    // RDNA1, RDNA3, CDNA: Use portable software fallback
    #define DOT4(a, b, acc) dot4_manual((a), (b), (acc))
#endif

#endif // ROCMFORGE_QUANT_COMMON_HIP
