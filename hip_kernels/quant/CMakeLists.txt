cmake_minimum_required(VERSION 3.18)
project(rocmerge_gpu_quant CXX HIP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROCm
find_package(hip REQUIRED)

# Common library (shared by all quantization kernels)
add_library(quant_common STATIC
    common.hip
)

# Set target properties
set_target_properties(quant_common PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

# Q4_K quantization kernel
add_library(q4_k_quantize STATIC
    q4_k_quantize.hip
)

# Link common library to Q4_K kernel
target_link_libraries(q4_k_quantize
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_k_quantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_K dequantization kernel
add_library(q4_k_dequantize STATIC
    q4_k_dequantize.hip
)

# Link common library to Q4_K dequantize kernel
target_link_libraries(q4_k_dequantize
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_k_dequantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_K verification kernel
add_library(q4_k_verify STATIC
    q4_k_verify.hip
)

# Link common library to Q4_K verify kernel
target_link_libraries(q4_k_verify
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_k_verify PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q8_0 quantization kernel
add_library(q8_0_quantize STATIC
    q8_0_quantize.hip
)

# Link common library to Q8_0 kernel
target_link_libraries(q8_0_quantize
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q8_0_quantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q8_0 dequantization kernel
add_library(q8_0_dequantize STATIC
    q8_0_dequantize.hip
)

# Link common library to Q8_0 dequantize kernel
target_link_libraries(q8_0_dequantize
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q8_0_dequantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q8_0 verification kernel
add_library(q8_0_verify STATIC
    q8_0_verify.hip
)

# Link common library to Q8_0 verify kernel
target_link_libraries(q8_0_verify
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q8_0_verify PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q8_0 GEMV kernel (Q8_0 × f32 matrix-vector multiplication)
add_library(q8_0_gemv STATIC
    q8_0_gemv.hip
)

# Link common library to Q8_0 GEMV kernel
target_link_libraries(q8_0_gemv
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q8_0_gemv PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_0 quantization kernel
add_library(q4_0_quantize STATIC q4_0_quantize.hip)
target_link_libraries(q4_0_quantize quant_common)
set_target_properties(q4_0_quantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_0 dequantization kernel
add_library(q4_0_dequantize STATIC q4_0_dequantize.hip)
target_link_libraries(q4_0_dequantize quant_common)
set_target_properties(q4_0_dequantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_1 quantization kernel
add_library(q4_1_quantize STATIC q4_1_quantize.hip)
target_link_libraries(q4_1_quantize quant_common)
set_target_properties(q4_1_quantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_1 dequantization kernel
add_library(q4_1_dequantize STATIC q4_1_dequantize.hip)
target_link_libraries(q4_1_dequantize quant_common)
set_target_properties(q4_1_dequantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_0 verification kernel
add_library(q4_0_verify STATIC q4_0_verify.hip)
target_link_libraries(q4_0_verify quant_common)
set_target_properties(q4_0_verify PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_1 verification kernel
add_library(q4_1_verify STATIC q4_1_verify.hip)
target_link_libraries(q4_1_verify quant_common)
set_target_properties(q4_1_verify PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_0 GEMV kernel (Q4_0 × f32 matrix-vector multiplication)
add_library(q4_0_gemv STATIC
    q4_0_gemv.hip
)

# Link common library to Q4_0 GEMV kernel
target_link_libraries(q4_0_gemv
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_0_gemv PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# GQA QKV Fusion kernel (Q4_0 GQA-aware QKV projection + RoPE)
add_library(q4_0_fused_qkv_rope_gqa STATIC
    q4_0_fused_qkv_rope_gqa.hip
)

# Link common library to GQA QKV fusion kernel
target_link_libraries(q4_0_fused_qkv_rope_gqa
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_0_fused_qkv_rope_gqa PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# DP4A-optimized fused QKV+RoPE kernel (RDNA2+ only)
# Provides 1.5-2× speedup using __builtin_amdgcn_sdot4 for 4-way int8 multiply-accumulate
# Trade-off: 0.4% noise from on-the-fly activation quantization vs 2× speedup
add_library(q4_0_fused_norm_qkv_rope_dp4a STATIC
    q4_0_fused_norm_qkv_rope_dp4a.hip
)

# Link common library to DP4A kernel
target_link_libraries(q4_0_fused_norm_qkv_rope_dp4a
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_0_fused_norm_qkv_rope_dp4a PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_1 GEMV kernel (Q4_1 × f32 matrix-vector multiplication)
add_library(q4_1_gemv STATIC
    q4_1_gemv.hip
)

# Link common library to Q4_1 GEMV kernel
target_link_libraries(q4_1_gemv
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_1_gemv PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Experimental FFN microkernel (kept separate from production dispatch)
add_library(q4_1_ffn_experimental STATIC
    q4_1_ffn_experimental.hip
)

target_link_libraries(q4_1_ffn_experimental
    quant_common
)

set_target_properties(q4_1_ffn_experimental PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# REMOVED: Vulkan-style GEMV (moved to old/ folder - not AMD HIP compliant)

# REMOVED: Vulkan-style Q4_K GEMV (moved to old/ folder - not AMD HIP compliant)

# Q6_K GEMV kernel (Q6_K × f32 matrix-vector multiplication)
add_library(q6_k_gemv STATIC
    q6_k_gemv.hip
)

target_link_libraries(q6_k_gemv
    quant_common
)

set_target_properties(q6_k_gemv PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# GEMM Kernels (Batched)
add_library(q4_0_gemm STATIC q4_0_gemm.hip)
target_link_libraries(q4_0_gemm quant_common)
set_target_properties(q4_0_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Batched Q4_0 GEMM kernel for prefill processing
add_library(batched_q4_0_gemm STATIC batched_q4_0.hip)
target_link_libraries(batched_q4_0_gemm quant_common)
set_target_properties(batched_q4_0_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Batched Q4_1 GEMM kernel for prefill processing
add_library(batched_q4_1_gemm STATIC batched_q4_1.hip)
target_link_libraries(batched_q4_1_gemm quant_common)
set_target_properties(batched_q4_1_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Q4_0 fused gate-up projection kernel (FFN layer optimization)
add_library(q4_0_fused STATIC q4_0_fused.hip)
target_link_libraries(q4_0_fused quant_common)
set_target_properties(q4_0_fused PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Q4_0 fused gate-up projection kernel with Q8_0 activation quantization
add_library(q4_0_fused_q8 STATIC q4_0_fused_q8.hip)
target_link_libraries(q4_0_fused_q8 quant_common)
set_target_properties(q4_0_fused_q8 PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(q4_1_gemm STATIC q4_1_gemm.hip)
target_link_libraries(q4_1_gemm quant_common)
set_target_properties(q4_1_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(q8_0_gemm STATIC q8_0_gemm.hip)
target_link_libraries(q8_0_gemm quant_common)
set_target_properties(q8_0_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(q4_k_gemm STATIC q4_k_gemm.hip)
target_link_libraries(q4_k_gemm quant_common)
set_target_properties(q4_k_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Q4_K GEMV kernel (vulkan-style, stream-compatible)
add_library(q4_k_gemv_vulkan_style STATIC q4_k_gemv_vulkan_style.hip)
target_link_libraries(q4_k_gemv_vulkan_style quant_common)
set_target_properties(q4_k_gemv_vulkan_style PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(q5_k_gemm STATIC q5_k_gemm.hip)
target_link_libraries(q5_k_gemm quant_common)
set_target_properties(q5_k_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(q6_k_gemm STATIC q6_k_gemm.hip)
target_link_libraries(q6_k_gemm quant_common)
set_target_properties(q6_k_gemm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Q6_K tile-based kernel (shared memory optimization)
add_library(q6_k_tile STATIC
    q6_k_tile.hip
)

# Link common library to Q6_K tile kernel
target_link_libraries(q6_k_tile
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q6_k_tile PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q6_K test library (for numerical accuracy validation)
add_library(q6_k_test STATIC
    q6_k_test.hip
)

# Link common library to Q6_K test kernel
target_link_libraries(q6_k_test
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q6_k_test PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q6_K vectorized kernel (Phase 2: Vectorized bit extraction)
add_library(q6_k_vectorized STATIC
    q6_k_vectorized.hip
)

# Link common library to Q6_K vectorized kernel
target_link_libraries(q6_k_vectorized
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q6_k_vectorized PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Test kernel library (for Phase 1 verification)
add_library(test_quant STATIC
    test_kernel.hip
)

# Link common library to test kernel
target_link_libraries(test_quant
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(test_quant PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q4_K GEMV kernel (Q4_K × f32 matrix-vector multiplication)
add_library(q4_k_gemv STATIC
    q4_k_gemv.hip
)

# Link common library to Q4_K GEMV kernel
target_link_libraries(q4_k_gemv
    quant_common
)

# Set output directory to match Cargo expectations
set_target_properties(q4_k_gemv PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q5_K quantization kernel
add_library(q5_k_quantize STATIC
    q5_k_quantize.hip
)

target_link_libraries(q5_k_quantize
    quant_common
)

set_target_properties(q5_k_quantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q5_K dequantization kernel
add_library(q5_k_dequantize STATIC
    q5_k_dequantize.hip
)

target_link_libraries(q5_k_dequantize
    quant_common
)

set_target_properties(q5_k_dequantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q5_K verification kernel
add_library(q5_k_verify STATIC
    q5_k_verify.hip
)

target_link_libraries(q5_k_verify
    quant_common
)

set_target_properties(q5_k_verify PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q5_K GEMV kernel (Q5_K × f32 matrix-vector multiplication)
add_library(q5_k_gemv STATIC
    q5_k_gemv.hip
)

target_link_libraries(q5_k_gemv
    quant_common
)

set_target_properties(q5_k_gemv PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q6_K quantization kernel
add_library(q6_k_quantize STATIC
    q6_k_quantize.hip
)

target_link_libraries(q6_k_quantize
    quant_common
)

set_target_properties(q6_k_quantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q6_K dequantization kernel
add_library(q6_k_dequantize STATIC
    q6_k_dequantize.hip
)

target_link_libraries(q6_k_dequantize
    quant_common
)

set_target_properties(q6_k_dequantize PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Q6_K verification kernel
add_library(q6_k_verify STATIC
    q6_k_verify.hip
)

target_link_libraries(q6_k_verify
    quant_common
)

set_target_properties(q6_k_verify PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# RFM GPU-Accelerated Unpack Kernels
add_library(rfm_unpack STATIC
    rfm_unpack.hip
)

target_link_libraries(rfm_unpack
    quant_common
)

set_target_properties(rfm_unpack PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# RDNA3 WMMA Matrix Core Acceleration
add_library(wmma_matmul STATIC
    wmma_matmul.hip
)

target_link_libraries(wmma_matmul
    quant_common
)

set_target_properties(wmma_matmul PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Print build info
message(STATUS "ROCm GPU Quantization Build Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/lib")

