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

# Print build info
message(STATUS "ROCm GPU Quantization Build Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/lib")
