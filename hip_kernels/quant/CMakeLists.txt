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

# Print build info
message(STATUS "ROCm GPU Quantization Build Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/lib")
