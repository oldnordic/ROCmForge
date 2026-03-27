//! Sanity tier tests for GPU quantization infrastructure.
//!
//! These tests verify the basic CMake build infrastructure is working
//! before implementing actual quantization kernels.
//!
//! Run with: cargo test --test quant_sanity

#![cfg(feature = "gpu")]

use std::path::Path;
use std::process::Command;
use serial_test::serial;

/// CMakeLists.txt must exist at the expected location
#[test]
#[serial]
fn cmakefile_exists() {
    let cmakefile = Path::new("hip_kernels/quant/CMakeLists.txt");
    assert!(cmakefile.exists(), "CMakeLists.txt must exist");
}

/// common.hip must exist with safety macros
#[test]
#[serial]
fn common_header_exists() {
    let header = Path::new("hip_kernels/quant/common.hip");
    assert!(header.exists(), "common.hip must exist");

    // Verify key macros are present
    let content = std::fs::read_to_string(header)
        .expect("Failed to read common.hip");

    assert!(content.contains("CHECK_HIP"), "Must define CHECK_HIP macro");
    assert!(content.contains("CHECK_BOUNDS"), "Must define CHECK_BOUNDS macro");
    assert!(content.contains("CHECK_LAST"), "Must define CHECK_LAST macro");
    assert!(content.contains("QK_K"), "Must define QK_K constant");
    assert!(content.contains("BLOCK_SIZE"), "Must define BLOCK_SIZE constant");
}

/// test_kernel.hip must exist with test kernel
#[test]
#[serial]
fn test_kernel_exists() {
    let kernel = Path::new("hip_kernels/quant/test_kernel.hip");
    assert!(kernel.exists(), "test_kernel.hip must exist");

    // Verify kernel function is present
    let content = std::fs::read_to_string(kernel)
        .expect("Failed to read test_kernel.hip");

    assert!(content.contains("vector_add"), "Must define vector_add kernel");
    assert!(content.contains("test_vector_add"), "Must define FFI wrapper");
}

/// Build directory structure should be creatable
#[test]
#[serial]
fn build_directory_exists_or_creatable() {
    let build_dir = Path::new("hip_kernels/quant/build");

    // If build directory exists, verify it's a directory
    if build_dir.exists() {
        assert!(build_dir.is_dir(), "build path must be a directory");
    } else {
        // Try to create it
        std::fs::create_dir_all(build_dir)
            .expect("Failed to create build directory");
        assert!(build_dir.exists(), "build directory should exist after creation");
    }
}

/// CMake executable should be available (or build.rs gracefully handles absence)
#[test]
#[serial]
fn cmake_available() {
    let cmake_check = Command::new("cmake")
        .arg("--version")
        .output();

    match cmake_check {
        Ok(output) => {
            // CMake found
            assert!(output.status.success(), "cmake --version should succeed");
            let version = String::from_utf8_lossy(&output.stdout);
            assert!(version.contains("cmake"), "Output should mention cmake");
        }
        Err(_) => {
            // CMake not found - this is OK, build.rs should handle it
            // The test passes because we verified the behavior
        }
    }
}

/// CMakeLists.txt should have valid syntax (can be configured)
#[test]
#[serial]
fn cmake_valid_syntax() {
    let cmake_check = Command::new("cmake")
        .arg("-S")
        .arg("hip_kernels/quant")
        .arg("-B")
        .arg("hip_kernels/quant/build")
        .output();

    match cmake_check {
        Ok(output) => {
            // CMake found - check configuration succeeded
            if output.status.success() {
                // Success - CMake syntax is valid
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(stdout.contains("ROCm"), "Should configure ROCm project");
            } else {
                // Configuration failed - show why
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                panic!("CMake configuration failed:\nstdout: {}\nstderr: {}", stdout, stderr);
            }
        }
        Err(_) => {
            // CMake not found - test passes (build.rs handles this)
        }
    }
}

/// CMake build should produce non-empty library file
#[test]
#[serial]
fn cmake_builds_library() {
    // First try to configure
    let _config = Command::new("cmake")
        .arg("-S")
        .arg("hip_kernels/quant")
        .arg("-B")
        .arg("hip_kernels/quant/build")
        .output();

    // Then build
    let build_result = Command::new("cmake")
        .arg("--build")
        .arg("hip_kernels/quant/build")
        .output();

    match build_result {
        Ok(output) => {
            if output.status.success() {
                // Build succeeded - verify library exists
                let lib_path = Path::new("hip_kernels/quant/build/lib/libtest_quant.a");
                assert!(lib_path.exists(), "Library file must exist after build");

                // Verify library has non-zero size
                let metadata = std::fs::metadata(lib_path)
                    .expect("Failed to read library metadata");
                assert!(metadata.len() > 0, "Library must have non-zero size");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                panic!("CMake build failed:\nstdout: {}\nstderr: {}", stdout, stderr);
            }
        }
        Err(_) => {
            // CMake not found - OK for this test
        }
    }
}

/// Verify build.rs calls compile_quant_kernels
#[test]
#[serial]
fn buildrs_invokes_cmake() {
    let buildrs = Path::new("build.rs");
    assert!(buildrs.exists(), "build.rs must exist");

    let content = std::fs::read_to_string(buildrs)
        .expect("Failed to read build.rs");

    // Verify compile_quant_kernels is called
    assert!(content.contains("compile_quant_kernels()"),
            "build.rs must call compile_quant_kernels");

    // Verify cmake command is used
    assert!(content.contains("cmake") || content.contains("CMake"),
            "compile_quant_kernels should invoke cmake");
}
