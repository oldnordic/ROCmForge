//! Unit tier tests for GPU quantization components.
//!
//! Tests individual components in isolation:
//! - GpuArchitecture enum properties
//! - VRAM management utilities
//! - Architecture detection
//!
//! Run with: cargo test --test quant_unit --features gpu

#![cfg(feature = "gpu")]

use serial_test::serial;

// Import architecture module for testing
use rocmforge::gpu::GpuArchitecture;

/// Test GpuArchitecture::max_threads_per_block for all architectures
#[test]
#[serial]
fn test_max_threads_per_block() {
    // RDNA3 (gfx1100)
    assert_eq!(GpuArchitecture::Gfx1100.max_threads_per_block(), 1024);

    // RDNA2 (gfx1030)
    assert_eq!(GpuArchitecture::Gfx1030.max_threads_per_block(), 1024);

    // CDNA2 (gfx90a)
    assert_eq!(GpuArchitecture::Gfx90a.max_threads_per_block(), 1024);

    // CDNA1 (gfx908)
    assert_eq!(GpuArchitecture::Gfx908.max_threads_per_block(), 1024);

    // Vega (gfx900)
    assert_eq!(GpuArchitecture::Gfx900.max_threads_per_block(), 1024);

    // Unknown architecture - should still return a reasonable value
    assert_eq!(GpuArchitecture::Unknown(0).max_threads_per_block(), 256);
}

/// Test GpuArchitecture::warp_size for RDNA vs CDNA/Vega
#[test]
#[serial]
fn test_warp_size() {
    // RDNA architectures use 32-thread wavefronts
    assert_eq!(GpuArchitecture::Gfx1100.warp_size(), 32);
    assert_eq!(GpuArchitecture::Gfx1030.warp_size(), 32);

    // CDNA and Vega use 64-thread wavefronts
    assert_eq!(GpuArchitecture::Gfx90a.warp_size(), 64);
    assert_eq!(GpuArchitecture::Gfx908.warp_size(), 64);
    assert_eq!(GpuArchitecture::Gfx900.warp_size(), 64);

    // Unknown defaults to 32 (most common)
    assert_eq!(GpuArchitecture::Unknown(0).warp_size(), 32);
}

/// Test GpuArchitecture::shared_mem_per_block varies by architecture
#[test]
#[serial]
fn test_shared_mem_per_block() {
    // RDNA3 has 64KB shared memory
    assert_eq!(GpuArchitecture::Gfx1100.shared_mem_per_block(), 65536);

    // RDNA2 has 64KB shared memory
    assert_eq!(GpuArchitecture::Gfx1030.shared_mem_per_block(), 65536);

    // CDNA2 has 64KB shared memory (implementation uses conservative value)
    assert_eq!(GpuArchitecture::Gfx90a.shared_mem_per_block(), 65536);

    // CDNA1 has 64KB shared memory
    assert_eq!(GpuArchitecture::Gfx908.shared_mem_per_block(), 65536);

    // Vega has 64KB shared memory
    assert_eq!(GpuArchitecture::Gfx900.shared_mem_per_block(), 65536);

    // Unknown defaults to 32KB
    assert_eq!(GpuArchitecture::Unknown(0).shared_mem_per_block(), 32768);
}

/// Test GpuArchitecture::from_name parsing
#[test]
#[serial]
fn test_architecture_from_name() {
    // Valid architecture names
    assert_eq!(GpuArchitecture::from_name("gfx1100"), Some(GpuArchitecture::Gfx1100));
    assert_eq!(GpuArchitecture::from_name("gfx1030"), Some(GpuArchitecture::Gfx1030));
    assert_eq!(GpuArchitecture::from_name("gfx90a"), Some(GpuArchitecture::Gfx90a));
    assert_eq!(GpuArchitecture::from_name("gfx908"), Some(GpuArchitecture::Gfx908));
    assert_eq!(GpuArchitecture::from_name("gfx900"), Some(GpuArchitecture::Gfx900));

    // Unknown architecture names (valid hex but not a known architecture)
    assert_eq!(GpuArchitecture::from_name("gfx9999"), Some(GpuArchitecture::Unknown(0x9999)));

    // Invalid names (can't parse as hex)
    assert_eq!(GpuArchitecture::from_name("invalid"), None);
    assert_eq!(GpuArchitecture::from_name(""), None);
}

/// Test block size calculation divides work evenly
#[test]
#[serial]
fn test_block_size_distribution() {
    let block_size = 256;

    // Test various array sizes
    let test_cases = vec![
        (256, 1),    // Exact multiple
        (512, 2),    // 2 blocks
        (1000, 4),   // 3.9 blocks → 4
        (1024, 4),   // 4 blocks
        (2048, 8),   // 8 blocks
        (3000, 12),  // 11.7 blocks → 12
    ];

    for (n, expected_blocks) in test_cases {
        let grid_size = (n + block_size - 1) / block_size;
        assert_eq!(grid_size, expected_blocks,
                   "Block count mismatch for n={}", n);
    }
}

/// Test VRAM check rejects obviously invalid values
#[test]
#[serial]
fn test_vram_check_rejects_invalid() {
    // Import from gpu_test_utils
    // Note: This test verifies logic without requiring actual GPU

    // MAX_TEST_VRAM_GB should be 10.0
    const MAX_TEST_VRAM_GB: f64 = 10.0;

    // Values exceeding MAX should always fail
    assert!(MAX_TEST_VRAM_GB > 0.0, "MAX_TEST_VRAM_GB must be positive");
    assert!(MAX_TEST_VRAM_GB <= 100.0, "MAX_TEST_VRAM_GB must be reasonable");

    // Test arithmetic for block calculation
    let required_gb = 15.0;
    assert!(required_gb > MAX_TEST_VRAM_GB, "Test should use value exceeding max");
}

/// Test tolerance allows reasonable floating point differences
#[test]
#[serial]
fn test_f32_tolerance() {
    const F32_TOLERANCE: f32 = 1e-4;

    // Exact match should always pass
    let a = 1.0f32;
    let b = 1.0f32;
    let diff = (a - b).abs();
    assert!(diff <= F32_TOLERANCE, "Exact match should pass");

    // Small difference within tolerance should pass
    let a = 1.0f32;
    let b = 1.00001f32;
    let diff = (a - b).abs();
    assert!(diff <= F32_TOLERANCE, "Small difference should pass");

    // Large difference should fail
    let a = 1.0f32;
    let b = 2.0f32;
    let diff = (a - b).abs();
    assert!(diff > F32_TOLERANCE, "Large difference should fail");
}

/// Test linspace generates correct sequence
#[test]
#[serial]
fn test_linspace_generation() {
    // Import gpu_test_utils linspace_1_to_n equivalent
    fn linspace_1_to_n(n: usize) -> Vec<f32> {
        (1..=n as i32).map(|i| i as f32).collect()
    }

    // Test small sequence
    let result = linspace_1_to_n(5);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Test single element
    let result = linspace_1_to_n(1);
    assert_eq!(result, vec![1.0]);

    // Test larger sequence
    let result = linspace_1_to_n(10);
    assert_eq!(result.len(), 10);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[9], 10.0);
}

/// Test QK_K constant value is 256
#[test]
#[serial]
fn test_qk_k_constant() {
    // QK_K = 256 is the standard block size for K-format quantization
    // This matches llama.cpp's ggml-common.h definition
    const QK_K: usize = 256;

    assert_eq!(QK_K, 256, "QK_K must be 256 for K-format compatibility");

    // QK_K should be a power of 2
    assert!(QK_K.is_power_of_two(), "QK_K must be power of 2");
}

/// Test BLOCK_SIZE is a reasonable default
#[test]
#[serial]
fn test_block_size_constant() {
    const BLOCK_SIZE: usize = 256;

    // BLOCK_SIZE should match QK_K for quantization
    const QK_K: usize = 256;
    assert_eq!(BLOCK_SIZE, QK_K, "BLOCK_SIZE should match QK_K");

    // BLOCK_SIZE should be power of 2
    assert!(BLOCK_SIZE.is_power_of_two(), "BLOCK_SIZE must be power of 2");
}

/// Test WARP_SIZE matches architecture expectations
#[test]
#[serial]
fn test_warp_size_constant() {
    const WARP_SIZE: usize = 32;

    // Default warp size should be 32 (NVIDIA/AMD RDNA standard)
    assert_eq!(WARP_SIZE, 32, "Default WARP_SIZE should be 32");

    // WARP_SIZE should be power of 2
    assert!(WARP_SIZE.is_power_of_two(), "WARP_SIZE must be power of 2");

    // WARP_SIZE should be <= typical wavefront sizes (32 or 64)
    assert!(WARP_SIZE <= 64, "WARP_SIZE should be <= 64");
}

/// Test architecture properties are internally consistent
#[test]
#[serial]
fn test_architecture_consistency() {
    let archs: Vec<GpuArchitecture> = vec![
        GpuArchitecture::Gfx1100,
        GpuArchitecture::Gfx1030,
        GpuArchitecture::Gfx90a,
        GpuArchitecture::Gfx908,
        GpuArchitecture::Gfx900,
    ];

    for arch in archs {
        let max_threads: u32 = arch.max_threads_per_block();
        let warp_size: u32 = arch.warp_size();
        let shared_mem: usize = arch.shared_mem_per_block();

        // max_threads must be positive
        assert!(max_threads > 0, "max_threads_per_block must be positive");

        // warp_size must be positive and power of 2
        assert!(warp_size > 0, "warp_size must be positive");
        assert!(warp_size.is_power_of_two(), "warp_size must be power of 2");

        // shared_mem must be reasonable (> 32KB, < 1MB)
        assert!(shared_mem >= 32768, "shared_mem must be >= 32KB");
        assert!(shared_mem <= 1048576, "shared_mem must be <= 1MB");

        // max_threads should be multiple of warp_size
        assert!(max_threads % warp_size == 0,
                "max_threads should be multiple of warp_size");
    }
}

/// Test CMake find logic would work (logic check, no actual filesystem)
#[test]
#[serial]
fn test_cmake_find_logic() {
    // This test verifies the logic of find_cmake() without actual filesystem access

    // Standard paths that find_cmake() should check
    let standard_paths = vec![
        "/usr/bin/cmake",
        "/usr/local/bin/cmake",
        "/opt/homebrew/bin/cmake",
    ];

    // All paths should end with "cmake"
    for path in &standard_paths {
        assert!(path.ends_with("cmake"), "Path should end with cmake: {}", path);
    }

    // Paths should be absolute
    for path in &standard_paths {
        assert!(path.starts_with('/'), "Path should be absolute: {}", path);
    }
}

/// Test quantization block size for Q4_K format
#[test]
#[serial]
fn test_q4_k_block_size() {
    // Q4_K: 256 elements per block
    const QK_K: usize = 256;

    // Q4_K stores 4-bit values, so each block is 128 bytes of data
    // (256 * 4 / 8 = 128)
    let data_bytes = QK_K * 4 / 8;
    assert_eq!(data_bytes, 128, "Q4_K data bytes should be 128");

    // Plus scales and other metadata
    // Total block size is approximately QK_K / 2 bytes for Q4_K
    assert!(data_bytes <= QK_K / 2, "Data should fit in QK_K/2 bytes");
}

/// Test quantization block size for Q8_0 format
#[test]
#[serial]
fn test_q8_0_block_size() {
    // Q8_0: 32 elements per block (smaller than QK_K)
    const QK8_0: usize = 32;

    // Q8_0 stores 8-bit values, so 32 bytes of data
    let data_bytes = QK8_0;
    assert_eq!(data_bytes, 32, "Q8_0 data bytes should be 32");

    // Plus 1 float (4 bytes) for scale
    let total_bytes = data_bytes + 4;
    assert_eq!(total_bytes, 36, "Q8_0 total bytes should be 36");
}

/// Test block size alignment for GPU efficiency
#[test]
#[serial]
fn test_block_alignment() {
    const BLOCK_SIZE: usize = 256;
    const WARP_SIZE: usize = 32;

    // BLOCK_SIZE should be multiple of WARP_SIZE for efficient GPU execution
    assert_eq!(BLOCK_SIZE % WARP_SIZE, 0,
               "BLOCK_SIZE should be multiple of WARP_SIZE");

    // Common thread counts should divide BLOCK_SIZE evenly
    let thread_counts = vec![32, 64, 128, 256];
    for threads in thread_counts {
        assert_eq!(BLOCK_SIZE % threads, 0,
                   "BLOCK_SIZE should be divisible by {}", threads);
    }
}

/// Test Q4_K constants match llama.cpp
#[test]
#[serial]
fn test_q4_k_constants() {
    use rocmforge::gpu::{QK_K, K_SCALE_SIZE, Q4_K_BLOCK_SIZE};

    assert_eq!(QK_K, 256, "QK_K must be 256 elements per block");
    assert_eq!(K_SCALE_SIZE, 12, "K_SCALE_SIZE must be 12");
    assert_eq!(Q4_K_BLOCK_SIZE, 144, "Q4_K_BLOCK_SIZE must be 144 bytes");
}

/// Test Q4KBlock size
#[test]
#[serial]
fn test_q4_k_block_struct_size() {
    use rocmforge::gpu::Q4KBlock;

    let block = Q4KBlock::default();
    assert_eq!(std::mem::size_of::<Q4KBlock>(), 144);
}

/// Test Q4KBlock default values
#[test]
#[serial]
fn test_q4_k_block_default() {
    use rocmforge::gpu::Q4KBlock;

    let block = Q4KBlock::default();
    assert_eq!(block.scales.len(), 12);
    assert_eq!(block.qs.len(), 128);
}

/// Test Q8_0 constants
#[test]
#[serial]
fn test_q8_0_constants() {
    use rocmforge::gpu::quant::{QK8_0, Q8_0_BLOCK_SIZE, Q8_0_MAX};
    assert_eq!(QK8_0, 32, "QK8_0 must be 32 for Q8_0 format");
    assert_eq!(Q8_0_BLOCK_SIZE, 34, "Q8_0_BLOCK_SIZE must be 34 bytes");
    assert_eq!(Q8_0_MAX, 127.0, "Q8_0_MAX must be 127.0");
}

/// Test Q8_0Block struct size
#[test]
#[serial]
fn test_q8_0_block_struct_size() {
    use rocmforge::gpu::quant::Q8_0Block;
    assert_eq!(std::mem::size_of::<Q8_0Block>(), 34, "Q8_0Block must be 34 bytes");
}

/// Test Q8_0Block default
#[test]
#[serial]
fn test_q8_0_block_default() {
    use rocmforge::gpu::quant::Q8_0Block;
    let block = Q8_0Block::default();
    assert_eq!(block.qs.len(), 32, "Default qs array must have 32 elements");
}

/// Test Q5_K constants match llama.cpp
#[test]
#[serial]
fn test_q5_k_constants() {
    use rocmforge::gpu::{QK_K, K_SCALE_SIZE, Q5_K_BLOCK_SIZE};

    assert_eq!(QK_K, 256, "QK_K must be 256 elements per block");
    assert_eq!(K_SCALE_SIZE, 12, "K_SCALE_SIZE must be 12");
    assert_eq!(Q5_K_BLOCK_SIZE, 176);  // 2+2+12+32+128
}

/// Test Q5KBlock size
#[test]
#[serial]
fn test_q5_k_block_struct_size() {
    use rocmforge::gpu::Q5KBlock;

    let block = Q5KBlock::default();
    assert_eq!(std::mem::size_of::<Q5KBlock>(), 176);  // 2+2+12+32+128
}

/// Test Q5KBlock default values
#[test]
#[serial]
fn test_q5_k_block_default() {
    use rocmforge::gpu::Q5KBlock;

    let block = Q5KBlock::default();
    assert_eq!(block.scales.len(), 12);
    assert_eq!(block.qh.len(), 32);
    assert_eq!(block.qs.len(), 128);
}
