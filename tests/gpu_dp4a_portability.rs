#![cfg(feature = "gpu")]
//! Portable DP4A numerical correctness tests
//!
//! Verifies that dot4_manual() produces identical results to hardware DP4A
//! on architectures where both are available (RDNA2).

use rocmforge::gpu::kernels::quant::{test_dot4_hardware, test_dot4_manual};
use rocmforge::gpu::GpuDevice;

#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    /// Test that dot4_manual matches hardware DP4A on RDNA2 (gfx1030)
    #[test]
    #[ignore] // Only run on RDNA2 hardware
    fn test_dot4_manual_matches_dp4a_rdna2() {
        // Skip on non-RDNA2 hardware
        let ctx = GpuDevice::init(0).unwrap();
        let device_name = ctx.get_name().unwrap_or_default();

        if !device_name.contains("gfx1030") {
            println!(
                "Skipping: Test requires RDNA2 (gfx1030), got {}",
                device_name
            );
            return;
        }

        // Test vectors: 4 pairs of int8 values packed into int32
        let test_cases = [
            // Simple cases
            ((0x00010203_i32, 0x00010203_i32, 0), 14), // 1*1 + 2*2 + 3*3 = 14
            ((-1_i32, 0x01010101, 0), -4),             // -1*1 + -1*1 + -1*1 + -1*1 = -4
            ((0x00000000_i32, -1_i32, 100), 100),      // 0 + 100 = 100
            // Mixed positive/negative
            ((0x7F808080_i32, 0x01010101, 0), -255), // 127*1 + -128*1 + -128*1 + -128*1 = -255
            ((0x01010101_i32, 0x7F808080, 0), -255), // Same as above (commutative)
            // Accumulator tests
            ((0x01010101_i32, 0x01010101, 10), 14), // 4 + 10 = 14
            ((-1_i32, -1_i32, -100), -104),         // -4 + -100 = -104
        ];

        for (i, ((a_packed, b_packed, acc), expected)) in test_cases.iter().enumerate() {
            // Call kernel with both DP4A and manual paths
            let result_dp4a = unsafe { test_dot4_hardware(*a_packed, *b_packed, *acc) };
            let result_manual = unsafe { test_dot4_manual(*a_packed, *b_packed, *acc) };

            assert_eq!(
                result_dp4a, *expected,
                "DP4A case {}: Expected {}, got {}",
                i, expected, result_dp4a
            );
            assert_eq!(
                result_manual, *expected,
                "Manual case {}: Expected {}, got {}",
                i, expected, result_manual
            );
            assert_eq!(
                result_dp4a, result_manual,
                "DP4A vs Manual mismatch case {}: DP4A={}, Manual={}",
                i, result_dp4a, result_manual
            );
        }
    }

    /// Test that portable DP4A produces valid results on RDNA3 (gfx1100)
    #[test]
    #[ignore] // Only run on RDNA3 hardware
    fn test_dot4_manual_correctness_rdna3() {
        let ctx = GpuDevice::init(0).unwrap();
        let device_name = ctx.get_name().unwrap_or_default();

        if !device_name.contains("gfx1100") {
            println!(
                "Skipping: Test requires RDNA3 (gfx1100), got {}",
                device_name
            );
            return;
        }

        // Same test cases as above, but only test manual path
        let test_cases = [
            ((0x00010203_i32, 0x00010203_i32, 0), 14),
            ((-1_i32, 0x01010101, 0), -4),
            ((0x00000000_i32, -1_i32, 100), 100),
            ((0x7F808080_i32, 0x01010101, 0), -255),
            ((0x01010101_i32, 0x7F808080, 0), -255),
            ((0x01010101_i32, 0x01010101, 10), 14),
            ((-1_i32, -1_i32, -100), -104),
        ];

        for (i, ((a_packed, b_packed, acc), expected)) in test_cases.iter().enumerate() {
            let result = unsafe { test_dot4_manual(*a_packed, *b_packed, *acc) };

            assert_eq!(
                result, *expected,
                "Manual case {}: Expected {}, got {}",
                i, expected, result
            );
        }
    }
}
