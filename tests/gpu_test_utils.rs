//! GPU test utilities for CPU vs GPU comparison.
//!
//! All functions use CPU reference implementations from src/cpu/ops.rs
//! as ground truth for GPU kernel correctness verification.

#![cfg(feature = "gpu")]

use serial_test::serial;

/// Tolerance for floating point comparison (1e-4 allows for GPU precision differences)
pub const F32_TOLERANCE: f32 = 1e-4;

/// Assert two f32 slices are approximately equal within tolerance.
pub fn assert_close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Slice lengths must match");

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tolerance || diff <= x.abs() * tolerance,
            "Mismatch at index {}: {} vs {} (diff={})",
            i, x, y, diff
        );
    }
}

/// Generate simple test data for kernels.
/// Returns a vector [1.0, 2.0, 3.0, ..., n as f32]
pub fn linspace_1_to_n(n: usize) -> Vec<f32> {
    (1..=n as i32).map(|i| i as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial]
    fn test_assert_close_with_equal_arrays() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_close(&a, &b, F32_TOLERANCE);
    }

    #[test]
    #[serial]
    fn test_assert_close_detects_difference() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 10.0]; // Different at index 2
        let result = std::panic::catch_unwind(|| {
            assert_close(&a, &b, F32_TOLERANCE);
        });
        assert!(result.is_err(), "Should detect mismatch");
    }
}
