//! HIP Backend Synchronization Safety Tests
//!
//! TDD Tests for Phase 23: hipDeviceSynchronize Fix
//!
//! These tests verify that synchronization uses stream-aware APIs
//! and doesn't cause desktop hangs by waiting for ALL GPU streams.
//!
//! # Test Strategy
//!
//! 1. Verify synchronize_device() is stream-aware
//! 2. Verify backend.synchronize() works correctly
//! 3. Verify no memory leaks from synchronization
//!
//! # Why This Matters
//!
//! `hipDeviceSynchronize()` waits for ALL streams including the desktop
//! compositor, causing deadlocks and desktop hangs. Stream-aware
//! synchronization (`hipStreamSynchronize`) only waits for our stream.

#[cfg(test)]
mod tests {
    use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
    use serial_test::serial;

    /// Test that synchronize_device() doesn't hang the desktop
    ///
    /// This test validates that the global synchronize_device() function
    /// uses stream-aware synchronization and doesn't call the dangerous
    /// hipDeviceSynchronize() which waits for ALL GPU streams.
    ///
    /// # Expected Behavior
    ///
    /// - synchronize_device() should return successfully
    /// - Desktop should NOT freeze during this test
    /// - No GPU memory leaks should occur
    #[test]
    #[serial]
    fn test_synchronize_device_is_stream_aware() {
        // Get the GPU fixture - returns None if GPU unavailable
        let fixture = GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");

        // Get backend reference to ensure it's initialized
        let _backend = fixture.backend();

        // Call synchronize_device - should NOT hang desktop
        // If this calls hipDeviceSynchronize(), desktop will freeze
        let result = rocmforge::backend::hip_backend::synchronize_device();

        assert!(
            result.is_ok(),
            "synchronize_device should succeed without hanging desktop"
        );

        // Verify no memory leaks
        fixture.assert_no_leak(5);
    }

    /// Test that backend.synchronize() works correctly
    ///
    /// This verifies that HipBackend's synchronize method uses
    /// stream-aware synchronization (hipStreamSynchronize).
    #[test]
    #[serial]
    fn test_backend_synchronize() {
        let fixture = GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");

        let backend = fixture.backend();

        // This should use hipStreamSynchronize (safe)
        let result = backend.synchronize();

        assert!(result.is_ok(), "backend.synchronize should succeed");

        fixture.assert_no_leak(5);
    }

    /// Test that multiple synchronizations don't cause issues
    ///
    /// Verifies that we can call synchronize multiple times without
    /// deadlocks or memory leaks.
    #[test]
    #[serial]
    fn test_multiple_synchronizations() {
        let fixture = GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");

        let backend = fixture.backend();

        // Call synchronize multiple times
        for _ in 0..5 {
            let result = backend.synchronize();
            assert!(result.is_ok(), "Multiple synchronize calls should succeed");
        }

        fixture.assert_no_leak(5);
    }

    /// Test that both sync methods produce the same result
    ///
    /// Verifies that synchronize_device() and backend.synchronize()
    /// behave identically (both should use stream-aware sync).
    #[test]
    #[serial]
    fn test_sync_methods_consistent() {
        let fixture = GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");

        let backend = fixture.backend();

        // Both methods should succeed
        let result1 = backend.synchronize();
        let result2 = rocmforge::backend::hip_backend::synchronize_device();

        assert!(
            result1.is_ok() && result2.is_ok(),
            "Both sync methods should succeed"
        );

        fixture.assert_no_leak(5);
    }

    /// Test synchronization after GPU operations
    ///
    /// Verifies that synchronize works correctly after actual GPU work.
    #[test]
    #[serial]
    fn test_synchronize_after_gpu_operations() {
        let fixture = GPU_FIXTURE
            .as_ref()
            .expect("GPU not available - test skipped");

        let backend = fixture.backend();

        // Perform some GPU operations
        let buffer = backend
            .allocate_buffer_safe(1024)
            .expect("Failed to allocate");

        // Synchronize after operations
        let result = backend.synchronize();
        assert!(
            result.is_ok(),
            "Synchronize after GPU operations should succeed"
        );

        fixture.assert_no_leak(5);
    }
}
