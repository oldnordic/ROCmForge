//! Common test utilities for GPU testing
//!
//! This module provides shared fixtures for GPU tests that:
//! - Check GPU availability before running
//! - Use a single shared backend (no multiple allocations)
//! - Check for memory leaks after each test

use crate::backend::HipBackend;
use once_cell::sync::Lazy;

/// Global GPU test fixture
///
/// This is initialized ONCE for all tests and shared across them.
/// If GPU is not available, tests will skip gracefully.
///
/// # Memory Safety
///
/// - Uses `HipBackend::new_checked()` which checks GPU availability first
/// - Tracks initial memory state for leak detection
/// - Returns `None` if GPU unavailable (graceful skip)
pub static GPU_FIXTURE: Lazy<Option<GpuTestFixture>> = Lazy::new(|| {
    if !HipBackend::gpu_available() {
        eprintln!("⚠️  WARNING: GPU not available - skipping GPU tests");
        eprintln!("To enable GPU tests, ensure:");
        eprintln!("  1. AMD GPU is present");
        eprintln!("  2. ROCm is installed (check with rocm-smi)");
        eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
        return None;
    }

    match GpuTestFixture::new() {
        Ok(fixture) => {
            eprintln!("✅ GPU Test Fixture initialized");
            eprintln!("   Device: {}", fixture.device_name());
            eprintln!("   Total Memory: {} MB", fixture.total_memory_mb());
            eprintln!("   Free Memory: {} MB", fixture.free_memory_mb());
            eprintln!("   Safe Alloc Limit: {} MB", fixture.safe_alloc_mb());
            Some(fixture)
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to initialize GPU test fixture: {}", e);
            eprintln!("   GPU tests will be skipped");
            None
        }
    }
});

pub struct GpuTestFixture {
    backend: std::sync::Arc<HipBackend>,
    initial_free_mb: usize,
    initial_total_mb: usize,
    device_name: String,
}

impl GpuTestFixture {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let backend = HipBackend::new_checked()?;
        let (free, total) = backend.get_memory_info()?;
        let device_name = backend.device().name.clone();

        Ok(Self {
            backend,
            initial_free_mb: free / 1024 / 1024,
            initial_total_mb: total / 1024 / 1024,
            device_name,
        })
    }

    /// Get the shared backend
    pub fn backend(&self) -> &std::sync::Arc<HipBackend> {
        &self.backend
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get total GPU memory in MB
    pub fn total_memory_mb(&self) -> usize {
        self.initial_total_mb
    }

    /// Get initial free memory in MB
    pub fn free_memory_mb(&self) -> usize {
        self.initial_free_mb
    }

    /// Get safe allocation limit in MB (70% of initial free)
    pub fn safe_alloc_mb(&self) -> usize {
        (self.initial_free_mb * 7) / 10
    }

    /// Check for memory leaks after test
    ///
    /// Tolerance is in percent (e.g., 5 = allow 5% variance)
    /// This accounts for memory fragmentation and driver overhead
    ///
    /// # Panics
    ///
    /// Panics if memory leaked exceeds tolerance
    pub fn assert_no_leak(&self, tolerance_percent: usize) {
        let (free, _total) = self
            .backend
            .get_memory_info()
            .expect("Failed to query GPU memory");

        let free_mb = free / 1024 / 1024;
        let leaked_mb = self.initial_free_mb.saturating_sub(free_mb);
        let tolerance_mb = (self.initial_total_mb * tolerance_percent) / 100;

        if leaked_mb > tolerance_mb {
            panic!(
                "🚨 GPU memory leak detected!\n\
                 Initial free: {} MB\n\
                 Current free: {} MB\n\
                 Leaked: {} MB\n\
                 Tolerance: {} MB ({}%)\n\
                 💡 Tip: Make sure DeviceTensors are dropped before end of test",
                self.initial_free_mb, free_mb, leaked_mb, tolerance_mb, tolerance_percent
            );
        }
    }

    /// Get current memory usage stats
    pub fn memory_stats(&self) -> (usize, usize) {
        match self.backend.get_memory_info() {
            Ok((free, total)) => (free / 1024 / 1024, total / 1024 / 1024),
            Err(_) => (0, 0),
        }
    }
}
