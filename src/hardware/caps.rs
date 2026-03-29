//! CPU capability detection using sysinfo crate.

use super::error::HardwareError;
use sysinfo::System;

/// SIMD feature support flags for kernel selection.
///
/// Used to dispatch to optimized kernels based on available CPU features.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdFeatures {
    /// AVX2 support (256-bit integer vectors)
    pub has_avx2: bool,
    /// AVX-512 support (512-bit vectors)
    pub has_avx512: bool,
    /// NEON support (ARM)
    pub has_neon: bool,
    /// SVE support (ARM scalable vectors)
    pub has_sve: bool,
}

impl SimdFeatures {
    /// Detect SIMD features at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let cpuid_avx2 = unsafe { __cpuid_count(0x00000007, 0) };
            let has_avx = unsafe { _xgetbv(0) } & 0x6 == 0x6;
            let has_avx2 = has_avx && (cpuid_avx2.ebx & (1 << 5) != 0);
            let has_avx512 = has_avx2 && (cpuid_avx2.ebx & (1 << 16) != 0);

            Self {
                has_avx2,
                has_avx512,
                has_neon: false,
                has_sve: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx2: false,
                has_avx512: false,
                has_neon: true, // Always available on aarch64
                has_sve: cfg!(target_feature = "sve"),
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// Get human-readable SIMD feature description.
    pub fn description(&self) -> &'static str {
        // Use the CPU features module for accurate detection
        crate::cpu::features::CpuFeatures::get()
            .description()
            .leak()
    }

    /// Get the kernel preference enum value.
    pub fn kernel_preference(&self) -> crate::cpu::features::KernelPreference {
        crate::cpu::features::CpuFeatures::get().kernel
    }
}

/// Detected CPU hardware capabilities.
///
/// Contains information about CPU topology, cache sizes, and memory.
pub struct CpuCapabilities {
    /// Physical CPU cores (NOT including hyperthreads).
    ///
    /// This is the number of actual physical cores available for compute.
    /// Hyperthreaded logical CPUs are NOT included.
    pub physical_cores: usize,
    /// Logical CPUs (physical_cores * threads_per_core).
    ///
    /// This is the total number of logical CPUs visible to the OS.
    pub logical_cpus: usize,
    /// L3 cache size in bytes.
    ///
    /// Zero if L3 cache size cannot be detected.
    /// Note: sysinfo 0.38+ does not provide cache information.
    /// Consider using hwloc-rs or other crates if cache sizes are needed.
    pub l3_cache_bytes: usize,
    /// L2 cache size in bytes.
    ///
    /// Zero if L2 cache size cannot be detected.
    /// Note: sysinfo 0.38+ does not provide cache information.
    /// Consider using hwloc-rs or other crates if cache sizes are needed.
    pub l2_cache_bytes: usize,
    /// Total system memory in bytes.
    ///
    /// This is the amount of RAM available to the system.
    pub total_memory_bytes: usize,
    /// SIMD feature support for kernel dispatch.
    pub simd: SimdFeatures,
}

impl CpuCapabilities {
    /// Detect hardware capabilities using sysinfo crate.
    ///
    /// Returns detected CPU topology, cache sizes, and memory.
    /// Falls back gracefully if certain information is unavailable.
    ///
    /// # Limitations
    ///
    /// sysinfo 0.38+ does not provide cache size information (L1/L2/L3).
    /// Cache fields will be set to 0, and BatchConfig will use fallback
    /// batch sizes (4MB default). For accurate cache detection, consider
    /// using hwloc-rs or other hardware topology detection crates.
    pub fn detect() -> Result<Self, HardwareError> {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Get physical core count (associated function in sysinfo 0.38+)
        let physical_cores = System::physical_core_count().ok_or_else(|| {
            HardwareError::DetectionFailed("could not detect physical core count".into())
        })?;

        if physical_cores == 0 {
            return Err(HardwareError::DetectionFailed(
                "physical core count is zero".into(),
            ));
        }

        // Get logical CPU count (cpus() returns &[Cpu])
        let logical_cpus = sys.cpus().len();
        if logical_cpus == 0 {
            return Err(HardwareError::DetectionFailed(
                "logical CPU count is zero".into(),
            ));
        }

        // Get total memory
        let total_memory_bytes = sys.total_memory() as usize;
        if total_memory_bytes == 0 {
            return Err(HardwareError::DetectionFailed(
                "total memory is zero".into(),
            ));
        }

        // Cache detection: sysinfo 0.38+ does NOT provide cache_sizes() method
        // on Cpu struct. Set to 0 to trigger fallback behavior in BatchConfig.
        // For production cache-aware batching, consider hwloc-rs crate.
        let (l3_cache_bytes, l2_cache_bytes) = (0, 0);

        // Detect SIMD features for optimized kernel dispatch
        let simd = SimdFeatures::detect();

        Ok(Self {
            physical_cores,
            logical_cpus,
            l3_cache_bytes,
            l2_cache_bytes,
            total_memory_bytes,
            simd,
        })
    }

    /// Get safe number of cores for compute operations.
    ///
    /// Returns physical cores only, ignoring hyperthreading.
    /// Using hyperthreads for pure compute typically does not improve performance
    /// and can increase contention.
    pub fn compute_cores(&self) -> usize {
        self.physical_cores
    }

    /// Check if L3 cache information is available.
    pub fn has_l3_cache(&self) -> bool {
        self.l3_cache_bytes > 0
    }

    /// Get L3 cache size in megabytes for display.
    pub fn l3_cache_mb(&self) -> f64 {
        self.l3_cache_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get total memory in gigabytes for display.
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_sensible_values() {
        let caps = CpuCapabilities::detect().unwrap();

        assert!(caps.physical_cores > 0, "physical_cores must be positive");
        assert!(
            caps.logical_cpus >= caps.physical_cores,
            "logical_cpus must be >= physical_cores"
        );
        assert!(
            caps.total_memory_bytes > 0,
            "total_memory_bytes must be positive"
        );

        // Cache sizes may be zero if undetectable
        // That's acceptable - we have fallback behavior
    }

    #[test]
    fn compute_cores_returns_physical_only() {
        let caps = CpuCapabilities::detect().unwrap();

        // Should always return physical cores, not logical
        assert_eq!(caps.compute_cores(), caps.physical_cores);
    }

    #[test]
    fn has_l3_cache_works() {
        let caps = CpuCapabilities::detect().unwrap();

        // Just ensure it returns a bool without panicking
        let _ = caps.has_l3_cache();
    }

    #[test]
    fn display_methods_no_panic() {
        let caps = CpuCapabilities::detect().unwrap();

        // These display methods should not panic even with zero cache
        let _mb = caps.l3_cache_mb();
        let _gb = caps.total_memory_gb();
    }
}
