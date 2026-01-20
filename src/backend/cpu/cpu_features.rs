//! Runtime CPU feature detection for SIMD dispatch
//!
//! Provides runtime detection of CPU SIMD capabilities (AVX-512, AVX2, NEON, SSE4.1)
//! using the raw-cpuid crate. Results are cached to avoid repeated CPUID calls.
//!
//! # Example
//!
//! ```rust
//! use rocmforge::backend::cpu::cpu_features::CpuFeatures;
//!
//! let features = CpuFeatures::detect();
//! if features.has_avx512f() {
//!     println!("AVX-512 Foundation available!");
//! } else if features.has_avx2() {
//!     println!("AVX2 available!");
//! }
//! ```

use once_cell::sync::Lazy;
use raw_cpuid::CpuId;
use std::fmt;

/// Cached CPU features detected at startup
static CPU_FEATURES: Lazy<CpuFeatures> = Lazy::new(CpuFeatures::detect);

/// CPU SIMD feature flags
///
/// Represents the SIMD capabilities available on the current CPU.
/// Detection happens once at startup via raw-cpuid and results are cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// AVX-512 Foundation (512-bit SIMD, 16 floats per vector)
    pub avx512f: bool,
    /// AVX2 (256-bit SIMD, 8 floats per vector)
    pub avx2: bool,
    /// SSE4.1 (128-bit SIMD, 4 floats per vector)
    pub sse41: bool,
    /// ARM NEON (128-bit SIMD, 4 floats per vector)
    pub neon: bool,
    /// CPU architecture (x86_64, aarch64, or other)
    pub arch: CpuArch,
}

/// CPU architecture enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    X86_64,
    Aarch64,
    Other,
}

impl fmt::Display for CpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuArch::X86_64 => write!(f, "x86_64"),
            CpuArch::Aarch64 => write!(f, "aarch64"),
            CpuArch::Other => write!(f, "unknown"),
        }
    }
}

impl fmt::Display for CpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CpuFeatures({}", self.arch)?;
        if self.avx512f {
            write!(f, " +AVX512F")?;
        }
        if self.avx2 {
            write!(f, " +AVX2")?;
        }
        if self.sse41 {
            write!(f, " +SSE4.1")?;
        }
        if self.neon {
            write!(f, " +NEON")?;
        }
        write!(f, ")")
    }
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    ///
    /// This function queries the CPU for available SIMD features using
    /// the CPUID instruction on x86_64 or compile-time knowledge on aarch64.
    ///
    /// Results are cached in `CPU_FEATURES` static - call `get()` instead
    /// of running detection multiple times.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let features = CpuFeatures::detect();
    /// println!("CPU: {}", features);
    /// ```
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64()
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::detect_other()
        }
    }

    /// Get cached CPU features (detected once at startup)
    ///
    /// This is the preferred way to access CPU features as it avoids
    /// repeated CPUID calls.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let features = CpuFeatures::get();
    /// if features.has_avx512f() {
    ///     // Use AVX-512 code path
    /// }
    /// ```
    #[inline]
    pub fn get() -> Self {
        *CPU_FEATURES
    }

    /// Detect x86_64 CPU features using CPUID
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        let cpuid = CpuId::new();

        // Check for AVX-512F and AVX2 - both in ExtendedFeatures (leaf 0x07)
        let extended_features = cpuid.get_extended_feature_info();
        let avx512f = extended_features
            .as_ref()
            .map(|info| info.has_avx512f())
            .unwrap_or(false);
        let avx2 = extended_features
            .as_ref()
            .map(|info| info.has_avx2())
            .unwrap_or(false);

        // Check for SSE4.1 - in FeatureInfo (leaf 0x01)
        let sse41 = cpuid
            .get_feature_info()
            .map(|info| info.has_sse41())
            .unwrap_or(false);

        Self {
            avx512f,
            avx2,
            sse41,
            neon: false,
            arch: CpuArch::X86_64,
        }
    }

    /// Detect aarch64 CPU features (NEON is always available on ARMv8+)
    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64() -> Self {
        Self {
            avx512f: false,
            avx2: false,
            sse41: false,
            neon: true, // NEON is mandatory on ARMv8+
            arch: CpuArch::Aarch64,
        }
    }

    /// Fallback for other architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_other() -> Self {
        Self {
            avx512f: false,
            avx2: false,
            sse41: false,
            neon: false,
            arch: CpuArch::Other,
        }
    }

    /// Check if AVX-512 Foundation is available
    ///
    /// AVX-512 provides 512-bit vectors (16 floats per vector) for ~2x
    /// theoretical speedup over AVX2.
    #[inline]
    pub fn has_avx512f(&self) -> bool {
        self.avx512f
    }

    /// Check if AVX2 is available
    ///
    /// AVX2 provides 256-bit vectors (8 floats per vector) for ~2x
    /// theoretical speedup over SSE4.1/scalar.
    #[inline]
    pub fn has_avx2(&self) -> bool {
        self.avx2
    }

    /// Check if SSE4.1 is available
    ///
    /// SSE4.1 provides 128-bit vectors (4 floats per vector) for ~4x
    /// theoretical speedup over scalar.
    #[inline]
    pub fn has_sse41(&self) -> bool {
        self.sse41
    }

    /// Check if NEON is available (ARM only)
    ///
    /// NEON provides 128-bit vectors (4 floats per vector).
    #[inline]
    pub fn has_neon(&self) -> bool {
        self.neon
    }

    /// Get the optimal SIMD width for f32 operations
    ///
    /// Returns the number of f32 values that can be processed in a single
    /// SIMD instruction based on available CPU features.
    ///
    /// # Returns
    ///
    /// * 16 for AVX-512 (512-bit / 32-bit)
    /// * 8 for AVX2 (256-bit / 32-bit)
    /// * 4 for NEON/SSE4.1 (128-bit / 32-bit)
    /// * 1 for scalar fallback
    pub fn optimal_f32_width(&self) -> usize {
        if self.avx512f {
            16
        } else if self.avx2 {
            8
        } else if self.neon || self.sse41 {
            4
        } else {
            1
        }
    }

    /// Log CPU features at startup
    ///
    /// Prints detected CPU features using the tracing infrastructure.
    pub fn log_features(&self) {
        tracing::info!("CPU Architecture: {}", self.arch);
        tracing::info!(
            "SIMD Features: AVX-512F={}, AVX2={}, SSE4.1={}, NEON={}",
            self.avx512f,
            self.avx2,
            self.sse41,
            self.neon
        );
        tracing::info!(
            "Optimal f32 SIMD width: {} elements per vector",
            self.optimal_f32_width()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_features_detect() {
        let features = CpuFeatures::detect();
        // Just verify detection doesn't panic
        match features.arch {
            CpuArch::X86_64 => {
                // On x86_64, at least SSE4.1 should be available on modern CPUs
                // But we don't fail the test if running on very old hardware
            }
            CpuArch::Aarch64 => {
                // NEON should be available on ARMv8+
                assert!(features.neon, "NEON should be available on aarch64");
            }
            CpuArch::Other => {}
        }
    }

    #[test]
    fn test_cpu_features_get() {
        let features1 = CpuFeatures::get();
        let features2 = CpuFeatures::get();
        // Should return the same cached value
        assert_eq!(features1, features2);
    }

    #[test]
    fn test_optimal_f32_width() {
        let features = CpuFeatures::get();
        let width = features.optimal_f32_width();
        // Width should be a power of 2 (1, 4, 8, or 16)
        assert!(width == 1 || width == 4 || width == 8 || width == 16);
    }

    #[test]
    fn test_cpu_features_display() {
        let features = CpuFeatures::get();
        let display = format!("{}", features);
        // Display should contain architecture
        assert!(display.contains("x86_64") || display.contains("aarch64") || display.contains("unknown"));
    }

    #[test]
    fn test_cpu_features_methods() {
        let features = CpuFeatures::get();
        // Just verify methods don't panic
        let _ = features.has_avx512f();
        let _ = features.has_avx2();
        let _ = features.has_sse41();
        let _ = features.has_neon();
    }

    #[test]
    fn test_mock_features_for_testing() {
        // Create mock features for testing dispatch logic
        let mock_avx512 = CpuFeatures {
            avx512f: true,
            avx2: true,
            sse41: true,
            neon: false,
            arch: CpuArch::X86_64,
        };
        assert!(mock_avx512.has_avx512f());
        assert_eq!(mock_avx512.optimal_f32_width(), 16);

        let mock_avx2 = CpuFeatures {
            avx512f: false,
            avx2: true,
            sse41: true,
            neon: false,
            arch: CpuArch::X86_64,
        };
        assert!(!mock_avx2.has_avx512f());
        assert!(mock_avx2.has_avx2());
        assert_eq!(mock_avx2.optimal_f32_width(), 8);

        let mock_scalar = CpuFeatures {
            avx512f: false,
            avx2: false,
            sse41: false,
            neon: false,
            arch: CpuArch::Other,
        };
        assert_eq!(mock_scalar.optimal_f32_width(), 1);
    }
}
