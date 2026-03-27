//! CPU SIMD feature detection with caching.
//!
//! Detects available CPU instruction sets at startup and caches results:
//! - x86: SSE, SSE2, SSE3, SSSE3, AVX, AVX2, AVX-512, AVX-VNNI, FMA
//! - ARM: NEON, SVE, SVE2
//!
//! Uses std::arch module for runtime detection with OnceLock for caching.

use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Detected CPU SIMD features for kernel dispatch.
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    /// SSE2 support (baseline for 64-bit x86)
    pub has_sse2: bool,
    /// SSE3 support
    pub has_sse3: bool,
    /// SSSE3 support (useful for vector operations)
    pub has_ssse3: bool,
    /// AVX (Advanced Vector Extensions)
    pub has_avx: bool,
    /// AVX2 (256-bit integer operations)
    pub has_avx2: bool,
    /// AVX-512 (512-bit vectors)
    pub has_avx512: bool,
    /// AVX-VNNI (vector neural network instructions)
    pub has_avxvnni: bool,
    /// FMA (Fused Multiply-Add)
    pub has_fma: bool,
    /// ARM NEON (128-bit SIMD)
    pub has_neon: bool,
    /// ARM SVE (Scalable Vector Extension)
    pub has_sve: bool,
    /// ARM SVE2
    pub has_sve2: bool,
    /// Detected kernel preference
    pub kernel: KernelPreference,
}

/// Kernel dispatch preference based on available features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPreference {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// SSE2 kernels (128-bit vectors)
    Sse2,
    /// SSSE3 kernels (horizontal ops, shuffles)
    Ssse3,
    /// AVX kernels (256-bit float vectors)
    Avx,
    /// AVX2 kernels (256-bit integer vectors)
    Avx2,
    /// AVX-VNNI kernels (AVX2 vector neural network instructions)
    AvxVnni,
    /// AVX-512 VNNI kernels (AVX-512 vector neural network instructions)
    Avx512Vnni,
    /// AVX-512 kernels (512-bit vectors)
    Avx512,
    /// ARM NEON kernels (128-bit vectors)
    Neon,
    /// ARM SVE kernels (scalable vectors)
    Sve,
    /// ARM SVE2 kernels
    Sve2,
}

/// Global cached CPU features (detected once at startup).
static CACHED_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

impl CpuFeatures {
    /// Get cached CPU features (detected once at startup).
    ///
    /// This is the preferred method for accessing CPU features in hot paths,
    /// as it avoids repeated CPUID checks.
    ///
    /// # Returns
    ///
    /// Detected features with kernel preference selected.
    #[inline]
    pub fn get() -> &'static Self {
        CACHED_FEATURES.get_or_init(|| Self::detect())
    }

    /// Detect CPU SIMD features at runtime.
    ///
    /// On x86_64: uses CPUID to detect SSE/AVX extensions.
    /// On ARM: uses cfg to determine NEON/SVE support (runtime detection limited).
    ///
    /// # Returns
    ///
    /// Detected features with kernel preference selected.
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
            // Fallback for unsupported architectures
            Self {
                has_sse2: false,
                has_sse3: false,
                has_ssse3: false,
                has_avx: false,
                has_avx2: false,
                has_avx512: false,
                has_fma: false,
                has_neon: false,
                has_sve: false,
                has_sve2: false,
                kernel: KernelPreference::Scalar,
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        let cpuid = unsafe { __cpuid(0x00000001) };

        let has_sse2 = cpuid.edx & (1 << 26) != 0;
        let has_sse3 = cpuid.ecx & (1 << 0) != 0;
        let has_ssse3 = cpuid.ecx & (1 << 9) != 0;
        let has_fma = cpuid.ecx & (1 << 12) != 0;

        // AVX detection requires checking both CPUID feature flag and OS support
        let has_avx = cpuid.ecx & (1 << 28) != 0 && Self::is_xgetbv_enabled();

        // AVX2 detection (CPUID 0x00000007, subleaf 0)
        let cpuid_avx2 = unsafe { __cpuid_count(0x00000007, 0) };
        let has_avx2 = has_avx && (cpuid_avx2.ebx & (1 << 5) != 0);

        // AVX-VNNI variants detection (CPUID 0x00000007, subleaf 0, ECX bits)
        // ECX bit 3: AVX512_VNNI (AVX-512 VNNI) - AMD Zen 4+
        // ECX bit 4: AVX_VNNI (AVX2 VNNI) - Intel Cascade Lake+
        let has_avx2_vnni = has_avx2 && (cpuid_avx2.ecx & (1 << 4) != 0);
        let has_avx512_vnni = has_avx2 && (cpuid_avx2.ebx & (1 << 16) != 0) && (cpuid_avx2.ecx & (1 << 3) != 0);

        // AVX-512 detection (multiple flags)
        // Need: F (foundation), CD (conflict detection), BW (byte/word), DQ, VL (vector length)
        let has_avx512f = cpuid_avx2.ebx & (1 << 16) != 0;
        let has_avx512cd = cpuid_avx2.ebx & (1 << 28) != 0;
        let has_avx512bw = cpuid_avx2.ebx & (1 << 30) != 0;
        let has_avx512vl = cpuid_avx2.ebx & (1 << 31) != 0;
        let has_avx512dq = cpuid_avx2.ebx & (1 << 17) != 0;

        // AVX-512 VNNI requires AVX-512F + VNNI (ECX bit 3)
        let has_avx512 = has_avx512f && has_avx512cd && has_avx512bw && has_avx512vl && has_avx512dq;
        let has_avx512_vnni = has_avx512 && (cpuid_avx2.ecx & (1 << 3) != 0);

        let kernel = Self::select_kernel_x86(
            has_sse2, has_ssse3, has_avx, has_avx2, has_avx512, has_avx2_vnni, has_avx512_vnni,
        );

        Self {
            has_sse2,
            has_sse3,
            has_ssse3,
            has_avx,
            has_avx2,
            has_avx512,
            // Report AVX-VNNI for AVX2 VNNI (Intel)
            // AVX-512 VNNI is tracked separately via has_avx512_vnni in kernel selection
            has_avxvnni: has_avx2_vnni,
            has_fma,
            has_neon: false,
            has_sve: false,
            has_sve2: false,
            kernel,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64() -> Self {
        // NEON is always available on aarch64
        let has_neon = true;

        // SVE/SVE2 require cfg and runtime detection
        let has_sve = cfg!(target_feature = "sve");
        let has_sve2 = cfg!(target_feature = "sve2");

        // For ARM, prefer NEON kernels with SVE/SVE2 fallback
        let kernel = if has_sve2 {
            KernelPreference::Sve2
        } else if has_sve {
            KernelPreference::Sve
        } else {
            KernelPreference::Neon
        };

        Self {
            has_sse2: false,
            has_sse3: false,
            has_ssse3: false,
            has_avx: false,
            has_avx2: false,
            has_avx512: false,
            has_avxvnni: false,
            has_fma: false,
            has_neon,
            has_sve,
            has_sve2,
            kernel,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn select_kernel_x86(
        has_sse2: bool,
        has_ssse3: bool,
        has_avx: bool,
        has_avx2: bool,
        has_avx512: bool,
        has_avx2_vnni: bool,
        has_avx512_vnni: bool,
    ) -> KernelPreference {
        // Priority order for best performance:
        // 1. AVX-512 VNNI (Zen 4, Ice Lake+) - fastest for quantized inference
        // 2. AVX2 VNNI (Cascade Lake+) - good for quantized workloads
        // 3. AVX-512 (512-bit vectors)
        // 4. AVX2 (256-bit integer vectors)
        // 5. AVX (256-bit float vectors)
        // 6. SSSE3 (horizontal ops)
        // 7. SSE2 (baseline)
        // 8. Scalar (fallback)

        if has_avx512_vnni {
            KernelPreference::Avx512Vnni
        } else if has_avx2_vnni {
            KernelPreference::AvxVnni
        } else if has_avx512 {
            KernelPreference::Avx512
        } else if has_avx2 {
            KernelPreference::Avx2
        } else if has_avx {
            KernelPreference::Avx
        } else if has_ssse3 {
            KernelPreference::Ssse3
        } else if has_sse2 {
            KernelPreference::Sse2
        } else {
            KernelPreference::Scalar
        }
    }

    /// Check if the OS supports XGETBV for AVX feature detection.
    ///
    /// AVX requires both CPU support and OS support (to save/restore 256-bit YMM registers).
    #[cfg(target_arch = "x86_64")]
    fn is_xgetbv_enabled() -> bool {
        unsafe {
            let xcr0 = _xgetbv(0);
            // Bit 1: SSE state, Bit 2: AVX state
            (xcr0 & 0x6) == 0x6
        }
    }

    /// Get human-readable description of the detected features.
    pub fn description(&self) -> String {
        let mut features = Vec::new();

        if self.has_avx512 && self.kernel == KernelPreference::Avx512Vnni {
            features.push("AVX-512 VNNI");
        }
        if self.has_avx512 && !matches!(self.kernel, KernelPreference::Avx512Vnni) {
            features.push("AVX-512");
        }
        if self.has_avxvnni {
            features.push("AVX-VNNI");
        }
        if self.has_avx2 {
            features.push("AVX2");
        }
        if self.has_avx {
            features.push("AVX");
        }
        if self.has_ssse3 {
            features.push("SSSE3");
        }
        if self.has_sse3 {
            features.push("SSE3");
        }
        if self.has_sse2 {
            features.push("SSE2");
        }
        if self.has_fma {
            features.push("FMA");
        }
        if self.has_sve2 {
            features.push("SVE2");
        }
        if self.has_sve {
            features.push("SVE");
        }
        if self.has_neon {
            features.push("NEON");
        }

        if features.is_empty() {
            "Scalar".to_string()
        } else {
            features.join(", ")
        }
    }
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_features_does_not_panic() {
        let features = CpuFeatures::detect();
        // Just ensure it runs without panicking
        let _ = features.description();
    }

    #[test]
    fn default_is_detect() {
        let detected = CpuFeatures::detect();
        let default = CpuFeatures::default();
        // Default should be same as detect
        assert_eq!(
            detected.has_avx2, default.has_avx2,
            "default should match detect"
        );
    }

    #[test]
    fn description_is_readable() {
        let features = CpuFeatures::detect();
        let desc = features.description();
        // Should not be empty (at least "Scalar")
        assert!(!desc.is_empty());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_has_sse2() {
        // All x86_64 CPUs must support SSE2
        let features = CpuFeatures::detect();
        assert!(features.has_sse2, "x86_64 requires SSE2 support");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_has_neon() {
        // All aarch64 CPUs must support NEON
        let features = CpuFeatures::detect();
        assert!(features.has_neon, "aarch64 requires NEON support");
    }
}
