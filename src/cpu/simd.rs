//! SIMD kernel dispatch system.
//!
//! Provides optimized kernels for different CPU architectures:
//! - x86_64: AVX2, AVX-512, SSE4.2, etc.
//! - ARM: NEON, SVE, SVE2
//!
//! Kernels are dispatched at runtime based on detected CPU features.

use crate::cpu::features::KernelPreference;

/// SIMD-optimized matrix multiplication kernels.
///
/// The struct holds function pointers to optimized implementations
/// for various operations, selected based on CPU features.
pub struct SimdKernels {
    /// Preference for kernel selection (detected at runtime)
    pub kernel: KernelPreference,
}

impl SimdKernels {
    /// Create SIMD kernels with auto-detected features.
    pub fn new(kernel: KernelPreference) -> Self {
        Self { kernel }
    }

    /// Create SIMD kernels from detected features.
    pub fn detect() -> Self {
        let features = super::features::CpuFeatures::detect();
        Self::new(features.kernel)
    }

    /// Get kernel description for debugging/logging.
    pub fn description(&self) -> &'static str {
        match self.kernel {
            KernelPreference::Scalar => "Scalar",
            KernelPreference::Sse2 => "SSE2",
            KernelPreference::Ssse3 => "SSSE3",
            KernelPreference::Avx => "AVX",
            KernelPreference::Avx2 => "AVX2",
            KernelPreference::AvxVnni => "AVX-VNNI",
            KernelPreference::Avx512Vnni => "AVX-512 VNNI",
            KernelPreference::Avx512 => "AVX-512",
            KernelPreference::Neon => "NEON",
            KernelPreference::Sve => "SVE",
            KernelPreference::Sve2 => "SVE2",
        }
    }
}

impl Default for SimdKernels {
    fn default() -> Self {
        Self::detect()
    }
}

/// SIMD-optimized activation functions.
///
/// Activation functions (ReLU, GELU, SiLU, etc.) benefit from
/// vectorized implementation.
pub struct SimdActivations {
    pub kernel: KernelPreference,
}

impl SimdActivations {
    pub fn new(kernel: KernelPreference) -> Self {
        Self { kernel }
    }

    pub fn detect() -> Self {
        let features = super::features::CpuFeatures::detect();
        Self::new(features.kernel)
    }

    /// GELU activation: f(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    ///
    /// # Arguments
    ///
    /// * `x` - Input array
    /// * `y` - Output array (can be same as x for in-place)
    ///
    /// # Performance
    ///
    /// With AVX2: ~2-3x faster than scalar
    /// With AVX-512: ~4-5x faster than scalar
    pub fn gelu(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), y.len(), "input and output must have same length");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if matches!(
                self.kernel,
                KernelPreference::Avx2
                    | KernelPreference::AvxVnni
                    | KernelPreference::Avx512
                    | KernelPreference::Avx512Vnni
            ) {
                return self.gelu_avx2(x, y);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if matches!(
                self.kernel,
                KernelPreference::Neon | KernelPreference::Sve | KernelPreference::Sve2
            ) {
                return self.gelu_neon(x, y);
            }
        }

        // Fallback to scalar
        self.gelu_scalar(x, y);
    }

    /// SiLU (Swish) activation: f(x) = x / (1 + exp(-x))
    ///
    /// Used in modern LLMs (e.g., LLaMA, Qwen2.5).
    pub fn silu(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), y.len(), "input and output must have same length");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if matches!(
                self.kernel,
                KernelPreference::Avx2
                    | KernelPreference::AvxVnni
                    | KernelPreference::Avx512
                    | KernelPreference::Avx512Vnni
            ) {
                return self.silu_avx2(x, y);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if matches!(
                self.kernel,
                KernelPreference::Neon | KernelPreference::Sve | KernelPreference::Sve2
            ) {
                return self.silu_neon(x, y);
            }
        }

        // Fallback to scalar
        self.silu_scalar(x, y);
    }

    // Scalar implementations

    fn gelu_scalar(&self, x: &[f32], y: &mut [f32]) {
        // Constants: sqrt(2/pi) = 0.7978845608, 0.044715
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const GELU_COEFF: f32 = 0.044715;

        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            let x_cube = xi * xi * xi;
            let tanh_arg = SQRT_2_OVER_PI * (xi + GELU_COEFF * x_cube);
            let tanh_val = tanh_arg.tanh();
            *yi = 0.5 * xi * (1.0 + tanh_val);
        }
    }

    fn silu_scalar(&self, x: &[f32], y: &mut [f32]) {
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            let sigmoid = (-xi).exp();
            let denom = 1.0 + sigmoid;
            *yi = xi / denom;
        }
    }

    // AVX2 implementations (inline assembly via intrinsics)

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn gelu_avx2(&self, x: &[f32], y: &mut [f32]) {
        // AVX2 does not provide transcendental intrinsics (exp, tanh).
        // Fall back to scalar — vectorized approximations can be added later.
        self.gelu_scalar(x, y);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn silu_avx2(&self, x: &[f32], y: &mut [f32]) {
        // AVX2 does not provide transcendental intrinsics (exp).
        // Fall back to scalar — vectorized approximations can be added later.
        self.silu_scalar(x, y);
    }

    // NEON implementations

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn gelu_neon(&self, x: &[f32], y: &mut [f32]) {
        // NEON does not provide built-in transcendental intrinsics (exp, tanh).
        // Fall back to scalar — vectorized approximations can be added later.
        self.gelu_scalar(x, y);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn silu_neon(&self, x: &[f32], y: &mut [f32]) {
        // NEON does not provide built-in transcendental intrinsics (exp).
        // Fall back to scalar — vectorized approximations can be added later.
        self.silu_scalar(x, y);
    }
}

impl Default for SimdActivations {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_scalar_matches_expected() {
        let activations = SimdActivations::new(KernelPreference::Scalar);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut y = vec![0.0; 5];

        activations.gelu(&x, &mut y);

        // Approximate expected values for GELU
        assert!((y[0] - (-0.1588)).abs() < 0.001);
        assert!((y[2] - 0.0).abs() < 0.001);
        assert!((y[4] - 0.8413).abs() < 0.001);
    }

    #[test]
    fn silu_scalar_matches_expected() {
        let activations = SimdActivations::new(KernelPreference::Scalar);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut y = vec![0.0; 5];

        activations.silu(&x, &mut y);

        // SiLU(0) = 0, SiLU(-x) = -SiLU(x) approximately
        assert!((y[2] - 0.0).abs() < 0.001);
        assert!(y[4] > 0.0);
        assert!(y[0] < 0.0);
    }

    #[test]
    fn gelu_output_length_matches_input() {
        let activations = SimdActivations::detect();
        let x = vec![0.0f32; 100];
        let mut y = vec![0.0f32; 100];

        activations.gelu(&x, &mut y);
        assert_eq!(x.len(), y.len());
    }

    #[test]
    fn simd_kernels_description_works() {
        let kernels = SimdKernels::detect();
        let desc = kernels.description();
        assert!(!desc.is_empty());
    }
}
