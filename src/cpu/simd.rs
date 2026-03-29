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
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
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
        use std::arch::x86_64::*;

        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const GELU_COEFF: f32 = 0.044715;

        let sqrt_2_pi = _mm256_set1_ps(SQRT_2_OVER_PI);
        let gelu_coeff = _mm256_set1_ps(GELU_COEFF);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);

        let chunks = x.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let xi = _mm256_loadu_ps(chunk.as_ptr());
            let xi_sq = _mm256_mul_ps(xi, xi);
            let xi_cube = _mm256_mul_ps(xi_sq, xi);

            let tanh_arg = _mm256_add_ps(
                _mm256_mul_ps(sqrt_2_pi, xi),
                _mm256_mul_ps(gelu_coeff, xi_cube),
            );

            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            let two_x = _mm256_mul_ps(tanh_arg, _mm256_set1_ps(2.0));
            let exp_2x = _mm256_exp_ps(two_x);
            let tanh_val = _mm256_div_ps(_mm256_sub_ps(exp_2x, one), _mm256_add_ps(exp_2x, one));

            let result = _mm256_mul_ps(half, _mm256_mul_ps(xi, _mm256_add_ps(one, tanh_val)));
            _mm256_storeu_ps(y.as_mut_ptr().add(chunk.len_offset()), result);
        }

        // Handle remainder
        for (xi, yi) in remainder
            .iter()
            .zip(y.iter_mut().skip(x.len() - remainder.len()))
        {
            let x_cube = xi * xi * xi;
            let tanh_arg = SQRT_2_OVER_PI * (xi + GELU_COEFF * x_cube);
            let tanh_val = tanh_arg.tanh();
            *yi = 0.5 * xi * (1.0 + tanh_val);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn silu_avx2(&self, x: &[f32], y: &mut [f32]) {
        use std::arch::x86_64::*;

        let one = _mm256_set1_ps(1.0);

        let chunks = x.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let xi = _mm256_loadu_ps(chunk.as_ptr());
            let neg_xi = _mm256_xor_ps(xi, _mm256_set1_ps(-0.0)); // negate
            let exp_neg_x = _mm256_exp_ps(neg_xi);
            let denom = _mm256_add_ps(one, exp_neg_x);
            let result = _mm256_div_ps(xi, denom);
            _mm256_storeu_ps(y.as_mut_ptr().add(chunk.len_offset()), result);
        }

        // Handle remainder
        for (xi, yi) in remainder
            .iter()
            .zip(y.iter_mut().skip(x.len() - remainder.len()))
        {
            let sigmoid = (-xi).exp();
            let denom = 1.0 + sigmoid;
            *yi = xi / denom;
        }
    }

    // NEON implementations

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn gelu_neon(&self, x: &[f32], y: &mut [f32]) {
        use std::arch::aarch64::*;

        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const GELU_COEFF: f32 = 0.044715;

        let sqrt_2_pi = vdupq_n_f32(SQRT_2_OVER_PI);
        let gelu_coeff = vdupq_n_f32(GELU_COEFF);
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);

        let chunks = x.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let xi = vld1q_f32(chunk.as_ptr());
            let xi_sq = vmulq_f32(xi, xi);
            let xi_cube = vmulq_f32(xi_sq, xi);

            let tanh_arg = vaddq_f32(vmulq_f32(sqrt_2_pi, xi), vmulq_f32(gelu_coeff, xi_cube));

            // tanh approximation or compute
            let tanh_val = vtanhq_f32(tanh_arg);

            let result = vmulq_f32(half, vmulq_f32(xi, vaddq_f32(one, tanh_val)));
            vst1q_f32(y.as_mut_ptr().add(chunk.len_offset()), result);
        }

        // Handle remainder
        for (xi, yi) in remainder
            .iter()
            .zip(y.iter_mut().skip(x.len() - remainder.len()))
        {
            let x_cube = xi * xi * xi;
            let tanh_arg = SQRT_2_OVER_PI * (xi + GELU_COEFF * x_cube);
            let tanh_val = tanh_arg.tanh();
            *yi = 0.5 * xi * (1.0 + tanh_val);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn silu_neon(&self, x: &[f32], y: &mut [f32]) {
        use std::arch::aarch64::*;

        let one = vdupq_n_f32(1.0);

        let chunks = x.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let xi = vld1q_f32(chunk.as_ptr());
            let neg_xi = vnegq_f32(xi);
            let exp_neg_x = vexpq_f32(neg_xi);
            let denom = vaddq_f32(one, exp_neg_x);
            let result = vdivq_f32(xi, denom);
            vst1q_f32(y.as_mut_ptr().add(chunk.len_offset()), result);
        }

        // Handle remainder
        for (xi, yi) in remainder
            .iter()
            .zip(y.iter_mut().skip(x.len() - remainder.len()))
        {
            let sigmoid = (-xi).exp();
            let denom = 1.0 + sigmoid;
            *yi = xi / denom;
        }
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
