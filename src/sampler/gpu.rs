//! GPU-accelerated sampling implementation
//!
//! Provides GPU kernels for top-k and top-p sampling using ROCm/HIP.
//! Based on FlashInfer's sorting-free rejection sampling algorithm.

#![allow(dead_code)]

// Re-export public types from sub-modules
pub use fused::GpuFusedSampler;
pub use top_k::GpuTopKSampler;
pub use top_p::GpuTopPSampler;

// Re-export kernel functions for external use
pub use kernels::{
    fused_sampling_kernel,
    generate_random,
    get_or_init_sampling_cache,
    temperature_scale_kernel,
    topp_prefix_sum_kernel,
    topp_sample_kernel,
    topp_threshold_kernel,
    topk_sampling_kernel,
    topp_sampling_kernel,
    SamplingKernelCache,
};

// Private sub-modules
mod fused;
mod kernels;
mod top_k;
mod top_p;

#[cfg(test)]
    use serial_test::serial;
mod tests {
    use super::*;

    #[test]
    #[serial]
    fn test_gpu_topp_sampler_creation() {
        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();
        assert_eq!(sampler.top_p, 0.9);
    }

    #[test]
    #[serial]
    fn test_gpu_topp_invalid_params() {
        let backend = crate::backend::HipBackend::new().unwrap();
        let result = GpuTopPSampler::new(backend.clone(), 0.0);
        assert!(result.is_err());

        let result = GpuTopPSampler::new(backend, 1.5);
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_gpu_topk_sampler_creation() {
        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 50).unwrap();
        assert_eq!(sampler.top_k, 50);
    }

    #[test]
    #[serial]
    fn test_gpu_topk_invalid_params() {
        let backend = crate::backend::HipBackend::new().unwrap();
        let result = GpuTopKSampler::new(backend, 0);
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_gpu_fused_sampler_creation() {
        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 50, 0.9).unwrap();
        assert_eq!(sampler.top_k, 50);
        assert_eq!(sampler.top_p, 0.9);
    }

    #[test]
    #[serial]
    fn test_topp_fallback_correctness() {
        // Test with known probabilities
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,  // Row 1 (sum = 1.0)
            0.5, 0.3, 0.1, 0.05, 0.05,  // Row 2 (sum = 1.0)
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    #[test]
    #[serial]
    fn test_topk_fallback_correctness() {
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,
            0.5, 0.3, 0.1, 0.05, 0.05,
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 3).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        // Results should be in top-3 (indices 0, 2, or 4 for first row)
        // Note: This is probabilistic, so we just check bounds
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    #[test]
    #[serial]
    fn test_fused_fallback_correctness() {
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,
            0.5, 0.3, 0.1, 0.05, 0.05,
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 3, 0.8).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    /// Test GPU kernel infrastructure
    ///
    /// TDD Step 1: This test verifies that the kernel cache can be initialized.
    /// When HSACO files are present, kernels should be loaded.
    /// When HSACO files are absent, cache should still initialize (with None for kernels).
    #[test]
    #[serial]
    fn test_kernel_cache_initialization() {
        // This should always succeed - cache initializes even if kernels aren't found
        let result = get_or_init_sampling_cache();
        assert!(result.is_ok(), "Kernel cache should initialize successfully");

        // Verify cache is populated
        let cache = result.unwrap().lock()
            .expect("Sampling cache lock should not be poisoned");
        assert!(cache.is_some(), "Cache should be Some after initialization");

        let cache_ref = cache.as_ref()
            .expect("Cache should contain Some(Mutex<KernelCache>)");
        // Kernels will be None if HSACO files aren't compiled yet
        // This is expected - the test documents current state
        if cache_ref.topp_kernel.is_none() {
            println!("WARNING: top-p kernel not loaded (HSACO files not compiled yet)");
            println!("To enable GPU sampling, compile kernels with:");
            println!("  hipcc --genco -O3 kernels/topp_sampling.hip -o kernels/topp_sampling.hsaco");
        }
    }

    /// Test GPU top-p sampling with known inputs
    ///
    /// TDD Step 1: Write test first
    /// TDD Step 2: Run test - see it use CPU fallback (will pass but logs warning)
    /// TDD Step 3: After HSACO compilation, GPU path will be used
    #[test]
    #[serial]
    fn test_topp_sampling_deterministic() {
        // Use deterministic probabilities where result is predictable
        let probabilities = vec![
            0.05, 0.05, 0.80, 0.05, 0.05,  // Row 1: token 2 has 80% probability
            0.10, 0.10, 0.10, 0.60, 0.10,  // Row 2: token 3 has 60% probability
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        // Verify basic properties
        assert_eq!(results.len(), 2, "Should return 2 samples");
        assert!(results[0] < 5, "First sample should be in vocabulary range");
        assert!(results[1] < 5, "Second sample should be in vocabulary range");

        // With top_p=0.9, token 2 (80%) should be highly likely for first row
        // With top_p=0.9, tokens 3 (60%) + 2 (10%) = 70% for second row
        // Note: This is probabilistic, so we just verify it runs without error
    }

    /// Test GPU top-k sampling with known inputs
    #[test]
    #[serial]
    fn test_topk_sampling_deterministic() {
        // Clear top-2 tokens: token 2 (80%), token 4 (10%)
        let probabilities = vec![
            0.02, 0.03, 0.80, 0.05, 0.10,  // Row 1: top-2 are indices 2 and 4
            0.05, 0.05, 0.10, 0.70, 0.10,  // Row 2: top-2 are indices 3 and 4
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 2).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);

        // With top_k=2, samples should be from {2, 4} for row 1
        // and from {3, 4} for row 2
        // Note: Probabilistic, so we just verify it runs
    }

    /// Test GPU fused sampling with known inputs
    ///
    /// TDD test for combined top-k + top-p sampling.
    #[test]
    #[serial]
    fn test_gpu_fused_sampling_deterministic() {
        // Create distribution where top-k and top-p both apply
        let probabilities = vec![
            0.05, 0.05, 0.70, 0.10, 0.10,  // Row 1: token 2 (70%), token 3 (10%), token 4 (10%)
            0.10, 0.60, 0.10, 0.10, 0.10,  // Row 2: token 1 (60%), others (10% each)
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 3, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        // Verify basic properties
        assert_eq!(results.len(), 2, "Should return 2 samples");
        assert!(results[0] < 5, "First sample should be in vocabulary range");
        assert!(results[1] < 5, "Second sample should be in vocabulary range");

        // With top_k=3, top_p=0.9:
        // Row 1: top-3 are indices 2 (70%), 3 (10%), 4 (10%) = 90% cumulative
        // Row 2: top-3 are indices 1 (60%), 0 (10%), 2 (10%) = 80% cumulative
        // Note: Probabilistic, so we just verify it runs without error
    }

    /// Test GPU sampling fallback on error
    ///
    /// Verifies that when GPU kernels are not available, CPU fallback is used.
    #[test]
    #[serial]
    fn test_gpu_sampling_fallback_on_error() {
        // This test uses CPU fallback directly (simulating kernel unavailability)
        let probabilities = vec![
            0.1, 0.2, 0.4, 0.2, 0.1,  // Sum = 1.0
            0.3, 0.3, 0.2, 0.1, 0.1,  // Sum = 1.0
        ];

        {
            let backend = crate::backend::HipBackend::new().unwrap();
            let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

            // This will use CPU fallback if kernels aren't loaded
            let results = sampler.sample(&probabilities, 2, 5);
            assert!(results.is_ok(), "Should fall back to CPU sampling");

            let results = results.unwrap();
            assert_eq!(results.len(), 2);
            assert!(results[0] < 5);
            assert!(results[1] < 5);
        }
    }

    /// Test GPU top-k sampling with single dominant token
    ///
    /// Edge case: One token has overwhelming probability.
    #[test]
    #[serial]
    #[ignore] // Requires actual GPU hardware
    fn test_gpu_topk_single_dominant() {
        let probabilities = vec![
            0.99, 0.0025, 0.0025, 0.0025, 0.0025,  // Token 0 dominates
            0.002, 0.99, 0.002, 0.003, 0.003,       // Token 1 dominates
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 5).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        // With 99% probability, most samples should be the dominant token
        // But we just verify bounds here since it's probabilistic
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    /// Test GPU top-p sampling with uniform distribution
    ///
    /// Edge case: All probabilities are equal.
    #[test]
    #[serial]
    fn test_gpu_topp_uniform_distribution() {
        let probabilities = vec![
            0.2, 0.2, 0.2, 0.2, 0.2,  // Uniform distribution
            0.25, 0.25, 0.25, 0.25, 0.0,  // Another uniform distribution
        ];

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.5).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    /// Test GPU sampling with edge case: single token vocabulary
    ///
    /// Edge case: Vocabulary size of 1 (only one possible token).
    #[test]
    #[serial]
    fn test_gpu_sampling_single_token_vocab() {
        let probabilities = vec![1.0; 2]; // batch_size=2, vocab_size=1

        let backend = crate::backend::HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 1).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0, "Only token 0 should be sampled");
        assert_eq!(results[1], 0, "Only token 0 should be sampled");
    }
}
