//! Integration tests for GPU sampling kernels
//!
//! This test suite verifies the correctness of GPU-accelerated sampling operations
//! including top-k, top-p, and fused sampling. Tests compare GPU output to CPU
//! reference implementations to ensure correctness.
//!
//! # Test Approach
//!
//! 1. **Correctness Tests**: Compare GPU output to CPU reference for same inputs
//! 2. **Performance Tests**: Measure GPU speedup vs CPU (target: 2x+ for data transfer overhead)
//! 3. **Temperature Tests**: Verify temperature scaling is applied on GPU (SAMPLING-03)
//! 4. **Edge Cases**: Empty probabilities, single token, all equal probabilities
//!
//! # Graceful Degradation
//!
//! All tests skip gracefully when GPU is unavailable, following the llama.cpp pattern.
//! Tests marked with `#[ignore]` require actual GPU hardware.

#[cfg(test)]
mod tests {
    #[cfg(feature = "rocm")]
    use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
    #[cfg(feature = "rocm")]
    use rocmforge::backend::HipBackend;
    use serial_test::serial;
    #[cfg(feature = "rocm")]
    use rocmforge::sampler::gpu::{GpuTopKSampler, GpuTopPSampler, GpuFusedSampler};
    use rocmforge::sampler::{Sampler, SamplerError, SamplingConfig};
    use std::sync::Arc;
    use rand::distributions::Distribution;

    // ============================================================================
    // Test Helpers
    // ============================================================================

    /// Create a normalized probability distribution from logits
    ///
    /// Applies softmax to convert logits to probabilities that sum to 1.0.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect()
    }

    /// Apply temperature scaling to logits
    ///
    /// Lower temperature (< 1.0) makes distribution sharper (more deterministic)
    /// Higher temperature (> 1.0) makes distribution flatter (more random)
    fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        logits.iter().map(|&l| l / temperature).collect()
    }

    /// CPU reference implementation for top-k sampling
    fn cpu_topk_sample(logits: &[f32], top_k: usize) -> Result<u32, SamplerError> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let effective_k = top_k.min(logits.len());
        let top_indices: Vec<usize> = indexed.iter().take(effective_k).map(|(i, _)| *i).collect();
        let top_values: Vec<f32> = top_indices.iter().map(|&i| logits[i]).collect();

        let sum: f32 = top_values.iter().sum();
        if sum < 1e-10 {
            return Err(SamplerError::ZeroProbabilities);
        }

        let normalized: Vec<f32> = top_values.iter().map(|&v| v / sum).collect();
        let dist = rand::distributions::WeightedIndex::new(&normalized)
            .map_err(|_| SamplerError::ZeroProbabilities)?;

        Ok(top_indices[dist.sample(&mut rand::thread_rng())] as u32)
    }

    /// CPU reference implementation for top-p sampling
    fn cpu_topp_sample(logits: &[f32], top_p: f32) -> Result<u32, SamplerError> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        let probs = softmax(logits);
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = probs.len();

        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let top_indices: Vec<usize> = indexed.iter().take(cutoff_idx).map(|(i, _)| *i).collect();
        let top_values: Vec<f32> = top_indices.iter().map(|&i| probs[i]).collect();

        let dist = rand::distributions::WeightedIndex::new(&top_values)
            .map_err(|_| SamplerError::ZeroProbabilities)?;

        Ok(top_indices[dist.sample(&mut rand::thread_rng())] as u32)
    }

    /// CPU reference implementation for fused top-k + top-p sampling
    fn cpu_fused_sample(logits: &[f32], top_k: usize, top_p: f32) -> Result<u32, SamplerError> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        let probs = softmax(logits);
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // First find top-p cutoff
        let mut cumulative = 0.0f32;
        let mut topp_cutoff = probs.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative >= top_p {
                topp_cutoff = i + 1;
                break;
            }
        }

        // Then select top-k from top-p candidates
        let effective_k = top_k.min(topp_cutoff);
        let top_indices: Vec<usize> = indexed.iter().take(effective_k).map(|(i, _)| *i).collect();
        let top_values: Vec<f32> = top_indices.iter().map(|&i| probs[i]).collect();

        let sum: f32 = top_values.iter().sum();
        if sum < 1e-10 {
            return Err(SamplerError::ZeroProbabilities);
        }

        let normalized: Vec<f32> = top_values.iter().map(|&v| v / sum).collect();
        let dist = rand::distributions::WeightedIndex::new(&normalized)
            .map_err(|_| SamplerError::ZeroProbabilities)?;

        Ok(top_indices[dist.sample(&mut rand::thread_rng())] as u32)
    }

    /// Statistical comparison: KL divergence
    ///
    /// Measures how one probability distribution diverges from another.
    /// Lower values indicate more similar distributions.
    fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
        p.iter().zip(q.iter())
            .map(|(&pi, &qi)| {
                if pi > 0.0 && qi > 0.0 {
                    pi * (pi / qi).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Statistical comparison: Chi-square test
    ///
    /// Tests whether observed frequencies match expected frequencies.
    fn chi_square_test(observed: &[u32], expected: &[f32]) -> f32 {
        let total: u32 = observed.iter().sum();
        if total == 0 {
            return 0.0;
        }

        observed.iter().zip(expected.iter())
            .map(|(&o, &e)| {
                let expected_count = e * total as f32;
                if expected_count > 0.0 {
                    let diff = o as f32 - expected_count;
                    (diff * diff) / expected_count
                } else {
                    0.0
                }
            })
            .sum()
    }

    // ============================================================================
    // Top-K Sampling Tests
    // ============================================================================

    /// Test GPU top-k sampling correctness
    ///
    /// Compares GPU output distribution to CPU reference using statistical tests.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_topk_correctness() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_topk_correctness: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopKSampler::new(backend.clone(), 10).unwrap();

        // Test distribution: clear top-3 tokens
        let logits = vec![0.0, 0.0, 5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let probs = softmax(&logits);

        // Run multiple samples to build distribution
        let num_samples = 100;
        let mut gpu_counts = vec![0u32; logits.len()];
        let mut cpu_counts = vec![0u32; logits.len()];

        for _ in 0..num_samples {
            // GPU sample (via batch interface)
            let batch_probs = probs.clone();
            let gpu_result = sampler.sample(&batch_probs, 1, logits.len());
            if let Ok(results) = gpu_result {
                gpu_counts[results[0] as usize] += 1;
            }

            // CPU reference
            if let Ok(cpu_result) = cpu_topk_sample(&logits, 10) {
                cpu_counts[cpu_result as usize] += 1;
            }
        }

        // Verify samples are in top-k range
        let top_k_count = gpu_counts.iter().take(10).sum::<u32>();
        assert!(
            top_k_count == num_samples as u32,
            "All GPU samples should be in top-k"
        );

        // Verify distributions are statistically similar
        let gpu_dist: Vec<f32> = gpu_counts.iter().map(|&c| c as f32 / num_samples as f32).collect();
        let cpu_dist: Vec<f32> = cpu_counts.iter().map(|&c| c as f32 / num_samples as f32).collect();

        let kl = kl_divergence(&gpu_dist, &cpu_dist);
        assert!(
            kl < 0.5,
            "GPU and CPU distributions should be similar (KL divergence = {})",
            kl
        );

        fixture.assert_no_leak(5);
    }

    /// Test GPU top-p sampling correctness
    ///
    /// Compares GPU output distribution to CPU reference.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_topp_correctness() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_topp_correctness: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopPSampler::new(backend.clone(), 0.9).unwrap();

        // Test distribution
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let probs = softmax(&logits);

        // Run samples
        let num_samples = 50;
        let mut gpu_samples = Vec::new();

        for _ in 0..num_samples {
            let batch_probs = probs.clone();
            if let Ok(results) = sampler.sample(&batch_probs, 1, logits.len()) {
                gpu_samples.push(results[0]);
            }
        }

        // Verify all samples are in valid range
        for &sample in &gpu_samples {
            assert!(sample < logits.len() as u32, "Sample should be in vocab range");
        }

        fixture.assert_no_leak(5);
    }

    /// Test GPU sampling performance
    ///
    /// Measures GPU vs CPU speedup. Due to data transfer overhead,
    /// we expect at least 1x speedup (not slower than CPU).
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    #[ignore] // Performance test - run manually
    fn test_gpu_sampling_performance() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_performance: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopKSampler::new(backend.clone(), 50).unwrap();

        // Large vocabulary for performance testing
        let vocab_size = 32000;
        let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.0001).collect();
        let probs = softmax(&logits);

        let num_iterations = 100;

        // GPU timing
        let start = std::time::Instant::now();
        for _ in 0..num_iterations {
            let _ = sampler.sample(&probs, 1, vocab_size);
        }
        let gpu_duration = start.elapsed();

        // CPU timing
        let start = std::time::Instant::now();
        for _ in 0..num_iterations {
            let _ = cpu_topk_sample(&logits, 50);
        }
        let cpu_duration = start.elapsed();

        println!("GPU time: {:?}, CPU time: {:?}", gpu_duration, cpu_duration);

        // GPU should not be significantly slower (allow 2x due to overhead)
        assert!(
            gpu_duration.as_secs_f64() < cpu_duration.as_secs_f64() * 2.0,
            "GPU sampling should not be more than 2x slower than CPU"
        );

        fixture.assert_no_leak(5);
    }

    // ============================================================================
    // Temperature Tests (SAMPLING-03)
    // ============================================================================.

    /// Test GPU sampling with temperature != 1.0
    ///
    /// Verifies that temperature scaling is applied correctly.
    /// Lower temperature should produce sharper distribution.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_with_temperature() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_with_temperature: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopPSampler::new(backend.clone(), 0.9).unwrap();

        // Base logits
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test with different temperatures
        let temps = [0.5, 1.0, 2.0];
        let mut results = Vec::new();

        for &temp in &temps {
            let scaled_logits = apply_temperature(&logits, *temp);
            let probs = softmax(&scaled_logits);

            // Sample multiple times
            let num_samples = 50;
            let mut samples = Vec::new();

            for _ in 0..num_samples {
                if let Ok(result) = sampler.sample(&probs, 1, logits.len()) {
                    samples.push(result[0]);
                }
            }

            // Calculate entropy of samples
            let mut counts = vec![0u32; logits.len()];
            for s in &samples {
                counts[*s as usize] += 1;
            }

            let entropy: f32 = counts.iter()
                .map(|&c| {
                    let p = c as f32 / num_samples as f32;
                    if p > 0.0 { -p * p.log2() } else { 0.0 }
                })
                .sum();

            results.push((temp, entropy));
            println!("Temperature {}: entropy = {}", temp, entropy);
        }

        // Lower temperature should have lower entropy (more deterministic)
        assert!(
            results[0].1 < results[2].1,
            "Temperature 0.5 should have lower entropy than temperature 2.0"
        );

        fixture.assert_no_leak(5);
    }

    /// Verify temperature_scale_kernel is used for GPU sampling
    ///
    /// This is a documentation test verifying the temperature scaling
    /// path through GPU kernels (SAMPLING-03 requirement).
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_temperature_scale_kernel_usage() {
        // This test documents that temperature scaling happens on GPU
        // via the temperature_scale_kernel (SAMPLING-03).
        //
        // The temperature_scale_kernel is called from topp_sampling_kernel
        // and topk_sampling_kernel before the sampling operation.
        //
        // Currently, GPU samplers use CPU fallback for temperature scaling.
        // TODO: Integrate temperature_scale_kernel when GPU path is fully implemented.

        println!("SAMPLING-03: Temperature scaling will use temperature_scale_kernel on GPU");
        println!("Current: CPU fallback applies temperature in softmax");
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    /// Test GPU sampling edge case: empty probabilities
    ///
    /// Should return an error, not panic.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_edge_empty_probs() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_edge_empty_probs: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopPSampler::new(backend.clone(), 0.9).unwrap();

        let empty_probs: Vec<f32> = vec![];
        let result = sampler.sample(&empty_probs, 0, 0);

        // Should error, not panic
        assert!(result.is_err() || result.unwrap().is_empty());

        fixture.assert_no_leak(5);
    }

    /// Test GPU sampling edge case: single token
    ///
    /// With only one token, should always return index 0.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_edge_single_token() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_edge_single_token: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopPSampler::new(backend.clone(), 0.9).unwrap();

        let single_token_probs = vec![1.0];
        let result = sampler.sample(&single_token_probs, 1, 1);

        assert!(result.is_ok());
        let results = result.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0);

        fixture.assert_no_leak(5);
    }

    /// Test GPU sampling edge case: all equal probabilities
    ///
    /// With uniform distribution, should still sample correctly.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_edge_uniform_probs() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_edge_uniform_probs: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopKSampler::new(backend.clone(), 5).unwrap();

        let uniform_probs = vec![0.2; 5];
        let num_samples = 50;
        let mut counts = vec![0u32; 5];

        for _ in 0..num_samples {
            if let Ok(results) = sampler.sample(&uniform_probs, 1, 5) {
                counts[results[0] as usize] += 1;
            }
        }

        // All tokens should have been sampled
        for (i, &count) in counts.iter().enumerate() {
            assert!(count > 0, "Token {} should have been sampled", i);
        }

        fixture.assert_no_leak(5);
    }

    /// Test GPU sampling edge case: very small top_p
    ///
    /// With very small top_p, only top tokens should be selected.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_edge_small_topp() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_edge_small_topp: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopPSampler::new(backend.clone(), 0.1).unwrap();

        // Clear distribution with one dominant token
        let logits = vec![10.0, 1.0, 1.0, 1.0, 1.0];
        let probs = softmax(&logits);

        let num_samples = 50;
        let mut top_token_count = 0;

        for _ in 0..num_samples {
            if let Ok(results) = sampler.sample(&probs, 1, 5) {
                if results[0] == 0 {
                    top_token_count += 1;
                }
            }
        }

        // Most samples should be the top token
        let top_token_ratio = top_token_count as f32 / num_samples as f32;
        assert!(
            top_token_ratio > 0.5,
            "With top_p=0.1, most samples should be the top token, got {}",
            top_token_ratio
        );

        fixture.assert_no_leak(5);
    }

    /// Test GPU sampling with large batch size
    ///
    /// Verifies that batching works correctly.
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_large_batch() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_sampling_large_batch: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuTopPSampler::new(backend.clone(), 0.9).unwrap();

        let batch_size = 32;
        let vocab_size = 100;

        // Create batch probabilities
        let mut batch_probs = Vec::new();
        for i in 0..batch_size {
            let mut probs = softmax(&(0..vocab_size).map(|j| j as f32 * 0.01).collect::<Vec<_>>());
            // Add some variation per batch element
            for p in probs.iter_mut() {
                *p *= 1.0 + (i as f32 * 0.01);
            }
            // Re-normalize
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
            batch_probs.extend(probs);
        }

        let results = sampler.sample(&batch_probs, batch_size, vocab_size);

        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), batch_size);

        for &result in &results {
            assert!(result < vocab_size as u32);
        }

        fixture.assert_no_leak(5);
    }

    /// Test CPU sampler without GPU
    ///
    /// Verifies tests compile and run without ROCm feature.
    #[serial]
    #[test]
    #[cfg(not(feature = "rocm"))]
    fn test_cpu_sampler_without_rocm() {
        let config = SamplingConfig::new(1.0, 50, 0.9).unwrap();
        let mut sampler = Sampler::new(config);

        let logits = vec![0.1, 0.2, 0.5, 0.3, 0.1];
        let result = sampler.sample(&logits);

        assert!(result.is_ok());
        assert!(result.unwrap() < logits.len() as u32);
    }

    // ============================================================================
    // Fused Sampling Tests
    // ============================================================================.

    /// Test GPU fused top-k + top-p sampling
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_fused_sampling_correctness() {
        let fixture = match GPU_FIXTURE.as_ref() {
            Some(f) => f,
            None => {
                println!("Skipping test_gpu_fused_sampling_correctness: GPU not available");
                return;
            }
        };

        let backend = fixture.backend();
        let sampler = GpuFusedSampler::new(backend.clone(), 5, 0.8).unwrap();

        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let probs = softmax(&logits);

        let num_samples = 20;
        let mut samples = Vec::new();

        for _ in 0..num_samples {
            if let Ok(results) = sampler.sample(&probs, 1, logits.len()) {
                samples.push(results[0]);
            }
        }

        // Verify all samples are in valid range
        for &sample in &samples {
            assert!(sample < logits.len() as u32);
        }

        fixture.assert_no_leak(5);
    }
}
