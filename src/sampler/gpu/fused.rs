//! GPU fused top-k + top-p sampling implementation
//!
//! Fused sampling combines both top-k and top-p filtering in a single
//! kernel launch for improved efficiency.

#![allow(dead_code)]

use crate::backend::hip_backend::{HipBackend, HipBuffer};
use crate::sampler::{SamplerError, SamplerResult};
use rand::Rng;
use std::sync::Arc;

/// GPU sampler for fused top-k + top-p sampling
#[derive(Debug, Clone)]
pub struct GpuFusedSampler {
    backend: Arc<HipBackend>,
    top_k: usize,
    top_p: f32,
}

impl GpuFusedSampler {
    /// Create a new GPU fused sampler
    pub fn new(backend: Arc<HipBackend>, top_k: usize, top_p: f32) -> SamplerResult<Self> {
        if top_k == 0 {
            return Err(SamplerError::InvalidTopK(top_k));
        }
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplerError::InvalidTopP(top_p));
        }

        Ok(GpuFusedSampler { backend, top_k, top_p })
    }

    /// Sample using fused top-k + top-p on GPU
    pub fn sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        // TDD: Try GPU path first, fall back to CPU if kernels not available or on error
        match self.try_gpu_sample(probabilities, batch_size, vocab_size) {
            Ok(results) => Ok(results),
            Err(e) => {
                tracing::debug!("GPU fused sampling failed, falling back to CPU: {}", e);
                self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
            }
        }
    }

    /// Try to sample using GPU kernels
    ///
    /// Uses the fused top-k + top-p sampling kernel which combines both
    /// filtering operations in a single kernel launch for efficiency.
    fn try_gpu_sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        use super::kernels::{fused_sampling_kernel, generate_random};
        use crate::backend::hip_backend::BLOCK_SIZE;

        tracing::debug!("GpuFusedSampler::try_gpu_sample: batch_size={}, vocab_size={}, top_k={}, top_p={}",
            batch_size, vocab_size, self.top_k, self.top_p);

        // Check if kernel is loaded
        let cache_ref = super::kernels::get_or_init_sampling_cache()
            .map_err(|e| {
                tracing::error!("Failed to get kernel cache: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let cache = cache_ref.lock()
            .map_err(|e| {
                tracing::error!("Failed to lock sampling cache: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        let cache_ref = cache.as_ref()
            .ok_or_else(|| {
                tracing::warn!("Sampling cache not initialized");
                SamplerError::InvalidTopK(0)
            })?;

        let fused_kernel = cache_ref.fused_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("fused_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("fused kernel loaded, allocating GPU buffers");

        // Allocate GPU buffers
        let total_elements = batch_size * vocab_size;
        let probs_bytes = total_elements * std::mem::size_of::<f32>();
        let random_bytes = batch_size * std::mem::size_of::<f32>();
        let output_bytes = batch_size * std::mem::size_of::<u32>();

        let probs_gpu = HipBuffer::new(probs_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate probs buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let random_gpu = HipBuffer::new(random_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate random buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let output_gpu = HipBuffer::new(output_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate output buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("GPU buffers allocated, copying data");

        // Copy probabilities to GPU
        probs_gpu.copy_from_host(probabilities)
            .map_err(|e| {
                tracing::error!("Failed to copy probs to GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        // Generate random values on CPU and copy to GPU
        let random_values: Vec<f32> = generate_random(&self.backend, batch_size);
        random_gpu.copy_from_host(&random_values)
            .map_err(|e| {
                tracing::error!("Failed to copy random to GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("Data copied to GPU, launching fused kernel");

        // Launch fused kernel
        let probs_ptr = probs_gpu.as_ptr() as *const f32;
        let random_ptr = random_gpu.as_ptr() as *const f32;
        let output_ptr = output_gpu.as_mut_ptr() as *mut u32;

        unsafe {
            // Launch kernel directly using backend
            let grid_dim = (batch_size as u32, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut probs_arg = probs_ptr;
            let mut random_values_arg = random_ptr;
            let mut output_arg = output_ptr;
            let mut top_k_arg = self.top_k as u32;
            let mut top_p_arg = self.top_p;
            let mut batch_size_arg = batch_size as u32;
            let mut vocab_size_arg = vocab_size as u32;

            let args: &[*mut std::ffi::c_void] = &[
                &mut probs_arg as *mut _ as *mut std::ffi::c_void,
                &mut random_values_arg as *mut _ as *mut std::ffi::c_void,
                &mut output_arg as *mut _ as *mut std::ffi::c_void,
                &mut top_k_arg as *mut _ as *mut std::ffi::c_void,
                &mut top_p_arg as *mut _ as *mut std::ffi::c_void,
                &mut batch_size_arg as *mut _ as *mut std::ffi::c_void,
                &mut vocab_size_arg as *mut _ as *mut std::ffi::c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                fused_kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch fused kernel: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        }

        tracing::debug!("Kernel launched, synchronizing");

        // Synchronize and copy results back
        self.backend.synchronize()
            .map_err(|e| {
                tracing::error!("Failed to synchronize: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("Synchronized, copying results back");

        let mut results = vec![0u32; batch_size];
        output_gpu.copy_to_host(&mut results)
            .map_err(|e| {
                tracing::error!("Failed to copy output from GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("GPU fused sampling complete: {:?}", results);

        Ok(results)
    }

    /// CPU fallback implementation for testing
    fn sample_cpu_fallback(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        let mut results = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..batch_size {
            let row_offset = batch_idx * vocab_size;
            let row_probs = &probabilities[row_offset..row_offset + vocab_size];

            // First apply top-p to get candidate set
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative = 0.0f32;
            let mut topp_cutoff = vocab_size;

            for (i, &(_, p)) in sorted_probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    topp_cutoff = i + 1;
                    break;
                }
            }

            // Among top-p tokens, select top-k
            let effective_k = self.top_k.min(topp_cutoff);
            let topk_indices: Vec<usize> = sorted_probs
                .iter()
                .take(effective_k)
                .map(|(i, _)| *i)
                .collect();

            let topk_values: Vec<f32> = topk_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            // Renormalize and sample
            let sum: f32 = topk_values.iter().sum();
            if sum < 1e-10f32 {
                return Err(SamplerError::ZeroProbabilities);
            }

            let normalized: Vec<f32> = topk_values.iter().map(|&v| v / sum).collect();

            let dist = rand::distributions::WeightedIndex::new(&normalized)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = topk_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}
