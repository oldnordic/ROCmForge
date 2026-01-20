//! GPU top-p (nucleus) sampling implementation
//!
//! Top-p sampling restricts sampling to the smallest set of tokens whose
//! cumulative probability exceeds a threshold.

#![allow(dead_code)]

use crate::backend::hip_backend::{HipBackend, HipBuffer};
use crate::sampler::{SamplerError, SamplerResult};
use rand::Rng;
use std::sync::Arc;

/// GPU sampler for top-p (nucleus) sampling with temperature support
#[derive(Debug, Clone)]
pub struct GpuTopPSampler {
    backend: Arc<HipBackend>,
    top_p: f32,
    temperature: f32,
}

impl GpuTopPSampler {
    /// Create a new GPU top-p sampler with default temperature (1.0)
    pub fn new(backend: Arc<HipBackend>, top_p: f32) -> SamplerResult<Self> {
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplerError::InvalidTopP(top_p));
        }

        Ok(GpuTopPSampler { backend, top_p, temperature: 1.0 })
    }

    /// Set temperature for sampling
    ///
    /// Temperature < 1.0 makes sampling more deterministic (sharper distribution)
    /// Temperature > 1.0 makes sampling more random (flatter distribution)
    /// Temperature = 1.0 is no scaling (default)
    ///
    /// Note: Temperature scaling is applied BEFORE softmax in the GPU pipeline.
    /// For top-p sampling with logits, apply temperature before calling sample().
    pub fn with_temperature(mut self, temperature: f32) -> SamplerResult<Self> {
        if temperature <= 0.0 {
            return Err(SamplerError::InvalidTemperature(temperature));
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Sample from probabilities using top-p filtering on GPU
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
                tracing::debug!("GPU top-p sampling failed, falling back to CPU: {}", e);
                self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
            }
        }
    }

    /// Try to sample using GPU kernels
    ///
    /// Uses 3-kernel pipeline for top-p sampling:
    /// 1. (Optional) temperature_scale_kernel - applies temperature scaling
    /// 2. topp_prefix_sum_kernel - computes CDF
    /// 3. topp_threshold_kernel - finds threshold index
    /// 4. topp_sample_kernel - samples tokens
    fn try_gpu_sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        use super::kernels::{generate_random, temperature_scale_kernel, topp_prefix_sum_kernel, topp_sample_kernel, topp_threshold_kernel};
        use crate::backend::hip_backend::BLOCK_SIZE;

        tracing::debug!("GpuTopPSampler::try_gpu_sample: batch_size={}, vocab_size={}, top_p={}, temperature={}",
            batch_size, vocab_size, self.top_p, self.temperature);

        // Check if all 3 kernels are loaded
        let cache_ref = super::kernels::get_or_init_sampling_cache()
            .map_err(|e| {
                tracing::error!("Failed to get kernel cache: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let cache = cache_ref.lock()
            .map_err(|e| {
                tracing::error!("Failed to lock sampling cache: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        let cache_ref = cache.as_ref()
            .ok_or_else(|| {
                tracing::warn!("Sampling cache not initialized");
                SamplerError::InvalidTopP(0.0)
            })?;

        // Check all 3 kernels for multi-kernel pipeline
        let prefix_sum_kernel = cache_ref.topp_prefix_sum_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("topp_prefix_sum_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopP(0.0)
            })?;
        let threshold_kernel = cache_ref.topp_threshold_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("topp_threshold_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopP(0.0)
            })?;
        let sample_kernel = cache_ref.topp_sample_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("topp_sample_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("All top-p kernels loaded, allocating GPU buffers");

        // Allocate GPU buffers for 3-kernel pipeline
        let total_elements = batch_size * vocab_size;
        let probs_bytes = total_elements * std::mem::size_of::<f32>();
        let prefix_sum_bytes = total_elements * std::mem::size_of::<f32>();
        let threshold_bytes = batch_size * std::mem::size_of::<i32>();
        let random_bytes = batch_size * std::mem::size_of::<f32>();
        let output_bytes = batch_size * std::mem::size_of::<i32>();

        let probs_gpu = HipBuffer::new(probs_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate probs buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let prefix_sum_gpu = HipBuffer::new(prefix_sum_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate prefix_sum buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let threshold_gpu = HipBuffer::new(threshold_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate threshold buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let random_gpu = HipBuffer::new(random_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate random buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let output_gpu = HipBuffer::new(output_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate output buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("GPU buffers allocated, copying data");

        // Copy probabilities to GPU
        probs_gpu.copy_from_host(probabilities)
            .map_err(|e| {
                tracing::error!("Failed to copy probs to GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        // Generate random values on CPU and copy to GPU
        let random_values: Vec<f32> = generate_random(&self.backend, batch_size);
        random_gpu.copy_from_host(&random_values)
            .map_err(|e| {
                tracing::error!("Failed to copy random to GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("Data copied to GPU, launching 3-kernel pipeline");

        // Apply temperature scaling if temperature != 1.0
        let probs_ptr = if self.temperature != 1.0 {
            // Check if temperature scale kernel is available
            let temp_scale_kernel = cache_ref.temperature_scale_kernel.as_ref()
                .ok_or_else(|| {
                    tracing::warn!("temperature_scale_kernel not loaded, falling back to CPU for temperature scaling");
                    SamplerError::InvalidTopP(0.0)
                })?;

            let probs_mut_ptr = probs_gpu.as_mut_ptr() as *mut f32;

            unsafe {
                temperature_scale_kernel(
                    &self.backend,
                    probs_mut_ptr,
                    self.temperature,
                    batch_size as u32,
                    vocab_size as u32,
                ).map_err(|e| {
                    tracing::error!("Failed to launch temperature scale kernel: {:?}", e);
                    SamplerError::InvalidTopP(0.0)
                })?;
            }

            tracing::debug!("Temperature scaling applied: temp={}", self.temperature);
            probs_mut_ptr as *const f32
        } else {
            probs_gpu.as_ptr() as *const f32
        };

        // Kernel 1: Compute prefix sum (CDF)
        let prefix_sum_ptr = prefix_sum_gpu.as_mut_ptr() as *mut f32;

        unsafe {
            // Launch kernel 1: prefix sum
            let grid_dim = (batch_size as u32, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut probs_arg = probs_ptr;
            let mut prefix_sum_arg = prefix_sum_ptr;
            let mut batch_size_arg = batch_size as u32;
            let mut vocab_size_arg = vocab_size as u32;

            let args: &[*mut std::ffi::c_void] = &[
                &mut probs_arg as *mut _ as *mut std::ffi::c_void,
                &mut prefix_sum_arg as *mut _ as *mut std::ffi::c_void,
                &mut batch_size_arg as *mut _ as *mut std::ffi::c_void,
                &mut vocab_size_arg as *mut _ as *mut std::ffi::c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                prefix_sum_kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch prefix sum kernel: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

            // Kernel 2: Find threshold
            let threshold_ptr = threshold_gpu.as_mut_ptr() as *mut i32;

            let mut prefix_sum_arg2 = prefix_sum_ptr;
            let mut threshold_arg = threshold_ptr;
            let mut top_p_arg = self.top_p;

            let args2: &[*mut std::ffi::c_void] = &[
                &mut prefix_sum_arg2 as *mut _ as *mut std::ffi::c_void,
                &mut threshold_arg as *mut _ as *mut std::ffi::c_void,
                &mut top_p_arg as *mut _ as *mut std::ffi::c_void,
                &mut batch_size_arg as *mut _ as *mut std::ffi::c_void,
                &mut vocab_size_arg as *mut _ as *mut std::ffi::c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                threshold_kernel,
                grid_dim,
                block_dim,
                args2,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch threshold kernel: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

            // Kernel 3: Sample tokens
            let random_ptr = random_gpu.as_ptr() as *const f32;
            let output_ptr = output_gpu.as_mut_ptr() as *mut i32;

            let mut threshold_idx_arg = threshold_ptr;
            let mut random_arg = random_ptr;
            let mut output_arg = output_ptr;

            let args3: &[*mut std::ffi::c_void] = &[
                &mut prefix_sum_arg2 as *mut _ as *mut std::ffi::c_void,
                &mut threshold_idx_arg as *mut _ as *mut std::ffi::c_void,
                &mut random_arg as *mut _ as *mut std::ffi::c_void,
                &mut output_arg as *mut _ as *mut std::ffi::c_void,
                &mut batch_size_arg as *mut _ as *mut std::ffi::c_void,
                &mut vocab_size_arg as *mut _ as *mut std::ffi::c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                sample_kernel,
                grid_dim,
                block_dim,
                args3,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch sample kernel: {:?}", e);
                    SamplerError::InvalidTopP(0.0)
            })?;
        }

        tracing::debug!("Kernels launched, synchronizing");

        // Synchronize and copy results back
        self.backend.synchronize()
            .map_err(|e| {
                tracing::error!("Failed to synchronize: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("Synchronized, copying results back");

        let mut results_i32 = vec![0i32; batch_size];
        output_gpu.copy_to_host(&mut results_i32)
            .map_err(|e| {
                tracing::error!("Failed to copy output from GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        // Convert i32 to u32
        let results: Vec<u32> = results_i32.into_iter().map(|v| v as u32).collect();

        tracing::debug!("GPU sampling complete: {:?}", results);

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

            // Find top-p cutoff
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative = 0.0f32;
            let mut cutoff_idx = vocab_size;

            for (i, &(_, p)) in sorted_probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Sample from top-p tokens
            let top_indices: Vec<usize> = sorted_probs
                .iter()
                .take(cutoff_idx)
                .map(|(i, _)| *i)
                .collect();

            let top_values: Vec<f32> = top_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            let dist = rand::distributions::WeightedIndex::new(&top_values)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = top_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}
