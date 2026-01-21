//! GPU backend implementation for attention computation using ROCm/HIP

use crate::attention::{compute, AttentionError, AttentionResult};
use crate::backend::{DeviceTensor, HipBackend, HipBlasHandle, HipBuffer};
use crate::loader::mmap_loader::TensorShape;
use crate::tensor::matmul::matmul_f32;

/// GPU backend for attention computation
pub struct GpuBackend {
    _handle: HipBlasHandle,
}

impl GpuBackend {
    /// Create new GPU backend
    pub fn new() -> Result<Self, AttentionError> {
        let handle = HipBlasHandle::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP BLAS handle: {}", e))
        })?;

        Ok(GpuBackend { _handle: handle })
    }

    /// Compute attention using GPU implementation
    pub fn forward(
        dim: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        dropout: Option<f32>,
    ) -> AttentionResult<Vec<f32>> {
        let batch_size = q.len() / (dim * dim);
        let seq_len = dim;

        if q.len() != k.len() || q.len() != v.len() {
            return Err(AttentionError::ShapeMismatch(
                "Q, K, V must have same shape".to_string(),
            ));
        }

        let scale = 1.0 / (dim as f32).sqrt();

        // Create HIP backend for GPU operations
        let backend = HipBackend::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
        })?;

        // Create HIP BLAS handle for GPU operations
        let handle = HipBlasHandle::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP BLAS handle: {}", e))
        })?;

        // CRITICAL: Associate hipBLAS handle with backend's stream
        //
        // Without this, hipBLAS uses the default stream while custom kernels use
        // the backend's custom stream. This causes synchronization issues and hangs.
        // See docs/STREAM_SYNCHRONIZATION.md for details.
        handle.set_stream(backend.stream().as_ptr()).map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to set hipBLAS stream: {}", e))
        })?;

        // Allocate GPU buffers
        let q_gpu = HipBuffer::new(std::mem::size_of_val(q)).map_err(|e| {
            AttentionError::MemoryAllocation(format!("Failed to allocate Q buffer: {}", e))
        })?;
        let k_gpu = HipBuffer::new(std::mem::size_of_val(k)).map_err(|e| {
            AttentionError::MemoryAllocation(format!("Failed to allocate K buffer: {}", e))
        })?;
        let v_gpu = HipBuffer::new(std::mem::size_of_val(v)).map_err(|e| {
            AttentionError::MemoryAllocation(format!("Failed to allocate V buffer: {}", e))
        })?;

        // Copy data to GPU
        q_gpu
            .copy_from_host(q)
            .map_err(|e| AttentionError::MemoryCopy(format!("Failed to copy Q to GPU: {}", e)))?;
        k_gpu
            .copy_from_host(k)
            .map_err(|e| AttentionError::MemoryCopy(format!("Failed to copy K to GPU: {}", e)))?;
        v_gpu
            .copy_from_host(v)
            .map_err(|e| AttentionError::MemoryCopy(format!("Failed to copy V to GPU: {}", e)))?;

        // Compute QK^T on GPU with fused scaling
        let mut scores = vec![0.0f32; batch_size * seq_len * seq_len];
        {
            let scores_gpu =
                HipBuffer::new(batch_size * seq_len * seq_len * std::mem::size_of::<f32>())
                    .map_err(|e| {
                        AttentionError::MemoryAllocation(format!(
                            "Failed to allocate scores buffer: {}",
                            e
                        ))
                    })?;

            // QK^T matrix multiplication: (batch_size, seq_len, dim) x (batch_size, dim, seq_len) -> (batch_size, seq_len, seq_len)
            for b in 0..batch_size {
                let q_offset = b * seq_len * dim;
                let k_offset = b * dim * seq_len;
                let scores_offset = b * seq_len * seq_len;

                // Create sub-buffers for this batch
                let q_batch =
                    HipBuffer::new(seq_len * dim * std::mem::size_of::<f32>()).map_err(|e| {
                        AttentionError::MemoryAllocation(format!(
                            "Failed to create Q batch buffer: {}",
                            e
                        ))
                    })?;
                let k_batch =
                    HipBuffer::new(dim * seq_len * std::mem::size_of::<f32>()).map_err(|e| {
                        AttentionError::MemoryAllocation(format!(
                            "Failed to create K batch buffer: {}",
                            e
                        ))
                    })?;

                // Copy batch data to GPU
                q_batch
                    .copy_from_host(&q[q_offset..q_offset + seq_len * dim])
                    .map_err(|e| {
                        AttentionError::MemoryCopy(format!("Failed to copy Q batch to GPU: {}", e))
                    })?;
                k_batch
                    .copy_from_host(&k[k_offset..k_offset + dim * seq_len])
                    .map_err(|e| {
                        AttentionError::MemoryCopy(format!("Failed to copy K batch to GPU: {}", e))
                    })?;

                let scores_batch = matmul_f32(
                    &backend,
                    &handle,
                    &q_batch,
                    &k_batch,
                    seq_len as i32,
                    seq_len as i32,
                    dim as i32,
                )
                .map_err(|e| AttentionError::GpuOperation(format!("GPU matmul failed: {}", e)))?;

                // Copy batch result to correct location in scores buffer
                let mut batch_scores = vec![0.0f32; seq_len * seq_len];
                backend.copy_from_device_safe(&scores_batch, &mut batch_scores).map_err(|e| {
                    AttentionError::MemoryCopy(format!("Failed to copy batch scores: {}", e))
                })?;

                // Copy to CPU scores array
                for (i, &val) in batch_scores.iter().enumerate() {
                    scores[scores_offset + i] = val;
                }
            }

            // Apply scaling on GPU
            unsafe {
                crate::attention::kernels::scale_gpu_kernel(
                    scores_gpu.as_ptr() as *mut f32,
                    scale,
                    batch_size as u32,
                    seq_len as u32,
                );
            }

            // CRITICAL: Synchronize after kernel launch before buffer goes out of scope
            // Without this sync, the kernel may still be executing when scores_gpu is dropped,
            // causing use-after-free and race conditions.
            if let Err(e) = crate::backend::hip_backend::synchronize_device() {
                return Err(AttentionError::GpuOperation(format!(
                    "GPU synchronization failed after scale kernel: {}",
                    e
                )));
            }
        }

        // Apply mask if provided
        if let Some(mask_data) = mask {
            if mask_data.len() != batch_size * seq_len * seq_len {
                return Err(AttentionError::ShapeMismatch(
                    "Mask shape mismatch".to_string(),
                ));
            }

            // Apply masking on GPU
            {
                let backend = HipBackend::new().map_err(|e| {
                    AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
                })?;

                let scores_gpu = HipBuffer::new(scores.len() * std::mem::size_of::<f32>())
                    .map_err(|e| {
                        AttentionError::MemoryAllocation(format!(
                            "Failed to allocate scores buffer for masking: {}",
                            e
                        ))
                    })?;
                let mask_gpu = HipBuffer::new(std::mem::size_of_val(mask_data)).map_err(|e| {
                    AttentionError::MemoryAllocation(format!(
                        "Failed to allocate mask buffer: {}",
                        e
                    ))
                })?;

                scores_gpu.copy_from_host(&scores).map_err(|e| {
                    AttentionError::MemoryCopy(format!(
                        "Failed to copy scores to GPU for masking: {}",
                        e
                    ))
                })?;
                mask_gpu.copy_from_host(mask_data).map_err(|e| {
                    AttentionError::MemoryCopy(format!("Failed to copy mask to GPU: {}", e))
                })?;

                // Launch mask kernel
                unsafe {
                    crate::attention::kernels::mask_gpu_kernel(
                        scores_gpu.as_ptr() as *mut f32,
                        mask_gpu.as_ptr() as *const f32,
                        batch_size as u32,
                        seq_len as u32,
                    );
                }

                // CRITICAL: Synchronize after kernel launch before using results
                // Ensures kernel completes before copying data back to host.
                if let Err(e) = crate::backend::hip_backend::synchronize_device() {
                    return Err(AttentionError::GpuOperation(format!(
                        "GPU synchronization failed after mask kernel: {}",
                        e
                    )));
                }

                backend.copy_from_device_safe(&scores_gpu, &mut scores).map_err(|e| {
                    AttentionError::MemoryCopy(format!(
                        "Failed to copy masked scores to host: {}",
                        e
                    ))
                })?;
            }
        }

        // Apply softmax row-wise on GPU
        {
            let backend = HipBackend::new().map_err(|e| {
                AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
            })?;

            let scores_gpu =
                HipBuffer::new(scores.len() * std::mem::size_of::<f32>()).map_err(|e| {
                    AttentionError::MemoryAllocation(format!(
                        "Failed to allocate scores buffer for softmax: {}",
                        e
                    ))
                })?;
            scores_gpu.copy_from_host(&scores).map_err(|e| {
                AttentionError::MemoryCopy(format!(
                    "Failed to copy scores to GPU for softmax: {}",
                    e
                ))
            })?;

            // Launch softmax kernel
            unsafe {
                crate::attention::kernels::softmax_gpu_kernel(
                    scores_gpu.as_ptr() as *mut f32,
                    batch_size as u32,
                    seq_len as u32,
                );
            }

            // CRITICAL: Synchronize after kernel launch before using results
            // Ensures kernel completes before copying data back to host.
            if let Err(e) = crate::backend::hip_backend::synchronize_device() {
                return Err(AttentionError::GpuOperation(format!(
                    "GPU synchronization failed after softmax kernel: {}",
                    e
                )));
            }

            backend.copy_from_device_safe(&scores_gpu, &mut scores).map_err(|e| {
                AttentionError::MemoryCopy(format!("Failed to copy softmax results to host: {}", e))
            })?;
        }

        // Apply dropout if provided
        if let Some(dropout_prob) = dropout {
            compute::apply_dropout(&mut scores, dropout_prob, 42);
        }

        // Compute final output: scores * V on GPU
        let mut output = vec![0.0f32; batch_size * seq_len * dim];
        {
            let scores_gpu =
                HipBuffer::new(scores.len() * std::mem::size_of::<f32>()).map_err(|e| {
                    AttentionError::MemoryAllocation(format!(
                        "Failed to allocate scores buffer: {}",
                        e
                    ))
                })?;
            scores_gpu.copy_from_host(&scores).map_err(|e| {
                AttentionError::MemoryCopy(format!("Failed to copy scores to GPU: {}", e))
            })?;
            let output_gpu = HipBuffer::new(
                batch_size * seq_len * dim * std::mem::size_of::<f32>(),
            )
            .map_err(|e| {
                AttentionError::MemoryAllocation(format!("Failed to allocate output buffer: {}", e))
            })?;

            // scores * V matrix multiplication: (batch_size, seq_len, seq_len) x (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
            for b in 0..batch_size {
                let scores_offset = b * seq_len * seq_len;
                let v_offset = b * seq_len * dim;
                let output_offset = b * seq_len * dim;

                // Create sub-buffers for this batch
                let scores_batch = HipBuffer::new(seq_len * seq_len * std::mem::size_of::<f32>())
                    .map_err(|e| {
                    AttentionError::MemoryAllocation(format!(
                        "Failed to create scores batch buffer: {}",
                        e
                    ))
                })?;
                let v_batch =
                    HipBuffer::new(seq_len * dim * std::mem::size_of::<f32>()).map_err(|e| {
                        AttentionError::MemoryAllocation(format!(
                            "Failed to create V batch buffer: {}",
                            e
                        ))
                    })?;

                // Copy batch data to GPU
                scores_batch
                    .copy_from_host(&scores[scores_offset..scores_offset + seq_len * seq_len])
                    .map_err(|e| {
                        AttentionError::MemoryCopy(format!(
                            "Failed to copy scores batch to GPU: {}",
                            e
                        ))
                    })?;
                v_batch
                    .copy_from_host(&v[v_offset..v_offset + seq_len * dim])
                    .map_err(|e| {
                        AttentionError::MemoryCopy(format!("Failed to copy V batch to GPU: {}", e))
                    })?;

                let output_batch = matmul_f32(
                    &backend,
                    &handle,
                    &scores_batch,
                    &v_batch,
                    seq_len as i32,
                    dim as i32,
                    seq_len as i32,
                )
                .map_err(|e| AttentionError::GpuOperation(format!("GPU matmul failed: {}", e)))?;

                // Copy batch result to correct location in output buffer
                let mut batch_output = vec![0.0f32; seq_len * dim];
                backend.copy_from_device_safe(&output_batch, &mut batch_output).map_err(|e| {
                    AttentionError::MemoryCopy(format!("Failed to copy batch output: {}", e))
                })?;

                // Copy to CPU output array
                for (i, &val) in batch_output.iter().enumerate() {
                    output[output_offset + i] = val;
                }
            }

            // Copy output back to CPU
            backend.copy_from_device_safe(&output_gpu, &mut output).map_err(|e| {
                AttentionError::MemoryCopy(format!("Failed to copy output to host: {}", e))
            })?;
        }

        Ok(output)
    }

    /// GPU forward pass using DeviceTensor inputs for zero-copy computation
    pub fn forward_device(
        dim: usize,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
        dropout: Option<f32>,
    ) -> AttentionResult<DeviceTensor> {
        // For now, fallback to host-based computation using DeviceTensor data
        // This establishes the integration pattern before optimizing for full GPU operation
        let backend = HipBackend::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
        })?;

        let mut q_host = vec![0.0f32; q.len()];
        backend.copy_from_device_safe(q.buffer(), &mut q_host)
            .map_err(|e| AttentionError::MemoryCopy(format!("Failed to copy Q to host: {}", e)))?;
        let mut k_host = vec![0.0f32; k.len()];
        backend.copy_from_device_safe(k.buffer(), &mut k_host)
            .map_err(|e| AttentionError::MemoryCopy(format!("Failed to copy K to host: {}", e)))?;
        let mut v_host = vec![0.0f32; v.len()];
        backend.copy_from_device_safe(v.buffer(), &mut v_host)
            .map_err(|e| AttentionError::MemoryCopy(format!("Failed to copy V to host: {}", e)))?;
        let mask_host = mask
            .map(|m| {
                let mut data = vec![0.0f32; m.len()];
                backend.copy_from_device_safe(m.buffer(), &mut data).map_err(|e| {
                    AttentionError::MemoryCopy(format!("Failed to copy mask to host: {}", e))
                })?;
                Ok(data)
            })
            .transpose()?;

        // Use existing GPU forward implementation with host data
        let output = Self::forward(
            dim,
            &q_host,
            &k_host,
            &v_host,
            mask_host.as_deref(),
            dropout,
        )?;

        // Convert output back to DeviceTensor
        let backend = HipBackend::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
        })?;
        let shape = TensorShape::from_dims(&[output.len()]);
        DeviceTensor::from_host_vec(&backend, output, shape).map_err(|e| {
            AttentionError::MemoryAllocation(format!("Failed to create output tensor: {}", e))
        })
    }
}
