//! Paged Attention GPU Kernel Implementation
//!
//! Implements paged attention computation for non-contiguous KV cache blocks.
//! This is the core of PagedAttention-style memory management for LLM inference.

#![allow(deprecated)] // TODO: Migrate from to_host_vec() to copy_from_device_safe() (Phase 13-03-02)

use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use std::sync::Arc;

/// Configuration for paged attention kernel
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Block size (tokens per block)
    pub block_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl PagedAttentionConfig {
    /// Create a new paged attention configuration
    pub fn new(block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            block_size,
            num_heads,
            head_dim,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> HipResult<()> {
        if self.block_size == 0 {
            return Err(HipError::GenericError("Block size must be > 0".to_string()));
        }
        if self.num_heads == 0 {
            return Err(HipError::GenericError(
                "Number of heads must be > 0".to_string(),
            ));
        }
        if self.head_dim == 0 {
            return Err(HipError::GenericError(
                "Head dimension must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Paged attention kernels for GPU computation
pub struct PagedAttentionKernels {
    backend: Arc<HipBackend>,
    #[allow(dead_code)] // Reserved for future paged attention configuration
    config: PagedAttentionConfig,
    kernel_compiled: bool,
}

impl PagedAttentionKernels {
    /// Create new paged attention kernels
    ///
    /// # Arguments
    /// * `backend` - HIP backend for GPU operations
    /// * `config` - Kernel configuration
    pub fn new(backend: &Arc<HipBackend>, config: &PagedAttentionConfig) -> HipResult<Self> {
        config.validate()?;

        // For now, mark kernel as compiled (we'll use CPU fallback)
        // In a full implementation, this would compile HIP kernels
        Ok(Self {
            backend: Arc::clone(backend),
            config: config.clone(),
            kernel_compiled: true,
        })
    }

    /// Check if kernel is compiled
    pub fn is_compiled(&self) -> bool {
        self.kernel_compiled
    }

    /// Compute paged attention with standard MHA (multi-head attention)
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, num_heads, head_dim]
    /// * `k_blocks` - Key blocks (non-contiguous)
    /// * `v_blocks` - Value blocks (non-contiguous)
    /// * `block_indices` - Block index for each position [seq_len]
    /// * `block_offsets` - Offset within block for each position [seq_len]
    /// * `output` - Output tensor [seq_len, num_heads, head_dim]
    pub fn compute_paged_attention(
        &self,
        q: &DeviceTensor,
        k_blocks: &[DeviceTensor],
        v_blocks: &[DeviceTensor],
        block_indices: &[u32],
        block_offsets: &[usize],
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        // Validate inputs
        self.validate_paged_attention_inputs(
            q,
            k_blocks,
            v_blocks,
            block_indices,
            block_offsets,
            output,
        )?;

        // For now, use CPU fallback implementation
        // This will be replaced with GPU kernel in Phase 3b
        self.compute_paged_attention_cpu_fallback(
            q,
            k_blocks,
            v_blocks,
            block_indices,
            block_offsets,
            output,
        )
    }

    /// Compute paged attention with MQA (multi-query attention)
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, num_q_heads, head_dim]
    /// * `k_blocks` - Key blocks [block_size, num_kv_heads, head_dim]
    /// * `v_blocks` - Value blocks [block_size, num_kv_heads, head_dim]
    /// * `block_indices` - Block index for each position [seq_len]
    /// * `block_offsets` - Offset within block for each position [seq_len]
    /// * `num_kv_heads` - Number of KV heads (typically 1 or 2 for MQA)
    /// * `output` - Output tensor [seq_len, num_q_heads, head_dim]
    pub fn compute_paged_attention_mqa(
        &self,
        q: &DeviceTensor,
        k_blocks: &[DeviceTensor],
        v_blocks: &[DeviceTensor],
        block_indices: &[u32],
        block_offsets: &[usize],
        num_kv_heads: usize,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        // Validate MQA inputs
        self.validate_mqa_inputs(
            q,
            k_blocks,
            v_blocks,
            block_indices,
            block_offsets,
            num_kv_heads,
            output,
        )?;

        // For now, use CPU fallback implementation
        self.compute_paged_attention_mqa_cpu_fallback(
            q,
            k_blocks,
            v_blocks,
            block_indices,
            block_offsets,
            num_kv_heads,
            output,
        )
    }

    /// Validate paged attention inputs
    fn validate_paged_attention_inputs(
        &self,
        q: &DeviceTensor,
        k_blocks: &[DeviceTensor],
        v_blocks: &[DeviceTensor],
        block_indices: &[u32],
        block_offsets: &[usize],
        output: &DeviceTensor,
    ) -> HipResult<()> {
        let q_shape = q.shape();
        let output_shape = output.shape();

        // Validate Q shape
        if q_shape.dims().len() != 3 {
            return Err(HipError::GenericError(
                "Q must be 3D [seq_len, num_heads, head_dim]".to_string(),
            ));
        }

        let seq_len = q_shape.dims()[0];
        let num_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

        // Validate block_indices length
        if block_indices.len() != seq_len {
            return Err(HipError::GenericError(format!(
                "block_indices length {} must match seq_len {}",
                block_indices.len(),
                seq_len
            )));
        }

        // Validate block_offsets length
        if block_offsets.len() != seq_len {
            return Err(HipError::GenericError(format!(
                "block_offsets length {} must match seq_len {}",
                block_offsets.len(),
                seq_len
            )));
        }

        // Validate K and V blocks
        if k_blocks.len() != v_blocks.len() {
            return Err(HipError::GenericError(
                "Number of K blocks must match number of V blocks".to_string(),
            ));
        }

        if k_blocks.is_empty() {
            return Err(HipError::GenericError(
                "At least one K/V block required".to_string(),
            ));
        }

        // Validate each block shape
        for (i, (k_block, v_block)) in k_blocks.iter().zip(v_blocks.iter()).enumerate() {
            let k_shape = k_block.shape();
            let v_shape = v_block.shape();

            if k_shape.dims().len() != 3 || v_shape.dims().len() != 3 {
                return Err(HipError::GenericError(format!(
                    "Block {} K/V must be 3D [block_size, num_heads, head_dim]",
                    i
                )));
            }

            let k_block_size = k_shape.dims()[0];
            let k_num_heads = k_shape.dims()[1];
            let k_head_dim = k_shape.dims()[2];

            let v_block_size = v_shape.dims()[0];
            let v_num_heads = v_shape.dims()[1];
            let v_head_dim = v_shape.dims()[2];

            if k_block_size != v_block_size {
                return Err(HipError::GenericError(format!(
                    "Block {} K/V block_size mismatch: {} vs {}",
                    i, k_block_size, v_block_size
                )));
            }

            if k_num_heads != num_heads {
                return Err(HipError::GenericError(format!(
                    "Block {} K num_heads {} must match Q num_heads {}",
                    i, k_num_heads, num_heads
                )));
            }

            if v_num_heads != num_heads {
                return Err(HipError::GenericError(format!(
                    "Block {} V num_heads {} must match Q num_heads {}",
                    i, v_num_heads, num_heads
                )));
            }

            if k_head_dim != head_dim || v_head_dim != head_dim {
                return Err(HipError::GenericError(format!(
                    "Block {} K/V head_dim must match Q head_dim {}",
                    i, head_dim
                )));
            }
        }

        // Validate output shape
        if output_shape.dims() != [seq_len, num_heads, head_dim] {
            return Err(HipError::GenericError(format!(
                "Output shape {:?} must match Q shape {:?}",
                output_shape.dims(),
                q_shape.dims()
            )));
        }

        Ok(())
    }

    /// Validate MQA inputs
    fn validate_mqa_inputs(
        &self,
        q: &DeviceTensor,
        k_blocks: &[DeviceTensor],
        v_blocks: &[DeviceTensor],
        block_indices: &[u32],
        block_offsets: &[usize],
        num_kv_heads: usize,
        output: &DeviceTensor,
    ) -> HipResult<()> {
        let q_shape = q.shape();
        let output_shape = output.shape();

        // Validate Q shape
        if q_shape.dims().len() != 3 {
            return Err(HipError::GenericError(
                "Q must be 3D [seq_len, num_q_heads, head_dim]".to_string(),
            ));
        }

        let seq_len = q_shape.dims()[0];
        let num_q_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

        // Validate block_indices length
        if block_indices.len() != seq_len {
            return Err(HipError::GenericError(format!(
                "block_indices length {} must match seq_len {}",
                block_indices.len(),
                seq_len
            )));
        }

        // Validate block_offsets length
        if block_offsets.len() != seq_len {
            return Err(HipError::GenericError(format!(
                "block_offsets length {} must match seq_len {}",
                block_offsets.len(),
                seq_len
            )));
        }

        // Validate K and V blocks
        if k_blocks.len() != v_blocks.len() {
            return Err(HipError::GenericError(
                "Number of K blocks must match number of V blocks".to_string(),
            ));
        }

        if k_blocks.is_empty() {
            return Err(HipError::GenericError(
                "At least one K/V block required".to_string(),
            ));
        }

        // Validate each block shape for MQA
        for (i, (k_block, v_block)) in k_blocks.iter().zip(v_blocks.iter()).enumerate() {
            let k_shape = k_block.shape();
            let v_shape = v_block.shape();

            if k_shape.dims().len() != 3 || v_shape.dims().len() != 3 {
                return Err(HipError::GenericError(format!(
                    "Block {} K/V must be 3D [block_size, num_kv_heads, head_dim]",
                    i
                )));
            }

            let k_num_heads = k_shape.dims()[1];
            let v_num_heads = v_shape.dims()[1];
            let k_head_dim = k_shape.dims()[2];
            let v_head_dim = v_shape.dims()[2];

            if k_num_heads != num_kv_heads {
                return Err(HipError::GenericError(format!(
                    "Block {} K num_heads {} must match num_kv_heads {}",
                    i, k_num_heads, num_kv_heads
                )));
            }

            if v_num_heads != num_kv_heads {
                return Err(HipError::GenericError(format!(
                    "Block {} V num_heads {} must match num_kv_heads {}",
                    i, v_num_heads, num_kv_heads
                )));
            }

            if k_head_dim != head_dim || v_head_dim != head_dim {
                return Err(HipError::GenericError(format!(
                    "Block {} K/V head_dim must match Q head_dim {}",
                    i, head_dim
                )));
            }
        }

        // Validate output shape
        if output_shape.dims() != [seq_len, num_q_heads, head_dim] {
            return Err(HipError::GenericError(format!(
                "Output shape {:?} must match Q shape {:?}",
                output_shape.dims(),
                q_shape.dims()
            )));
        }

        Ok(())
    }

    /// CPU fallback for paged attention (for testing)
    fn compute_paged_attention_cpu_fallback(
        &self,
        q: &DeviceTensor,
        k_blocks: &[DeviceTensor],
        v_blocks: &[DeviceTensor],
        block_indices: &[u32],
        block_offsets: &[usize],
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        let q_shape = q.shape();
        let seq_len = q_shape.dims()[0];
        let num_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

        // Copy Q to host
        let q_host = q.to_host_vec()?;

        // Copy all K/V blocks to host
        let mut k_blocks_host = Vec::new();
        let mut v_blocks_host = Vec::new();

        for (k_block, v_block) in k_blocks.iter().zip(v_blocks.iter()) {
            k_blocks_host.push(k_block.to_host_vec()?);
            v_blocks_host.push(v_block.to_host_vec()?);
        }

        // Compute paged attention on CPU
        let mut output_host = vec![0.0f32; seq_len * num_heads * head_dim];

        for i in 0..seq_len {
            let block_idx = block_indices[i] as usize;
            let _offset = block_offsets[i];

            // Get K/V for this position from the correct block
            let _k_block = &k_blocks_host[block_idx];
            let _v_block = &v_blocks_host[block_idx];

            for h in 0..num_heads {
                for d in 0..head_dim {
                    // Compute attention score for this (query_pos, head, dim)
                    let q_idx = i * num_heads * head_dim + h * head_dim + d;
                    let q_val = q_host[q_idx];

                    // Simple attention: just accumulate weighted sum
                    // In real implementation, this would compute softmax(scores) @ V
                    let mut sum = 0.0f32;

                    // Iterate over all positions in all blocks
                    for (j, &b_idx) in block_indices.iter().enumerate() {
                        let b_offset = block_offsets[j];
                        let k_block_j = &k_blocks_host[b_idx as usize];
                        let v_block_j = &v_blocks_host[b_idx as usize];

                        let k_idx = b_offset * num_heads * head_dim + h * head_dim + d;
                        let v_idx = b_offset * num_heads * head_dim + h * head_dim + d;

                        if k_idx < k_block_j.len() && v_idx < v_block_j.len() {
                            let score = q_val * k_block_j[k_idx];
                            sum = score * v_block_j[v_idx];
                        }
                    }

                    let output_idx = i * num_heads * head_dim + h * head_dim + d;
                    output_host[output_idx] = sum;
                }
            }
        }

        // Copy output back to GPU
        let output_shape = output.shape().clone();
        *output = DeviceTensor::from_host_vec(&self.backend, output_host, output_shape)?;

        Ok(())
    }

    /// CPU fallback for MQA paged attention
    fn compute_paged_attention_mqa_cpu_fallback(
        &self,
        q: &DeviceTensor,
        k_blocks: &[DeviceTensor],
        v_blocks: &[DeviceTensor],
        block_indices: &[u32],
        block_offsets: &[usize],
        num_kv_heads: usize,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        let q_shape = q.shape();
        let seq_len = q_shape.dims()[0];
        let num_q_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

        // Copy Q to host
        let q_host = q.to_host_vec()?;

        // Copy all K/V blocks to host
        let mut k_blocks_host = Vec::new();
        let mut v_blocks_host = Vec::new();

        for (k_block, v_block) in k_blocks.iter().zip(v_blocks.iter()) {
            k_blocks_host.push(k_block.to_host_vec()?);
            v_blocks_host.push(v_block.to_host_vec()?);
        }

        // Compute MQA paged attention on CPU
        let mut output_host = vec![0.0f32; seq_len * num_q_heads * head_dim];

        for i in 0..seq_len {
            let block_idx = block_indices[i] as usize;
            let _offset = block_offsets[i];

            // Get K/V for this position from the correct block
            let _k_block = &k_blocks_host[block_idx];
            let _v_block = &v_blocks_host[block_idx];

            for qh in 0..num_q_heads {
                // Map query head to KV head (for MQA)
                let kv_h = qh % num_kv_heads;

                for d in 0..head_dim {
                    // Compute attention score for this (query_pos, query_head, dim)
                    let q_idx = i * num_q_heads * head_dim + qh * head_dim + d;
                    let q_val = q_host[q_idx];

                    // Simple attention: just accumulate weighted sum
                    let mut sum = 0.0f32;

                    // Iterate over all positions in all blocks
                    for (j, &b_idx) in block_indices.iter().enumerate() {
                        let b_offset = block_offsets[j];
                        let k_block_j = &k_blocks_host[b_idx as usize];
                        let v_block_j = &v_blocks_host[b_idx as usize];

                        let k_idx = b_offset * num_kv_heads * head_dim + kv_h * head_dim + d;
                        let v_idx = b_offset * num_kv_heads * head_dim + kv_h * head_dim + d;

                        if k_idx < k_block_j.len() && v_idx < v_block_j.len() {
                            let score = q_val * k_block_j[k_idx];
                            sum = score * v_block_j[v_idx];
                        }
                    }

                    let output_idx = i * num_q_heads * head_dim + qh * head_dim + d;
                    output_host[output_idx] = sum;
                }
            }
        }

        // Copy output back to GPU
        let output_shape = output.shape().clone();
        *output = DeviceTensor::from_host_vec(&self.backend, output_host, output_shape)?;

        Ok(())
    }
}

// HIP Kernel Source (for future implementation)
//
// NOTE: This kernel source is a template for future GPU implementation of paged attention.
// It is not currently used - the paged attention module falls back to CPU computation.
// TODO: Implement HIPRTC compilation and launch of this kernel (see Phase 3 docs).
#[allow(dead_code)]
const PAGED_ATTENTION_KERNEL: &str = r#"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void paged_attention_kernel(
    const T* __restrict__ q,           // [seq_len, num_heads, head_dim]
    const T** __restrict__ k_blocks,   // [num_blocks][block_size, num_heads, head_dim]
    const T** __restrict__ v_blocks,   // [num_blocks][block_size, num_heads, head_dim]
    const int32_t* __restrict__ block_indices,  // [seq_len]
    const int32_t* __restrict__ block_offsets,   // [seq_len]
    T* __restrict__ output,           // [seq_len, num_heads, head_dim]
    int seq_len,
    int num_heads,
    int head_dim,
    int block_size
) {
    // Each thread block processes one query position
    int pos_idx = blockIdx.x;
    if (pos_idx >= seq_len) return;

    int head_idx = blockIdx.y;
    if (head_idx >= num_heads) return;

    // Get block for this position
    int block_idx = block_indices[pos_idx];
    int offset = block_offsets[pos_idx];

    // Get K/V pointers for this block
    const T* k_block = k_blocks[block_idx];
    const T* v_block = v_blocks[block_idx];

    // Shared memory for attention scores
    extern __shared__ float scores[];

    int tid = threadIdx.x;

    // Load Q
    float q_val = 0.0f;
    if (tid < head_dim) {
        q_val = static_cast<float>(q[pos_idx * num_heads * head_dim + head_idx * head_dim + tid]);
    }

    // Compute attention scores by iterating through sequence
    float max_score = -1e20f;
    float sum_exp = 0.0f;

    for (int j = 0; j < seq_len; ++j) {
        int j_block_idx = block_indices[j];
        int j_offset = block_offsets[j];

        const T* k_block_j = k_blocks[j_block_idx];

        float score = 0.0f;
        if (tid < head_dim) {
            float k_val = static_cast<float>(k_block_j[j_offset * num_heads * head_dim + head_idx * head_dim + tid]);
            score = q_val * k_val;
        }

        // Reduce across threads
        __syncthreads();
        if (tid < head_dim) {
            scores[tid] = score;
        }
        __syncthreads();

        float total_score = 0.0f;
        if (tid == 0) {
            for (int d = 0; d < head_dim; ++d) {
                total_score += scores[d];
            }
            total_score /= __sqrtf(static_cast<float>(head_dim));
            scores[0] = total_score;
        }
        __syncthreads();

        if (tid == 0) {
            if (j == 0) {
                max_score = scores[0];
            } else {
                max_score = fmaxf(max_score, scores[0]);
            }
        }
        __syncthreads();
    }

    // Compute softmax and weighted sum with V
    for (int j = 0; j < seq_len; ++j) {
        int j_block_idx = block_indices[j];
        int j_offset = block_offsets[j];

        const T* k_block_j = k_blocks[j_block_idx];
        const T* v_block_j = v_blocks[j_block_idx];

        float score = 0.0f;
        if (tid < head_dim) {
            float k_val = static_cast<float>(k_block_j[j_offset * num_heads * head_dim + head_idx * head_dim + tid]);
            score = q_val * k_val;
        }

        // Reduce across threads
        __syncthreads();
        if (tid < head_dim) {
            scores[tid] = score;
        }
        __syncthreads();

        float total_score = 0.0f;
        if (tid == 0) {
            for (int d = 0; d < head_dim; ++d) {
                total_score += scores[d];
            }
            total_score = total_score / __sqrtf(static_cast<float>(head_dim)) - max_score;
            total_score = __expf(total_score);
            scores[0] = total_score;
            sum_exp += total_score;
        }
        __syncthreads();

        // Accumulate weighted V
        if (tid < head_dim) {
            float v_val = static_cast<float>(v_block_j[j_offset * num_heads * head_dim + head_idx * head_dim + tid]);
            float weight = scores[0];
            if (j == seq_len - 1) {
                weight /= sum_exp;
            }
            atomicAdd(&output[pos_idx * num_heads * head_dim + head_idx * head_dim + tid], weight * v_val);
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = PagedAttentionConfig::new(16, 4, 32);
        assert!(config.validate().is_ok());

        let invalid_config = PagedAttentionConfig::new(0, 4, 32);
        assert!(invalid_config.validate().is_err());
    }
}
