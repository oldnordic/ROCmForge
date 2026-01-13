//! GPU Attention Implementation - Phase E
//!
//! Implements scaled dot-product attention with HIP kernels:
//! - QK^T matrix multiplication with hip_flash_attention_qk_kernel.hip
//! - Scaling by sqrt(head_dim)
//! - Causal masking with hip_flash_attention_softmax_kernel.hip
//! - Softmax computation with hip_flash_attention_softmax_kernel.hip
//! - V contraction with hip_flash_attention_v_kernel.hip
//!
//! Uses scratch buffers and integrates with KV cache.

use crate::backend::{
    hip_blas::{self, HipBlasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T},
    DeviceTensor, HipBackend, HipError, HipResult,
};
#[cfg(feature = "rocm")]
use crate::backend::{HipKernel, HipModule};
use crate::loader::TensorShape;
use crate::model::kv_cache::KVCache;
use crate::tensor::matmul::matmul_f32;
#[cfg(feature = "rocm")]
use once_cell::sync::OnceCell;
use std::ffi::c_void;

/// GPU Attention Kernels for Phase E
///
/// Provides HIP kernel implementations for attention computation:
/// - QK^T computation kernel
/// - Softmax with causal masking kernel  
/// - Attention-weighted V computation kernel
pub struct HipAttentionKernels {
    backend: HipBackend,
    blas_handle: HipBlasHandle,
    // HIP kernel modules will be loaded here
    qk_kernel: Option<crate::backend::hip_backend::HipModule>,
    softmax_kernel: Option<crate::backend::hip_backend::HipModule>,
    v_kernel: Option<crate::backend::hip_backend::HipModule>,
    #[cfg(feature = "rocm")]
    attention_softmax_kernel: OnceCell<CompiledKernel>,
    #[cfg(feature = "rocm")]
    causal_mask_kernel: OnceCell<CompiledKernel>,
}

#[cfg(feature = "rocm")]
struct CompiledKernel {
    module: HipModule,
    kernel: HipKernel,
}

impl HipAttentionKernels {
    /// Create new HIP attention kernels
    ///
    /// Loads HIP kernels for attention computation.
    /// For now, creates placeholder implementation.
    pub fn new(backend: &HipBackend) -> HipResult<Self> {
        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        // CRITICAL: Associate hipBLAS handle with the backend's HIP stream
        //
        // Without this, hipBLAS uses the default stream while our custom HIP kernels
        // (softmax, causal_mask) use the backend's custom stream. This causes
        // synchronization issues and hangs when copy_to_host() calls hipDeviceSynchronize().
        //
        // hipDeviceSynchronize() waits for operations on the custom stream, but hipBLAS
        // operations on the default stream are still pending, causing the D2H copy to
        // read incomplete data and hang.
        //
        // See: https://github.com/ROCm/hip/issues/3370
        // See: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html
        blas_handle
            .set_stream(backend.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        Ok(HipAttentionKernels {
            backend: backend.clone(),
            blas_handle,
            qk_kernel: None, // Will be loaded from .hip files
            softmax_kernel: None,
            v_kernel: None,
            #[cfg(feature = "rocm")]
            attention_softmax_kernel: OnceCell::new(),
            #[cfg(feature = "rocm")]
            causal_mask_kernel: OnceCell::new(),
        })
    }

    #[cfg(feature = "rocm")]
    fn compile_attention_softmax_kernel(&self) -> HipResult<CompiledKernel> {
        let code = hiprtc::compile_kernel("attention_softmax", ATTENTION_SOFTMAX_KERNEL)?;
        let module = self.backend.load_module_from_data(&code)?;
        let kernel = self
            .backend
            .get_kernel_function(&module, "attention_softmax")?;
        Ok(CompiledKernel { module, kernel })
    }

    #[cfg(feature = "rocm")]
    fn get_attention_softmax_kernel(&self) -> HipResult<&CompiledKernel> {
        self.attention_softmax_kernel
            .get_or_try_init(|| self.compile_attention_softmax_kernel())
    }

    #[cfg(feature = "rocm")]
    fn compile_causal_mask_kernel(&self) -> HipResult<CompiledKernel> {
        let code = hiprtc::compile_kernel("causal_mask", CAUSAL_MASK_KERNEL)?;
        let module = self.backend.load_module_from_data(&code)?;
        let kernel = self
            .backend
            .get_kernel_function(&module, "causal_mask_kernel")?;
        Ok(CompiledKernel { module, kernel })
    }

    #[cfg(feature = "rocm")]
    fn get_causal_mask_kernel(&self) -> HipResult<&CompiledKernel> {
        self.causal_mask_kernel
            .get_or_try_init(|| self.compile_causal_mask_kernel())
    }

    /// Compute QK^T using HIP kernel
    ///
    /// Computes Q @ K^T for attention scores.
    /// Shape: Q[seq_q, num_heads, head_dim] @ K[seq_k, num_heads, head_dim]^T -> [seq_q, seq_k]
    pub fn compute_qk_t(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        // Validate input shapes
        let q_shape = q.shape();
        let k_shape = k.shape();

        if q_shape.dims().len() != 3 || k_shape.dims().len() != 3 {
            return Err(HipError::GenericError(
                "Q and K must be 3D tensors [seq, num_heads, head_dim]".to_string(),
            ));
        }

        let (_seq_q, num_heads_q, head_dim_q) =
            (q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2]);
        let (_seq_k, num_heads_k, head_dim_k) =
            (k_shape.dims()[0], k_shape.dims()[1], k_shape.dims()[2]);

        if num_heads_q != num_heads_k || head_dim_q != head_dim_k {
            return Err(HipError::GenericError(
                "Q and K must have matching num_heads and head_dim".to_string(),
            ));
        }

        // Prefer GPU implementation, fall back to CPU if needed
        if let Err(err) = self.compute_qk_t_gemm(q, k, output) {
            tracing::warn!("hipBLAS QK^T fallback to CPU: {}", err);
            self.compute_qk_t_cpu_fallback(q, k, output)
        } else {
            Ok(())
        }
    }

    fn compute_qk_t_gemm(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        let q_shape = q.shape();
        let k_shape = k.shape();

        if q_shape.dims().len() != 3 || k_shape.dims().len() != 3 {
            return Err(HipError::GenericError(
                "Q and K must be 3D tensors [seq, num_heads, head_dim]".to_string(),
            ));
        }

        let seq_q = q_shape.dims()[0];
        let seq_k = k_shape.dims()[0];
        let num_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

        if k_shape.dims()[1] != num_heads || k_shape.dims()[2] != head_dim {
            return Err(HipError::GenericError(
                "Q and K must have matching num_heads and head_dim".to_string(),
            ));
        }

        let total_dim = num_heads * head_dim;
        if output.shape().dims() != [seq_q, seq_k] {
            return Err(HipError::GenericError(format!(
                "Output shape mismatch: expected [{}, {}], got {:?}",
                seq_q,
                seq_k,
                output.shape().dims()
            )));
        }

        hip_blas::sgemm(
            &self.blas_handle,
            HIPBLAS_OP_T,
            HIPBLAS_OP_N,
            seq_k as i32,
            seq_q as i32,
            total_dim as i32,
            1.0,
            k.buffer().as_ptr() as *const f32,
            total_dim as i32,
            q.buffer().as_ptr() as *const f32,
            total_dim as i32,
            0.0,
            output.buffer().as_ptr() as *mut f32,
            seq_k as i32,
        )
        .map_err(|e| HipError::GenericError(format!("hipBLAS sgemm failed: {}", e)))?;

        Ok(())
    }

    /// Apply causal mask to attention scores
    ///
    /// Masks future positions (j > i) with -inf for causal attention.
    pub fn apply_causal_mask(
        &self,
        attention: &mut DeviceTensor,
        seq_len: usize,
        cache_len: usize,
    ) -> HipResult<()> {
        #[cfg(feature = "rocm")]
        {
            if let Err(err) = self.apply_causal_mask_gpu(attention, seq_len, cache_len) {
                tracing::warn!("hip attention mask fallback to CPU: {}", err);
            } else {
                return Ok(());
            }
        }

        self.apply_causal_mask_cpu_fallback(attention, seq_len, cache_len)
    }

    #[cfg(feature = "rocm")]
    fn apply_causal_mask_gpu(
        &self,
        attention: &mut DeviceTensor,
        seq_len: usize,
        cache_len: usize,
    ) -> HipResult<()> {
        let attention_shape = attention.shape();
        let dims = attention_shape.dims();

        // For causal mask with KV cache, we typically have:
        // - seq_len: current query length
        // - cache_len: total key length in cache (>= seq_len)
        // The attention tensor is [seq_len, cache_len]

        if dims.len() == 2 {
            // Simple 2D case: [seq_len, cache_len]
            let kernel = self.get_causal_mask_kernel()?;

            let grid_dim = (seq_len as u32, 1, 1);
            let block_dim = 32; // WARP_SIZE

            let _seq_len_i32 = seq_len as i32;
            let mut cache_len_i32 = cache_len as i32;
            let mut attention_ptr = attention.buffer().as_ptr();

            let args = [
                &mut attention_ptr as *mut _ as *mut c_void,
                &mut 1i32 as *mut _ as *mut c_void, // batch_size = 1
                &mut cache_len_i32 as *mut _ as *mut c_void,
                &mut 1i32 as *mut _ as *mut c_void, // num_heads = 1
            ];

            self.backend.launch_kernel_with_module_shared(
                &kernel.kernel,
                grid_dim,
                (block_dim, 1, 1),
                &args,
                0, // shared memory
            )
        } else if dims.len() == 4 {
            // 4D case: [batch_size, num_heads, seq_len, cache_len]
            let batch_size = dims[0];
            let num_heads = dims[1];

            let kernel = self.get_causal_mask_kernel()?;

            let grid_dim = (seq_len as u32, num_heads as u32, batch_size as u32);
            let block_dim = 32; // WARP_SIZE

            let mut batch_size_i32 = batch_size as i32;
            let _seq_len_i32 = seq_len as i32;
            let mut cache_len_i32 = cache_len as i32;
            let mut num_heads_i32 = num_heads as i32;
            let mut attention_ptr = attention.buffer().as_ptr();

            let args = [
                &mut attention_ptr as *mut _ as *mut c_void,
                &mut batch_size_i32 as *mut _ as *mut c_void,
                &mut cache_len_i32 as *mut _ as *mut c_void,
                &mut num_heads_i32 as *mut _ as *mut c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                &kernel.kernel,
                grid_dim,
                (block_dim, 1, 1),
                &args,
                0, // shared memory
            )
        } else {
            Err(HipError::GenericError(format!(
                "Unexpected attention tensor shape: {:?}",
                dims
            )))
        }
    }

    /// Compute softmax of attention scores
    ///
    /// Computes row-wise softmax with numerical stability.
    pub fn compute_softmax(
        &self,
        attention: &mut DeviceTensor,
        temp_buffer: &DeviceTensor,
    ) -> HipResult<()> {
        #[cfg(feature = "rocm")]
        {
            if let Err(err) = self.compute_softmax_gpu(attention) {
                tracing::warn!("hip attention softmax fallback to CPU: {}", err);
            } else {
                return Ok(());
            }
        }

        let _ = temp_buffer;
        self.compute_softmax_cpu_fallback(attention, temp_buffer)
    }

    #[cfg(feature = "rocm")]
    fn compute_softmax_gpu(&self, attention: &mut DeviceTensor) -> HipResult<()> {
        let kernel = self.get_attention_softmax_kernel()?;
        let shape = attention.shape();
        if shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "Attention tensor must be 2D [rows, cols]".to_string(),
            ));
        }

        let rows = shape.dims()[0] as u32;
        let cols = shape.dims()[1] as u32;

        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let mut block_dim = 1;
        while block_dim < cols && block_dim < 256 {
            block_dim <<= 1;
        }
        block_dim = block_dim.min(256).max(1);

        let shared_bytes = (block_dim as usize * std::mem::size_of::<f32>()) as u32;

        let mut rows_i32 = rows as i32;
        let mut cols_i32 = cols as i32;
        let mut scores_ptr = attention.buffer().as_ptr();
        let args = [
            &mut scores_ptr as *mut _ as *mut c_void,
            &mut rows_i32 as *mut _ as *mut c_void,
            &mut cols_i32 as *mut _ as *mut c_void,
        ];

        self.backend.launch_kernel_with_module_shared(
            &kernel.kernel,
            (rows, 1, 1),
            (block_dim, 1, 1),
            &args,
            shared_bytes,
        )
    }

    /// Compute attention-weighted V: attention @ V
    ///
    /// Computes final attention output by weighting V with attention scores.
    pub fn compute_attention_weighted_v(
        &self,
        attention: &DeviceTensor,
        v: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        // Prefer GPU implementation
        match self.compute_attention_weighted_v_gemm(attention, v, output) {
            Ok(_) => Ok(()),
            Err(err) => {
                tracing::warn!("hipBLAS attention*V fallback to CPU: {}", err);
                self.compute_attention_weighted_v_cpu_fallback(attention, v, output)
            }
        }
    }

    fn compute_attention_weighted_v_gemm(
        &self,
        attention: &DeviceTensor,
        v: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        let att_shape = attention.shape();
        if att_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "Attention tensor must be 2D [seq_q, seq_k]".to_string(),
            ));
        }

        let seq_q = att_shape.dims()[0];
        let seq_k = att_shape.dims()[1];

        let v_shape = v.shape();
        if v_shape.dims().len() != 3 {
            return Err(HipError::GenericError(
                "Value tensor must be 3D [seq_k, num_heads, head_dim]".to_string(),
            ));
        }

        let v_seq = v_shape.dims()[0];
        let num_heads = v_shape.dims()[1];
        let head_dim = v_shape.dims()[2];

        if v_seq != seq_k {
            return Err(HipError::GenericError(format!(
                "Value tensor seq_len {} must match attention K length {}",
                v_seq, seq_k
            )));
        }

        let out_shape = output.shape();
        if out_shape.dims() != [seq_q, num_heads, head_dim] {
            return Err(HipError::GenericError(format!(
                "Output tensor shape {:?} must be [seq_q, num_heads, head_dim]",
                out_shape.dims()
            )));
        }

        let hidden_size = num_heads * head_dim;

        let result_buffer = matmul_f32(
            &self.blas_handle,
            attention.buffer(),
            v.buffer(),
            seq_q as i32,
            hidden_size as i32,
            seq_k as i32,
        )
        .map_err(|e| HipError::GenericError(format!("hipBLAS attention matmul failed: {}", e)))?;

        output
            .buffer()
            .copy_from_buffer(&result_buffer)
            .map_err(|e| {
                HipError::GenericError(format!("Failed to copy attention result: {}", e))
            })?;

        Ok(())
    }

    /// Complete attention computation pipeline
    ///
    /// Full attention: QK^T -> scale -> mask -> softmax -> @V
    pub fn compute_attention(
        &self,
        q: &DeviceTensor,
        _attention_scores: &DeviceTensor,
        softmax_temp: &DeviceTensor,
        kv_cache: &KVCache,
        layer_id: usize,
        current_seq_len: usize,
    ) -> HipResult<DeviceTensor> {
        // Get K and V from cache
        let (k_tensor, v_tensor) = kv_cache.retrieve(layer_id, current_seq_len)?;

        // Validate shapes
        let q_shape = q.shape();
        let (seq_len, num_heads, head_dim) =
            (q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2]);

        // Compute QK^T
        let attention_shape = TensorShape::from_dims(&[seq_len, current_seq_len]);
        let mut attention_scores = DeviceTensor::empty(&self.backend, attention_shape)?;
        self.compute_qk_t(q, &k_tensor, &mut attention_scores)?;

        // Scale by 1/sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        self.scale_attention(&mut attention_scores, scale)?;

        // Apply causal mask
        self.apply_causal_mask(&mut attention_scores, seq_len, current_seq_len)?;

        // Compute softmax
        self.compute_softmax(&mut attention_scores, softmax_temp)?;

        // Compute attention @ V
        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut output = DeviceTensor::empty(&self.backend, output_shape)?;
        self.compute_attention_weighted_v(&attention_scores, &v_tensor, &mut output)?;

        Ok(output)
    }

    /// Scale attention scores by factor
    fn scale_attention(&self, attention: &mut DeviceTensor, scale: f32) -> HipResult<()> {
        self.backend.scale_inplace(attention, scale)
    }

    /// CPU fallback for QK^T computation
    fn compute_qk_t_cpu_fallback(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        let q_host = q.to_host_vec()?;
        let k_host = k.to_host_vec()?;
        let q_shape = q.shape();
        let k_shape = k.shape();

        let seq_q = q_shape.dims()[0];
        let seq_k = k_shape.dims()[0];
        let num_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

        let mut output_host = vec![0.0f32; seq_q * seq_k];

        // Compute QK^T: for each query position and key position
        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut sum = 0.0f32;
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        let q_idx = i * num_heads * head_dim + h * head_dim + d;
                        let k_idx = j * num_heads * head_dim + h * head_dim + d;
                        sum += q_host[q_idx] * k_host[k_idx];
                    }
                }
                output_host[i * seq_k + j] = sum;
            }
        }

        let shape = output.shape().clone();
        *output = DeviceTensor::from_host_vec(&self.backend, output_host, shape)?;
        Ok(())
    }

    /// CPU fallback for causal masking
    fn apply_causal_mask_cpu_fallback(
        &self,
        attention: &mut DeviceTensor,
        seq_len: usize,
        cache_len: usize,
    ) -> HipResult<()> {
        let mut attention_host = attention.to_host_vec()?;

        for i in 0..seq_len {
            for j in 0..cache_len {
                if j > i {
                    attention_host[i * cache_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        let shape = attention.shape().clone();
        *attention = DeviceTensor::from_host_vec(&self.backend, attention_host, shape)?;
        Ok(())
    }

    /// CPU fallback for softmax computation
    fn compute_softmax_cpu_fallback(
        &self,
        attention: &mut DeviceTensor,
        _temp_buffer: &DeviceTensor,
    ) -> HipResult<()> {
        let mut attention_host = attention.to_host_vec()?;
        let attention_shape = attention.shape();
        let rows = attention_shape.dims()[0];
        let cols = attention_shape.dims()[1];

        // Compute softmax row by row
        for i in 0..rows {
            let row_start = i * cols;
            let row_end = row_start + cols;

            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in row_start..row_end {
                if attention_host[j] > max_val {
                    max_val = attention_host[j];
                }
            }

            // Compute exp and sum
            let mut sum = 0.0f32;
            for j in row_start..row_end {
                if attention_host[j] != f32::NEG_INFINITY {
                    attention_host[j] = (attention_host[j] - max_val).exp();
                    sum += attention_host[j];
                } else {
                    attention_host[j] = 0.0f32;
                }
            }

            // Normalize
            if sum > 0.0f32 {
                for j in row_start..row_end {
                    attention_host[j] /= sum;
                }
            }
        }

        let shape = attention.shape().clone();
        *attention = DeviceTensor::from_host_vec(&self.backend, attention_host, shape)?;
        Ok(())
    }

    /// CPU fallback for attention @ V computation
    fn compute_attention_weighted_v_cpu_fallback(
        &self,
        attention: &DeviceTensor,
        v: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        let attention_host = attention.to_host_vec()?;
        let v_host = v.to_host_vec()?;
        let attention_shape = attention.shape();
        let v_shape = v.shape();

        let seq_len = attention_shape.dims()[0];
        let cache_len = attention_shape.dims()[1];
        let num_heads = v_shape.dims()[1];
        let head_dim = v_shape.dims()[2];

        let mut output_host = vec![0.0f32; seq_len * num_heads * head_dim];

        // Compute attention @ V
        for i in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for k in 0..cache_len {
                        let attention_idx = i * cache_len + k;
                        let v_idx = k * num_heads * head_dim + h * head_dim + d;
                        sum += attention_host[attention_idx] * v_host[v_idx];
                    }
                    let output_idx = i * num_heads * head_dim + h * head_dim + d;
                    output_host[output_idx] = sum;
                }
            }
        }

        let shape = output.shape().clone();
        *output = DeviceTensor::from_host_vec(&self.backend, output_host, shape)?;
        Ok(())
    }
}

#[cfg(feature = "rocm")]
const ATTENTION_SOFTMAX_KERNEL: &str = r#"
extern "C" __global__ void attention_softmax(float* scores, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float max_val = -3.402823466e+38f;  // -FLT_MAX, largest negative finite float
    for (int col = tid; col < cols; col += blockDim.x) {
        float val = scores[row * cols + col];
        if (val > max_val) {
            max_val = val;
        }
    }
    shared[tid] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float val = scores[row * cols + col];
        float exp_val = expf(val - max_val);
        scores[row * cols + col] = exp_val;
        local_sum += exp_val;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    float sum_val = shared[0] + 1e-9f;
    __syncthreads();

    for (int col = tid; col < cols; col += blockDim.x) {
        scores[row * cols + col] = scores[row * cols + col] / sum_val;
    }
}
"#;

#[cfg(feature = "rocm")]
#[allow(dead_code)]
const ATTENTION_MASK_KERNEL: &str = r#"
extern "C" __global__ void attention_mask(float* scores, int rows, int cols) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }
    if (col > row) {
        scores[row * cols + col] = -INFINITY;
    }
}
"#;

#[cfg(feature = "rocm")]
const CAUSAL_MASK_KERNEL: &str = r#"
#include <hip/hip_runtime.h>

constexpr int WARP_SIZE = 32;

extern "C" __global__ void causal_mask_kernel(
    float* __restrict__ attention,
    const int batch_size,
    const int seq_len,
    const int num_heads
) {
    const int query_pos = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || query_pos >= seq_len) {
        return;
    }

    const int batch_head_offset = batch_idx * num_heads * seq_len * seq_len
                                + head_idx * seq_len * seq_len;
    const int row_offset = batch_head_offset + query_pos * seq_len;

    // Only mask future positions (upper triangle)
    // Preserve current and past positions (lower triangle)
    for (int key_pos = tid; key_pos < seq_len; key_pos += WARP_SIZE) {
        if (key_pos > query_pos) {
            // Mask future position
            attention[row_offset + key_pos] = -(__builtin_inff());
        }
        // else: keep original value (don't modify)
    }
}
"#;

#[cfg(feature = "rocm")]
mod hiprtc {
    use super::{HipError, HipResult};
    use std::ffi::{c_char, c_void, CString};
    use std::ptr;

    type HiprtcProgram = *mut c_void;
    const HIPRTC_SUCCESS: i32 = 0;

    #[link(name = "hiprtc")]
    extern "C" {
        fn hiprtcCreateProgram(
            prog: *mut HiprtcProgram,
            src: *const c_char,
            name: *const c_char,
            num_headers: i32,
            headers: *const *const c_char,
            include_names: *const *const c_char,
        ) -> i32;
        fn hiprtcCompileProgram(
            prog: HiprtcProgram,
            num_options: i32,
            options: *const *const c_char,
        ) -> i32;
        fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size_ret: *mut usize) -> i32;
        fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut c_char) -> i32;
        fn hiprtcGetCodeSize(prog: HiprtcProgram, code_size_ret: *mut usize) -> i32;
        fn hiprtcGetCode(prog: HiprtcProgram, code: *mut c_char) -> i32;
        fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> i32;
    }

    pub fn compile_kernel(name: &str, source: &str) -> HipResult<Vec<u8>> {
        let name_c = CString::new(name)
            .map_err(|e| HipError::KernelLoadFailed(format!("Invalid kernel name: {}", e)))?;
        let source_c = CString::new(source)
            .map_err(|e| HipError::KernelLoadFailed(format!("Invalid kernel source: {}", e)))?;

        let mut program: HiprtcProgram = ptr::null_mut();
        let create_result = unsafe {
            hiprtcCreateProgram(
                &mut program,
                source_c.as_ptr(),
                name_c.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
        };

        if create_result != HIPRTC_SUCCESS {
            return Err(HipError::KernelLoadFailed(
                "hiprtcCreateProgram failed".to_string(),
            ));
        }

        // UNWRAP: Static string literal without null bytes, always safe
        let option = CString::new("--std=c++17").unwrap();
        let options = [option.as_ptr()];
        let compile_result =
            unsafe { hiprtcCompileProgram(program, options.len() as i32, options.as_ptr()) };

        if compile_result != HIPRTC_SUCCESS {
            let log = get_program_log(program);
            unsafe { hiprtcDestroyProgram(&mut program) };
            return Err(HipError::KernelLoadFailed(format!(
                "hiprtcCompileProgram failed: {}",
                log.unwrap_or_else(|| "unknown error".to_string())
            )));
        }

        let mut code_size: usize = 0;
        let size_result = unsafe { hiprtcGetCodeSize(program, &mut code_size) };
        if size_result != HIPRTC_SUCCESS {
            unsafe { hiprtcDestroyProgram(&mut program) };
            return Err(HipError::KernelLoadFailed(
                "hiprtcGetCodeSize failed".to_string(),
            ));
        }

        let mut code = vec![0u8; code_size];
        let code_result = unsafe { hiprtcGetCode(program, code.as_mut_ptr() as *mut c_char) };
        unsafe { hiprtcDestroyProgram(&mut program) };

        if code_result != HIPRTC_SUCCESS {
            return Err(HipError::KernelLoadFailed(
                "hiprtcGetCode failed".to_string(),
            ));
        }

        Ok(code)
    }

    fn get_program_log(program: HiprtcProgram) -> Option<String> {
        let mut size: usize = 0;
        if unsafe { hiprtcGetProgramLogSize(program, &mut size) } != HIPRTC_SUCCESS || size == 0 {
            return None;
        }

        let mut buffer = vec![0u8; size];
        if unsafe { hiprtcGetProgramLog(program, buffer.as_mut_ptr() as *mut c_char) }
            != HIPRTC_SUCCESS
        {
            return None;
        }

        Some(String::from_utf8_lossy(&buffer).into_owned())
    }
}

/// GPU attention implementation
impl HipBackend {
    /// Compute attention using GPU kernels
    ///
    /// Performs full attention computation:
    /// 1. QK^T matrix multiplication
    /// 2. Scaling by sqrt(head_dim)
    /// 3. Mask application (if needed)
    /// 4. Softmax computation
    /// 5. V contraction
    ///
    /// Uses scratch buffers for intermediate results.
    pub fn compute_attention(
        &self,
        q: &DeviceTensor,
        _attention_scores: &DeviceTensor,
        softmax_temp: &DeviceTensor,
        kv_cache: &KVCache,
        layer_id: usize,
        current_seq_len: usize,
    ) -> HipResult<DeviceTensor> {
        // Validate input shapes
        let q_shape = q.shape();
        if q_shape.dims().len() != 3 {
            return Err(HipError::GenericError(
                "Q tensor must be 3D [seq_len, num_heads, head_dim]".to_string(),
            ));
        }

        let (seq_len, num_heads, head_dim) =
            (q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2]);

        // Retrieve K and V from KV cache
        let (k_tensor, v_tensor) = kv_cache.retrieve(layer_id, seq_len)?;

        let _k_shape = k_tensor.shape();
        let _v_shape = v_tensor.shape();

        // Compute QK^T: [seq_len, cache_seq_len]
        let attention_shape = TensorShape::from_dims(&[seq_len, current_seq_len]);
        let mut attention_output = DeviceTensor::empty(self, attention_shape.clone())?;

        self.compute_qk_t(q, &k_tensor, &mut attention_output)?;

        // Scale by sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        self.scale_attention(&mut attention_output, scale)?;

        // Apply causal mask (lower triangular)
        self.apply_causal_mask(&mut attention_output, seq_len, current_seq_len)?;

        // Compute softmax
        self.compute_softmax(&mut attention_output, softmax_temp)?;

        // Multiply by V: [seq_len, num_heads, head_dim]
        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut final_output = DeviceTensor::empty(self, output_shape)?;

        self.compute_attention_weighted_v(&attention_output, &v_tensor, &mut final_output)?;

        Ok(final_output)
    }

    /// Compute QK^T matrix multiplication
    fn compute_qk_t(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        // Q: [seq_len_q, num_heads, head_dim] -> reshape to [seq_len_q, num_heads * head_dim]
        // K: [seq_len_k, num_heads, head_dim] -> reshape to [seq_len_k, num_heads * head_dim]
        // Output: [seq_len_q, seq_len_k]

        let q_shape = q.shape();
        let k_shape = k.shape();

        let seq_q = q_shape.dims()[0];
        let seq_k = k_shape.dims()[0];
        let num_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

        let _q_flat_size = seq_q * num_heads * head_dim;
        let _k_flat_size = seq_k * num_heads * head_dim;

        // Reshape Q and K for batched matrix multiplication
        let q_reshaped = self.reshape_for_qk(q, seq_q, num_heads, head_dim)?;
        let k_reshaped = self.reshape_k_for_qk(k, seq_k, num_heads, head_dim)?;

        // Perform batched GEMM: Q @ K^T
        self.batched_gemm(
            &q_reshaped,
            &k_reshaped,
            output,
            seq_q,
            seq_k,
            num_heads,
            head_dim,
        )?;

        Ok(())
    }

    /// Reshape Q tensor for QK^T computation
    fn reshape_for_qk(
        &self,
        q: &DeviceTensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<DeviceTensor> {
        // Reshape from [seq_len, num_heads, head_dim] to [seq_len, num_heads * head_dim]
        let flat_size = seq_len * num_heads * head_dim;
        let q_host = q.to_host_vec()?;

        let mut q_flat = vec![0.0f32; flat_size];
        for i in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let q_idx = i * num_heads * head_dim + h * head_dim + d;
                    let flat_idx = i * (num_heads * head_dim) + h * head_dim + d;
                    q_flat[flat_idx] = q_host[q_idx];
                }
            }
        }

        let flat_shape = TensorShape::from_dims(&[seq_len, num_heads * head_dim]);
        DeviceTensor::from_host_vec(self, q_flat, flat_shape)
    }

    /// Reshape K tensor for QK^T computation  
    fn reshape_k_for_qk(
        &self,
        k: &DeviceTensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<DeviceTensor> {
        // Reshape from [seq_len, num_heads, head_dim] to [seq_len, num_heads * head_dim]
        let flat_size = seq_len * num_heads * head_dim;
        let k_host = k.to_host_vec()?;

        let mut k_flat = vec![0.0f32; flat_size];
        for i in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let k_idx = i * num_heads * head_dim + h * head_dim + d;
                    let flat_idx = i * (num_heads * head_dim) + h * head_dim + d;
                    k_flat[flat_idx] = k_host[k_idx];
                }
            }
        }

        let flat_shape = TensorShape::from_dims(&[seq_len, num_heads * head_dim]);
        DeviceTensor::from_host_vec(self, k_flat, flat_shape)
    }

    /// Scale attention scores by 1/sqrt(head_dim)
    fn scale_attention(&self, attention: &mut DeviceTensor, scale: f32) -> HipResult<()> {
        let mut attention_host = attention.to_host_vec()?;

        for i in 0..attention_host.len() {
            attention_host[i] *= scale;
        }

        let shape = attention.shape().clone();
        *attention = DeviceTensor::from_host_vec(self, attention_host, shape)?;
        Ok(())
    }

    /// Apply causal mask to attention scores
    fn apply_causal_mask(
        &self,
        attention: &mut DeviceTensor,
        seq_len: usize,
        cache_len: usize,
    ) -> HipResult<()> {
        let mut attention_host = attention.to_host_vec()?;

        for i in 0..seq_len {
            for j in 0..cache_len {
                if j > i {
                    // Apply causal mask: future tokens cannot attend to past tokens
                    attention_host[i * cache_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        let shape = attention.shape().clone();
        *attention = DeviceTensor::from_host_vec(self, attention_host, shape)?;
        Ok(())
    }

    /// Compute softmax of attention scores
    fn compute_softmax(
        &self,
        attention: &mut DeviceTensor,
        _temp_buffer: &DeviceTensor,
    ) -> HipResult<()> {
        let attention_shape = attention.shape();

        if attention_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "Attention must be 2D for softmax".to_string(),
            ));
        }

        let seq_len = attention_shape.dims()[0];
        let cache_len = attention_shape.dims()[1];

        let mut attention_host = attention.to_host_vec()?;

        // Compute softmax row by row
        for i in 0..seq_len {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..cache_len {
                let idx = i * cache_len + j;
                if attention_host[idx] > max_val {
                    max_val = attention_host[idx];
                }
            }

            // Compute exp and sum
            let mut sum = 0.0f32;
            for j in 0..cache_len {
                let idx = i * cache_len + j;
                attention_host[idx] = (attention_host[idx] - max_val).exp();
                sum += attention_host[idx];
            }

            // Normalize
            for j in 0..cache_len {
                let idx = i * cache_len + j;
                attention_host[idx] /= sum;
            }
        }

        let shape = attention.shape().clone();
        *attention = DeviceTensor::from_host_vec(self, attention_host, shape)?;
        Ok(())
    }

    /// Compute attention-weighted V: attention @ V
    fn compute_attention_weighted_v(
        &self,
        attention: &DeviceTensor,
        v: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        // attention: [seq_len, cache_len]
        // V: [cache_len, num_heads, head_dim] -> reshape to [cache_len, num_heads * head_dim]
        // output: [seq_len, num_heads, head_dim]

        let attention_shape = attention.shape();
        let v_shape = v.shape();

        let seq_len = attention_shape.dims()[0];
        let cache_len = attention_shape.dims()[1];
        let num_heads = v_shape.dims()[1];
        let head_dim = v_shape.dims()[2];

        // Reshape V for matrix multiplication
        let v_flat_size = cache_len * num_heads * head_dim;
        let v_host = v.to_host_vec()?;

        let mut v_flat = vec![0.0f32; v_flat_size];
        for i in 0..cache_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let v_idx = i * num_heads * head_dim + h * head_dim + d;
                    let flat_idx = i * (num_heads * head_dim) + h * head_dim + d;
                    v_flat[flat_idx] = v_host[v_idx];
                }
            }
        }

        let v_flat_shape = TensorShape::from_dims(&[cache_len, num_heads * head_dim]);
        let v_reshaped = DeviceTensor::from_host_vec(self, v_flat, v_flat_shape)?;

        // Perform attention @ V
        self.gemm_attention_v(attention, &v_reshaped, output, seq_len, num_heads, head_dim)?;

        Ok(())
    }

    /// Perform batched GEMM for QK^T computation
    fn batched_gemm(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        output: &mut DeviceTensor,
        seq_q: usize,
        seq_k: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        // For now, implement as simple matrix multiplication
        // In a real implementation, this would use GPU batched GEMM

        let q_host = q.to_host_vec()?;
        let k_host = k.to_host_vec()?;
        let mut output_host = output.to_host_vec()?;

        // Q: [seq_q, num_heads * head_dim]
        // K: [seq_k, num_heads * head_dim] -> need K^T: [num_heads * head_dim, seq_k]
        // Output: [seq_q, seq_k]

        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut sum = 0.0f32;
                for l in 0..(num_heads * head_dim) {
                    sum += q_host[i * (num_heads * head_dim) + l]
                        * k_host[j * (num_heads * head_dim) + l];
                }
                output_host[i * seq_k + j] = sum;
            }
        }

        let shape = output.shape().clone();
        *output = DeviceTensor::from_host_vec(self, output_host, shape)?;
        Ok(())
    }

    /// Perform attention @ V matrix multiplication
    fn gemm_attention_v(
        &self,
        attention: &DeviceTensor,
        v: &DeviceTensor,
        output: &mut DeviceTensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        let attention_host = attention.to_host_vec()?;
        let v_host = v.to_host_vec()?;
        let mut output_host = output.to_host_vec()?;

        // attention: [seq_len, cache_len]
        // v: [cache_len, num_heads * head_dim]
        // output: [seq_len, num_heads * head_dim]

        let attention_shape = attention.shape();
        let cache_len = attention_shape.dims()[1];

        for i in 0..seq_len {
            for j in 0..(num_heads * head_dim) {
                let mut sum = 0.0f32;
                for k in 0..cache_len {
                    sum +=
                        attention_host[i * cache_len + k] * v_host[k * (num_heads * head_dim) + j];
                }
                output_host[i * (num_heads * head_dim) + j] = sum;
            }
        }

        let shape = output.shape().clone();
        *output = DeviceTensor::from_host_vec(self, output_host, shape)?;
        Ok(())
    }
}

// Include causal mask tests
#[cfg(test)]
#[cfg(feature = "rocm")]
include!("causal_mask_tests.rs");
