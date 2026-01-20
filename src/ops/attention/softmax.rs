//! Attention Operations
//!
//! Re-exports from the original attention_gpu.rs module.
//! This file contains the attention computation pipeline implementations.

#![allow(deprecated)]


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

#[cfg(feature = "rocm")]
use super::hiprtc;
#[cfg(feature = "rocm")]
use super::kernels::CompiledKernel;

/// QK^T matrix multiplication operation
pub struct QkMatmul {
    backend: HipBackend,
    blas_handle: HipBlasHandle,
}

impl QkMatmul {
    pub fn new(backend: &HipBackend) -> HipResult<Self> {
        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        blas_handle
            .set_stream(backend.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        Ok(QkMatmul {
            backend: backend.clone(),
            blas_handle,
        })
    }

    pub fn backend(&self) -> &HipBackend {
        &self.backend
    }

    pub fn blas_handle(&self) -> &HipBlasHandle {
        &self.blas_handle
    }

    pub fn compute(
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

        let (_seq_q, num_heads_q, head_dim_q) =
            (q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2]);
        let (_seq_k, num_heads_k, head_dim_k) =
            (k_shape.dims()[0], k_shape.dims()[1], k_shape.dims()[2]);

        if num_heads_q != num_heads_k || head_dim_q != head_dim_k {
            return Err(HipError::GenericError(
                "Q and K must have matching num_heads and head_dim".to_string(),
            ));
        }

        self.compute_gemm(q, k, output)
    }

    fn compute_gemm(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        let q_shape = q.shape();
        let k_shape = k.shape();

        let seq_q = q_shape.dims()[0];
        let seq_k = k_shape.dims()[0];
        let num_heads = q_shape.dims()[1];
        let head_dim = q_shape.dims()[2];

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
}

/// Causal mask operation
pub struct CausalMaskOp {
    backend: HipBackend,
    #[cfg(feature = "rocm")]
    causal_mask_kernel: OnceCell<CompiledKernel>,
}

impl CausalMaskOp {
    pub fn new(backend: &HipBackend) -> Self {
        CausalMaskOp {
            backend: backend.clone(),
            #[cfg(feature = "rocm")]
            causal_mask_kernel: OnceCell::new(),
        }
    }

    pub fn apply(
        &self,
        attention: &mut DeviceTensor,
        seq_len: usize,
        cache_len: usize,
    ) -> HipResult<()> {
        #[cfg(feature = "rocm")]
        {
            if let Err(err) = self.apply_gpu(attention, seq_len, cache_len) {
                tracing::warn!("hip attention mask fallback to CPU: {}", err);
            } else {
                return Ok(());
            }
        }

        self.apply_cpu_fallback(attention, seq_len, cache_len)
    }

    #[cfg(feature = "rocm")]
    fn compile_causal_mask_kernel(&self) -> HipResult<CompiledKernel> {
        let code = hiprtc::compile_kernel("causal_mask", super::kernels::CAUSAL_MASK_KERNEL)?;
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

    #[cfg(feature = "rocm")]
    fn apply_gpu(
        &self,
        attention: &mut DeviceTensor,
        seq_len: usize,
        cache_len: usize,
    ) -> HipResult<()> {
        let attention_shape = attention.shape();
        let dims = attention_shape.dims();

        if dims.len() == 2 {
            let kernel = self.get_causal_mask_kernel()?;
            let grid_dim = (seq_len as u32, 1, 1);
            let block_dim = 32;

            let _seq_len_i32 = seq_len as i32;
            let mut cache_len_i32 = cache_len as i32;
            let mut attention_ptr = attention.buffer().as_ptr();

            let args = [
                &mut attention_ptr as *mut _ as *mut c_void,
                &mut 1i32 as *mut _ as *mut c_void,
                &mut cache_len_i32 as *mut _ as *mut c_void,
                &mut 1i32 as *mut _ as *mut c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                &kernel.kernel,
                grid_dim,
                (block_dim, 1, 1),
                &args,
                0,
            )
        } else if dims.len() == 4 {
            let kernel = self.get_causal_mask_kernel()?;
            let batch_size = dims[0];
            let num_heads = dims[1];

            let grid_dim = (seq_len as u32, num_heads as u32, batch_size as u32);
            let block_dim = 32;

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
                0,
            )
        } else {
            Err(HipError::GenericError(format!(
                "Unexpected attention tensor shape: {:?}",
                dims
            )))
        }
    }

    fn apply_cpu_fallback(
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
}

/// Attention softmax operation
pub struct AttentionSoftmax {
    backend: HipBackend,
    #[cfg(feature = "rocm")]
    attention_softmax_kernel: OnceCell<CompiledKernel>,
}

impl AttentionSoftmax {
    pub fn new(backend: &HipBackend) -> Self {
        AttentionSoftmax {
            backend: backend.clone(),
            #[cfg(feature = "rocm")]
            attention_softmax_kernel: OnceCell::new(),
        }
    }

    pub fn compute(
        &self,
        attention: &mut DeviceTensor,
        _temp_buffer: &DeviceTensor,
    ) -> HipResult<()> {
        #[cfg(feature = "rocm")]
        {
            if let Err(err) = self.compute_gpu(attention) {
                tracing::warn!("hip attention softmax fallback to CPU: {}", err);
            } else {
                return Ok(());
            }
        }

        self.compute_cpu_fallback(attention, _temp_buffer)
    }

    #[cfg(feature = "rocm")]
    fn compile_attention_softmax_kernel(&self) -> HipResult<CompiledKernel> {
        let code = hiprtc::compile_kernel("attention_softmax", super::kernels::ATTENTION_SOFTMAX_KERNEL)?;
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
    fn compute_gpu(&self, attention: &mut DeviceTensor) -> HipResult<()> {
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

    fn compute_cpu_fallback(
        &self,
        attention: &mut DeviceTensor,
        _temp_buffer: &DeviceTensor,
    ) -> HipResult<()> {
        let mut attention_host = attention.to_host_vec()?;
        let attention_shape = attention.shape();
        let rows = attention_shape.dims()[0];
        let cols = attention_shape.dims()[1];

        for i in 0..rows {
            let row_start = i * cols;
            let row_end = row_start + cols;

            let mut max_val = f32::NEG_INFINITY;
            for j in row_start..row_end {
                if attention_host[j] > max_val {
                    max_val = attention_host[j];
                }
            }

            let mut sum = 0.0f32;
            for j in row_start..row_end {
                if attention_host[j] != f32::NEG_INFINITY {
                    attention_host[j] = (attention_host[j] - max_val).exp();
                    sum += attention_host[j];
                } else {
                    attention_host[j] = 0.0f32;
                }
            }

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
}

/// Weighted matmul operation
pub struct WeightedMatmul {
    backend: HipBackend,
    blas_handle: HipBlasHandle,
}

impl WeightedMatmul {
    pub fn new(qk_matmul: &QkMatmul) -> HipResult<Self> {
        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        blas_handle
            .set_stream(qk_matmul.backend().stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        Ok(WeightedMatmul {
            backend: qk_matmul.backend().clone(),
            blas_handle,
        })
    }

    pub fn compute(
        &self,
        attention: &DeviceTensor,
        v: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        match self.compute_gemm(attention, v, output) {
            Ok(_) => Ok(()),
            Err(err) => {
                tracing::warn!("hipBLAS attention*V fallback to CPU: {}", err);
                self.compute_cpu_fallback(attention, v, output)
            }
        }
    }

    fn compute_gemm(
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
            &self.backend,
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

    fn compute_cpu_fallback(
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

/// Complete attention computation
pub fn compute_attention(
    qk_matmul: &QkMatmul,
    causal_mask: &CausalMaskOp,
    softmax: &AttentionSoftmax,
    q: &DeviceTensor,
    kv_cache: &KVCache,
    layer_id: usize,
    current_seq_len: usize,
) -> HipResult<DeviceTensor> {
    let backend = qk_matmul.backend();

    let q_shape = q.shape();
    if q_shape.dims().len() != 3 {
        return Err(HipError::GenericError(
            "Q tensor must be 3D [seq_len, num_heads, head_dim]".to_string(),
        ));
    }

    let (seq_len, num_heads, head_dim) =
        (q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2]);

    let (k_tensor, v_tensor) = kv_cache.retrieve(layer_id, current_seq_len)?;

    let attention_shape = TensorShape::from_dims(&[seq_len, current_seq_len]);
    let mut attention_scores = DeviceTensor::empty(backend, attention_shape)?;
    qk_matmul.compute(q, &k_tensor, &mut attention_scores)?;

    backend.scale_inplace(&mut attention_scores, 1.0 / (head_dim as f32).sqrt())?;

    causal_mask.apply(&mut attention_scores, seq_len, current_seq_len)?;

    let temp_buffer = DeviceTensor::empty(backend, TensorShape::from_dims(&[1, 1]))?;
    softmax.compute(&mut attention_scores, &temp_buffer)?;

    let weighted_matmul = WeightedMatmul::new(qk_matmul)?;
    let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
    let mut output = DeviceTensor::empty(backend, output_shape)?;
    weighted_matmul.compute(&attention_scores, &v_tensor, &mut output)?;

    Ok(output)
}

#[cfg(test)]
#[cfg(feature = "rocm")]
include!("causal_mask_tests.rs");
