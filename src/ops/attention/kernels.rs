//! GPU Attention Kernel Compilation and Management

use crate::backend::{
    hip_blas::{HipBlasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T},
    DeviceTensor, HipBackend, HipError, HipResult,
};
#[cfg(feature = "rocm")]
use crate::backend::{HipKernel, HipModule};
#[cfg(feature = "rocm")]
use once_cell::sync::OnceCell;
use crate::tensor::matmul::matmul_f32;

#[cfg(feature = "rocm")]
use super::hiprtc;

#[cfg(feature = "rocm")]
pub struct CompiledKernel {
    #[allow(dead_code)]
    module: HipModule,
    pub kernel: HipKernel,
}

pub struct HipAttentionKernels {
    backend: HipBackend,
    blas_handle: HipBlasHandle,
    #[cfg(feature = "rocm")]
    attention_softmax_kernel: OnceCell<CompiledKernel>,
    #[cfg(feature = "rocm")]
    causal_mask_kernel: OnceCell<CompiledKernel>,
}

impl HipAttentionKernels {
    pub fn new(backend: &HipBackend) -> HipResult<Self> {
        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        blas_handle
            .set_stream(backend.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        Ok(HipAttentionKernels {
            backend: backend.clone(),
            blas_handle,
            #[cfg(feature = "rocm")]
            attention_softmax_kernel: OnceCell::new(),
            #[cfg(feature = "rocm")]
            causal_mask_kernel: OnceCell::new(),
        })
    }

    pub fn backend(&self) -> &HipBackend {
        &self.backend
    }

    pub fn blas_handle(&self) -> &HipBlasHandle {
        &self.blas_handle
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
    pub fn get_attention_softmax_kernel(&self) -> HipResult<&CompiledKernel> {
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
    pub fn get_causal_mask_kernel(&self) -> HipResult<&CompiledKernel> {
        self.causal_mask_kernel
            .get_or_try_init(|| self.compile_causal_mask_kernel())
    }

    /// Compute QK^T using hipBLAS
    pub fn compute_qk_t(
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

        self.compute_qk_t_gemm(q, k, output)
    }

    fn compute_qk_t_gemm(
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

        crate::backend::hip_blas::sgemm(
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
        use std::ffi::c_void;

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

    /// Compute softmax of attention scores
    pub fn compute_softmax(
        &self,
        attention: &mut DeviceTensor,
        _temp_buffer: &DeviceTensor,
    ) -> HipResult<()> {
        

        #[cfg(feature = "rocm")]
        {
            if let Err(err) = self.compute_softmax_gpu(attention) {
                tracing::warn!("hip attention softmax fallback to CPU: {}", err);
            } else {
                return Ok(());
            }
        }

        let _ = _temp_buffer;
        self.compute_softmax_cpu_fallback(attention, _temp_buffer)
    }

    #[cfg(feature = "rocm")]
    fn compute_softmax_gpu(&self, attention: &mut DeviceTensor) -> HipResult<()> {
        use std::ffi::c_void;

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

    fn compute_softmax_cpu_fallback(
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

    /// Compute attention-weighted V: attention @ V
    pub fn compute_attention_weighted_v(
        &self,
        attention: &DeviceTensor,
        v: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
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
pub(crate) const ATTENTION_SOFTMAX_KERNEL: &str = r#"
extern "C" __global__ void attention_softmax(float* scores, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float max_val = -3.402823466e+38f;
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
pub(crate) const CAUSAL_MASK_KERNEL: &str = r#"
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

    for (int key_pos = tid; key_pos < seq_len; key_pos += WARP_SIZE) {
        if (key_pos > query_pos) {
            attention[row_offset + key_pos] = -(__builtin_inff());
        }
    }
}
"#;
