//! GPU Attention Kernel Compilation and Management

use crate::backend::{HipBackend, HipError, HipResult};
#[cfg(feature = "rocm")]
use crate::backend::{HipKernel, HipModule};
#[cfg(feature = "rocm")]
use once_cell::sync::OnceCell;

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
    #[cfg(feature = "rocm")]
    attention_softmax_kernel: OnceCell<CompiledKernel>,
    #[cfg(feature = "rocm")]
    causal_mask_kernel: OnceCell<CompiledKernel>,
}

impl HipAttentionKernels {
    pub fn new(backend: &HipBackend) -> Self {
        HipAttentionKernels {
            backend: backend.clone(),
            #[cfg(feature = "rocm")]
            attention_softmax_kernel: OnceCell::new(),
            #[cfg(feature = "rocm")]
            causal_mask_kernel: OnceCell::new(),
        }
    }

    pub fn backend(&self) -> &HipBackend {
        &self.backend
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
