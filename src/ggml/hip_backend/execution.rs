//! ggml IR operation execution for HIP backend.
//!
//! This module provides the main execution entry point for GGML IR operations.
//! Individual operation implementations are delegated to the op_dispatch module.

use crate::ggml::{GgmlResult, Op, TensorId};

use super::HipGgmlBackend;

// op_dispatch module provides the execute_* operation implementations
// Both execution.rs and op_dispatch.rs define impl HipGgmlBackend blocks
// that get merged by the compiler since they're in the same crate module

impl HipGgmlBackend {
    /// Execute a ggml IR operation.
    ///
    /// This is the main entry point for executing GGML operations. It validates
    /// the operation type and dispatches to the appropriate implementation in
    /// the op_dispatch module.
    pub fn execute_op(
        &mut self,
        op: &Op,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> GgmlResult<()> {
        eprintln!(">>> execute_op: op={:?}", op);
        match op {
            Op::GetRows => {
                self.execute_get_rows(inputs, outputs)
            }
            Op::MatMul => {
                self.execute_matmul(inputs, outputs)
            }
            Op::Add => {
                self.execute_add(inputs, outputs)
            }
            Op::Scale { factor } => {
                self.execute_scale(inputs, outputs, *factor)
            }
            Op::LayerNorm { eps } => {
                self.execute_layernorm(inputs, outputs, *eps)
            }
            Op::RmsNorm { eps } => {
                self.execute_rmsnorm(inputs, outputs, *eps)
            }
            Op::Rope => {
                self.execute_rope(inputs, outputs)
            }
            Op::Softmax => {
                self.execute_softmax(inputs, outputs)
            }
            Op::Attention => {
                self.execute_attention(inputs, outputs)
            }
            Op::Mask => {
                self.execute_mask(inputs, outputs)
            }
            Op::SwiGlu => {
                self.execute_swiglu(inputs, outputs)
            }
            Op::MlpSwiglu => {
                self.execute_mlp_swiglu(inputs, outputs)
            }
            Op::SplitQkv => {
                self.execute_split_qkv(inputs, outputs)
            }
            Op::View | Op::Reshape => {
                self.execute_view_reshape(inputs, outputs)
            }
            Op::Copy => {
                self.execute_copy(inputs, outputs)
            }
            Op::MatMulQ4_0 => {
                self.execute_matmul_q4_0(inputs, outputs)
            }
            Op::MatMulQ8_0 => {
                self.execute_matmul_q8_0(inputs, outputs)
            }
            Op::Accumulate { offset } => {
                self.execute_accumulate(inputs, outputs, *offset)
            }
        }
    }
}
