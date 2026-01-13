//! Backend interface for ggml IR execution.

use crate::ggml::{GgmlResult, Op, TensorDesc, TensorId};

pub trait GgmlBackend {
    type Buffer;

    fn alloc(&mut self, desc: &TensorDesc) -> GgmlResult<()>;
    fn bind(&mut self, desc: &TensorDesc, buffer: Self::Buffer) -> GgmlResult<()>;
    fn free(&mut self, id: TensorId) -> GgmlResult<()>;
    fn tensor_desc(&self, id: TensorId) -> Option<&TensorDesc>;
    fn buffer(&self, id: TensorId) -> Option<&Self::Buffer>;
    fn buffer_mut(&mut self, id: TensorId) -> Option<&mut Self::Buffer>;

    fn execute_op(
        &mut self,
        op: &Op,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> GgmlResult<()>;

    fn synchronize(&mut self) -> GgmlResult<()>;
}
