//! CPU reference backend for ggml IR.

use crate::ggml::{GgmlBackend, GgmlError, GgmlResult, Op, TensorDesc, TensorId};
use std::collections::HashMap;

#[derive(Default)]
pub struct CpuBackend {
    tensors: HashMap<TensorId, (TensorDesc, Vec<f32>)>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GgmlBackend for CpuBackend {
    type Buffer = Vec<f32>;

    fn alloc(&mut self, desc: &TensorDesc) -> GgmlResult<()> {
        let buffer = vec![0.0; desc.element_count()];
        self.tensors.insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn bind(&mut self, desc: &TensorDesc, buffer: Self::Buffer) -> GgmlResult<()> {
        self.tensors.insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn free(&mut self, id: TensorId) -> GgmlResult<()> {
        self.tensors.remove(&id);
        Ok(())
    }

    fn tensor_desc(&self, id: TensorId) -> Option<&TensorDesc> {
        self.tensors.get(&id).map(|(desc, _)| desc)
    }

    fn buffer(&self, id: TensorId) -> Option<&Self::Buffer> {
        self.tensors.get(&id).map(|(_, buf)| buf)
    }

    fn buffer_mut(&mut self, id: TensorId) -> Option<&mut Self::Buffer> {
        self.tensors.get_mut(&id).map(|(_, buf)| buf)
    }

    fn execute_op(
        &mut self,
        op: &Op,
        _inputs: &[TensorId],
        _outputs: &[TensorId],
    ) -> GgmlResult<()> {
        Err(GgmlError::Unimplemented(format!(
            "CPU backend op not implemented: {:?}",
            op
        )))
    }

    fn synchronize(&mut self) -> GgmlResult<()> {
        Ok(())
    }
}
