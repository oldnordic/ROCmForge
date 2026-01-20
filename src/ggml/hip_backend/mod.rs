//! HIP backend for ggml IR.

pub mod backend;
pub mod buffer;
pub mod ops;
pub mod tuning;
mod capability;
mod execution;

pub use backend::HipGgmlBackend;
pub use tuning::{GpuArchitecture, KernelTuning};

use crate::ggml::{GgmlBackend, GgmlError, GgmlResult, TensorDesc, TensorId};
use std::sync::Arc;

impl GgmlBackend for HipGgmlBackend {
    type Buffer = crate::backend::HipBuffer;

    fn alloc(&mut self, desc: &TensorDesc) -> GgmlResult<()> {
        let bytes = desc.byte_size();

        // Get backend reference first to avoid borrow conflicts
        let backend_ref = self.hip_backend().clone();

        // Skip allocation for zero-byte tensors (empty tensors from KV cache at current_len=0)
        // These are handled as views and don't need GPU buffers
        if bytes == 0 {
            let buffer = backend_ref.dummy_zero_buffer()
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
            self.tensors_mut().insert(desc.id, (desc.clone(), buffer));
            return Ok(());
        }

        // Try to allocate from pool if allocator is enabled
        let buffer = if let Some(alloc) = self.allocator() {
            let backend_ref_clone = backend_ref.clone();
            alloc.allocate(bytes, |size| {
                backend_ref_clone
                    .allocate_buffer(size)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e| GgmlError::Backend(e))?
        } else {
            backend_ref
                .allocate_buffer(bytes)
                .map_err(|e| GgmlError::Backend(e.to_string()))?
        };

        self.tensors_mut().insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn bind(&mut self, desc: &TensorDesc, buffer: Self::Buffer) -> GgmlResult<()> {
        self.tensors_mut().insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn free(&mut self, id: TensorId) -> GgmlResult<()> {
        if let Some((desc, buffer)) = self.tensors_mut().remove(&id) {
            // Return buffer to allocator if enabled
            if let Some(alloc) = self.allocator() {
                alloc.free(buffer, desc.byte_size());
            }
            // Otherwise buffer is dropped (deallocated)
        }
        Ok(())
    }

    fn tensor_desc(&self, id: TensorId) -> Option<&TensorDesc> {
        self.tensors().get(&id).map(|(desc, _)| desc)
    }

    fn buffer(&self, id: TensorId) -> Option<&Self::Buffer> {
        self.tensors().get(&id).map(|(_, buf)| buf)
    }

    fn buffer_mut(&mut self, id: TensorId) -> Option<&mut Self::Buffer> {
        self.tensors_mut().get_mut(&id).map(|(_, buf)| buf)
    }

    fn execute_op(
        &mut self,
        op: &crate::ggml::Op,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> GgmlResult<()> {
        self.execute_op(op, inputs, outputs)
    }

    fn synchronize(&mut self) -> GgmlResult<()> {
        // PHASE 01 FIX: Actually synchronize instead of being a no-op
        //
        // Previously this was a no-op that just returned Ok(()).
        // This caused hangs because GPU operations queued on the backend's
        // stream weren't completing before the caller tried to read results.
        self.hip_backend()
            .synchronize()
            .map_err(|e| GgmlError::Backend(e.to_string()))
    }
}
