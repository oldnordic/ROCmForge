//! Graph executor for ggml IR.

use crate::ggml::{GgmlBackend, GgmlResult, Graph};

pub fn execute_graph<B: GgmlBackend>(backend: &mut B, graph: &Graph) -> GgmlResult<()> {
    for desc in &graph.tensors {
        if desc.is_view() {
            continue;
        }
        if backend.buffer(desc.id).is_none() {
            backend.alloc(desc)?;
        }
    }
    for node in &graph.nodes {
        backend.execute_op(&node.op, &node.inputs, &node.outputs)?;
    }
    backend.synchronize()
}
