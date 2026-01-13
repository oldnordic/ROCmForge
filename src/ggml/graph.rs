//! Graph structures for ggml IR.

use crate::ggml::{Op, TensorDesc, TensorId};

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub tensors: Vec<TensorDesc>,
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(&mut self, mut desc: TensorDesc) -> TensorId {
        let id = TensorId(self.tensors.len());
        desc.id = id;
        self.tensors.push(desc);
        id
    }

    pub fn add_node(&mut self, op: Op, inputs: Vec<TensorId>, outputs: Vec<TensorId>) {
        self.nodes.push(Node { op, inputs, outputs });
    }
}
