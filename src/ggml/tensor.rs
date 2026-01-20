//! Tensor descriptors for ggml IR.

use crate::ggml::{layout::Layout, shape::element_count};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    I32,
    U32,
    F32,
    F16,
    Q8_0,
    Q4_0,
    Mxfp4,
    Mxfp6,
}

#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub layout: Layout,
    pub strides: Vec<usize>,
    pub byte_offset: usize,
    pub view_of: Option<TensorId>,
}




impl TensorDesc {
    pub fn new(shape: Vec<usize>, dtype: DType, layout: Layout) -> Self {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride: usize = 1;
        for dim in shape.iter().rev() {
            strides.push(stride);
            stride = stride.saturating_mul(*dim);
        }
        strides.reverse();
        Self {
            id: TensorId(0),
            shape,
            dtype,
            layout,
            strides,
            byte_offset: 0,
            view_of: None,
        }
    }

    pub fn view_of(mut self, source: TensorId, byte_offset: usize) -> Self {
        self.view_of = Some(source);
        self.byte_offset = byte_offset;
        self
    }

    pub fn is_view(&self) -> bool {
        self.view_of.is_some()
    }

    pub fn element_count(&self) -> usize {
        element_count(&self.shape)
    }

    pub fn element_size(&self) -> usize {
        match self.dtype {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::I32 => 4,
            DType::U32 => 4,
            DType::Q8_0 => 1,
            DType::Q4_0 => 1,
            DType::Mxfp4 | DType::Mxfp6 => 1,
        }
    }

    pub fn byte_size(&self) -> usize {
        self.element_count().saturating_mul(self.element_size())
    }

    pub fn set_shape(&mut self, shape: Vec<usize>) {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride: usize = 1;
        for dim in shape.iter().rev() {
            strides.push(stride);
            stride = stride.saturating_mul(*dim);
        }
        strides.reverse();
        self.shape = shape;
        self.strides = strides;
    }
}
