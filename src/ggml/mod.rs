//! ggml-style IR and backend contracts.

pub mod backend;
pub mod cpu_backend;
pub mod executor;
pub mod graph;
pub mod layout;
pub mod op;
pub mod shape;
pub mod tensor;
pub mod validate;

pub mod hip_backend;

#[derive(Debug, Clone)]
pub enum GgmlError {
    InvalidShape(String),
    InvalidLayout(String),
    Backend(String),
    Unimplemented(String),
}

pub type GgmlResult<T> = Result<T, GgmlError>;

pub use backend::GgmlBackend;
pub use graph::{Graph, Node};
pub use layout::Layout;
pub use op::Op;
pub use tensor::{DType, TensorDesc, TensorId};
