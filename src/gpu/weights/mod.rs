//! GPU weight storage in VRAM.

mod buffer;
mod layer;
mod metadata;
mod model;
mod upload;

pub use buffer::{GpuBuffer, GpuPinnedBuffer};
pub use layer::GpuLayerWeights;
pub use metadata::{TensorRole, WeightMeta};
pub use model::GpuModelWeights;
