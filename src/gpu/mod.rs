//! AMD GPU inference backend with HIP.

mod error;
mod ffi;
mod detect;

pub use error::{GpuError, GpuResult};
pub use detect::GpuCapabilities;
