//! AMD GPU inference backend with HIP.

mod error;
mod ffi;
mod detect;
mod device;

pub use error::{GpuError, GpuResult};
pub use detect::GpuCapabilities;
pub use device::GpuDevice;
