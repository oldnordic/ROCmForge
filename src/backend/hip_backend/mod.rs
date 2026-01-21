//! ROCm/HIP backend for GPU kernel execution
//!
//! This module provides the HIP backend implementation for AMD GPUs.

mod async_ops;
mod backend;
mod device;
mod device_context;
mod error;
mod event;
mod ffi;
pub mod memory;  // pub for HipBufferInner access in backend.rs
mod module;
mod runtime;
mod stream;

#[cfg(test)]
mod matmul_sync_tests;

// Re-export memory module types for backward compatibility
pub use memory::HipBuffer;
pub use memory::HipBufferInner;

// Public API re-exports
pub use async_ops::{AsyncLoader, NUM_UPLOAD_STREAMS};
pub use backend::{DeviceTensor, HipBackend, synchronize_device};
pub use device::{HipDevice, HipDeviceProp, hipUUID, get_error_string};
pub use error::{HipError, HipResult};
pub use event::HipEvent;
pub use module::{HipKernel, HipModule};
pub use runtime::ModelRuntime;
pub use stream::HipStream;

// Re-export FFI functions (for test access)
pub use ffi::{
    hipGetDeviceCount,
    hipGetDeviceProperties,
};
