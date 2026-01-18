//! ROCm/HIP backend for GPU kernel execution
//!
//! This module provides the HIP backend implementation for AMD GPUs.

mod backend;

// Public API re-exports from backend module
pub use backend::{
    // Error types
    HipError, HipResult,
    // Device types
    HipDevice, HipDeviceProp, hipUUID,
    // Stream and event
    HipStream, HipEvent,
    // Buffer types (HipBufferInner is private)
    HipBuffer,
    // Module and kernel
    HipModule, HipKernel,
    // Main backend
    HipBackend,
    // Device tensor
    DeviceTensor,
    // Model runtime
    ModelRuntime,
    // Async loader
    AsyncLoader,
    // Utility functions
    synchronize_device,
    // FFI functions (re-exported for test access)
    // Note: These are pub in the extern "C" block but need explicit re-export
    hipGetDeviceCount,
    hipGetDeviceProperties,
};
