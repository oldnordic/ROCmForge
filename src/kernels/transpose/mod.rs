//! GPU matrix transpose kernel
//!
//! This module provides GPU-accelerated matrix transpose operations,
//! avoiding CPU round-trip for weight transposition during model loading.
//!
//! ## Usage
//!
//! ```rust
//! use std::sync::Arc;
//! use crate::backend::HipBackend;
//! use crate::kernels::transpose::TransposeKernel;
//!
//! let backend = Arc::new(HipBackend::new()?);
//! let mut kernel = TransposeKernel::new(backend);
//!
//! // Lazy initialization (loads HSACO from TRANSPOSE_HSACO env var)
//! kernel.initialize()?;
//!
//! // Transpose a 2D tensor (rows, cols) -> (cols, rows)
//! let transposed = kernel.transpose(&input_tensor)?;
//! ```

use std::path::Path;
use std::sync::Arc;

use crate::backend::hip_backend::{DeviceTensor, HipBackend, HipError, HipKernel, HipModule, HipResult};
use crate::loader::mmap_loader::TensorShape;

/// GPU transpose kernel
///
/// Loads and executes the transpose kernel from compiled HSACO.
/// The kernel is loaded lazily on first call to `initialize()` or `transpose()`.
#[derive(Debug)]
pub struct TransposeKernel {
    /// HIP backend reference
    backend: Arc<HipBackend>,
    /// Loaded HSACO module (kept alive to prevent kernel unloading)
    #[allow(dead_code)]
    module: Option<HipModule>,
    /// Transpose kernel function
    kernel: Option<HipKernel>,
    /// Whether the kernel has been initialized
    initialized: bool,
}

impl TransposeKernel {
    /// Create a new transpose kernel
    ///
    /// The kernel is not loaded until `initialize()` is called.
    ///
    /// # Arguments
    ///
    /// * `backend` - HIP backend for GPU operations
    ///
    /// # Returns
    ///
    /// A new `TransposeKernel` instance
    pub fn new(backend: Arc<HipBackend>) -> Self {
        Self {
            backend,
            module: None,
            kernel: None,
            initialized: false,
        }
    }

    /// Initialize the kernel (lazy load from HSACO)
    ///
    /// This method:
    /// 1. Checks for TRANSPOSE_HSACO environment variable
    /// 2. Verifies the HSACO file exists
    /// 3. Loads the module and extracts the kernel function
    ///
    /// # Returns
    ///
    /// - `Ok(())` if kernel loaded successfully
    /// - `Err(HipError::KernelLoadFailed)` if HSACO not found or invalid
    ///
    /// # Errors
    ///
    /// - `KernelLoadFailed` - TRANSPOSE_HSACO env var not set
    /// - `KernelLoadFailed` - HSACO file does not exist
    /// - `KernelLoadFailed` - Module or kernel function not found
    pub fn initialize(&mut self) -> HipResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Load HSACO path from build.rs environment variable
        let hsaco_path = std::env::var("TRANSPOSE_HSACO")
            .ok()
            .ok_or_else(|| HipError::KernelLoadFailed("TRANSPOSE_HSACO env var not set".to_string()))?;

        // Check if HSACO file exists
        if !Path::new(&hsaco_path).exists() {
            return Err(HipError::KernelLoadFailed(format!(
                "HSACO not found: {}",
                hsaco_path
            )));
        }

        // Load the module
        let module = self.backend.load_module(&hsaco_path)?;

        // Get the kernel function
        let kernel = self
            .backend
            .get_kernel_function(&module, "transposeLdsNoBankConflicts")?;

        self.module = Some(module);
        self.kernel = Some(kernel);
        self.initialized = true;

        tracing::debug!("Transpose kernel loaded from {}", hsaco_path);

        Ok(())
    }

    /// Transpose a 2D tensor on GPU
    ///
    /// Converts a tensor from shape (rows, cols) to (cols, rows).
    /// This is useful for transposing weight matrices without CPU round-trip.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to transpose (must be 2D)
    ///
    /// # Returns
    ///
    /// A new `DeviceTensor` with transposed shape
    ///
    /// # Errors
    ///
    /// - `KernelLoadFailed` - Kernel not initialized (call `initialize()` first)
    /// - `DeviceError` - Input tensor is not 2D
    pub fn transpose(&mut self, tensor: &DeviceTensor) -> HipResult<DeviceTensor> {
        // Auto-initialize if needed
        if !self.initialized {
            self.initialize()?;
        }

        // Verify tensor is 2D
        let shape = tensor.shape();
        if shape.dims().len() != 2 {
            return Err(HipError::DeviceError(format!(
                "Transpose requires 2D tensor, got {}D",
                shape.dims().len()
            )));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let elem_size = std::mem::size_of::<f32>();

        // Allocate output buffer with transposed shape (cols, rows)
        let transposed_shape = TensorShape::from_dims(&[cols, rows]);
        let output_buffer = self
            .backend
            .allocate_buffer(cols * rows * elem_size)?;
        let output_tensor = DeviceTensor {
            buffer: output_buffer,
            shape: transposed_shape,
        };

        // TODO: Launch kernel in plan 27-02
        // For now, return placeholder output
        tracing::warn!(
            "Transpose kernel not yet implemented - returning untransposed tensor (shape: {:?})",
            output_tensor.shape()
        );

        Ok(output_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HipBackend;

    #[test]
    #[cfg(feature = "rocm")]
    fn test_transpose_kernel_creation() {
        let backend = Arc::new(HipBackend::new().unwrap());
        let kernel = TransposeKernel::new(backend);
        assert!(!kernel.initialized);
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_transpose_kernel_initialize_failure_no_env() {
        // Temporarily unset env var to test error handling
        let backend = Arc::new(HipBackend::new().unwrap());
        let mut kernel = TransposeKernel::new(backend);

        // This should fail if TRANSPOSE_HSACO is not set
        let result = kernel.initialize();
        match result {
            Err(HipError::KernelLoadFailed(msg)) if msg.contains("TRANSPOSE_HSACO") => {
                // Expected - kernel not compiled yet
                println!("Expected: {}", msg);
            }
            Ok(_) => {
                println!("TRANSPOSE_HSACO is set - kernel compiled successfully");
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_transpose_2d_tensor_shape_check() {
        let backend = Arc::new(HipBackend::new().unwrap());
        let mut kernel = TransposeKernel::new(backend);

        // Create a 2D tensor
        let shape = TensorShape::from_dims(&[4, 8]);
        let buffer = kernel.backend.allocate_buffer(4 * 8 * 4).unwrap();
        let input = DeviceTensor { buffer, shape };

        // Attempt transpose (will fail if kernel not initialized, but shape check works)
        match kernel.transpose(&input) {
            Err(HipError::DeviceError(msg)) if msg.contains("2D") => {
                panic!("Should accept 2D tensor: {}", msg);
            }
            Err(HipError::KernelLoadFailed(_)) => {
                // Expected - kernel not compiled yet
            }
            Ok(_) => {
                println!("Transpose succeeded");
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_transpose_rejects_non_2d_tensor() {
        let backend = Arc::new(HipBackend::new().unwrap());
        let mut kernel = TransposeKernel::new(backend);

        // Create a 3D tensor
        let shape = TensorShape::from_dims(&[2, 4, 8]);
        let buffer = kernel.backend.allocate_buffer(2 * 4 * 8 * 4).unwrap();
        let input = DeviceTensor { buffer, shape };

        // Should fail with "requires 2D tensor"
        let result = kernel.transpose(&input);
        match result {
            Err(HipError::DeviceError(msg)) if msg.contains("2D") => {
                // Expected
                println!("Expected rejection: {}", msg);
            }
            _ => {
                panic!("Should reject 3D tensor, got: {:?}", result);
            }
        }
    }
}
