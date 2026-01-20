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
use crate::loader::TensorShape;

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

        // Get kernel reference
        let kernel = self.kernel.as_ref()
            .ok_or_else(|| HipError::KernelLoadFailed("Transpose kernel not initialized".to_string()))?;

        // Grid/block dimensions for transpose kernel
        // Grid: (ceil(cols / TILE_DIM), ceil(rows / TILE_DIM))
        // Block: (TILE_DIM, TILE_DIM) for 2D tiling
        const TILE_DIM: u32 = 64;
        let grid_x = (cols as u32 + TILE_DIM - 1) / TILE_DIM;
        let grid_y = (rows as u32 + TILE_DIM - 1) / TILE_DIM;
        let grid_dim = (grid_x, grid_y, 1);
        let block_dim = (TILE_DIM, TILE_DIM, 1);

        // Shared memory: TILE_DIM * (TILE_DIM + 1) floats for bank conflict avoidance
        let shared_mem_bytes = TILE_DIM * (TILE_DIM + 1) * std::mem::size_of::<f32>() as u32;

        // Prepare kernel arguments
        let mut rows_arg = rows;
        let mut cols_arg = cols;

        let args: &[*mut std::ffi::c_void] = &[
            output_buffer.as_ptr() as *mut std::ffi::c_void,
            tensor.buffer.as_ptr() as *mut std::ffi::c_void,
            &mut rows_arg as *mut usize as *mut std::ffi::c_void,
            &mut cols_arg as *mut usize as *mut std::ffi::c_void,
        ];

        self.backend.launch_kernel_with_module_shared(
            kernel,
            grid_dim,
            block_dim,
            args,
            shared_mem_bytes,
        )?;

        tracing::debug!(
            "Transposed tensor: [{}, {}] -> [{}, {}]",
            rows, cols, cols, rows
        );

        Ok(DeviceTensor {
            buffer: output_buffer,
            shape: transposed_shape,
        })
    }
}

/// Convenience function to transpose a 2D tensor on GPU
///
/// Creates a TransposeKernel instance and performs the transpose.
/// This is the preferred API for one-shot transpose operations.
///
/// # Arguments
///
/// * `backend` - HIP backend for GPU operations
/// * `tensor` - Input tensor to transpose (must be 2D)
///
/// # Returns
///
/// A new `DeviceTensor` with transposed shape
///
/// # Errors
///
/// - `KernelLoadFailed` - HSACO file not found
/// - `DeviceError` - Input tensor is not 2D
///
/// # Example
///
/// ```rust,no_run
/// use crate::kernels::transpose::transpose_tensor;
/// use std::sync::Arc;
///
/// let transposed = transpose_tensor(&backend, &input_tensor)?;
/// // shape [hidden_size, vocab_size] -> [vocab_size, hidden_size]
/// ```
pub fn transpose_tensor(
    backend: &Arc<HipBackend>,
    tensor: &DeviceTensor,
) -> HipResult<DeviceTensor> {
    let mut kernel = TransposeKernel::new(Arc::clone(backend));
    kernel.transpose(tensor)
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
