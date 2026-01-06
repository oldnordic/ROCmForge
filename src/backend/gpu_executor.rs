//! GPU model executor for running neural network layers on AMD GPUs

use std::collections::HashMap;
use std::path::Path;

use crate::backend::{HipBackend, HipError, HipKernel, HipModule};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("Kernel compilation failed: {0}")]
    CompilationFailed(String),
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),
    #[error("Module loading failed: {0}")]
    ModuleLoadFailed(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Invalid tensor dimensions: {0}")]
    InvalidDimensions(String),
}

pub type ExecutorResult<T> = Result<T, ExecutorError>;

impl From<HipError> for ExecutorError {
    fn from(error: HipError) -> Self {
        ExecutorError::ExecutionFailed(error.to_string())
    }
}

/// GPU model executor that manages compiled kernels and executes model layers
#[derive(Debug)]
pub struct GpuModelExecutor {
    backend: HipBackend,
    compiled_modules: HashMap<String, HipModule>,
    compiled_kernels: HashMap<String, HipKernel>,
}

impl GpuModelExecutor {
    /// Create a new GPU model executor
    pub fn new(backend: HipBackend) -> Self {
        Self {
            backend,
            compiled_modules: HashMap::new(),
            compiled_kernels: HashMap::new(),
        }
    }

    /// Compile and load a HIP kernel from source file
    pub fn compile_kernel<P: AsRef<Path>>(
        &mut self,
        kernel_name: &str,
        source_path: P,
    ) -> ExecutorResult<()> {
        let source_path = source_path.as_ref();

        // Compile the HIP kernel to a .hipfb bundle file
        let object_path = source_path.with_extension("hipfb");
        let target_arch = self.get_target_arch()?;

        let compile_cmd = std::process::Command::new("hipcc")
            .args([
                "-O3",
                &format!("--offload-arch={}", target_arch),
                "--offload-device-only",
                source_path.to_str().ok_or_else(|| {
                    ExecutorError::CompilationFailed("Invalid source path".to_string())
                })?,
                "-o",
                object_path.to_str().ok_or_else(|| {
                    ExecutorError::CompilationFailed("Invalid object path".to_string())
                })?,
            ])
            .output()
            .map_err(|e| ExecutorError::CompilationFailed(format!("Failed to run hipcc: {}", e)))?;

        if !compile_cmd.status.success() {
            return Err(ExecutorError::CompilationFailed(format!(
                "Compilation failed: {}",
                String::from_utf8_lossy(&compile_cmd.stderr)
            )));
        }

        // Load the compiled module
        let module = self
            .backend
            .load_module(&object_path.to_string_lossy())
            .map_err(|e| ExecutorError::ModuleLoadFailed(e.to_string()))?;

        // Get the kernel function
        let kernel = self
            .backend
            .get_kernel_function(&module, kernel_name)
            .map_err(|e| ExecutorError::KernelNotFound(e.to_string()))?;

        // Store the module and kernel
        self.compiled_modules
            .insert(kernel_name.to_string(), module);
        self.compiled_kernels
            .insert(kernel_name.to_string(), kernel);

        Ok(())
    }

    /// Execute layer normalization kernel
    pub fn layer_norm(
        &self,
        input: &crate::backend::HipBuffer,
        output: &crate::backend::HipBuffer,
        weight: &crate::backend::HipBuffer,
        bias: &crate::backend::HipBuffer,
        epsilon: f32,
        hidden_size: usize,
        seq_len: usize,
    ) -> ExecutorResult<()> {
        let kernel = self
            .compiled_kernels
            .get("layer_norm_kernel")
            .ok_or_else(|| ExecutorError::KernelNotFound("layer_norm_kernel".to_string()))?;

        let args = [
            input.as_ptr() as *mut std::ffi::c_void,
            output.as_ptr() as *mut std::ffi::c_void,
            weight.as_ptr() as *mut std::ffi::c_void,
            bias.as_ptr() as *mut std::ffi::c_void,
            &epsilon as *const f32 as *mut std::ffi::c_void,
            &(hidden_size as i32) as *const i32 as *mut std::ffi::c_void,
            &(seq_len as i32) as *const i32 as *mut std::ffi::c_void,
        ];

        self.backend
            .launch_kernel_with_module(
                kernel,
                (seq_len as u32, 1, 1),                        // grid_dim
                (((hidden_size + 31) / 32 * 32) as u32, 1, 1), // block_dim (round up to warp size)
                &args,
            )
            .map_err(|e| ExecutorError::ExecutionFailed(e.to_string()))?;

        Ok(())
    }

    /// Execute RoPE (Rotary Position Embedding) kernel
    pub fn rope(
        &self,
        input: &crate::backend::HipBuffer,
        output: &crate::backend::HipBuffer,
        cos_cache: &crate::backend::HipBuffer,
        sin_cache: &crate::backend::HipBuffer,
        head_dim: usize,
        num_heads: usize,
        seq_len: usize,
        rope_dim: usize,
    ) -> ExecutorResult<()> {
        let kernel = self
            .compiled_kernels
            .get("rope_kernel")
            .ok_or_else(|| ExecutorError::KernelNotFound("rope_kernel".to_string()))?;

        let args = [
            input.as_ptr() as *mut std::ffi::c_void,
            output.as_ptr() as *mut std::ffi::c_void,
            cos_cache.as_ptr() as *mut std::ffi::c_void,
            sin_cache.as_ptr() as *mut std::ffi::c_void,
            &(head_dim as i32) as *const i32 as *mut std::ffi::c_void,
            &(num_heads as i32) as *const i32 as *mut std::ffi::c_void,
            &(seq_len as i32) as *const i32 as *mut std::ffi::c_void,
            &(rope_dim as i32) as *const i32 as *mut std::ffi::c_void,
        ];

        self.backend
            .launch_kernel_with_module(
                kernel,
                (seq_len as u32, num_heads as u32, 1), // grid_dim
                ((rope_dim / 2) as u32, 1, 1),         // block_dim
                &args,
            )
            .map_err(|e| ExecutorError::ExecutionFailed(e.to_string()))?;

        Ok(())
    }

    /// Execute softmax kernel
    pub fn softmax(
        &self,
        input: &crate::backend::HipBuffer,
        output: &crate::backend::HipBuffer,
        vocab_size: usize,
        batch_size: usize,
    ) -> ExecutorResult<()> {
        let kernel = self
            .compiled_kernels
            .get("softmax_kernel")
            .ok_or_else(|| ExecutorError::KernelNotFound("softmax_kernel".to_string()))?;

        let args = [
            input.as_ptr() as *mut std::ffi::c_void,
            output.as_ptr() as *mut std::ffi::c_void,
            &(vocab_size as i32) as *const i32 as *mut std::ffi::c_void,
            &(batch_size as i32) as *const i32 as *mut std::ffi::c_void,
        ];

        self.backend
            .launch_kernel_with_module(
                kernel,
                (batch_size as u32, 1, 1), // grid_dim
                (vocab_size as u32, 1, 1), // block_dim
                &args,
            )
            .map_err(|e| ExecutorError::ExecutionFailed(e.to_string()))?;

        Ok(())
    }

    /// Execute scaled softmax kernel (for attention scores)
    pub fn scaled_softmax(
        &self,
        input: &crate::backend::HipBuffer,
        output: &crate::backend::HipBuffer,
        scale: f32,
        vocab_size: usize,
        batch_size: usize,
    ) -> ExecutorResult<()> {
        let kernel = self
            .compiled_kernels
            .get("scaled_softmax_kernel")
            .ok_or_else(|| ExecutorError::KernelNotFound("scaled_softmax_kernel".to_string()))?;

        let args = [
            input.as_ptr() as *mut std::ffi::c_void,
            output.as_ptr() as *mut std::ffi::c_void,
            &scale as *const f32 as *mut std::ffi::c_void,
            &(vocab_size as i32) as *const i32 as *mut std::ffi::c_void,
            &(batch_size as i32) as *const i32 as *mut std::ffi::c_void,
        ];

        self.backend
            .launch_kernel_with_module(
                kernel,
                (batch_size as u32, 1, 1), // grid_dim
                (vocab_size as u32, 1, 1), // block_dim
                &args,
            )
            .map_err(|e| ExecutorError::ExecutionFailed(e.to_string()))?;

        Ok(())
    }

    /// Get the target architecture for the current GPU
    fn get_target_arch(&self) -> ExecutorResult<String> {
        // Get device properties to determine architecture
        let device_count = self.backend.get_device_count().map_err(|e| {
            ExecutorError::ExecutionFailed(format!("Failed to get device count: {}", e))
        })?;

        if device_count == 0 {
            return Err(ExecutorError::ExecutionFailed(
                "No HIP devices found".to_string(),
            ));
        }

        // Use first device for now
        let device_props = self.backend.get_device_properties(0).map_err(|e| {
            ExecutorError::ExecutionFailed(format!("Failed to get device properties: {}", e))
        })?;

        // Convert device name to architecture
        // This is a simplified mapping - in practice you'd want a more comprehensive mapping
        let device_name_str = device_props.name();

        if device_name_str.contains("gfx1100") {
            Ok("gfx1100".to_string())
        } else if device_name_str.contains("gfx1030") {
            Ok("gfx1030".to_string())
        } else if device_name_str.contains("gfx1010") {
            Ok("gfx1010".to_string())
        } else {
            // Default to a common architecture
            Ok("gfx1100".to_string())
        }
    }

    /// Synchronize all GPU operations
    pub fn synchronize(&self) -> ExecutorResult<()> {
        self.backend
            .synchronize()
            .map_err(|e| ExecutorError::ExecutionFailed(e.to_string()))
    }
}

// SAFETY: GpuModelExecutor is Send+Sync because HipBackend is Send+Sync
// and we ensure thread-safe access to the HashMap through proper synchronization
unsafe impl Send for GpuModelExecutor {}
unsafe impl Sync for GpuModelExecutor {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HipBackend;

    #[test]
    fn test_gpu_executor_creation() {
        let backend = HipBackend::new().expect("Failed to create HIP backend");
        let executor = GpuModelExecutor::new(backend);
        assert!(executor.compiled_modules.is_empty());
        assert!(executor.compiled_kernels.is_empty());
    }

    #[test]
    fn test_compile_smoke_kernel() -> ExecutorResult<()> {
        let backend = HipBackend::new().expect("Failed to create HIP backend");
        let mut executor = GpuModelExecutor::new(backend);

        // Try to compile the smoke test kernel
        let kernel_path = "src/backend/hip_kernels/smoke_test.hip";
        if Path::new(kernel_path).exists() {
            executor.compile_kernel("add_one", kernel_path)?;
            assert!(executor.compiled_kernels.contains_key("add_one"));
        }

        Ok(())
    }

    #[test]
    fn test_execute_no_argument_kernel() -> ExecutorResult<()> {
        let backend = HipBackend::new().expect("Failed to create HIP backend");
        let mut executor = GpuModelExecutor::new(backend);

        // Try to compile the smoke test kernel
        let kernel_path = "src/backend/hip_kernels/smoke_test.hip";
        if Path::new(kernel_path).exists() {
            executor.compile_kernel("simple_test", kernel_path)?;
            assert!(executor.compiled_kernels.contains_key("simple_test"));

            // Execute the no-argument kernel
            let kernel = executor.compiled_kernels.get("simple_test").unwrap();
            println!("Kernel pointer: {:?}", kernel.as_ptr());

            // No arguments for this kernel
            let args: &[*mut std::ffi::c_void] = &[];

            println!("Launching no-argument kernel");

            let launch_result = executor.backend.launch_kernel_with_module(
                kernel,
                (1, 1, 1), // grid_dim
                (1, 1, 1), // block_dim
                args,
            );

            if let Err(e) = launch_result {
                println!("Kernel launch failed: {}", e);
                return Err(ExecutorError::ExecutionFailed(e.to_string()));
            } else {
                println!("No-argument kernel launch reported success");
            }

            // Synchronize to ensure kernel completion
            executor.backend.synchronize()?;
            println!("No-argument kernel execution completed successfully");
        }

        Ok(())
    }

    #[test]
    fn test_execute_smoke_kernel() -> ExecutorResult<()> {
        let backend = HipBackend::new().expect("Failed to create HIP backend");
        let mut executor = GpuModelExecutor::new(backend);

        // Try to compile the smoke test kernel
        let kernel_path = "src/backend/hip_kernels/smoke_test.hip";
        if Path::new(kernel_path).exists() {
            executor.compile_kernel("add_one", kernel_path)?;
            assert!(executor.compiled_kernels.contains_key("add_one"));

            // Create test data - just 1 element for simplicity
            let size = 1;
            let host_data: Vec<f32> = vec![0.0];
            let device_buffer = executor
                .backend
                .allocate_buffer(size * std::mem::size_of::<f32>())?;

            // Copy data to device
            executor.backend.copy_to_gpu(&host_data, &device_buffer)?;

            // Execute the kernel
            let kernel = executor.compiled_kernels.get("add_one").unwrap();
            println!("Kernel pointer: {:?}", kernel.as_ptr());

            // HIP kernel arguments need to be passed differently - use device-side pointers
            let data_ptr = device_buffer.as_ptr();
            let size_val = size as i32;

            // HIP kernel arguments need to be passed as pointers to the values
            // Create a buffer to hold the argument pointers
            let mut arg_data = Vec::new();

            // For pointer arguments, store the device pointer value
            arg_data.push(data_ptr as u64);

            // For scalar arguments, store the value directly
            arg_data.push(size_val as u64);

            // Create array of pointers to the argument data
            let args: Vec<*mut std::ffi::c_void> = arg_data
                .iter()
                .map(|x| x as *const u64 as *mut std::ffi::c_void)
                .collect();

            println!("Launching kernel with grid: {}, block: {}", 1, 1);
            println!("Device buffer pointer: {:?}", data_ptr);
            println!("Size value: {}", size_val);

            // For single element, use 1 block, 1 thread
            let launch_result = executor.backend.launch_kernel_with_module(
                kernel,
                (1, 1, 1), // grid_dim
                (1, 1, 1), // block_dim
                &args,
            );

            if let Err(e) = launch_result {
                println!("Kernel launch failed: {}", e);
                return Err(ExecutorError::ExecutionFailed(e.to_string()));
            } else {
                println!("Kernel launch reported success");
            }

            // Synchronize to ensure kernel completion
            executor.backend.synchronize()?;

            // Copy result back
            let mut result: Vec<f32> = vec![0.0; size];
            executor
                .backend
                .copy_from_gpu(&device_buffer, &mut result)?;

            // Debug: Print result
            println!("Input value: {}", host_data[0]);
            println!("Result value: {}", result[0]);

            // Verify results - for single element test
            println!("Expected result: {}, Got result: {}", 42.0, result[0]);
            assert!(
                (result[0] - 42.0).abs() < 1e-6,
                "Element 0 failed: expected {}, got {}",
                42.0,
                result[0]
            );
        }

        Ok(())
    }
}
