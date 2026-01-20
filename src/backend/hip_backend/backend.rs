//! HIP backend main implementation

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::ptr;

use crate::backend::hip_backend::device::{get_error_string, HipDevice, HipDeviceProp};
use crate::backend::hip_backend::error::{HipError, HipResult};
use crate::backend::hip_backend::ffi;
use crate::backend::hip_backend::memory::HipBuffer;
use crate::backend::hip_backend::module::{HipKernel, HipModule};
use crate::backend::hip_backend::stream::HipStream;
use crate::loader::mmap_loader::TensorShape;

// NOTE: #[repr(C)] is NOT used here because HipBackend contains Arc<T>
// which is NOT C-compatible. Using repr(C) would cause ABI violations.
// See docs/deep_crash_analysis.md for details.
#[derive(Debug)]
pub struct HipBackend {
    device: HipDevice,
    stream: Arc<HipStream>,
}

// Manual Clone implementation that clones the Arc (shared ownership)
// This is safe because Arc ensures the stream is only destroyed once
impl Clone for HipBackend {
    fn clone(&self) -> Self {
        HipBackend {
            device: self.device.clone(),
            stream: Arc::clone(&self.stream),
        }
    }
}

// WORKAROUND: Singleton pattern to avoid ABI issues with returning HipBackend
static GLOBAL_BACKEND: Mutex<Option<Arc<HipBackend>>> = Mutex::new(None);
static GLOBAL_INIT_CALLED: AtomicBool = AtomicBool::new(false);

impl HipBackend {
    /// Phase 20: Check if GPU is available WITHOUT initializing HIP backend
    ///
    /// This performs a lightweight check to see if:
    /// - HIP runtime is available (amdhip64 library present)
    /// - At least one GPU device is present
    ///
    /// Returns false if:
    /// - No GPU device present
    /// - HIP runtime not installed
    /// - hipInit() fails
    ///
    /// This is safe to call from anywhere - it won't crash if GPU isn't available.
    pub fn gpu_available() -> bool {
        use std::sync::atomic::AtomicBool;

        static CHECKED: AtomicBool = AtomicBool::new(false);
        static AVAILABLE: AtomicBool = AtomicBool::new(false);
        static INIT: Once = Once::new();

        INIT.call_once(|| {
            // Use catch_unwind to prevent panics from propagating
            let result = std::panic::catch_unwind(|| {
                unsafe {
                    // Try to initialize HIP (lightweight check)
                    let init_result = ffi::hipInit(0);
                    if init_result != ffi::HIP_SUCCESS {
                        tracing::debug!(
                            "HIP not available: hipInit failed with code {}",
                            init_result
                        );
                        return false;
                    }

                    // Try to get device count
                    let mut count: i32 = 0;
                    let count_result = ffi::hipGetDeviceCount(&mut count);
                    if count_result != ffi::HIP_SUCCESS {
                        tracing::debug!(
                            "HIP not available: hipGetDeviceCount failed with code {}",
                            count_result
                        );
                        return false;
                    }

                    let available = count > 0;
                    tracing::debug!("GPU available: {} ({} device(s))", available, count);
                    available
                }
            })
            .unwrap_or(false);

            AVAILABLE.store(result, Ordering::Release);
            CHECKED.store(true, Ordering::Release);
        });

        AVAILABLE.load(Ordering::Acquire)
    }

    /// Phase 20: Create backend only if GPU is available
    ///
    /// This is the safe version of `new()` that returns a clear error
    /// instead of crashing if GPU is not available.
    pub fn new_checked() -> HipResult<Arc<Self>> {
        if !Self::gpu_available() {
            return Err(HipError::DeviceNotFound);
        }
        Self::new()
    }

    /// Create a new HIP backend singleton (thread-safe)
    /// Returns Arc<HipBackend> for shared ownership across the codebase
    pub fn new() -> HipResult<Arc<Self>> {
        // Double-checked locking pattern for singleton initialization
        if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {
            return Ok(GLOBAL_BACKEND
                .lock()
                .map_err(|e| {
                    HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e))
                })?
                .as_ref()
                .map(Arc::clone)
                .expect("Global backend initialized but not set"));
        }

        // Initialize under lock
        let mut guard = GLOBAL_BACKEND
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e)))?;
        if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {
            return Ok(guard
                .as_ref()
                .map(Arc::clone)
                .expect("Global backend initialized but not set"));
        }

        // Initialize HIP
        Self::initialize_hip()?;

        // Detect AMD GPU
        let device = Self::detect_amd_gpu()?;

        // CRITICAL: Set the detected device as current HIP context
        // Without this, HIP may use a different device context, causing allocation failures
        eprintln!(
            ">>> HipBackend::new: Setting device to {} ({})",
            device.device_id, device.name
        );
        let set_result = unsafe { ffi::hipSetDevice(device.device_id) };
        if set_result != ffi::HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to set device {}: hipSetDevice returned {}",
                device.device_id, set_result
            )));
        }
        tracing::debug!(
            "HipBackend::new: Device {} set successfully",
            device.device_id
        );

        // Create stream wrapped in Arc for shared ownership
        let stream = Arc::new(HipStream::new()?);

        let backend = Arc::new(HipBackend { device, stream });
        *guard = Some(backend.clone());
        // CRITICAL: Set flag BEFORE releasing lock to prevent race condition
        // Other threads check GLOBAL_INIT_CALLED before acquiring lock
        GLOBAL_INIT_CALLED.store(true, Ordering::Release);
        drop(guard); // Explicitly release lock before returning

        Ok(backend)
    }

    fn detect_amd_gpu() -> HipResult<HipDevice> {
        let mut count: i32 = 0;
        let result = unsafe { ffi::hipGetDeviceCount(&mut count) };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::DeviceNotFound);
        }

        if count == 0 {
            return Err(HipError::DeviceNotFound);
        }

        // Find best GPU (prefer discrete graphics cards)
        let mut best_device = 0;
        let mut max_memory = 0;

        for device_id in 0..count {
            let mut props = HipDeviceProp::default();
            let result = unsafe { ffi::hipGetDeviceProperties(&mut props, device_id) };

            if result == ffi::HIP_SUCCESS {
                println!(
                    "Device {}: {} - {}MB VRAM",
                    device_id,
                    props.name(),
                    props.total_global_mem() / (1024 * 1024)
                );

                // Prefer device with most memory (likely discrete GPU)
                if props.total_global_mem() > max_memory {
                    max_memory = props.total_global_mem();
                    best_device = device_id;
                }
            }
        }

        // Get properties for best device
        let mut props = HipDeviceProp::default();
        let result = unsafe { ffi::hipGetDeviceProperties(&mut props, best_device) };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to get device properties: {}",
                result
            )));
        }

        Ok(HipDevice {
            device_id: best_device,
            name: props.name(),
            memory: props.total_global_mem() as usize,
            compute_units: props.multi_processor_count(),
        })
    }

    fn initialize_hip() -> HipResult<()> {
        let result = unsafe { ffi::hipInit(0) };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::InitializationFailed(format!(
                "hipInit failed with code {}",
                result
            )));
        }

        Ok(())
    }

    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    pub fn stream(&self) -> &HipStream {
        &self.stream
    }

    /// Get available GPU memory information
    pub fn get_memory_info(&self) -> HipResult<(usize, usize)> {
        let mut free: usize = 0;
        let mut total: usize = 0;

        let result = unsafe { ffi::hipMemGetInfo(&mut free, &mut total) };

        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryQueryFailed(format!(
                "hipMemGetInfo failed with code {}",
                result
            )));
        }

        Ok((free, total))
    }

    /// Get GPU health status for monitoring
    ///
    /// Returns a tuple of (available, total) memory in bytes, or an error if GPU is unavailable.
    /// This is a non-failing version used for health checks.
    pub fn get_gpu_status(&self) -> (Option<(usize, usize)>, Option<String>) {
        match self.get_memory_info() {
            Ok((free, total)) => (Some((free, total)), None),
            Err(e) => (None, Some(e.to_string())),
        }
    }

    /// Check if we have enough memory for model loading
    ///
    /// Queries available GPU memory and compares against required bytes.
    /// Adds a 10% safety margin for driver overhead and alignment.
    pub fn check_memory_for_model(&self, needed_bytes: usize) -> HipResult<(bool, usize, usize)> {
        let (free, _total) = self.get_memory_info()?;

        // Add 10% safety margin for driver overhead
        // Minimum 100MB margin to account for alignment and driver allocations
        let safety_margin = (needed_bytes / 10).max(1024 * 1024 * 100);
        let needed_with_margin = needed_bytes.saturating_add(safety_margin);

        let has_enough = free >= needed_with_margin;
        let free_mb = free / 1024 / 1024;
        let needed_mb = needed_with_margin / 1024 / 1024;

        Ok((has_enough, free_mb, needed_mb))
    }

    /// Check if an allocation of given size is safe
    ///
    /// Returns true if size < 70% of currently free GPU memory.
    pub fn can_allocate(&self, size: usize) -> HipResult<bool> {
        let (free, _total) = self.get_memory_info()?;

        // Safety margin: use only 70% of free memory
        let safe_threshold = (free * 7) / 10;

        Ok(size <= safe_threshold)
    }

    /// Allocate buffer with conservative memory check
    pub fn allocate_buffer_safe(&self, size: usize) -> HipResult<HipBuffer> {
        // First check if allocation is safe
        if !self.can_allocate(size)? {
            let (free, total) = self.get_memory_info()?;
            let safe_threshold = (free * 7) / 10;

            return Err(HipError::MemoryAllocationFailed(format!(
                "Requested {} bytes ({} MB) exceeds safe threshold {} bytes ({} MB)\n\
                 Free GPU memory: {} MB / Total: {} MB",
                size,
                size / 1024 / 1024,
                safe_threshold,
                safe_threshold / 1024 / 1024,
                free / 1024 / 1024,
                total / 1024 / 1024
            )));
        }

        self.allocate_buffer(size)
    }

    /// Get safe allocation size for testing
    pub fn safe_alloc_size(&self) -> HipResult<usize> {
        let (free, _) = self.get_memory_info()?;
        Ok((free * 7) / 10)
    }

    /// Copy from GPU to host using stream-aware synchronization (SAFE)
    pub fn copy_from_device_safe<T>(
        &self,
        gpu_buffer: &HipBuffer,
        host_data: &mut [T],
    ) -> HipResult<()> {
        gpu_buffer.copy_to_host_with_stream(host_data, self.stream.as_ptr())
    }

    /// Execute a fallible operation with retry logic for recoverable errors
    pub fn retry_operation<F, T>(
        &self,
        mut operation: F,
        context: &str,
    ) -> HipResult<T>
    where
        F: FnMut() -> HipResult<T>,
    {
        let max_retries = 3;
        let initial_delay_ms = 10u64;
        let backoff_multiplier = 2.0f64;
        let max_delay_ms = 1000u64;

        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation() {
                Ok(result) => {
                    if attempt > 0 {
                        tracing::info!(
                            context,
                            attempt,
                            "GPU operation succeeded after retry"
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);

                    let is_recoverable = last_error
                        .as_ref()
                        .map(|e| e.is_recoverable())
                        .unwrap_or(false);

                    if !is_recoverable || attempt >= max_retries {
                        break;
                    }

                    let base_delay = initial_delay_ms as f64
                        * backoff_multiplier.powi(attempt as i32);
                    let delay_ms = base_delay.min(max_delay_ms as f64) as u64;

                    tracing::warn!(
                        context,
                        attempt,
                        next_attempt = attempt + 1,
                        delay_ms,
                        error = %last_error.as_ref().unwrap(),
                        "GPU operation failed, retrying with exponential backoff"
                    );

                    std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Allocate buffer with automatic retry on temporary failures
    pub fn allocate_buffer_with_retry(&self, size: usize) -> HipResult<HipBuffer> {
        self.retry_operation(
            || self.allocate_buffer(size),
            "allocate_buffer"
        )
    }

    /// Copy from device with automatic retry on temporary failures
    pub fn copy_from_device_with_retry<T>(
        &self,
        buffer: &HipBuffer,
        data: &mut [T],
    ) -> HipResult<()> {
        self.retry_operation(
            || self.copy_from_device(buffer, data),
            "copy_from_device"
        )
    }

    /// Allocate buffer
    pub fn allocate_buffer(&self, size: usize) -> HipResult<HipBuffer> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);
        let count = ALLOC_COUNTER.fetch_add(1, Ordering::SeqCst);
        eprintln!(
            ">>> allocate_buffer #{}: requesting {} bytes ({} MB)",
            count,
            size,
            size / 1024 / 1024
        );
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();

        let result = unsafe { ffi::hipMalloc(&mut ptr, size) };
        if result != ffi::HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to allocate device memory: {}",
                result
            )));
        }

        let buffer = HipBuffer::from_raw_parts(ptr, size, 0);
        println!(
            "DEBUG: allocate_buffer: created buffer with size {} bytes",
            buffer.size()
        );
        Ok(buffer)
    }

    /// Create a dummy zero-byte buffer for empty tensors
    pub fn dummy_zero_buffer(&self) -> HipResult<HipBuffer> {
        Ok(HipBuffer::from_raw_parts(ptr::null_mut(), 0, 0))
    }

    /// Launch kernel (placeholder)
    pub fn launch_kernel(
        &self,
        kernel_name: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        _args: &[*mut std::ffi::c_void],
    ) -> HipResult<()> {
        println!(
            "Launching kernel '{}' with grid {:?} and block {:?}",
            kernel_name, grid_dim, block_dim
        );
        Ok(())
    }

    /// Synchronize the backend stream
    pub fn synchronize(&self) -> HipResult<()> {
        self.stream.synchronize()
    }

    /// Copy data from host to device
    pub fn copy_to_device<T>(&self, buffer: &HipBuffer, data: &[T]) -> HipResult<()> {
        buffer.copy_from_host_with_stream(data, self.stream.as_ptr())
    }

    /// Copy data from device to host
    pub fn copy_from_device<T>(&self, buffer: &HipBuffer, data: &mut [T]) -> HipResult<()> {
        buffer.copy_to_host_with_stream(data, self.stream.as_ptr())?;
        self.stream.synchronize()
    }

    /// Load module from path
    pub fn load_module(&self, path: &str) -> HipResult<HipModule> {
        HipModule::load_from_path(path)
    }

    /// Load module from data
    pub fn load_module_from_data(&self, data: &[u8]) -> HipResult<HipModule> {
        HipModule::load_from_data(data)
    }

    /// Get kernel from module path
    pub fn get_kernel(&self, module_path: &str, kernel_name: &str) -> HipResult<HipKernel> {
        let module = self.load_module(module_path)?;
        self.get_kernel_function(&module, kernel_name)
    }

    /// Get kernel function from module
    pub fn get_kernel_function(
        &self,
        module: &HipModule,
        kernel_name: &str,
    ) -> HipResult<HipKernel> {
        HipKernel::from_module(module, kernel_name)
    }

    /// Get device count
    pub fn get_device_count(&self) -> HipResult<i32> {
        let mut count: i32 = 0;
        let result = unsafe { ffi::hipGetDeviceCount(&mut count) };

        if result != ffi::HIP_SUCCESS {
            let error_msg = get_error_string(result);
            return Err(HipError::DeviceError(format!(
                "Failed to get device count: {}",
                error_msg
            )));
        }

        Ok(count)
    }

    /// Get device properties
    pub fn get_device_properties(&self, device_id: i32) -> HipResult<HipDeviceProp> {
        let mut props = HipDeviceProp::default();
        let result = unsafe { ffi::hipGetDeviceProperties(&mut props, device_id) };

        if result != ffi::HIP_SUCCESS {
            let error_msg = get_error_string(result);
            return Err(HipError::DeviceError(format!(
                "Failed to get device properties: {}",
                error_msg
            )));
        }

        Ok(props)
    }

    /// Allocate GPU buffer
    pub fn alloc_gpu_buffer<T>(&self, len: usize) -> HipResult<HipBuffer> {
        let size = len * std::mem::size_of::<T>();
        self.allocate_buffer(size)
    }

    /// Copy to GPU
    pub fn copy_to_gpu<T>(&self, host_data: &[T], gpu_buffer: &HipBuffer) -> HipResult<()> {
        gpu_buffer.copy_from_host(host_data)
    }

    /// Copy from GPU
    pub fn copy_from_gpu<T>(&self, gpu_buffer: &HipBuffer, host_data: &mut [T]) -> HipResult<()> {
        self.copy_from_device_safe(gpu_buffer, host_data)
    }

    /// Add two tensors in-place
    pub fn add_inplace(&self, input: &DeviceTensor, output: &mut DeviceTensor) -> HipResult<()> {
        if input.shape().dims() != output.shape().dims() {
            return Err(HipError::GenericError(
                "Input and output tensors must have matching shapes".to_string(),
            ));
        }

        let len = input.len();
        if len == 0 {
            return Ok(());
        }

        let handle = crate::backend::hip_blas::HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        crate::backend::hip_blas::saxpy(
            &handle,
            len as i32,
            1.0,
            input.buffer().as_ptr() as *const f32,
            1,
            output.buffer().as_ptr() as *mut f32,
            1,
        )
        .map_err(|e| HipError::GenericError(format!("hipBLAS saxpy failed: {}", e)))?;

        Ok(())
    }

    /// Scale tensor in-place
    pub fn scale_inplace(&self, tensor: &mut DeviceTensor, scale: f32) -> HipResult<()> {
        if tensor.len() == 0 {
            return Ok(());
        }

        let handle = crate::backend::hip_blas::HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        crate::backend::hip_blas::sscal(
            &handle,
            tensor.len() as i32,
            scale,
            tensor.buffer().as_ptr() as *mut f32,
            1,
        )
        .map_err(|e| HipError::GenericError(format!("hipBLAS sscal failed: {}", e)))?;

        Ok(())
    }

    /// Add row bias to tensor
    pub fn add_row_bias(&self, tensor: &mut DeviceTensor, bias: &DeviceTensor) -> HipResult<()> {
        let tensor_shape = tensor.shape().dims();
        if tensor_shape.len() != 2 {
            return Err(HipError::GenericError(
                "Tensor for row bias must be 2D".to_string(),
            ));
        }
        let rows = tensor_shape[0];
        let cols = tensor_shape[1];

        let bias_shape = bias.shape().dims();
        if !(bias_shape.len() == 1 && bias_shape[0] == cols)
            && !(bias_shape.len() == 2 && bias_shape[0] == 1 && bias_shape[1] == cols)
        {
            return Err(HipError::GenericError(format!(
                "Bias shape {:?} incompatible with tensor shape {:?}",
                bias_shape, tensor_shape
            )));
        }

        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let handle = crate::backend::hip_blas::HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        let bias_ptr = bias.buffer().as_ptr() as *const f32;
        let mut row_ptr = tensor.buffer().as_ptr() as *mut f32;
        let stride = cols;

        for _ in 0..rows {
            crate::backend::hip_blas::saxpy(&handle, cols as i32, 1.0f32, bias_ptr, 1, row_ptr, 1)
                .map_err(|e| HipError::GenericError(format!("hipBLAS saxpy failed: {}", e)))?;

            let current = row_ptr as usize;
            row_ptr = current.checked_add(stride).ok_or_else(|| {
                HipError::GenericError(format!(
                    "Pointer arithmetic overflow in add_bias_to_rows (current=0x{:x}, stride={})",
                    current, stride
                ))
            })? as *mut f32;
        }

        Ok(())
    }

    /// Launch kernel with module
    pub fn launch_kernel_with_module(
        &self,
        kernel: &HipKernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[*mut std::ffi::c_void],
    ) -> HipResult<()> {
        self.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0)
    }

    /// Launch kernel with module and shared memory
    pub fn launch_kernel_with_module_shared(
        &self,
        kernel: &HipKernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[*mut std::ffi::c_void],
        shared_mem_bytes: u32,
    ) -> HipResult<()> {
        tracing::trace!("launch_kernel_with_module_shared: Launching kernel with grid={:?}, block={:?}, shared_mem={}, args_len={}",
                       grid_dim, block_dim, shared_mem_bytes, args.len());

        if grid_dim.0 == 0 || grid_dim.1 == 0 || grid_dim.2 == 0 {
            tracing::error!("launch_kernel_with_module_shared: Invalid grid dimensions: {:?}", grid_dim);
            return Err(HipError::KernelLaunchFailed("Grid dimensions cannot be zero".to_string()));
        }
        if block_dim.0 == 0 || block_dim.1 == 0 || block_dim.2 == 0 {
            tracing::error!("launch_kernel_with_module_shared: Invalid block dimensions: {:?}", block_dim);
            return Err(HipError::KernelLaunchFailed("Block dimensions cannot be zero".to_string()));
        }

        let total_threads = (grid_dim.0 * grid_dim.1 * grid_dim.2 * block_dim.0 * block_dim.1 * block_dim.2) as u64;
        if total_threads > 1_000_000_000 {
            tracing::warn!("launch_kernel_with_module_shared: Very large kernel launch: {} total threads", total_threads);
        }

        tracing::trace!("launch_kernel_with_module_shared: Calling hipModuleLaunchKernel");
        let result = unsafe {
            ffi::hipModuleLaunchKernel(
                kernel.as_ptr(),
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                shared_mem_bytes,
                self.stream.stream,
                args.as_ptr() as *mut *mut std::ffi::c_void,
                ptr::null_mut(),
            )
        };
        tracing::trace!("launch_kernel_with_module_shared: hipModuleLaunchKernel returned {}", result);

        if result != ffi::HIP_SUCCESS {
            let error_msg = get_error_string(result);
            tracing::error!("launch_kernel_with_module_shared: Kernel launch failed with code {}: {}", result, error_msg);
            return Err(HipError::KernelLaunchFailed(format!(
                "Kernel launch failed: {}",
                error_msg
            )));
        }

        tracing::trace!("launch_kernel_with_module_shared: Kernel launched successfully");
        Ok(())
    }

    /// Create scratch buffers
    pub fn create_scratch_buffers(
        &self,
        config: &crate::model::config::ModelConfig,
    ) -> HipResult<crate::backend::scratch::ScratchBufferManager> {
        crate::backend::scratch::ScratchBufferManager::new(
            self,
            config.num_attention_heads,
            config.hidden_size,
            config.head_dim,
            config.max_position_embeddings,
        )
        .map_err(|e| HipError::GenericError(format!("Scratch buffer creation failed: {}", e)))
    }

    /// Create model runtime
    pub fn create_model_runtime(
        &self,
        config: &crate::model::config::ModelConfig,
    ) -> HipResult<super::runtime::ModelRuntime> {
        super::runtime::ModelRuntime::new_with_config(config.clone())
    }

    /// MLP (SwiGLU) forward pass
    pub fn mlp_swiglu(
        &self,
        hidden_states: &DeviceTensor,
        gate_weight: &DeviceTensor,
        up_weight: &DeviceTensor,
        down_weight: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> HipResult<()> {
        use crate::backend::hip_blas::HipBlasHandle;
        use crate::tensor::matmul::matmul_f32;

        let hidden_shape = hidden_states.shape();
        let gate_shape = gate_weight.shape();
        let up_shape = up_weight.shape();
        let down_shape = down_weight.shape();
        let output_shape = output.shape();

        if hidden_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "hidden_states must be 2D [seq_len, hidden_size]".to_string(),
            ));
        }

        if gate_shape.dims().len() != 2 || gate_shape.dims()[0] != hidden_shape.dims()[1] {
            return Err(HipError::GenericError(
                "gate_weight must be 2D [hidden_size, intermediate_size]".to_string(),
            ));
        }

        if up_shape.dims().len() != 2 || up_shape.dims()[0] != hidden_shape.dims()[1] {
            return Err(HipError::GenericError(
                "up_weight must be 2D [hidden_size, intermediate_size]".to_string(),
            ));
        }

        if down_shape.dims().len() != 2 || down_shape.dims()[1] != hidden_shape.dims()[1] {
            return Err(HipError::GenericError(
                "down_weight must be 2D [intermediate_size, hidden_size]".to_string(),
            ));
        }

        if output_shape.dims() != hidden_shape.dims() {
            return Err(HipError::GenericError(
                "output must match hidden_states shape [seq_len, hidden_size]".to_string(),
            ));
        }

        let (seq_len, hidden_size) = (hidden_shape.dims()[0], hidden_shape.dims()[1]);
        let intermediate_size = gate_shape.dims()[1];

        if up_shape.dims()[1] != intermediate_size || down_shape.dims()[0] != intermediate_size {
            return Err(HipError::GenericError(
                "All intermediate dimensions must match".to_string(),
            ));
        }

        let blas_handle = HipBlasHandle::new().map_err(|e| {
            HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
        })?;

        blas_handle
            .set_stream(self.stream().as_ptr())
            .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;

        let gate_buffer = matmul_f32(
            self,
            &blas_handle,
            hidden_states.buffer(),
            gate_weight.buffer(),
            seq_len as i32,
            intermediate_size as i32,
            hidden_size as i32,
        )
        .map_err(|e| HipError::GenericError(format!("Gate projection failed: {}", e)))?;

        let up_buffer = matmul_f32(
            self,
            &blas_handle,
            hidden_states.buffer(),
            up_weight.buffer(),
            seq_len as i32,
            intermediate_size as i32,
            hidden_size as i32,
        )
        .map_err(|e| HipError::GenericError(format!("Up projection failed: {}", e)))?;

        self.synchronize()?;

        {
            let swiglu_buffer = HipBuffer::new((seq_len * intermediate_size) * std::mem::size_of::<f32>())
                .map_err(|e| HipError::GenericError(format!("Failed to allocate SwiGLU buffer: {}", e)))?;

            unsafe {
                crate::mlp::kernels::swiglu_gpu_kernel(
                    self,
                    gate_buffer.as_ptr() as *const f32,
                    up_buffer.as_ptr() as *const f32,
                    swiglu_buffer.as_mut_ptr() as *mut f32,
                    seq_len as u32,
                    intermediate_size as u32,
                )
                .map_err(|e| HipError::GenericError(format!("SwiGLU GPU kernel failed: {}", e)))?;
            }

            self.synchronize()
                .map_err(|e| HipError::GenericError(format!("GPU synchronization failed: {}", e)))?;

            let final_buffer = matmul_f32(
                self,
                &blas_handle,
                &swiglu_buffer,
                down_weight.buffer(),
                seq_len as i32,
                hidden_size as i32,
                intermediate_size as i32,
            )
            .map_err(|e| HipError::GenericError(format!("Down projection failed: {}", e)))?;

            output.buffer().copy_from_buffer(&final_buffer)
                .map_err(|e| HipError::GenericError(format!("Final copy failed: {}", e)))?;
        }

        #[cfg(not(feature = "rocm"))]
        {
            let mut gate_host = vec![0.0f32; (seq_len * intermediate_size) as usize];
            let mut up_host = vec![0.0f32; (seq_len * intermediate_size) as usize];

            self.copy_from_device_safe(&gate_buffer, &mut gate_host)?;
            self.copy_from_device_safe(&up_buffer, &mut up_host)?;

            let mut swiglu_host = vec![0.0f32; (seq_len * intermediate_size) as usize];
            for i in 0..swiglu_host.len() {
                let gate_val = gate_host[i];
                let up_val = up_host[i];
                let sigmoid_up = 1.0 / (1.0 + (-up_val).exp());
                let swish_up = up_val * sigmoid_up;
                swiglu_host[i] = gate_val * swish_up;
            }

            let swiglu_buffer = HipBuffer::new(swiglu_host.len() * std::mem::size_of::<f32>())?;
            swiglu_buffer.copy_from_host(&swiglu_host)?;

            let final_buffer = matmul_f32(
                self,
                &blas_handle,
                &swiglu_buffer,
                down_weight.buffer(),
                seq_len as i32,
                hidden_size as i32,
                intermediate_size as i32,
            )
            .map_err(|e| HipError::GenericError(format!("Down projection failed: {}", e)))?;

            let mut output_host = vec![0.0f32; (seq_len * hidden_size) as usize];
            self.copy_from_device_safe(&final_buffer, &mut output_host)?;
            output.buffer().copy_from_host(&output_host)?;
        }

        Ok(())
    }

    /// Layer normalization forward pass
    pub fn layernorm(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        output: &mut DeviceTensor,
        eps: f32,
    ) -> HipResult<()> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let output_shape = output.shape();

        if input_shape.dims().is_empty() {
            return Err(HipError::GenericError(
                "input must have at least 1 dimension".to_string(),
            ));
        }

        let last_dim = *input_shape.dims().last().ok_or_else(|| {
            HipError::GenericError("input must have at least one dimension".to_string())
        })?;
        if weight_shape.dims() != [last_dim] {
            return Err(HipError::GenericError(
                "weight must match last dimension of input".to_string(),
            ));
        }

        if let Some(bias_tensor) = bias {
            let bias_shape = bias_tensor.shape();
            if bias_shape.dims() != weight_shape.dims() {
                return Err(HipError::GenericError(
                    "bias must match weight dimensions".to_string(),
                ));
            }
        }

        if output_shape.dims() != input_shape.dims() {
            return Err(HipError::GenericError(
                "output must match input shape".to_string(),
            ));
        }

        let total_elements = input_shape.total_elements();
        let last_dim_size = last_dim;
        let num_rows = total_elements / last_dim_size;

        let mut input_host = vec![0.0f32; total_elements];
        self.copy_from_device_safe(input.buffer(), &mut input_host)?;

        let mut weight_host = vec![0.0f32; last_dim_size];
        self.copy_from_device_safe(weight.buffer(), &mut weight_host)?;

        let bias_host = if let Some(bias_tensor) = bias {
            let mut bias_data = vec![0.0f32; last_dim_size];
            self.copy_from_device_safe(bias_tensor.buffer(), &mut bias_data)?;
            Some(bias_data)
        } else {
            None
        };

        let mut output_host = vec![0.0f32; total_elements];
        for row_idx in 0..num_rows {
            let start_idx = row_idx * last_dim_size;
            let end_idx = start_idx + last_dim_size;
            let row = &input_host[start_idx..end_idx];

            let mean: f32 = row.iter().sum::<f32>() / last_dim_size as f32;
            let variance: f32 =
                row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last_dim_size as f32;
            let std = (variance + eps).sqrt();

            for (i, &x) in row.iter().enumerate() {
                let normalized = (x - mean) / std;
                let scaled = normalized * weight_host[i];
                output_host[start_idx + i] = if let Some(ref bias_data) = bias_host {
                    scaled + bias_data[i]
                } else {
                    scaled
                };
            }
        }

        output.buffer().copy_from_host(&output_host)?;

        Ok(())
    }
}

/// Synchronize device globally using STREAM-AWARE synchronization
pub fn synchronize_device() -> HipResult<()> {
    let backend = GLOBAL_BACKEND
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e)))?
        .as_ref()
        .map(Arc::clone)
        .ok_or_else(|| {
            HipError::DeviceError(
                "HIP backend not initialized - call HipBackend::new() first".to_string(),
            )
        })?;

    backend.stream.synchronize()
}

/// Device tensor type
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub buffer: HipBuffer,
    pub shape: TensorShape,
}

impl DeviceTensor {
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn buffer(&self) -> &HipBuffer {
        &self.buffer
    }

    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    pub fn len(&self) -> usize {
        self.shape.total_elements()
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.buffer.as_ptr() as *const f32
    }

    pub fn empty(backend: &HipBackend, shape: TensorShape) -> HipResult<Self> {
        let total_bytes = shape.total_elements() * std::mem::size_of::<f32>();
        let buffer = backend.allocate_buffer(total_bytes)?;

        let result = unsafe { ffi::hipMemset(buffer.as_ptr(), 0, total_bytes) };
        if result != ffi::HIP_SUCCESS {
            return Err(HipError::MemoryAllocationFailed(format!(
                "hipMemset failed: {}",
                result
            )));
        }

        Ok(DeviceTensor { buffer, shape })
    }

    pub fn from_host_vec(
        backend: &HipBackend,
        host_data: Vec<f32>,
        shape: TensorShape,
    ) -> HipResult<Self> {
        let total_bytes = host_data.len() * std::mem::size_of::<f32>();
        let buffer = backend.allocate_buffer(total_bytes)?;
        buffer.copy_from_host(&host_data)?;
        Ok(DeviceTensor { buffer, shape })
    }

    pub fn from_pool(
        pool: &HipBuffer,
        offset: usize,
        host_data: Vec<f32>,
        shape: TensorShape,
    ) -> HipResult<Self> {
        let total_bytes = host_data.len() * std::mem::size_of::<f32>();
        let buffer = pool.sub_buffer_view(offset, total_bytes)?;
        buffer.copy_from_host(&host_data)?;
        Ok(DeviceTensor { buffer, shape })
    }

    pub fn from_pool_with_backend(
        pool: &HipBuffer,
        offset: usize,
        host_data: Vec<f32>,
        shape: TensorShape,
        backend: &HipBackend,
    ) -> HipResult<Self> {
        let total_bytes = host_data.len() * std::mem::size_of::<f32>();
        let buffer = pool.sub_buffer_view(offset, total_bytes)?;
        buffer.copy_from_host_with_stream(&host_data, backend.stream().as_ptr())?;
        Ok(DeviceTensor { buffer, shape })
    }

    pub fn from_arena_slice(
        _backend: &HipBackend,
        arena_buffer: &HipBuffer,
        offset: usize,
        size: usize,
        shape: TensorShape,
    ) -> HipResult<Self> {
        let buffer_view = arena_buffer.sub_buffer_view(offset, size).map_err(|e| {
            HipError::MemoryAllocationFailed(format!(
                "Failed to create arena slice: offset={}, size={}, error={}",
                offset, size, e
            ))
        })?;
        Ok(DeviceTensor {
            buffer: buffer_view,
            shape,
        })
    }

    /// Create device tensor from memory-mapped weights
    pub fn from_mmap(
        backend: &HipBackend,
        mmap_weights: &crate::loader::mmap_loader::MmapWeights,
        shape: TensorShape,
        byte_offset: usize,
    ) -> HipResult<Self> {
        let total_elements = shape.total_elements();
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        let buffer = backend.allocate_buffer(total_bytes)?;

        let start_element = byte_offset / std::mem::size_of::<f32>();
        let end_element = start_element + total_elements;
        let f32_slice = mmap_weights.view_f32(start_element..end_element);

        buffer.copy_from_host(f32_slice)?;

        Ok(DeviceTensor { buffer, shape })
    }

    /// Create device tensor from pre-allocated buffer
    pub fn from_buffer(
        _backend: &HipBackend,
        buffer: HipBuffer,
        shape: TensorShape,
    ) -> HipResult<Self> {
        Ok(DeviceTensor { buffer, shape })
    }

    /// Copy device data to host vector
    #[deprecated(
        since = "0.23.0",
        note = "Use HipBackend::copy_from_device_safe() with explicit buffer access instead"
    )]
    #[allow(deprecated)]
    pub fn to_host_vec(&self) -> HipResult<Vec<f32>> {
        let mut host_data = vec![0.0f32; self.len()];
        unsafe {
            let ptr = host_data.as_mut_ptr() as *mut u8;
            let byte_size = self.len() * std::mem::size_of::<f32>();
            let byte_slice = std::slice::from_raw_parts_mut(ptr, byte_size);
            self.buffer.copy_to_host(byte_slice)?;
        }
        Ok(host_data)
    }

    pub fn copy_from_host(&mut self, host_data: &[f32]) -> HipResult<()> {
        if host_data.len() != self.len() {
            return Err(HipError::GenericError(format!(
                "Host data size {} does not match tensor size {}",
                host_data.len(),
                self.len()
            )));
        }
        self.buffer.copy_from_host(host_data)
    }

    pub fn copy_from_host_vec(&mut self, host_data: Vec<f32>) -> HipResult<()> {
        self.copy_from_host(&host_data)
    }

    pub fn copy_from_device_region(
        &mut self,
        dst_offset_elements: usize,
        src: &DeviceTensor,
        src_offset_elements: usize,
        len_elements: usize,
    ) -> HipResult<()> {
        let byte_len = len_elements * std::mem::size_of::<f32>();
        let dst_offset = dst_offset_elements * std::mem::size_of::<f32>();
        let src_offset = src_offset_elements * std::mem::size_of::<f32>();
        self.buffer
            .copy_from_buffer_region(dst_offset, src.buffer(), src_offset, byte_len)
    }
}
