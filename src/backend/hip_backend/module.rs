//! HIP module and kernel wrapper

use std::ffi::CString;
use std::ptr;

use crate::backend::hip_backend::device::get_error_string;
use crate::backend::hip_backend::error::HipResult;
use crate::backend::hip_backend::ffi;
use crate::backend::hip_backend::HipError;

// SAFETY: HipModule is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

/// HIP module wrapper
#[repr(C)]
#[derive(Debug)]
pub struct HipModule {
    module: *mut std::ffi::c_void,
}

impl HipModule {
    /// Create HipModule from raw pointer
    pub fn from_ptr(module: *mut std::ffi::c_void) -> Self {
        HipModule { module }
    }

    /// Get raw module pointer
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.module
    }

    /// Load HIP module from file path
    pub fn load_from_path(path: &str) -> HipResult<Self> {
        let path_cstr = CString::new(path)
            .map_err(|e| HipError::KernelLoadFailed(format!("Invalid path string: {}", e)))?;

        let mut module: *mut std::ffi::c_void = ptr::null_mut();
        let result = unsafe { ffi::hipModuleLoad(&mut module, path_cstr.as_ptr()) };

        if result != ffi::HIP_SUCCESS {
            let error_msg = get_error_string(result);
            return Err(HipError::KernelLoadFailed(format!(
                "Failed to load module '{}': {}",
                path, error_msg
            )));
        }

        Ok(HipModule::from_ptr(module))
    }

    /// Load HIP module from data
    pub fn load_from_data(data: &[u8]) -> HipResult<Self> {
        let mut module: *mut std::ffi::c_void = ptr::null_mut();
        let result = unsafe { ffi::hipModuleLoadData(&mut module, data.as_ptr() as *const std::ffi::c_void) };

        if result != ffi::HIP_SUCCESS {
            let error_msg = get_error_string(result);
            return Err(HipError::KernelLoadFailed(format!(
                "Failed to load module from data: {}",
                error_msg
            )));
        }

        Ok(HipModule::from_ptr(module))
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                ffi::hipModuleUnload(self.module);
            }
        }
    }
}

// SAFETY: HipKernel is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
// NOTE: #[repr(C)] is CRITICAL for FFI compatibility
unsafe impl Send for HipKernel {}
unsafe impl Sync for HipKernel {}

/// HIP kernel wrapper
#[repr(C)]
#[derive(Debug)]
pub struct HipKernel {
    func: *mut std::ffi::c_void,
}

impl HipKernel {
    /// Create HipKernel from raw pointer
    pub fn from_ptr(func: *mut std::ffi::c_void) -> Self {
        HipKernel { func }
    }

    /// Get raw kernel function pointer
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.func
    }

    /// Get kernel function from module
    pub fn from_module(module: &HipModule, kernel_name: &str) -> HipResult<Self> {
        let kernel_name_cstr = CString::new(kernel_name)
            .map_err(|e| HipError::KernelLoadFailed(format!("Invalid kernel name: {}", e)))?;

        let mut func: *mut std::ffi::c_void = ptr::null_mut();
        let result =
            unsafe { ffi::hipModuleGetFunction(&mut func, module.as_ptr(), kernel_name_cstr.as_ptr()) };

        if result != ffi::HIP_SUCCESS {
            let error_msg = get_error_string(result);
            return Err(HipError::KernelLoadFailed(format!(
                "Failed to get kernel '{}': {}",
                kernel_name, error_msg
            )));
        }

        Ok(HipKernel::from_ptr(func))
    }
}
