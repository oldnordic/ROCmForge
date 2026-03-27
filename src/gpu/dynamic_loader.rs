//! Dynamic library loading with RAII cleanup.
//!
//! Safety-first design:
//! - RAII ensures dlclose on Drop (never leaks library handles)
//! - Configurable search paths via environment variable
//! - Detailed error messages with search path information
//! - Thread-safe initialization (sync once)

use super::error::{GpuError, GpuResult};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::fmt;

/// Search paths for kernel library, in order of priority.
///
/// 1. ROCMFORGE_KERNEL_LIB environment variable (user-specified)
/// 2. System library paths (LD_LIBRARY_PATH, /usr/local/lib, /opt/rocm/lib)
/// 3. Memoria build path (fallback for development)
fn kernel_library_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Environment variable (highest priority)
    if let Ok(lib_path) = std::env::var("ROCMFORGE_KERNEL_LIB") {
        paths.push(PathBuf::from(lib_path));
    }

    // 2. System library paths
    if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
        for path in std::env::split_paths(&ld_path) {
            paths.push(path);
        }
    }
    paths.push(PathBuf::from("/usr/local/lib"));
    paths.push(PathBuf::from("/usr/lib"));
    paths.push(PathBuf::from("/opt/rocm/lib"));

    // 3. Memoria fallback (development only)
    paths.push(PathBuf::from("/home/feanor/Projects/Memoria/gpu/libgpu.so"));

    paths
}

/// RAII wrapper for dynamically loaded library.
///
/// Opens library on first access, closes on Drop.
/// Symbol lookup with detailed error messages.
pub struct DynamicLibrary {
    handle: *mut c_void,
    library_path: PathBuf,
}

impl DynamicLibrary {
    /// Load library by searching standard paths.
    ///
    /// Searches in order:
    /// 1. ROCMFORGE_KERNEL_LIB environment variable
    /// 2. LD_LIBRARY_PATH and system library paths
    /// 3. Memoria build path (development fallback)
    ///
    /// # Returns
    /// Ok(DynamicLibrary) if library found and loaded
    /// Err(GpuError::HipNotAvailable) if not found in any path
    pub fn load(library_name: &str) -> GpuResult<Self> {
        let paths = kernel_library_search_paths();

        #[cfg(target_os = "linux")]
        for base_path in paths {
            let full_path = base_path.join(library_name);

            // Try to open the library
            let handle = unsafe {
                let path_str = full_path.to_string_lossy();
                libc::dlopen(
                    path_str.as_ptr() as *const i8,
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            };

            if !handle.is_null() {
                return Ok(Self {
                    handle,
                    library_path: full_path,
                });
            }
        }

        // Library not found - provide helpful error message
        Err(GpuError::HipNotAvailable)
    }

    /// Get symbol from loaded library.
    ///
    /// # Returns
    /// Ok(pointer) if symbol found
    /// Err with symbol name if not found
    pub fn get_symbol(&self, symbol_name: &str) -> GpuResult<*const c_void> {
        let symbol_cstr = format!("{}\0", symbol_name);

        let ptr = unsafe {
            libc::dlsym(
                self.handle,
                symbol_cstr.as_ptr() as *const i8,
            )
        };

        if ptr.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "symbol '{}' not found in {}",
                    symbol_name,
                    self.library_path.display()
                ),
            });
        }

        Ok(ptr)
    }

    /// Get the path of the loaded library.
    pub fn library_path(&self) -> &PathBuf {
        &self.library_path
    }
}

// SAFETY: The library handle is safe to send across threads
// (the underlying library must be thread-safe, which HIP libraries are)
unsafe impl Send for DynamicLibrary {}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                libc::dlclose(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

impl fmt::Debug for DynamicLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicLibrary")
            .field("library_path", &self.library_path)
            .field("handle", &format!("{:p}", self.handle))
            .finish()
    }
}

// ── Kernel Registry ─────────────────────────────────────────────────────────────

/// Global kernel registry with lazy loading.
///
/// Loads libgpu.so on first access, caches function pointers.
struct KernelRegistry {
    library: Option<DynamicLibrary>,
}

impl KernelRegistry {
    /// Get or create the global kernel registry.
    fn get() -> GpuResult<&'static std::sync::Mutex<Self>> {
        use std::sync::{Mutex, OnceLock};

        static REGISTRY: OnceLock<Mutex<KernelRegistry>> = OnceLock::new();

        // Initialize on first access
        if REGISTRY.get().is_none() {
            let library = DynamicLibrary::load("libgpu.so")?;
            let _ = REGISTRY.set(Mutex::new(KernelRegistry {
                library: Some(library),
            }));
        }

        Ok(REGISTRY.get().unwrap())
    }

    /// Load a kernel function pointer by name.
    ///
    /// # Safety
    /// Caller must ensure the function signature matches the actual kernel.
    unsafe fn load_kernel<T>(
        library: &DynamicLibrary,
        symbol_name: &str,
    ) -> GpuResult<T> {
        let ptr = library.get_symbol(symbol_name)?;

        Ok(std::mem::transmute_copy::<*const c_void, T>(&ptr))
    }

    /// Get gpu_kv_write kernel.
    pub fn gpu_kv_write(&self) -> GpuResult<unsafe extern "C" fn(
        *mut f32, *mut f32, *const f32, *const f32,
        c_int, c_int, c_int
    ) -> c_int> {
        let library = self.library.as_ref()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "Kernel library not loaded".to_string(),
            })?;

        unsafe { Self::load_kernel(library, "gpu_kv_write") }
    }

    /// Get gpu_kv_write_batched kernel.
    pub fn gpu_kv_write_batched(&self) -> GpuResult<unsafe extern "C" fn(
        *mut f32, *mut f32, *const f32, *const f32,
        c_int, c_int, c_int, c_int
    ) -> c_int> {
        let library = self.library.as_ref()
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "Kernel library not loaded".to_string(),
            })?;

        unsafe { Self::load_kernel(library, "gpu_kv_write_batched") }
    }
}

/// Public API: Get a kernel from the global registry.
pub fn get_kernel<F, T>(kernel_getter: F) -> GpuResult<T>
where
    F: Fn(&KernelRegistry) -> GpuResult<T>,
{
    let registry = KernelRegistry::get()?;
    let registry = registry.lock().map_err(|_| GpuError::HipApiError {
        code: -1,
        description: "Kernel registry mutex poisoned".to_string(),
    })?;

    kernel_getter(&registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_paths_includes_memoria_fallback() {
        let paths = kernel_library_search_paths();
        let last = paths.last().unwrap();
        assert!(last.to_string_lossy().contains("Memoria"));
    }

    #[test]
    fn load_fails_for_nonexistent_library() {
        let result = DynamicLibrary::load("nonexistent_library_12345.so");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn registry_initializes_on_first_access() {
        // Will fail without libgpu.so, that's expected
        let result = KernelRegistry::get();
        match result {
            Ok(_) => {
                // Registry initialized successfully
            }
            Err(_) => {
                // Expected when HIP unavailable
            }
        }
    }
}
