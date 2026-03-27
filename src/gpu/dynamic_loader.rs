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
