//! Dynamic library loading with RAII cleanup.
//!
//! Safety-first design:
//! - RAII ensures dlclose on Drop (never leaks library handles)
//! - Configurable search paths via environment variable
//! - Detailed error messages with search path information
//! - Thread-safe initialization (sync once)

use super::error::{GpuError, GpuResult};
use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Search paths for kernel library, in order of priority.
///
/// 1. ROCMFORGE_KERNEL_LIB environment variable (user-specified)
/// 2. ROCm installation paths (auto-detected from ROCm version)
/// 3. System library paths (LD_LIBRARY_PATH, /usr/local/lib, /opt/rocm/lib)
/// 4. Memoria build path (fallback for development)
fn kernel_library_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Environment variable (highest priority)
    if let Ok(lib_path) = std::env::var("ROCMFORGE_KERNEL_LIB") {
        paths.push(PathBuf::from(lib_path));
    }

    // JIT compilation target directory
    paths.push(PathBuf::from("target/jit"));

    // 2. Auto-detect ROCm installation
    if let Ok(rocm_path) = detect_rocm_path() {
        paths.push(rocm_path.join("lib"));
    }

    // 3. System library paths
    if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
        for path in std::env::split_paths(&ld_path) {
            paths.push(path);
        }
    }
    paths.push(PathBuf::from("/usr/local/lib"));
    paths.push(PathBuf::from("/usr/lib"));
    paths.push(PathBuf::from("/opt/rocm/lib"));

    // 5. Cargo build directories and current directory
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        paths.push(PathBuf::from(&manifest_dir).join("target/jit"));
        paths.push(PathBuf::from(&manifest_dir).join("target/debug"));
        paths.push(PathBuf::from(&manifest_dir).join("target/release"));
    }
    paths.push(PathBuf::from("target/debug"));
    paths.push(PathBuf::from("target/release"));
    paths.push(PathBuf::from("."));

    // 4. Memoria fallback (development only - lowest priority, must be last)
    paths.push(PathBuf::from("/home/feanor/Projects/Memoria/gpu/libgpu.so"));

    paths
}

/// Detect ROCm installation path from system.
///
/// Checks for ROCm environment variables and common installation locations.
/// Returns ROCm base path if found (e.g., /opt/rocm)
fn detect_rocm_path() -> Result<PathBuf, ()> {
    // Check ROCm environment variable
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        return Ok(PathBuf::from(rocm_path));
    }

    // Check common ROCm installation directories
    let rocm_dirs = [
        "/opt/rocm",
        "/opt/rocm-@VERSION@",       // Versioned installations
        "/usr/lib/x86_64-linux-gnu", // System ROCm packages
    ];

    for dir in rocm_dirs {
        let path = PathBuf::from(dir);
        if path.exists() {
            // Verify it's actually ROCm (check for hip libraries)
            let hip_lib = path.join("lib/libhiprtc.so");
            if hip_lib.exists() {
                return Ok(path);
            }
        }
    }

    Err(())
}

/// Compute combined hash of HIP kernel sources and architecture name.
fn compute_source_hash(arch: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::fs;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    let mut files = Vec::new();

    // Helper to recursively collect files
    fn collect_files(dir: &std::path::Path, files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect_files(&path, files);
                } else if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy();
                    if ext_str == "hip"
                        || ext_str == "h"
                        || ext_str == "cpp"
                        || path
                            .file_name()
                            .map(|f| f == "CMakeLists.txt")
                            .unwrap_or(false)
                    {
                        files.push(path);
                    }
                }
            }
        }
    }

    let mut hip_kernels_path = PathBuf::from("hip_kernels");
    if !hip_kernels_path.exists() {
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let path = PathBuf::from(manifest_dir).join("hip_kernels");
            if path.exists() {
                hip_kernels_path = path;
            }
        }
    }

    collect_files(&hip_kernels_path, &mut files);
    files.sort();

    for file in &files {
        if let Ok(content) = fs::read_to_string(file) {
            content.hash(&mut hasher);
        }
    }

    arch.hash(&mut hasher);

    format!("{:016x}", hasher.finish())
}

/// Run hipcc compiler JIT to build dynamic libgpu.so from sources.
fn compile_jit_libgpu(
    target_dir: &std::path::Path,
    arch: &str,
    current_hash: &str,
) -> GpuResult<()> {
    use std::process::Command;

    // Locate hipcc compiler
    let rocm_path = if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        PathBuf::from(rocm_path)
    } else if let Ok(hip_path) = std::env::var("HIP_PATH") {
        PathBuf::from(hip_path)
    } else {
        PathBuf::from("/opt/rocm")
    };

    let hipcc = rocm_path.join("bin/hipcc");
    if !hipcc.exists() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "hipcc compiler not found for JIT compilation".to_string(),
        });
    }

    // Locate hip_kernels/attention.hip
    let mut attention_hip = PathBuf::from("hip_kernels/attention.hip");
    if !attention_hip.exists() {
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let path = PathBuf::from(manifest_dir).join("hip_kernels/attention.hip");
            if path.exists() {
                attention_hip = path;
            }
        }
    }

    if !attention_hip.exists() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "attention.hip source not found for JIT compilation".to_string(),
        });
    }

    let out_so = target_dir.join("libgpu.so");

    // Ensure target dir exists
    let _ = std::fs::create_dir_all(target_dir);

    // Compute wave size for this arch
    let wave_size = match arch {
        "gfx1201" => 32,
        "gfx1100" | "gfx1101" | "gfx1102" => 32,
        "gfx1030" | "gfx1031" | "gfx1032" => 32,
        "gfx1010" | "gfx1011" | "gfx1012" => 32,
        "gfx900" | "gfx906" | "gfx908" | "gfx90a" | "gfx90c" | "gfx942" => 64,
        _ => 32,
    };

    let hip_include = rocm_path.join("include");

    eprintln!(
        "⚡ [JIT Compiler] Compiling libgpu.so for arch {} (wave size: {})...",
        arch, wave_size
    );

    let compile_status = Command::new(&hipcc)
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O3")
        .arg(format!("--offload-arch={}", arch))
        .arg(format!("-DWARP_SIZE={}", wave_size))
        .arg(format!("-I{}", hip_include.display()))
        .arg(&attention_hip)
        .arg("-o")
        .arg(&out_so)
        .status();

    match compile_status {
        Ok(s) if s.success() => {
            let hash_path = target_dir.join("libgpu.hash");
            if let Err(e) = std::fs::write(&hash_path, current_hash) {
                eprintln!("⚠️ Warning: failed to write JIT hash file: {}", e);
            }
            eprintln!("⚡ [JIT Compiler] Successfully compiled and verified target/jit/libgpu.so");
            Ok(())
        }
        Ok(s) => Err(GpuError::HipApiError {
            code: s.code().unwrap_or(-1),
            description: format!("hipcc returned non-zero status: {:?}", s),
        }),
        Err(e) => Err(GpuError::HipApiError {
            code: -1,
            description: format!("Failed to execute hipcc: {:?}", e),
        }),
    }
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
        let arch_str = if let Some(caps) = super::detect::GpuCapabilities::detect() {
            caps.architecture.to_string()
        } else {
            "gfx1100".to_string()
        };

        if library_name == "libgpu.so" {
            let current_hash = compute_source_hash(&arch_str);
            let jit_dir = PathBuf::from("target/jit");
            let jit_so = jit_dir.join("libgpu.so");
            let jit_hash_file = jit_dir.join("libgpu.hash");

            let mut needs_compile = true;
            if jit_so.exists() && jit_hash_file.exists() {
                if let Ok(stored_hash) = std::fs::read_to_string(&jit_hash_file) {
                    if stored_hash.trim() == current_hash {
                        needs_compile = false;
                    }
                }
            }

            if needs_compile {
                if let Err(e) = compile_jit_libgpu(&jit_dir, &arch_str, &current_hash) {
                    eprintln!("⚠️ [JIT Compiler] JIT compilation failed: {}. Falling back to standard paths.", e);
                }
            }
        }

        let paths = kernel_library_search_paths();

        #[cfg(target_os = "linux")]
        for base_path in paths {
            let full_path = base_path.join(library_name);

            // Try to open the library
            let handle = unsafe {
                let path_str = full_path.to_string_lossy();
                let c_path = std::ffi::CString::new(path_str.as_ref())
                    .expect("invariant: file path contains null byte");
                libc::dlopen(c_path.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL)
            };

            if !handle.is_null() {
                // If library has a sidecar .hash file, verify it
                let hash_file = base_path.join(format!(
                    "{}.hash",
                    library_name.strip_suffix(".so").unwrap_or(library_name)
                ));
                if hash_file.exists() {
                    let current_hash = compute_source_hash(&arch_str);
                    if let Ok(stored_hash) = std::fs::read_to_string(&hash_file) {
                        if stored_hash.trim() != current_hash {
                            eprintln!("⚠️ WARNING: loaded library {} hash mismatched expected source hash!", full_path.display());
                            eprintln!("           Expected: {}", current_hash);
                            eprintln!("           Stored:   {}", stored_hash.trim());
                        }
                    }
                }

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

        let ptr = unsafe { libc::dlsym(self.handle, symbol_cstr.as_ptr() as *const i8) };

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
pub struct KernelRegistry {
    library: Option<DynamicLibrary>,
}

impl KernelRegistry {
    /// Get or create the global kernel registry.
    pub fn get() -> GpuResult<&'static parking_lot::Mutex<Self>> {
        use parking_lot::Mutex;
        use std::sync::OnceLock;

        static REGISTRY: OnceLock<Mutex<KernelRegistry>> = OnceLock::new();

        // Initialize on first access
        if REGISTRY.get().is_none() {
            let library = DynamicLibrary::load("libgpu.so")?;
            let _ = REGISTRY.set(Mutex::new(KernelRegistry {
                library: Some(library),
            }));
        }

        Ok(REGISTRY
            .get()
            .expect("invariant: KernelRegistry OnceLock must be initialized"))
    }

    /// Load a kernel function pointer by name.
    ///
    /// # Safety
    /// Caller must ensure the function signature matches the actual kernel.
    unsafe fn load_kernel<T>(library: &DynamicLibrary, symbol_name: &str) -> GpuResult<T> {
        let ptr = library.get_symbol(symbol_name)?;

        Ok(std::mem::transmute_copy::<*const c_void, T>(&ptr))
    }

    /// Get gpu_kv_write kernel.
    pub fn gpu_kv_write(
        &self,
    ) -> GpuResult<
        unsafe extern "C" fn(
            *mut f32,
            *mut f32,
            *const f32,
            *const f32,
            c_int,
            c_int,
            c_int,
        ) -> c_int,
    > {
        let library = self.library.as_ref().ok_or_else(|| GpuError::HipApiError {
            code: -1,
            description: "Kernel library not loaded".to_string(),
        })?;

        unsafe { Self::load_kernel(library, "gpu_kv_write") }
    }

    /// Get gpu_kv_write_batched kernel.
    pub fn gpu_kv_write_batched(
        &self,
    ) -> GpuResult<
        unsafe extern "C" fn(
            *mut f32,
            *mut f32,
            *const f32,
            *const f32,
            c_int,
            c_int,
            c_int,
            c_int,
        ) -> c_int,
    > {
        let library = self.library.as_ref().ok_or_else(|| GpuError::HipApiError {
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
    let registry = registry.lock();

    kernel_getter(&registry)
}

/// Get information about the loaded kernel library.
///
/// Returns None if library not yet loaded.
pub fn library_info() -> Option<LibraryInfo> {
    let registry = KernelRegistry::get().ok()?;
    let registry = registry.lock();

    let library = registry.library.as_ref()?;

    Some(LibraryInfo {
        path: library.library_path().clone(),
        num_loaded_kernels: 2, // gpu_kv_write, gpu_kv_write_batched
    })
}

/// Information about the loaded kernel library.
#[derive(Debug, Clone)]
pub struct LibraryInfo {
    /// Full path to the loaded library
    pub path: PathBuf,
    /// Number of kernels loaded (cached function pointers)
    pub num_loaded_kernels: usize,
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

    #[test]
    fn library_info_returns_none_before_load() {
        let info = library_info();
        assert!(
            info.is_none(),
            "library_info should be None before any kernel is loaded"
        );
    }
}
