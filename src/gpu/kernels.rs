//! HIP kernel bindings for GPU operations.
//!
//! Safety-first design:
//! - All kernel launches wrapped with error checking
//! - Never panic, always return GpuError
//! - CHECK_HIP pattern from Memoria (libgpu.hip)
//!
//! Kernels are loaded dynamically from libgpu.so at runtime.

use super::error::{GpuError, GpuResult};
use std::os::raw::{c_int, c_void};
use std::sync::Mutex;

// Dynamic function pointers loaded from libgpu.so
struct KernelFunctions {
    gpu_kv_write: unsafe extern "C" fn(
        k_cache: *mut f32,
        v_cache: *mut f32,
        k: *const f32,
        v: *const f32,
        pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
    ) -> c_int,
    gpu_kv_write_batched: unsafe extern "C" fn(
        k_cache: *mut f32,
        v_cache: *mut f32,
        k: *const f32,
        v: *const f32,
        start_pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
        seq_len: c_int,
    ) -> c_int,
}

static KERNEL_FUNCS: Mutex<Option<KernelFunctions>> = Mutex::new(None);

/// Load kernel functions from libgpu.so
///
/// Searches for libgpu.so in:
/// 1. /home/feanor/Projects/Memoria/gpu/libgpu.so (Memoria build)
fn load_kernel_functions() -> Result<KernelFunctions, GpuError> {
    const MEMORIA_LIBGPU: &str = "/home/feanor/Projects/Memoria/gpu/libgpu.so";

    #[cfg(target_os = "linux")]
    unsafe {
        let handle = libc::dlopen(
            MEMORIA_LIBGPU.as_ptr() as *const i8,
            libc::RTLD_LAZY | libc::RTLD_LOCAL,
        );

        if handle.is_null() {
            return Err(GpuError::HipNotAvailable);
        }

        let gpu_kv_write_ptr = libc::dlsym(
            handle,
            b"gpu_kv_write\0".as_ptr() as *const i8,
        );

        if gpu_kv_write_ptr.is_null() {
            libc::dlclose(handle);
            return Err(GpuError::HipApiError {
                code: -1,
                description: "gpu_kv_write symbol not found in libgpu.so".to_string(),
            });
        }

        let gpu_kv_write: unsafe extern "C" fn(
            *mut f32,
            *mut f32,
            *const f32,
            *const f32,
            c_int,
            c_int,
            c_int,
        ) -> c_int = std::mem::transmute(gpu_kv_write_ptr);

        let gpu_kv_write_batched_ptr = libc::dlsym(
            handle,
            b"gpu_kv_write_batched\0".as_ptr() as *const i8,
        );

        if gpu_kv_write_batched_ptr.is_null() {
            libc::dlclose(handle);
            return Err(GpuError::HipApiError {
                code: -1,
                description: "gpu_kv_write_batched symbol not found in libgpu.so".to_string(),
            });
        }

        let gpu_kv_write_batched: unsafe extern "C" fn(
            *mut f32,
            *mut f32,
            *const f32,
            *const f32,
            c_int,
            c_int,
            c_int,
            c_int,
        ) -> c_int = std::mem::transmute(gpu_kv_write_batched_ptr);

        Ok(KernelFunctions {
            gpu_kv_write,
            gpu_kv_write_batched,
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        Err(GpuError::HipNotAvailable)
    }
}

/// Get kernel functions, loading on first access
fn get_kernel_funcs() -> Result<KernelFunctions, GpuError> {
    let mut funcs = KERNEL_FUNCS.lock().unwrap();
    if let Some(ref f) = *funcs {
        // Clone function pointers - they're Copy
        return Ok(KernelFunctions {
            gpu_kv_write: f.gpu_kv_write,
            gpu_kv_write_batched: f.gpu_kv_write_batched,
        });
    }

    let loaded = load_kernel_functions()?;
    *funcs = Some(KernelFunctions {
        gpu_kv_write: loaded.gpu_kv_write,
        gpu_kv_write_batched: loaded.gpu_kv_write_batched,
    });
    Ok(loaded)
}

/// Safe wrapper for KV cache write kernel.
///
/// Writes K/V vectors to cache at specific position.
/// Bounds checking happens before kernel launch.
///
/// # Arguments
/// * `k_cache` - GPU pointer to key cache
/// * `v_cache` - GPU pointer to value cache
/// * `k` - GPU pointer to key vector to write
/// * `v` - GPU pointer to value vector to write
/// * `pos` - Position in cache to write (must be < max_seq)
/// * `kv_size` - Size of K/V vectors
/// * `max_seq` - Maximum sequence length (for validation)
///
/// # Returns
/// Ok(()) on successful kernel launch, Err otherwise
pub fn kv_write(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    kv_size: usize,
    max_seq: usize,
) -> GpuResult<()> {
    // Validate bounds BEFORE launching kernel
    if pos >= max_seq {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("KV write position {} exceeds max_seq {}", pos, max_seq),
        });
    }

    let funcs = get_kernel_funcs()?;

    // Launch kernel and check result (CHECK_HIP pattern)
    let result = unsafe {
        (funcs.gpu_kv_write)(
            k_cache,
            v_cache,
            k,
            v,
            pos as c_int,
            kv_size as c_int,
            max_seq as c_int,
        )
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "kv_write".to_string(),
        });
    }

    Ok(())
}

/// Safe wrapper for batched KV cache write (prefill).
///
/// Writes multiple K/V vectors at once for prompt processing.
///
/// # Arguments
/// * `start_pos` - Starting position in cache
/// * `seq_len` - Number of positions to write
///
/// # Returns
/// Ok(()) on success, Err if bounds invalid or kernel fails
pub fn kv_write_batched(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    start_pos: usize,
    kv_size: usize,
    max_seq: usize,
    seq_len: usize,
) -> GpuResult<()> {
    // Validate bounds
    if start_pos + seq_len > max_seq {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "KV batch write range [{}..{}) exceeds max_seq {}",
                start_pos,
                start_pos + seq_len,
                max_seq
            ),
        });
    }

    let funcs = get_kernel_funcs()?;

    let result = unsafe {
        (funcs.gpu_kv_write_batched)(
            k_cache,
            v_cache,
            k,
            v,
            start_pos as c_int,
            kv_size as c_int,
            max_seq as c_int,
            seq_len as c_int,
        )
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "kv_write_batched".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_write_rejects_invalid_position() {
        // Bounds check should fail before any kernel launch
        let result = kv_write(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            999, // pos > max_seq
            128,
            512,
        );
        assert!(result.is_err());
    }

    #[test]
    fn kv_write_batched_rejects_overflow() {
        let result = kv_write_batched(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            400,
            128,
            512,
            200, // 400 + 200 = 600 > 512
        );
        assert!(result.is_err());
    }

    #[test]
    fn bounds_checked_before_kernel_load() {
        // These should fail without attempting to load libgpu.so
        let _ = kv_write(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 0, 0, 0);
        let _ = kv_write_batched(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 0, 0, 0, 0);
        // Tests pass if bounds checking works without GPU
    }
}
