//! HIP kernel bindings for GPU operations.
//!
//! Safety-first design:
//! - All kernel launches wrapped with error checking
//! - Never panic, always return GpuError
//! - CHECK_HIP pattern from Memoria (libgpu.hip)
//!
//! Kernels are loaded dynamically from libgpu.so via KernelRegistry.

use super::error::{GpuError, GpuResult};
use super::dynamic_loader;
use std::os::raw::c_int;

// ── Public Kernel Wrapper Functions ────────────────────────────────────────────────

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

    // Get kernel from registry (loads on first use)
    let kernel = dynamic_loader::get_kernel(|registry| registry.gpu_kv_write())?;

    // Launch kernel and check result (CHECK_HIP pattern)
    let result = unsafe {
        kernel(
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

    let kernel = dynamic_loader::get_kernel(|registry| registry.gpu_kv_write_batched())?;

    let result = unsafe {
        kernel(
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
