//! GPU kernel wrappers organized by functionality.
//!
//! Safety-first design:
//! - All kernels validate bounds before launching
//! - All hipError_t return values checked
//! - Never panic, always return GpuError

pub mod norm;

pub use norm::{rms_norm, rms_norm_batched};

// Re-export attention kernels from dynamic_loader
// TODO: Replace with our own HIP kernels in Task 5
pub use crate::gpu::dynamic_loader::get_kernel;
use crate::gpu::error::{GpuError, GpuResult};
use std::os::raw::c_int;

/// KV cache write (temporary wrapper using dynamic_loader)
pub fn kv_write(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    kv_size: usize,
    max_seq: usize,
) -> GpuResult<()> {
    if pos >= max_seq {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("KV write position {} exceeds max_seq {}", pos, max_seq),
        });
    }

    // Temporarily use dynamic_loader's KernelRegistry
    let kernel = get_kernel(|registry| registry.gpu_kv_write())?;

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

/// Batched KV cache write (temporary wrapper using dynamic_loader)
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

    let kernel = get_kernel(|registry| registry.gpu_kv_write_batched())?;

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
