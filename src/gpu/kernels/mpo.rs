//! MPO (Matrix Product Operator) apply kernel dispatch.
//!
//! Wraps the HIP kernel for y = MPO * x where the MPO is a chain of site tensors.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipStream_t;

extern "C" {
    fn gpu_mpo_apply_f32(
        sites: *const f32,
        site_dims: *const u32,
        n_sites: i32,
        out_dim: i32,
        in_dim: i32,
        x: *const f32,
        y: *mut f32,
        stream: hipStream_t,
    ) -> i32;
}

/// Dispatch MPO apply: y = MPO * x
///
/// # Arguments
/// * `sites` — flattened site tensors, contiguous in memory
/// * `site_dims` — per-site dimensions [chi_left, d_i, chi_right] for each site,
///   packed as n_sites * 3 u32 values
/// * `n_sites` — number of sites in the MPO chain
/// * `out_dim` — output dimension (d_0, the physical dim of the first site)
/// * `in_dim` — input dimension (d_{n-1}, the physical dim of the last site)
/// * `x` — input vector [in_dim]
/// * `y` — output vector [out_dim]
/// * `stream` — HIP stream
pub fn dispatch_mpo_apply_f32(
    sites: *const f32,
    site_dims: *const u32,
    n_sites: usize,
    out_dim: usize,
    in_dim: usize,
    x: *const f32,
    y: *mut f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if sites.is_null() || site_dims.is_null() || x.is_null() || y.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "mpo_apply: null pointer argument".to_string(),
        });
    }
    if n_sites == 0 || out_dim == 0 || in_dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "mpo_apply: invalid dimensions n_sites={} out_dim={} in_dim={}",
                n_sites, out_dim, in_dim
            ),
        });
    }

    let code = unsafe {
        gpu_mpo_apply_f32(
            sites,
            site_dims,
            n_sites as i32,
            out_dim as i32,
            in_dim as i32,
            x,
            y,
            stream,
        )
    };
    if code != 0 {
        return Err(GpuError::HipApiError {
            code,
            description: format!("gpu_mpo_apply_f32 failed with code {}", code),
        });
    }
    Ok(())
}
