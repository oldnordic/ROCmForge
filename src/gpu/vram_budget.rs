//! VRAM budget management and safety constants.
//!
//! Single source of truth for desktop VRAM reservation and allocation limits.
//! All GPU allocation paths must go through this module to prevent stealing
//! memory from the display compositor.

use super::error::{GpuError, GpuResult};
use super::ffi;

/// VRAM reserved for desktop/compositor (multi-monitor setups).
///
/// This prevents allocations from stealing memory needed for display.
/// 4 GB is typical for multi-monitor 4K setups with desktop compositors.
pub const DESKTOP_VRAM_RESERVATION_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Safety margin for VRAM allocations (10% of free VRAM).
pub const VRAM_SAFETY_MARGIN_RATIO: f64 = 0.1;

/// Additional guardrail for full model loads.
///
/// Model loading performs many allocations back-to-back, so keep a larger
/// buffer than the one-off allocation guard.
pub const MODEL_LOAD_SAFE_RATIO: f64 = 0.7;

#[derive(Clone, Copy, Debug)]
pub struct VramBudget {
    pub device_id: i32,
    pub free_vram: usize,
    pub total_vram: usize,
    pub safe_allocation_size: usize,
    pub safe_model_load_limit: usize,
}

pub fn query_vram_budget(device_id: i32) -> GpuResult<VramBudget> {
    let (free_vram, total_vram) =
        ffi::hip_get_mem_info(device_id).map_err(|e| GpuError::HipApiError {
            code: -1,
            description: format!(
                "VRAM safety query failed for device {}: {}. Refusing unsafe GPU allocation.",
                device_id, e
            ),
        })?;
    let usable_vram = free_vram.saturating_sub(DESKTOP_VRAM_RESERVATION_BYTES);
    Ok(VramBudget {
        device_id,
        free_vram,
        total_vram,
        safe_allocation_size: (usable_vram as f64 * (1.0 - VRAM_SAFETY_MARGIN_RATIO)) as usize,
        safe_model_load_limit: (usable_vram as f64 * MODEL_LOAD_SAFE_RATIO) as usize,
    })
}

pub fn active_or_default_device_id() -> i32 {
    ffi::hip_get_device().unwrap_or(0)
}

pub fn check_model_load_headroom(
    budget: VramBudget,
    current_usage: usize,
    next_allocation: usize,
) -> GpuResult<()> {
    let projected = current_usage.saturating_add(next_allocation);
    if projected > budget.safe_model_load_limit {
        return Err(GpuError::ModelTooLarge {
            required: projected,
            available: budget.safe_model_load_limit,
            hint: format!(
                "Projected GPU weight load on device {} would use {} MB, exceeding the guarded load budget of {} MB ({} MB free, {} MB reserved for desktop, {} MB total VRAM).",
                budget.device_id,
                projected / (1024 * 1024),
                budget.safe_model_load_limit / (1024 * 1024),
                budget.free_vram / (1024 * 1024),
                DESKTOP_VRAM_RESERVATION_BYTES / (1024 * 1024),
                budget.total_vram / (1024 * 1024)
            ),
        });
    }
    Ok(())
}

/// Check whether a single allocation of `bytes` fits within the VRAM budget.
///
/// Returns `Ok(())` if the allocation is safe, or a descriptive error if not.
pub fn check_allocation_fits(device_id: i32, bytes: usize) -> GpuResult<()> {
    let budget = query_vram_budget(device_id)?;
    if bytes > budget.safe_allocation_size {
        return Err(GpuError::OutOfVram {
            requested: bytes,
            free: budget.free_vram,
            total: budget.total_vram,
        });
    }
    Ok(())
}
