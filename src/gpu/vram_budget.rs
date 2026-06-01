//! VRAM budget management and safety constants.
//!
//! Single source of truth for desktop VRAM reservation and allocation limits.
//! All GPU allocation paths must go through this module to prevent stealing
//! memory from the display compositor.

use super::error::{GpuError, GpuResult};
use super::ffi;
use std::sync::atomic::{AtomicUsize, Ordering};

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Default VRAM reserved for desktop/compositor.
///
/// Override with `ROCMFORGE_DESKTOP_VRAM_GB=<float>` (e.g. `2.0` for a
/// single-monitor setup).  4 GB is the default — conservative enough for
/// multi-monitor 4K compositors sharing the same discrete GPU.
pub const DESKTOP_VRAM_RESERVATION_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Return the effective desktop VRAM reservation, honouring the env override.
pub fn desktop_vram_reservation() -> usize {
    if let Ok(s) = std::env::var("ROCMFORGE_DESKTOP_VRAM_GB") {
        if let Ok(gb) = s.trim().parse::<f64>() {
            if gb >= 0.0 {
                return (gb * GB) as usize;
            }
        }
    }
    DESKTOP_VRAM_RESERVATION_BYTES
}

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
    let usable_vram = free_vram.saturating_sub(desktop_vram_reservation());
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
                desktop_vram_reservation() / (1024 * 1024),
                budget.total_vram / (1024 * 1024)
            ),
        });
    }
    Ok(())
}

// ── VramSession ───────────────────────────────────────────────────────────────────

/// Snapshot of VRAM state at inference startup.
///
/// Captures free/total VRAM before any model allocation, computes the desktop
/// reservation and the effective inference budget, and provides a pre-flight
/// check that aborts cleanly when the workload won't fit.
#[derive(Clone, Copy, Debug)]
pub struct VramSession {
    pub device_id: i32,
    /// Total VRAM on the device.
    pub total: usize,
    /// Free VRAM at session creation (before any rocmforge allocations).
    pub startup_free: usize,
    /// VRAM already occupied by other processes / compositor at startup.
    pub already_used: usize,
    /// VRAM held back for the desktop compositor (never touched by inference).
    pub desktop_reserved: usize,
    /// Effective budget for inference = startup_free - desktop_reserved.
    pub inference_budget: usize,
}

impl VramSession {
    /// Query the GPU and capture the current VRAM state.
    pub fn new(device_id: i32) -> GpuResult<Self> {
        let (free, total) =
            ffi::hip_get_mem_info(device_id).map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("VRAM startup query failed for device {}: {}", device_id, e),
            })?;
        let already_used = total.saturating_sub(free);
        let desktop_reserved = desktop_vram_reservation();
        let inference_budget = free.saturating_sub(desktop_reserved);
        Ok(VramSession {
            device_id,
            total,
            startup_free: free,
            already_used,
            desktop_reserved,
            inference_budget,
        })
    }

    /// Print a human-readable VRAM status table to stderr.
    ///
    /// `model_bytes`, `kv_bytes`, `scratch_bytes` are estimates from static
    /// methods on the weight/cache/scratch structs before any allocation.
    pub fn print_startup_report(&self, model_bytes: usize, kv_bytes: usize, scratch_bytes: usize) {
        let total_needed = model_bytes + kv_bytes + scratch_bytes;
        let fits = total_needed <= self.inference_budget;
        let status = if fits { "OK" } else { "OVER BUDGET" };
        eprintln!("  VRAM status (device {}):", self.device_id);
        eprintln!("    Total              {:>7.2} GB", self.total as f64 / GB);
        eprintln!(
            "    Used (other)       {:>7.2} GB",
            self.already_used as f64 / GB
        );
        eprintln!(
            "    Free               {:>7.2} GB",
            self.startup_free as f64 / GB
        );
        eprintln!(
            "    Desktop reserved   {:>7.2} GB  (ROCMFORGE_DESKTOP_VRAM_GB to change)",
            self.desktop_reserved as f64 / GB
        );
        eprintln!(
            "    For inference      {:>7.2} GB",
            self.inference_budget as f64 / GB
        );
        eprintln!("  Estimated usage:");
        eprintln!("    Model weights      {:>7.2} GB", model_bytes as f64 / GB);
        eprintln!("    KV cache           {:>7.2} GB", kv_bytes as f64 / GB);
        eprintln!(
            "    Scratch buffers    {:>7.2} GB",
            scratch_bytes as f64 / GB
        );
        eprintln!(
            "    Total required     {:>7.2} GB  [{}]",
            total_needed as f64 / GB,
            status
        );
    }

    /// Abort if `model_bytes + kv_bytes + scratch_bytes` exceeds the inference budget.
    pub fn check_fits(
        &self,
        model_bytes: usize,
        kv_bytes: usize,
        scratch_bytes: usize,
    ) -> GpuResult<()> {
        let total = model_bytes + kv_bytes + scratch_bytes;
        if total > self.inference_budget {
            return Err(GpuError::OutOfMemory {
                requested: total,
                available: self.inference_budget,
                hint: format!(
                    "Model ({:.2} GB) + KV cache ({:.2} GB) + scratch ({:.2} GB) = {:.2} GB \
                     exceeds inference budget of {:.2} GB ({:.2} GB free, {:.2} GB reserved for desktop). \
                     Try a smaller model, shorter context, or set ROCMFORGE_DESKTOP_VRAM_GB to a lower value.",
                    model_bytes as f64 / GB,
                    kv_bytes as f64 / GB,
                    scratch_bytes as f64 / GB,
                    total as f64 / GB,
                    self.inference_budget as f64 / GB,
                    self.startup_free as f64 / GB,
                    self.desktop_reserved as f64 / GB,
                ),
            });
        }
        Ok(())
    }
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

// ── Runtime VRAM tracking ────────────────────────────────────────────────────────

/// Process-wide counter of bytes currently allocated in GPU VRAM by rocmforge.
///
/// Updated by [`GpuBuffer::alloc`] and [`GpuBuffer::drop`] to give a live view
/// of how much VRAM the inference session has consumed.  This is *in addition*
/// to the pre-flight estimate — it tracks reality.
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Add `bytes` to the running total of allocated VRAM.
pub fn track_allocation(bytes: usize) {
    ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Subtract `bytes` from the running total of allocated VRAM.
pub fn track_deallocation(bytes: usize) {
    ALLOCATED_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// Return the number of bytes currently allocated by rocmforge.
pub fn current_allocated_bytes() -> usize {
    ALLOCATED_BYTES.load(Ordering::Relaxed)
}

/// Return a human-readable string of the current VRAM usage.
pub fn format_vram_usage() -> String {
    let bytes = current_allocated_bytes();
    format!("{:.2} GB", bytes as f64 / GB)
}

/// Pre-flight VRAM budget check for binaries.
///
/// Ensures the VRAM manager is initialized, queries the budget, and verifies
/// that there is sufficient free VRAM (greater than the desktop reservation).
/// If not, prints a clear diagnostic and exits with error code 1.
pub fn binary_vram_safety_preflight(device_id: i32) {
    match query_vram_budget(device_id) {
        Ok(budget) => {
            let reserved_gb = desktop_vram_reservation() as f64 / GB;
            let free_gb = budget.free_vram as f64 / GB;
            let safe_gb = budget.safe_allocation_size as f64 / GB;

            eprintln!(
                "⚡ [VRAM Manager] Free VRAM: {:.2} GB | Desktop Reserved: {:.2} GB | Safe Allocation Limit: {:.2} GB",
                free_gb, reserved_gb, safe_gb
            );

            if budget.free_vram <= desktop_vram_reservation() {
                eprintln!(
                    "❌ ERROR: Free VRAM ({:.2} GB) is less than or equal to the desktop VRAM reservation ({:.2} GB). Refusing unsafe GPU execution.",
                    free_gb, reserved_gb
                );
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!(
                "❌ ERROR querying VRAM budget: {}. Refusing unsafe GPU execution.",
                e
            );
            std::process::exit(1);
        }
    }
}
