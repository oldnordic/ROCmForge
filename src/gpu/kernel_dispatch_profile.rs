//! Kernel dispatch profiling — records which kernel variant was selected per call.
//!
//! Complements `profile.rs` (wall-clock timing) and `decode_profile.rs` (stage-level
//! decode timing) by tracking *which* kernel path was taken for every dispatch.
//! This is essential for verifying that env-var overrides and autotune selections
//! are actually affecting dispatch decisions at runtime.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A single dispatch decision record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRecord {
    /// Kernel family name (e.g. "gemv_q4_0_f32", "gemm_q8_0_f32")
    pub kernel_family: String,
    /// Specific variant selected (e.g. "wave32", "wave64", "dp4a", "baseline")
    pub variant: String,
    /// Quantization type string
    pub quant_type: String,
    /// Whether this was a prefill (seq_len > 1) or decode (seq_len == 1) dispatch
    pub is_prefill: bool,
    /// Call count for this (family, variant) pair
    pub calls: u64,
}

/// Per-dispatch profiler singleton.
pub struct KernelDispatchProfiler {
    records: HashMap<(String, String, String, bool), u64>,
}

impl KernelDispatchProfiler {
    fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    fn global() -> &'static Mutex<Self> {
        static INSTANCE: OnceLock<Mutex<KernelDispatchProfiler>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(Self::new()))
    }

    /// Record a kernel dispatch decision.
    ///
    /// # Arguments
    /// * `kernel_family` — e.g. "gemv_q4_0_f32", "gemm_q8_0_f32"
    /// * `variant` — e.g. "wave32", "wave64", "dp4a", "baseline", "autotune_v2"
    /// * `quant_type` — GGUF type name, e.g. "Q4_0", "Q8_0"
    /// * `is_prefill` — true if seq_len > 1, false for decode (seq_len == 1)
    pub fn record(kernel_family: &str, variant: &str, quant_type: &str, is_prefill: bool) {
        if let Ok(mut profiler) = Self::global().lock() {
            let key = (
                kernel_family.to_string(),
                variant.to_string(),
                quant_type.to_string(),
                is_prefill,
            );
            *profiler.records.entry(key).or_insert(0) += 1;
        }
    }

    /// Get all dispatch records as a Vec.
    pub fn get_records() -> Vec<DispatchRecord> {
        let profiler = match Self::global().lock() {
            Ok(p) => p,
            Err(poison) => poison.into_inner(),
        };
        profiler
            .records
            .iter()
            .map(
                |((family, variant, quant, prefill), calls)| DispatchRecord {
                    kernel_family: family.clone(),
                    variant: variant.clone(),
                    quant_type: quant.clone(),
                    is_prefill: *prefill,
                    calls: *calls,
                },
            )
            .collect()
    }

    /// Reset all recorded dispatch decisions.
    pub fn reset() {
        let mut profiler = match Self::global().lock() {
            Ok(p) => p,
            Err(poison) => poison.into_inner(),
        };
        profiler.records.clear();
    }

    /// Print a summary table of all dispatch decisions.
    pub fn print_summary() {
        let records = Self::get_records();
        if records.is_empty() {
            println!("\n=== Kernel Dispatch Profile (no dispatches recorded) ===");
            return;
        }

        println!("\n=== Kernel Dispatch Profile ===");
        println!(
            "{:<24} {:<16} {:<8} {:<10} {:>8}",
            "Kernel Family", "Variant", "Quant", "Mode", "Calls"
        );
        println!("{}", "-".repeat(80));

        let mut sorted = records;
        sorted.sort_by(|a, b| {
            b.calls
                .cmp(&a.calls)
                .then_with(|| a.kernel_family.cmp(&b.kernel_family))
        });

        for r in sorted {
            println!(
                "{:<24} {:<16} {:<8} {:<10} {:>8}",
                r.kernel_family,
                r.variant,
                r.quant_type,
                if r.is_prefill { "prefill" } else { "decode" },
                r.calls
            );
        }
    }
}

/// Convenience: record a GEMV dispatch decision.
pub fn record_gemv_dispatch(quant_type: &str, variant: &str) {
    KernelDispatchProfiler::record(
        &format!("gemv_{}_f32", quant_type.to_ascii_lowercase()),
        variant,
        quant_type,
        false,
    );
}

/// Convenience: record a GEMM dispatch decision.
pub fn record_gemm_dispatch(quant_type: &str, variant: &str) {
    KernelDispatchProfiler::record(
        &format!("gemm_{}_f32", quant_type.to_ascii_lowercase()),
        variant,
        quant_type,
        true,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_get() {
        KernelDispatchProfiler::reset();
        KernelDispatchProfiler::record("gemv_q4_0_f32", "wave32", "Q4_0", false);
        KernelDispatchProfiler::record("gemv_q4_0_f32", "wave32", "Q4_0", false);
        KernelDispatchProfiler::record("gemv_q4_0_f32", "wave64", "Q4_0", false);
        KernelDispatchProfiler::record("gemm_q8_0_f32", "baseline", "Q8_0", true);

        let records = KernelDispatchProfiler::get_records();
        assert_eq!(records.len(), 3);

        let wave32 = records
            .iter()
            .find(|r| r.variant == "wave32")
            .expect("wave32 record");
        assert_eq!(wave32.calls, 2);
        assert!(!wave32.is_prefill);

        let wave64 = records
            .iter()
            .find(|r| r.variant == "wave64")
            .expect("wave64 record");
        assert_eq!(wave64.calls, 1);

        let gemm = records
            .iter()
            .find(|r| r.kernel_family == "gemm_q8_0_f32")
            .expect("gemm record");
        assert!(gemm.is_prefill);
    }

    #[test]
    fn test_reset_clears_records() {
        KernelDispatchProfiler::reset();
        KernelDispatchProfiler::record("gemv_q4_0_f32", "wave32", "Q4_0", false);
        KernelDispatchProfiler::reset();
        let records = KernelDispatchProfiler::get_records();
        assert!(records.is_empty());
    }

    #[test]
    fn test_convenience_helpers() {
        KernelDispatchProfiler::reset();
        record_gemv_dispatch("Q4_0", "wave32");
        record_gemm_dispatch("Q8_0", "baseline");

        let records = KernelDispatchProfiler::get_records();
        assert_eq!(records.len(), 2);

        let gemv = records
            .iter()
            .find(|r| r.kernel_family == "gemv_q4_0_f32")
            .expect("gemv record");
        assert!(!gemv.is_prefill);

        let gemm = records
            .iter()
            .find(|r| r.kernel_family == "gemm_q8_0_f32")
            .expect("gemm record");
        assert!(gemm.is_prefill);
    }
}
