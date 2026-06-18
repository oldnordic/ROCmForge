//! INF-12: Batch decode orchestrator — run `gpu_full_forward_hybrid` for each active
//! slot in a `DecodeBatch` scheduler.
//!
//! This layer is **pure orchestration**: no GPU kernels, no allocations.
//! The caller provides one `GpuKvCache` + `GpuForwardScratch` per slot,
//! already allocated and initialised (prefill must have run before the first
//! `gpu_batch_decode` call).
//!
//! Design: one-shot per sequence — no batched kernel fusion at this level.
//! Future INF phases can replace the per-sequence loop with a single batched
//! kernel call.

use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::weights::CpuModelWeights;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::decode_scheduler::{DecodeBatch, DecodeBatchError};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::GpuResult;
use crate::gpu::forward::{gpu_full_forward_hybrid, GpuLogitsMode};
use crate::gpu::weights::GpuModelWeights;

/// Run one decode step for every active (`Decoding`) slot in `batch`.
///
/// `kv_cache` and `scratch` MUST have the same length as `batch.num_slots()`.
/// The caller must have already run prefill on each sequence so that the
/// KV cache is warm and the initial token is in the scratch buffers.
///
/// On success returns a vector of `(slot_idx, sequence_id, next_token)` for all
/// slots that produced a token this step. Slots that hit EOS are **not** in the
/// return vector — the caller should check `batch.is_slot_completed(idx)`.
///
/// ## Round-robin guarantees
///
/// Slots are processed in ascending index order. This is deterministic and
/// avoids starvation even when some sequences finish early.
///
/// ## GPU safety
///
/// Each slot runs `gpu_full_forward_hybrid` independently.  Because every slot
/// has its own scratch and KV cache, there is no buffer sharing and therefore
/// no inter-slot race condition.  Synchronization is per slot.
///
/// ## Typical usage
///
/// ```ignore
/// while batch.any_active() {
///     let results = gpu_full_forward_decode_step(
///         device, gpu_weights, cpu_weights,
///         &mut batch,
///         &mut kv_caches, &mut scratches, &mut host_scratches,
///         config,
///     )?;
///     for (slot_idx, seq_id, token) in results {
///         batch.advance_slot(slot_idx, Some(token))?;
///         emit(seq_id, token);
///     }
/// }
/// ```
pub fn gpu_full_forward_decode_step(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    batch: &mut DecodeBatch,
    kv_caches: &mut [GpuKvCache],
    scratches: &mut [GpuForwardScratch],
    host_scratches: &mut [CpuForwardScratch],
    config: &ModelConfig,
) -> GpuResult<Vec<(usize, u64, u32)>> {
    let num_slots = batch.num_slots();
    assert_eq!(
        kv_caches.len(),
        num_slots,
        "kv_cache count must match batch slots"
    );
    assert_eq!(
        scratches.len(),
        num_slots,
        "scratch count must match batch slots"
    );
    assert_eq!(
        host_scratches.len(),
        num_slots,
        "host_scratch count must match batch slots"
    );

    let mut results = Vec::new();

    for (slot_idx, seq_id, pos) in batch.active_slots() {
        let slot_idx = slot_idx;
        let kv = &mut kv_caches[slot_idx];
        let scratch = &mut scratches[slot_idx];
        let host = &mut host_scratches[slot_idx];

        // Get the last token from the batch slot
        let token_id = batch.last_token_for_slot(slot_idx).unwrap_or(0);

        let token_opt = gpu_full_forward_hybrid(
            device,
            gpu_weights,
            cpu_weights,
            kv,
            scratch,
            host,
            pos,
            config,
            GpuLogitsMode::GreedyArgmax,
            token_id,
        )?;

        if let Some(token) = token_opt {
            results.push((slot_idx, seq_id, token));
        }
    }

    Ok(results)
}
