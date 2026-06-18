//! INF-12: High-level decode infrastructure — allocate pooled buffers for up to N
//! concurrent sequences and drive the scheduler round-robin.
//!
//! `DecodeSession` owns per-slot `GpuKvCache`, `GpuForwardScratch`, and `CpuForwardScratch`
//! and provides a single `run_step()` call that advances every active sequence.

use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::weights::CpuModelWeights;
use crate::gpu::batch_decode::gpu_full_forward_decode_step;
use crate::gpu::cache::{GpuForwardScratch, GpuKvCache};
use crate::gpu::decode_scheduler::{DecodeBatch, DecodeBatchError, SequenceState};
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::forward::gpu_full_forward_hybrid;
use crate::gpu::forward::GpuLogitsMode;
use crate::gpu::weights::GpuModelWeights;

/// Per-slot decode state including GPU resources.
///
/// `position` is separate from `DecodeBatchSlot.position` because the
/// scheduler position counts decode steps only, while this position may
/// include the prefill length for prefix-caching (INF-14).
pub struct DecodeSlot {
    /// Slot index in the session.
    pub slot_idx: usize,
    /// Current full sequence length (prefill + decode tokens so far).
    pub position: usize,
    /// KV cache for this sequence.
    pub kv: GpuKvCache,
    /// GPU scratch buffers.
    pub scratch: GpuForwardScratch,
    /// Host scratch buffers for logits download / CPU fallback.
    pub host_scratch: CpuForwardScratch,
    /// Opaque request ID from the caller.
    pub sequence_id: u64,
    /// Lifecycle state mirrored from the scheduler.
    pub state: SequenceState,
}

impl DecodeSlot {
    /// Allocate a fresh slot. Fails if HIP is unavailable or VRAM is exhausted.
    pub fn new(
        device: &GpuDevice,
        slot_idx: usize,
        config: &ModelConfig,
        max_tokens: usize,
    ) -> GpuResult<Self> {
        let kv = GpuKvCache::new(config, max_tokens)?;
        let scratch = GpuForwardScratch::new(config)?;
        let host_scratch = CpuForwardScratch::new(config);
        Ok(Self {
            slot_idx,
            position: 0,
            kv,
            scratch,
            host_scratch,
            sequence_id: slot_idx as u64,
            state: SequenceState::Awaiting,
        })
    }
}

/// A decode session managing up to `num_slots` concurrent sequences.
///
/// ```ignore
/// let mut session = DecodeSession::new(device, config, num_slots)?;
/// // Prefill 3 sequences via prefill_graph.rs
///
/// let results = session.run_step(&gpu_weights, &cpu_weights, config, GpuLogitsMode::GreedyArgmax)?;
/// for (slot_idx, seq_id, token) in results {
///     println!("slot {} seq {} -> {}", slot_idx, seq_id, token);
/// }
/// ```
pub struct DecodeSession {
    batch: DecodeBatch,
    slots: Vec<DecodeSlot>,
}

impl DecodeSession {
    /// Allocate `num_slots` slots and a scheduler of the same capacity.
    ///
    /// Fails eagerly if any slot allocation fails (OOM, no GPU, etc.).
    pub fn new(device: &GpuDevice, config: &ModelConfig, num_slots: usize) -> GpuResult<Self> {
        let mut slots = Vec::with_capacity(num_slots);
        let batch = DecodeBatch::new(num_slots);
        for i in 0..num_slots {
            slots.push(DecodeSlot::new(device, i, config, config.max_seq_len)?);
        }
        Ok(Self { batch, slots })
    }

    /// Admit a sequence into the first free slot, returning the slot index.
    pub fn add_sequence(
        &mut self,
        seq_id: u64,
        max_tokens: usize,
    ) -> Result<usize, DecodeBatchError> {
        let slot = self.batch.add_sequence(seq_id, max_tokens)?;
        self.slots[slot].sequence_id = seq_id;
        self.slots[slot].state = SequenceState::Decoding;
        Ok(slot)
    }

    /// Remove a completed sequence by ID, resetting its slot.
    pub fn remove_sequence(&mut self, seq_id: u64) -> Result<usize, DecodeBatchError> {
        let slot = self.batch.remove_sequence(seq_id)?;
        self.slots[slot].position = 0;
        self.slots[slot].state = SequenceState::Awaiting;
        Ok(slot)
    }

    /// Run one decode token for every active slot.
    ///
    /// On success returns `(slot_idx, sequence_id, next_token)` for every
    /// sequence that advanced this step. Sequences that emitted EOS are NOT
    /// in the returned vector — check `session.is_slot_completed(i)`.
    pub fn run_step(
        &mut self,
        device: &GpuDevice,
        gpu_weights: &GpuModelWeights,
        cpu_weights: &CpuModelWeights,
        config: &ModelConfig,
        logits_mode: GpuLogitsMode,
    ) -> GpuResult<Vec<(usize, u64, u32)>> {
        // Rebuild position from DecodeBatch for every active slot before calling forward.
        // This is the pivot point where the scheduler (pure logic) meets the inference engine.
        for (sl_idx, seq_id, decode_pos) in self.batch.active_slots() {
            self.slots[sl_idx].position = decode_pos;
            self.slots[sl_idx].sequence_id = seq_id;
            self.slots[sl_idx].state = SequenceState::Decoding;
        }

        let num = self.batch.num_slots();
        let mut kvs: Vec<&mut GpuKvCache> = Vec::with_capacity(num);
        let mut scratches: Vec<&mut GpuForwardScratch> = Vec::with_capacity(num);
        let mut hosts: Vec<&mut CpuForwardScratch> = Vec::with_capacity(num);

        for slot in &mut self.slots {
            kvs.push(&mut slot.kv);
            scratches.push(&mut slot.scratch);
            hosts.push(&mut slot.host_scratch);
        }

        // Currently `gpu_full_forward_decode_step` takes `&mut [GpuKvCache]` etc.
        // This requires re-borrow or using the slice directly. Since we have
        // separate Vec<RefCell> we call inline logic.

        // Run decode forward for every active slot.
        // We collect active slot info first to avoid borrowing `self.slots` multiple times.
        let active: Vec<(usize, u64, usize)> = self.batch.active_slots().collect();
        let mut results = Vec::new();
        for (slot_idx, seq_id, pos) in active {
            let slot = &mut self.slots[slot_idx];
            let token_id = self.batch.last_token_for_slot(slot_idx).unwrap_or(0);
            let token_opt = gpu_full_forward_hybrid(
                device,
                gpu_weights,
                cpu_weights,
                &mut slot.kv,
                &mut slot.scratch,
                &mut slot.host_scratch,
                pos,
                config,
                logits_mode,
                token_id,
            )?;
            if let Some(token) = token_opt {
                results.push((slot_idx, seq_id, token));
            }
        }

        // Advance scheduler state for every slot that produced a token.
        for (slot_idx, _seq_id, token) in &results {
            let _ = self.batch.advance_slot(*slot_idx, Some(*token));
            if self.batch.is_slot_completed(*slot_idx) {
                self.slots[*slot_idx].state = SequenceState::Completed;
            }
        }

        Ok(results)
    }

    /// Is any slot still actively decoding?
    pub fn any_active(&self) -> bool {
        self.batch.any_active()
    }

    /// Is this slot awaiting admission?
    pub fn is_slot_awaiting(&self, idx: usize) -> bool {
        self.batch.slot_state(idx) == Some(SequenceState::Awaiting)
    }

    /// Is this slot actively decoding?
    pub fn is_slot_active(&self, idx: usize) -> bool {
        self.batch.is_slot_active(idx) == Some(true)
    }

    /// Has this slot emitted EOS and not been removed?
    pub fn is_slot_completed(&self, idx: usize) -> bool {
        self.batch.is_slot_completed(idx)
    }

    /// Current decode position (number of tokens generated so far).
    pub fn position(&self, idx: usize) -> usize {
        self.batch.position_for_slot(idx)
    }

    /// Last emitted token for a slot.
    pub fn last_token(&self, idx: usize) -> Option<u32> {
        self.batch.last_token_for_slot(idx)
    }

    /// Slot capacity.
    pub fn num_slots(&self) -> usize {
        self.batch.num_slots()
    }
}
