//! INF-12: Decode batch scheduler for continuous multi-sequence generation.
//!
//! A slot table manages up to `N` concurrent decoding sequences on one GPU.
//! Admission and eviction are out-of-band: the caller adds sequences and advances
//! them until they emit `EOS`, then removes them to free slots.
//!
//! This module is pure logic (no GPU kernels).  It coordinates multiple
//! `GpuKvCache` + `GpuForwardScratch` pairs that the caller owns.

use core::fmt;
use std::str::FromStr;

/// Terminal token used by the scheduler to detect sequence completion.
/// The default `EOS=2` matches the LLaMa / Gemma convention.
pub const DEFAULT_EOS_TOKEN: u32 = 2;

/// What a given batch slot is doing right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceState {
    /// Slot has never held a sequence.
    Awaiting,
    /// Sequence is active and has not yet emitted EOS.
    Decoding,
    /// Sequence emitted EOS. Slot is stale until removed.
    Completed,
    /// Slot was freed after removal.
    Freed,
}

impl SequenceState {
    pub fn as_str(&self) -> &'static str {
        match self {
            SequenceState::Awaiting => "awaiting",
            SequenceState::Decoding => "decoding",
            SequenceState::Completed => "completed",
            SequenceState::Freed => "freed",
        }
    }
}

impl fmt::Display for SequenceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SequenceState {
    type Err = DecodeBatchError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "awaiting" => Ok(SequenceState::Awaiting),
            "decoding" => Ok(SequenceState::Decoding),
            "completed" => Ok(SequenceState::Completed),
            "freed" => Ok(SequenceState::Freed),
            _ => Err(DecodeBatchError::InvalidState(s.to_string())),
        }
    }
}

/// Per-slot bookkeeping.  The GPU buffers (`GpuKvCache`, `GpuForwardScratch`)
/// are stored OUTSIDE this struct — the caller indexes `BatchHandle{> into its own pools.
#[derive(Clone, Debug)]
pub struct DecodeBatchSlot {
    /// Unique request ID (opaque to the scheduler).
    pub sequence_id: u64,
    /// Current lifecycle state.
    pub state: SequenceState,
    /// Number of tokens already generated (starts at 0 after prefill).
    pub position: usize,
    /// Maximum tokens to allow in this sequence before forced termination.
    pub max_tokens: usize,
    /// Last emitted token, if any.
    pub last_token: Option<u32>,
}

impl DecodeBatchSlot {
    /// Fresh slot awaiting a sequence.
    pub fn new(slot_id: u64) -> Self {
        Self {
            sequence_id: slot_id,
            state: SequenceState::Awaiting,
            position: 0,
            max_tokens: 0,
            last_token: None,
        }
    }

    /// Reset to empty, preserving only the slot index.
    pub fn clear(&mut self, slot_id: u64) {
        self.sequence_id = slot_id;
        self.state = SequenceState::Awaiting;
        self.position = 0;
        self.max_tokens = 0;
        self.last_token = None;
    }
}

/// Errors returned by the scheduler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeBatchError {
    NoFreeSlots,
    SlotNotFound(u64),
    InvalidState(String),
    SlotNotActive { slot: usize, state: SequenceState },
}

impl fmt::Display for DecodeBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeBatchError::NoFreeSlots => f.write_str("no free slots in decode batch"),
            DecodeBatchError::SlotNotFound(id) => {
                write!(f, "sequence id {id} not found in batch")
            }
            DecodeBatchError::InvalidState(s) => write!(f, "invalid sequence state: {s}"),
            DecodeBatchError::SlotNotActive { slot, state } => {
                write!(f, "slot {slot} is not active (state = {state})")
            }
        }
    }
}

impl std::error::Error for DecodeBatchError {}

/// Slot table for up to `NUM_SLOTS` concurrent sequences.
///
/// Design decisions (grounded in existing infra):
/// 1. **No GPU inside** — caller owns `GpuKvCache` / `GpuForwardScratch` arrays.
///    The scheduler only tells the caller *which* slots to run.
/// 2. **Round-robin by default** — active slots are yielded in ascending order.
///    The caller runs `gpu_layer_forward_hybrid` once per active slot per token.
/// 3. **EOS detection** — configurable per batch; default = 2.
/// 4. **Bounded lifetime** — `max_tokens` per sequence prevents runaway.
pub struct DecodeBatch {
    slots: Vec<DecodeBatchSlot>,
    /// Token signalling completion.
    eos_token: u32,
}

impl DecodeBatch {
    /// Create a scheduler with `num_slots` slots, all initially `Awaiting`.
    pub fn new(num_slots: usize) -> Self {
        let mut slots = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            slots.push(DecodeBatchSlot::new(i as u64));
        }
        Self {
            slots,
            eos_token: DEFAULT_EOS_TOKEN,
        }
    }

    /// Total capacity.
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// How many slots are currently `Decoding`.
    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state == SequenceState::Decoding)
            .count()
    }

    /// Is there at least one `Decoding` slot?
    pub fn any_active(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.state == SequenceState::Decoding)
    }

    /// Return the state of a given slot, if it exists.
    pub fn slot_state(&self, idx: usize) -> Option<SequenceState> {
        self.slots.get(idx).map(|s| s.state)
    }

    /// Is this slot currently active (`Decoding`)?
    pub fn is_slot_active(&self, idx: usize) -> Option<bool> {
        self.slots
            .get(idx)
            .map(|s| s.state == SequenceState::Decoding)
    }

    /// Has this slot emitted EOS and not been removed?
    pub fn is_slot_completed(&self, idx: usize) -> bool {
        self.slots
            .get(idx)
            .map(|s| s.state == SequenceState::Completed)
            .unwrap_or(false)
    }

    /// Admit a new sequence into the first free (`Awaiting` or `Freed`) slot.
    /// Returns the slot index on success.
    pub fn add_sequence(
        &mut self,
        seq_id: u64,
        max_tokens: usize,
    ) -> Result<usize, DecodeBatchError> {
        let free_idx = self
            .slots
            .iter()
            .position(|s| s.state == SequenceState::Awaiting || s.state == SequenceState::Freed);
        match free_idx {
            Some(idx) => {
                self.slots[idx].sequence_id = seq_id;
                self.slots[idx].state = SequenceState::Decoding;
                self.slots[idx].position = 0;
                self.slots[idx].max_tokens = max_tokens;
                self.slots[idx].last_token = None;
                Ok(idx)
            }
            None => Err(DecodeBatchError::NoFreeSlots),
        }
    }

    /// Remove a sequence by its opaque `sequence_id`, freeing the slot.
    pub fn remove_sequence(&mut self, seq_id: u64) -> Result<usize, DecodeBatchError> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.sequence_id == seq_id)
            .ok_or(DecodeBatchError::SlotNotFound(seq_id))?;
        self.slots[idx].clear(idx as u64);
        Ok(idx)
    }

    /// Advance one token in `slot_idx`.
    ///
    /// If `token` is `EOS` or `position >= max_tokens`, the slot transitions to `Completed`.
    pub fn advance_slot(
        &mut self,
        slot_idx: usize,
        token: Option<u32>,
    ) -> Result<(), DecodeBatchError> {
        let slot = self
            .slots
            .get_mut(slot_idx)
            .ok_or(DecodeBatchError::SlotNotFound(slot_idx as u64))?;
        if slot.state != SequenceState::Decoding {
            return Err(DecodeBatchError::SlotNotActive {
                slot: slot_idx,
                state: slot.state,
            });
        }
        slot.position += 1;
        slot.last_token = token;

        let eos = token.map_or(false, |t| t == self.eos_token);
        let reached_limit = slot.position >= slot.max_tokens;
        if eos || reached_limit {
            slot.state = SequenceState::Completed;
        }
        Ok(())
    }

    /// Current token position for a slot (number of decode steps taken).
    pub fn position_for_slot(&self, slot_idx: usize) -> usize {
        self.slots.get(slot_idx).map(|s| s.position).unwrap_or(0)
    }

    /// Last emitted token for a slot, if any.
    pub fn last_token_for_slot(&self, slot_idx: usize) -> Option<u32> {
        self.slots.get(slot_idx).and_then(|s| s.last_token)
    }

    /// Opaque sequence ID for a slot.
    pub fn sequence_id_for_slot(&self, slot_idx: usize) -> Option<u64> {
        self.slots.get(slot_idx).map(|s| s.sequence_id)
    }

    /// Call `f` once for every `Decoding` slot, passing (slot_idx, sequence_id, position).
    pub fn foreach_active<F>(&self, mut f: F)
    where
        F: FnMut(usize, u64, usize),
    {
        for (idx, slot) in self.slots.iter().enumerate() {
            if slot.state == SequenceState::Decoding {
                f(idx, slot.sequence_id, slot.position);
            }
        }
    }

    /// Iterate active slots for testing/inspection.
    pub fn active_slots(&self) -> impl Iterator<Item = (usize, u64, usize)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == SequenceState::Decoding)
            .map(|(idx, s)| (idx, s.sequence_id, s.position))
    }

    /// Force-set EOS token (for models with a non-default EOS ID).
    pub fn set_eos_token(&mut self, token: u32) {
        self.eos_token = token;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_roundtrip() {
        for state in [
            SequenceState::Awaiting,
            SequenceState::Decoding,
            SequenceState::Completed,
            SequenceState::Freed,
        ] {
            let s = state.as_str();
            let parsed: SequenceState = s.parse().expect("roundtrip");
            assert_eq!(parsed, state);
        }
    }

    #[test]
    fn full_lifecycle() {
        let mut b = DecodeBatch::new(2);
        let slot0 = b.add_sequence(100, 10).expect("add 100");
        let slot1 = b.add_sequence(101, 10).expect("add 101");
        assert_eq!(slot0, 0);
        assert_eq!(slot1, 1);

        b.advance_slot(slot0, Some(5)).expect("adv 0");
        b.advance_slot(slot1, Some(5)).expect("adv 1");
        assert_eq!(b.position_for_slot(0), 1);
        assert_eq!(b.position_for_slot(1), 1);

        b.advance_slot(0, Some(DEFAULT_EOS_TOKEN)).expect("eos 0");
        assert!(b.is_slot_completed(0));
        assert!(!b.is_slot_completed(1));

        b.remove_sequence(100).expect("rem 100");
        assert_eq!(b.slot_state(0), Some(SequenceState::Awaiting));
    }
}
