//! INF-12 + INF-13: Decode batch scheduler for continuous multi-sequence generation
//! with chunked prefill support.
//!
//! A slot table manages up to `N` concurrent sequences on one GPU.
//! Sequences can be admitted with or without a prompt.  Prompt-bearing sequences
//! enter the `Prefilling` state and are advanced chunk-by-chunk by the caller.
//! Once the prompt is fully prefilled they transition to `Decoding`.
//!
//! Admission and eviction are out-of-band: the caller adds sequences and
//! advances them until they emit `EOS`, then removes them to free slots.
//! This module is pure logic (no GPU kernels).

use core::fmt;
use std::str::FromStr;

/// Terminal token used by the scheduler to detect sequence completion.
/// The default `EOS=2` matches the LLaMa / Gemma convention.
pub const DEFAULT_EOS_TOKEN: u32 = 2;

/// Default chunk size (in tokens) for prefill steps.
pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 512;

/// Lifecycle state for a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// Slot is available — no sequence admitted.
    Awaiting,
    /// Slot is actively prefill-processing a prompt — output tokens not yet generated.
    Prefilling,
    /// Slot is actively generating decode tokens.
    Decoding,
    /// Slot emitted EOS or hit max_tokens — waiting for caller to remove it.
    Completed,
    /// Slot was removed (temporary state until next add_sequence overwrites it).
    Freed,
}

impl SequenceState {
    pub fn as_str(&self) -> &'static str {
        match self {
            SequenceState::Awaiting => "awaiting",
            SequenceState::Prefilling => "prefilling",
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
            "prefilling" => Ok(SequenceState::Prefilling),
            "decoding" => Ok(SequenceState::Decoding),
            "completed" => Ok(SequenceState::Completed),
            "freed" => Ok(SequenceState::Freed),
            _ => Err(DecodeBatchError::InvalidState(s.to_string())),
        }
    }
}

/// Per-slot bookkeeping.
#[derive(Clone, Debug)]
pub struct DecodeBatchSlot {
    /// Unique request ID (opaque to the scheduler).
    pub sequence_id: u64,
    /// Current lifecycle state.
    pub state: SequenceState,
    /// Number of tokens already processed/generated (starts at 0).
    pub position: usize,
    /// Maximum tokens to allow in this sequence before forced termination.
    pub max_tokens: usize,
    /// Prompt token IDs that are pending prefill (empty for decode-only slots).
    pub prompt_tokens: Vec<u32>,
    /// Number of prompt tokens already processed during chunked prefill.
    pub prefill_done: usize,
    /// Maximum number of tokens per prefill chunk.
    pub prefill_budget: usize,
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
            prompt_tokens: Vec::new(),
            prefill_done: 0,
            prefill_budget: DEFAULT_PREFILL_CHUNK_TOKENS,
            last_token: None,
        }
    }

    /// Reset to empty, preserving only the slot index.
    pub fn clear(&mut self, slot_id: u64) {
        self.sequence_id = slot_id;
        self.state = SequenceState::Awaiting;
        self.position = 0;
        self.max_tokens = 0;
        self.prompt_tokens.clear();
        self.prefill_done = 0;
        self.prefill_budget = DEFAULT_PREFILL_CHUNK_TOKENS;
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
    NotPrefilling,
    InvalidChunkSize,
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
            DecodeBatchError::NotPrefilling => f.write_str("slot is not in Prefilling state"),
            DecodeBatchError::InvalidChunkSize => f.write_str("prefill chunk size must be > 0"),
        }
    }
}

impl std::error::Error for DecodeBatchError {}

pub mod infra;
pub struct DecodeBatch {
    slots: Vec<DecodeBatchSlot>,
    /// Token signalling completion.
    eos_token: u32,
    /// Prefill chunk size.
    prefill_budget: usize,
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
            prefill_budget: DEFAULT_PREFILL_CHUNK_TOKENS,
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

    /// Is there at least one `Prefilling` slot?
    pub fn any_prefilling(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.state == SequenceState::Prefilling)
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

    /// Admit a decode-only sequence (no prompt) into the first free slot.
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
                self.slots[idx].prompt_tokens.clear();
                self.slots[idx].prefill_done = 0;
                self.slots[idx].prefill_budget = self.prefill_budget;
                self.slots[idx].last_token = None;
                Ok(idx)
            }
            None => Err(DecodeBatchError::NoFreeSlots),
        }
    }

    /// Admit a sequence with a prompt. The slot enters `Prefilling` state
    /// and must be advanced via `prefill_next_chunk` until empty.
    pub fn add_sequence_with_prompt(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<usize, DecodeBatchError> {
        let free_idx = self
            .slots
            .iter()
            .position(|s| s.state == SequenceState::Awaiting || s.state == SequenceState::Freed);
        match free_idx {
            Some(idx) => {
                self.slots[idx].sequence_id = seq_id;
                self.slots[idx].state = SequenceState::Prefilling;
                self.slots[idx].position = 0;
                self.slots[idx].max_tokens = max_tokens;
                self.slots[idx].prompt_tokens = prompt_tokens.to_vec();
                self.slots[idx].prefill_done = 0;
                self.slots[idx].prefill_budget = self.prefill_budget;
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

    /// Advance one token in a `Decoding` slot.
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

        let eos = token == Some(self.eos_token);
        let reached_limit = slot.position >= slot.max_tokens;
        if eos || reached_limit {
            slot.state = SequenceState::Completed;
        }
        Ok(())
    }

    /// Number of prompt tokens still awaiting prefill for `slot_idx`.
    pub fn prefill_remaining(&self, slot_idx: usize) -> usize {
        self.slots
            .get(slot_idx)
            .map(|s| s.prompt_tokens.len().saturating_sub(s.prefill_done))
            .unwrap_or(0)
    }

    /// Number of prompt tokens already processed for `slot_idx`.
    pub fn prefill_done(&self, slot_idx: usize) -> usize {
        self.slots
            .get(slot_idx)
            .map(|s| s.prefill_done)
            .unwrap_or(0)
    }

    /// Advance a `Prefilling` slot by processing up to `chunk_size` prompt tokens.
    ///
    /// Returns the number of prompt tokens *remaining* after this chunk.
    /// When 0 is returned the slot automatically transitions to `Decoding`.
    pub fn prefill_next_chunk(
        &mut self,
        slot_idx: usize,
        chunk_size: usize,
    ) -> Result<usize, DecodeBatchError> {
        let slot = self
            .slots
            .get_mut(slot_idx)
            .ok_or(DecodeBatchError::SlotNotFound(slot_idx as u64))?;
        if slot.state != SequenceState::Prefilling {
            return Err(DecodeBatchError::NotPrefilling);
        }
        if chunk_size == 0 {
            return Err(DecodeBatchError::InvalidChunkSize);
        }
        let remaining = slot.prompt_tokens.len().saturating_sub(slot.prefill_done);
        let consumed = chunk_size.min(remaining);
        slot.prefill_done += consumed;
        let still_remaining = slot.prompt_tokens.len().saturating_sub(slot.prefill_done);
        if still_remaining == 0 {
            slot.state = SequenceState::Decoding;
            slot.position = slot.prefill_done; // position tracks total sequence length
        }
        Ok(still_remaining)
    }

    /// Current token position for a slot.
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

    /// Call `f` once for every `Decoding` slot.
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

    /// Iterate active (`Decoding`) slots.
    pub fn active_slots(&self) -> impl Iterator<Item = (usize, u64, usize)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == SequenceState::Decoding)
            .map(|(idx, s)| (idx, s.sequence_id, s.position))
    }

    /// Iterate prefill (`Prefilling`) slots.
    pub fn prefill_slots(&self) -> impl Iterator<Item = (usize, u64, &[u32], usize)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == SequenceState::Prefilling)
            .map(|(idx, s)| {
                let chunk_start = s.prefill_done;
                let chunk_end = (s.prefill_done + s.prefill_budget).min(s.prompt_tokens.len());
                let remaining = &s.prompt_tokens[chunk_start..chunk_end];
                (idx, s.sequence_id, remaining, chunk_start)
            })
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
            SequenceState::Prefilling,
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
