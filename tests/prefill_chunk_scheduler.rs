//! INF-13: Chunked prefill scheduler tests — TDD RED phase.
//!
//! These tests exercise the prefill chunking state machine in `DecodeBatch`.
//! No real GPU kernels are launched; only scheduler state is driven.

#[cfg(feature = "gpu")]
mod prefill_tests {
    use rocmforge::gpu::decode_scheduler::{DecodeBatch, DecodeBatchError, SequenceState};

    #[test]
    fn add_sequence_with_prompt_stores_prefill_tokens() {
        let mut batch = DecodeBatch::new(4);
        let prompt = vec![1, 2, 3, 4, 5];
        let slot = batch
            .add_sequence_with_prompt(42, &prompt, 256)
            .expect("add seq with prompt");
        assert_eq!(slot, 0);
        assert_eq!(batch.prefill_remaining(slot), 5);
        assert_eq!(batch.prefill_done(slot), 0);
        assert_eq!(batch.slot_state(slot), Some(SequenceState::Prefilling));
    }

    #[test]
    fn prefill_next_chunk_advances_by_chunk_size() {
        let mut batch = DecodeBatch::new(4);
        let prompt = vec![1, 2, 3, 4, 5];
        let slot = batch
            .add_sequence_with_prompt(42, &prompt, 256)
            .expect("add seq with prompt");
        let remaining = batch.prefill_next_chunk(slot, 2).expect("chunk");
        assert_eq!(remaining, 3); // 5 - 2 = 3 remaining
        assert_eq!(batch.prefill_done(slot), 2);
        assert_eq!(batch.slot_state(slot), Some(SequenceState::Prefilling));
    }

    #[test]
    fn prefill_next_chunk_transitions_to_decoding_when_done() {
        let mut batch = DecodeBatch::new(4);
        let prompt = vec![1, 2, 3];
        let slot = batch
            .add_sequence_with_prompt(42, &prompt, 256)
            .expect("add seq with prompt");
        let remaining = batch.prefill_next_chunk(slot, 10).expect("chunk");
        assert_eq!(remaining, 0); // all 3 processed
        assert_eq!(batch.slot_state(slot), Some(SequenceState::Decoding));
        assert_eq!(batch.position_for_slot(slot), 3); // total sequence length after prefill
    }

    #[test]
    fn interleave_prefill_and_decode_slots() {
        let mut batch = DecodeBatch::new(4);
        let slot_prefill = batch
            .add_sequence_with_prompt(42, &vec![1, 2, 3, 4, 5], 256)
            .expect("add prefill");
        let slot_decode = batch.add_sequence(43, 256).expect("add decode");

        assert_eq!(
            batch.slot_state(slot_prefill),
            Some(SequenceState::Prefilling)
        );
        assert_eq!(batch.slot_state(slot_decode), Some(SequenceState::Decoding));

        assert_eq!(batch.prefill_slots().count(), 1);
        assert_eq!(batch.active_slots().count(), 1); // only decode slot

        // After prefilling the chunk, slot 0 should transition
        let remaining = batch.prefill_next_chunk(slot_prefill, 5).expect("chunk");
        assert_eq!(remaining, 0);
        assert_eq!(
            batch.slot_state(slot_prefill),
            Some(SequenceState::Decoding)
        );
        assert_eq!(batch.active_slots().count(), 2);
    }

    #[test]
    fn add_sequence_without_prompt_immediately_decoding() {
        let mut batch = DecodeBatch::new(4);
        let slot = batch.add_sequence(42, 256).expect("add seq");
        assert_eq!(batch.slot_state(slot), Some(SequenceState::Decoding));
        assert_eq!(batch.prefill_remaining(slot), 0);
        assert_eq!(batch.prefill_done(slot), 0);
    }

    #[test]
    fn prefill_budget_rejects_zero_chunk() {
        let mut batch = DecodeBatch::new(4);
        let prompt = vec![1, 2, 3];
        let slot = batch
            .add_sequence_with_prompt(42, &prompt, 256)
            .expect("add seq with prompt");
        let err = batch.prefill_next_chunk(slot, 0).unwrap_err();
        assert_eq!(err, DecodeBatchError::InvalidChunkSize);
    }

    #[test]
    fn remove_sequence_during_prefill_frees_slot() {
        let mut batch = DecodeBatch::new(4);
        let prompt = vec![1, 2, 3, 4, 5];
        batch
            .add_sequence_with_prompt(42, &prompt, 256)
            .expect("add seq with prompt");
        batch.prefill_next_chunk(0, 2).expect("prefill chunk");
        let slot = batch.remove_sequence(42).expect("remove seq");
        assert_eq!(slot, 0);
        // Slot should be available for reuse
        let new_slot = batch.add_sequence(99, 128).expect("add new seq");
        assert_eq!(new_slot, 0);
        assert_eq!(batch.slot_state(new_slot), Some(SequenceState::Decoding));
    }

    #[test]
    fn prefill_chunk_rejects_on_decode_slot() {
        let mut batch = DecodeBatch::new(4);
        let slot = batch.add_sequence(42, 256).expect("add seq");
        let err = batch.prefill_next_chunk(slot, 2).unwrap_err();
        assert_eq!(err, DecodeBatchError::NotPrefilling);
    }

    #[test]
    fn prefill_slot_has_zero_position_until_complete() {
        let mut batch = DecodeBatch::new(4);
        let prompt = vec![1, 2, 3, 4, 5];
        let slot = batch
            .add_sequence_with_prompt(42, &prompt, 256)
            .expect("add seq with prompt");
        // Position is 0 until prefill completes
        assert_eq!(batch.position_for_slot(slot), 0);
        batch.prefill_next_chunk(slot, 5).expect("prefill chunk");
        // After prefill, position reflects total sequence length
        assert_eq!(batch.position_for_slot(slot), 5);
    }

    #[test]
    fn prefill_sequence_advance_and_eos() {
        let mut batch = DecodeBatch::new(4);
        let prompt = vec![1, 2, 3];
        let slot = batch
            .add_sequence_with_prompt(42, &prompt, 5)
            .expect("add seq with prompt");
        batch.prefill_next_chunk(slot, 3).expect("prefill");
        assert_eq!(batch.slot_state(slot), Some(SequenceState::Decoding));
        assert_eq!(batch.position_for_slot(slot), 3);

        // Decode 1: position 4 (4 < 5)
        batch.advance_slot(slot, Some(7)).expect("decode 1");
        assert_eq!(batch.position_for_slot(slot), 4);

        // Decode 2: position 5 == max_tokens → Completed
        batch.advance_slot(slot, Some(7)).expect("decode 2");
        assert!(batch.is_slot_completed(slot));
        assert_eq!(batch.position_for_slot(slot), 5);
    }

    #[test]
    fn any_prefilling_counts_slots_needing_prefill() {
        let mut batch = DecodeBatch::new(4);
        assert!(!batch.any_prefilling());
        batch
            .add_sequence_with_prompt(42, &vec![1, 2, 3], 256)
            .expect("add prefill");
        assert!(batch.any_prefilling());
        batch.prefill_next_chunk(0, 3).expect("prefill");
        assert!(!batch.any_prefilling());
    }

    #[test]
    fn partial_prefill_then_resume() {
        let mut batch = DecodeBatch::new(4);
        let prompt: Vec<u32> = (0..100).collect();
        let slot = batch
            .add_sequence_with_prompt(42, &prompt, 256)
            .expect("add seq with prompt");

        let rem1 = batch.prefill_next_chunk(slot, 30).expect("chunk");
        assert_eq!(rem1, 70);
        assert_eq!(batch.prefill_done(slot), 30);

        let rem2 = batch.prefill_next_chunk(slot, 30).expect("chunk");
        assert_eq!(rem2, 40);
        assert_eq!(batch.prefill_done(slot), 60);

        let rem3 = batch.prefill_next_chunk(slot, 50).expect("chunk");
        assert_eq!(rem3, 0);
        assert_eq!(batch.prefill_done(slot), 100);
        assert_eq!(batch.position_for_slot(slot), 100);
        assert_eq!(batch.slot_state(slot), Some(SequenceState::Decoding));
    }

    #[test]
    fn slot_state_prefilling_roundtrip() {
        assert_eq!(SequenceState::Prefilling.as_str(), "prefilling");
        assert_eq!(
            "prefilling".parse::<SequenceState>(),
            Ok(SequenceState::Prefilling)
        );
    }
}
