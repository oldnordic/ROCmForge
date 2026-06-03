//! INF-12: Decode batch scheduler tests — TDD RED phase.
//!
//! These tests exercise multi-sequence scheduling on a single-GPU device.
//! No real kernels are launched; the GpuDevice and GpuBuffers are allocated
//! and the scheduler state is driven.

#[cfg(feature = "gpu")]
mod scheduler_tests {
    use rocmforge::gpu::decode_scheduler::{DecodeBatch, DecodeBatchError, SequenceState};

    #[test]
    fn new_batch_all_slots_awaiting() {
        let batch = DecodeBatch::new(4);
        assert_eq!(batch.num_slots(), 4);
        assert_eq!(batch.is_slot_active(0), Some(false));
        assert_eq!(batch.is_slot_active(1), Some(false));
        assert_eq!(batch.is_slot_active(2), Some(false));
        assert_eq!(batch.is_slot_active(3), Some(false));
    }

    #[test]
    fn add_sequence_fills_first_free_slot() {
        let mut batch = DecodeBatch::new(4);
        let slot = batch.add_sequence(42, 256).expect("add seq 42");
        assert_eq!(slot, 0);
        let slot = batch.add_sequence(43, 128).expect("add seq 43");
        assert_eq!(slot, 1);
    }

    #[test]
    fn add_sequence_fails_when_full() {
        let mut batch = DecodeBatch::new(1);
        let _ = batch.add_sequence(42, 256).expect("add first");
        let err = batch.add_sequence(43, 128).unwrap_err();
        assert_eq!(err, DecodeBatchError::NoFreeSlots);
    }

    #[test]
    fn remove_sequence_frees_slot() {
        let mut batch = DecodeBatch::new(4);
        let _ = batch.add_sequence(42, 256).expect("add seq 42");
        let _ = batch.remove_sequence(42).expect("remove seq 42");
        let slot = batch.add_sequence(99, 128).expect("add seq 99");
        assert_eq!(slot, 0);
    }

    #[test]
    fn slot_initial_pos_is_zero() {
        let mut batch = DecodeBatch::new(4);
        let _ = batch.add_sequence(42, 256);
        assert_eq!(batch.position_for_slot(0), 0);
    }

    #[test]
    fn advance_increments_position() {
        let mut batch = DecodeBatch::new(4);
        let _ = batch.add_sequence(42, 256);
        let _ = batch.advance_slot(0, Some(7));
        assert_eq!(batch.position_for_slot(0), 1);
        assert_eq!(batch.last_token_for_slot(0), Some(7));
    }

    #[test]
    fn advance_to_eos_marks_completed() {
        let mut batch = DecodeBatch::new(4);
        let _ = batch.add_sequence(42, 256);
        let e = batch.advance_slot(0, Some(2)); // assume 2 == EOS
        assert!(e.is_ok());
        assert!(batch.is_slot_completed(0));
        assert_eq!(batch.position_for_slot(0), 1);
        assert_eq!(batch.position_for_slot(0), 1);
        // Completed slot 0 blocks reuse, but other slots remain available
        let slot = batch.add_sequence(43, 256).expect("admit to second slot");
        assert_eq!(slot, 1);
    }

    #[test]
    fn any_active_after_add() {
        let mut batch = DecodeBatch::new(4);
        assert!(!batch.any_active());
        let _ = batch.add_sequence(42, 256);
        assert!(batch.any_active());
    }

    #[test]
    fn no_active_after_all_eos() {
        let mut batch = DecodeBatch::new(4);
        let _ = batch.add_sequence(42, 4);
        let _ = batch.add_sequence(43, 4);
        let _ = batch.advance_slot(0, Some(2)); // EOS
        let _ = batch.advance_slot(1, Some(2));
        assert!(!batch.any_active());
    }

    #[test]
    fn slot_state_enum_roundtrip() {
        assert_eq!(SequenceState::Awaiting.as_str(), "awaiting");
        assert_eq!(SequenceState::Decoding.as_str(), "decoding");
        assert_eq!(SequenceState::Completed.as_str(), "completed");
        assert_eq!(SequenceState::Freed.as_str(), "freed");
        assert_eq!(
            "awaiting".parse::<SequenceState>(),
            Ok(SequenceState::Awaiting)
        );
        assert_eq!(
            "decoding".parse::<SequenceState>(),
            Ok(SequenceState::Decoding)
        );
        assert_eq!(
            "completed".parse::<SequenceState>(),
            Ok(SequenceState::Completed)
        );
        assert_eq!("freed".parse::<SequenceState>(), Ok(SequenceState::Freed));
    }
}
