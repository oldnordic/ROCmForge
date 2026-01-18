//! Comprehensive TDD tests for scheduler module

use rocmforge::scheduler::{
    Batch, GenerationRequest, RequestState, Scheduler, SchedulerConfig, SchedulerError,
};
use std::time::Duration;

#[test]
fn test_request_creation() {
    let request = GenerationRequest::new(1, vec![1, 2, 3, 4, 5], 10, 0.8, 50, 0.9);

    assert_eq!(request.request_id, 1);
    assert_eq!(request.prompt_tokens, vec![1, 2, 3, 4, 5]);
    assert_eq!(request.max_tokens, 10);
    assert_eq!(request.temperature, 0.8);
    assert_eq!(request.top_k, 50);
    assert_eq!(request.top_p, 0.9);
    assert_eq!(request.state, RequestState::Pending);
    assert_eq!(request.total_tokens(), 5);
    assert!(!request.is_complete());
    assert!(request.created_at <= std::time::Instant::now());
}

#[test]
fn test_request_state_transitions() {
    let mut request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

    // Initial state
    assert_eq!(request.state, RequestState::Pending);
    assert!(request.started_at.is_none());
    assert!(request.completed_at.is_none());

    // Start processing
    let result = request.start_processing();
    assert!(result.is_ok());
    assert_eq!(request.state, RequestState::Processing);
    assert!(request.started_at.is_some());
    assert!(request.completed_at.is_none());

    // Add generated tokens
    for i in 0..10 {
        let result = request.add_generated_token(i);
        assert!(result.is_ok());
    }

    // Should be complete now
    assert_eq!(request.state, RequestState::Completed);
    assert!(request.completed_at.is_some());
    assert!(request.is_complete());
    assert_eq!(request.generated_tokens.len(), 10);

    // Verify processing time
    let processing_time = request.processing_time();
    assert!(processing_time.is_some());
    assert!(processing_time.unwrap() >= Duration::from_secs(0));
}

#[test]
fn test_invalid_state_transitions() {
    let mut request = GenerationRequest::new(1, vec![1], 10, 0.8, 50, 0.9);

    // Can't complete without starting
    let result = request.complete(None);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(SchedulerError::InvalidStateTransition)
    ));

    // Can't fail without starting
    let result = request.fail();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(SchedulerError::InvalidStateTransition)
    ));

    // Start processing
    request.start_processing().context("Failed to start processing")?;

    // Can't start again
    let result = request.start_processing();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(SchedulerError::InvalidStateTransition)
    ));

    // Complete
    request.complete(None).context("Failed to complete request")?;

    // Can't add tokens after completion
    let result = request.add_generated_token(42);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(SchedulerError::InvalidStateTransition)
    ));
}

#[test]
fn test_batch_creation() {
    let mut batch = Batch::new(1);

    assert_eq!(batch.batch_id, 1);
    assert!(batch.requests.is_empty());
    assert!(batch.is_empty());
    assert_eq!(batch.size(), 0);
    assert_eq!(batch.total_tokens(), 0);
    assert_eq!(batch.max_sequence_length(), 0);
    assert_eq!(batch.min_sequence_length(), 0);
    assert_eq!(batch.length_variance(), 0.0);

    // Add requests
    let request1 = GenerationRequest::new(1, vec![1, 2], 10, 0.8, 50, 0.9);
    let request2 = GenerationRequest::new(2, vec![1, 2, 3], 10, 0.8, 50, 0.9);

    batch.add_request(request1).context("Failed to add request")?;
    batch.add_request(request2).context("Failed to add request")?;

    assert_eq!(batch.size(), 2);
    assert!(!batch.is_empty());
    assert_eq!(batch.total_tokens(), 5); // 2 + 3
    assert_eq!(batch.max_sequence_length(), 3);
    assert_eq!(batch.min_sequence_length(), 2);
    assert!(batch.length_variance() > 0.0);
}

#[test]
fn test_batch_length_variance() {
    let mut batch = Batch::new(1);

    // Add requests with similar lengths
    batch
        .add_request(GenerationRequest::new(1, vec![1; 10], 10, 0.8, 50, 0.9))
        .unwrap();
    batch
        .add_request(GenerationRequest::new(2, vec![2; 12], 10, 0.8, 50, 0.9))
        .unwrap();
    batch
        .add_request(GenerationRequest::new(3, vec![3; 11], 10, 0.8, 50, 0.9))
        .unwrap();

    let variance = batch.length_variance();
    assert!(variance < 2.0); // Should be low variance

    // Add request with very different length
    batch
        .add_request(GenerationRequest::new(4, vec![4; 50], 10, 0.8, 50, 0.9))
        .unwrap();

    let variance2 = batch.length_variance();
    assert!(variance2 > variance); // Variance should increase
}

#[test]
fn test_scheduler_creation() {
    let config = SchedulerConfig::default();
    let scheduler = Scheduler::new(config);

    let stats = scheduler.get_queue_stats();
    assert_eq!(stats.pending_requests, 0);
    assert_eq!(stats.processing_requests, 0);
    assert_eq!(stats.completed_requests, 0);

    assert!(!scheduler.has_pending_requests());
    assert!(scheduler.can_create_batch());
}

#[test]
fn test_request_submission() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    // Submit first request
    let request_id1 = scheduler.submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9);
    assert!(request_id1.is_ok());
    assert_eq!(request_id1.unwrap(), 0);

    // Submit second request
    let request_id2 = scheduler.submit_request(vec![4, 5, 6], 15, 0.9, 40, 0.8);
    assert!(request_id2.is_ok());
    assert_eq!(request_id2.unwrap(), 1);

    let stats = scheduler.get_queue_stats();
    assert_eq!(stats.pending_requests, 2);
    assert_eq!(stats.processing_requests, 0);
    assert!(scheduler.has_pending_requests());
}

#[test]
fn test_queue_capacity_limit() {
    let config = SchedulerConfig {
        max_queue_size: 2,
        ..Default::default()
    };
    let mut scheduler = Scheduler::new(config);

    // Submit up to capacity
    scheduler.submit_request(vec![1], 10, 0.8, 50, 0.9).context("Failed to submit request")?;
    scheduler.submit_request(vec![2], 10, 0.8, 50, 0.9).context("Failed to submit request")?;

    // Should fail when exceeding capacity
    let result = scheduler.submit_request(vec![3], 10, 0.8, 50, 0.9);
    assert!(result.is_err());
    assert!(matches!(result, Err(SchedulerError::QueueCapacityExceeded)));
}

#[test]
fn test_cancel_pending_request() {
    let mut scheduler = Scheduler::new(SchedulerConfig::default());
    let request_id = scheduler
        .submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9)
        .unwrap();

    let cancelled = scheduler.cancel_request(request_id).context("Failed to cancel request")?;
    assert_eq!(cancelled.request_id, request_id);
    assert_eq!(cancelled.state, RequestState::Cancelled);
    assert_eq!(cancelled.finish_reason.as_deref(), Some("cancelled"));

    let stats = scheduler.get_queue_stats();
    assert_eq!(stats.pending_requests, 0);
    assert_eq!(stats.completed_requests, 1);
}

#[test]
fn test_cancel_processing_request() {
    let mut scheduler = Scheduler::new(SchedulerConfig::default());
    let request_id = scheduler
        .submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9)
        .unwrap();
    let batch = scheduler.create_batch().context("Failed to create batch")?;
    assert_eq!(batch.size(), 1);

    let cancelled = scheduler.cancel_request(request_id).context("Failed to cancel request")?;
    assert_eq!(cancelled.state, RequestState::Cancelled);
    assert_eq!(cancelled.finish_reason.as_deref(), Some("cancelled"));

    let stats = scheduler.get_queue_stats();
    assert_eq!(stats.processing_requests, 0);
    assert_eq!(stats.completed_requests, 1);
}

#[test]
fn test_batch_creation_with_scheduler() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    // Submit multiple requests
    for i in 0..3 {
        scheduler
            .submit_request(vec![i, i + 1, i + 2], 10, 0.8, 50, 0.9)
            .unwrap();
    }

    let batch = scheduler.create_batch();
    assert!(batch.is_ok());

    let batch = batch.unwrap();
    assert_eq!(batch.size(), 3);
    assert_eq!(batch.batch_id, 0);

    // Verify requests are in processing state
    for request in &batch.requests {
        assert_eq!(request.state, RequestState::Processing);
        assert!(request.started_at.is_some());
    }

    let stats = scheduler.get_queue_stats();
    assert_eq!(stats.pending_requests, 0);
    assert_eq!(stats.processing_requests, 3);
}

#[test]
fn test_batch_update() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    scheduler
        .submit_request(vec![1, 2, 3], 2, 0.8, 50, 0.9)
        .unwrap();
    let batch = scheduler.create_batch().context("Failed to create batch")?;

    // Simulate token generation
    let mut updated_batch = batch;
    for request in &mut updated_batch.requests {
        request.add_generated_token(42).context("Failed to add generated token")?;
        request.add_generated_token(43).context("Failed to add generated token")?; // Should complete
    }

    let completed = scheduler.update_batch(updated_batch).context("Failed to update batch")?;
    assert_eq!(completed.len(), 1);
    assert!(completed[0].is_complete());

    let stats = scheduler.get_queue_stats();
    assert_eq!(stats.processing_requests, 0);
    assert_eq!(stats.completed_requests, 1);
}

#[test]
fn test_request_retrieval() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    let request_id = scheduler
        .submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9)
        .unwrap();

    // Get request from pending queue
    let request = scheduler.get_request(request_id);
    assert!(request.is_ok());
    let request = request.unwrap();
    assert_eq!(request.request_id, request_id);
    assert_eq!(request.state, RequestState::Pending);

    // Create batch to move to processing
    let batch = scheduler.create_batch().context("Failed to create batch")?;

    // Get request from processing
    let request = scheduler.get_request(request_id);
    assert!(request.is_ok());
    let request = request.unwrap();
    assert_eq!(request.state, RequestState::Processing);

    // Complete the request
    let mut updated_batch = batch;
    for request in &mut updated_batch.requests {
        for i in 0..10 {
            let _ = request.add_generated_token(i);
        }
    }
    scheduler.update_batch(updated_batch).context("Failed to update batch")?;

    // Get request from completed
    let request = scheduler.get_request(request_id);
    assert!(request.is_ok());
    let request = request.unwrap();
    assert_eq!(request.state, RequestState::Completed);
}

#[test]
fn test_token_generation() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    let request_id = scheduler
        .submit_request(vec![1, 2, 3], 5, 0.8, 50, 0.9)
        .unwrap();

    // Create batch
    let batch = scheduler.create_batch().context("Failed to create batch")?;

    // Add generated token
    let result = scheduler.add_generated_token(request_id, 42);
    assert!(result.is_ok());

    // Verify token was added
    let request = scheduler.get_request(request_id).context("Failed to get request")?;
    assert_eq!(request.generated_tokens.len(), 1);
    assert_eq!(request.generated_tokens[0], 42);
}

#[test]
fn test_batch_size_limit() {
    let config = SchedulerConfig {
        max_batch_size: 2,
        ..Default::default()
    };
    let mut scheduler = Scheduler::new(config);

    // Submit more requests than batch size
    for i in 0..5 {
        scheduler
            .submit_request(vec![i; 5], 10, 0.8, 50, 0.9)
            .unwrap();
    }

    // Create batch - should only take max_batch_size requests
    let batch = scheduler.create_batch();
    assert!(batch.is_ok());

    let batch = batch.unwrap();
    assert_eq!(batch.size(), 2); // Limited by max_batch_size

    let stats = scheduler.get_queue_stats();
    assert_eq!(stats.pending_requests, 3); // 5 - 2
    assert_eq!(stats.processing_requests, 2);
}

#[test]
fn test_length_based_batching() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    // Submit requests with different lengths
    scheduler
        .submit_request(vec![1; 5], 10, 0.8, 50, 0.9)
        .unwrap(); // Short
    scheduler
        .submit_request(vec![2; 20], 10, 0.8, 50, 0.9)
        .unwrap(); // Long
    scheduler
        .submit_request(vec![3; 6], 10, 0.8, 50, 0.9)
        .unwrap(); // Short
    scheduler
        .submit_request(vec![4; 22], 10, 0.8, 50, 0.9)
        .unwrap(); // Long

    let batch = scheduler.create_batch().context("Failed to create batch")?;

    // Should group similar lengths together
    let lengths: Vec<usize> = batch.requests.iter().map(|r| r.total_tokens()).collect();

    // Check that lengths are sorted (similar lengths grouped)
    let mut sorted_lengths = lengths.clone();
    sorted_lengths.sort();
    assert_eq!(lengths, sorted_lengths);
}

// Property-based tests
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_batching_properties(
        num_requests in 1..20usize,
        max_batch_size in 1..8usize
    ) {
        let config = SchedulerConfig {
            max_batch_size,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        // Submit requests with varying lengths
        for i in 0..num_requests {
            let prompt_length = (i % 10) + 1;
            let prompt_tokens = vec![i as u32; prompt_length];

            scheduler.submit_request(
                prompt_tokens,
                10,
                0.8,
                50,
                0.9,
            ).unwrap();
        }

        // Create batches until no more pending requests
        let mut total_processed = 0;
        let mut batch_count = 0;

        while scheduler.has_pending_requests() {
            let batch = scheduler.create_batch().context("Failed to create batch")?;
            prop_assert!(batch.size() <= max_batch_size);
            total_processed += batch.size();
            batch_count += 1;

            // Complete all requests in batch
            let mut updated_batch = batch;
            for request in &mut updated_batch.requests {
                for _ in 0..request.max_tokens {
                    let _ = request.add_generated_token(42);
                }
            }

            scheduler.update_batch(updated_batch).context("Failed to update batch")?;
        }

        prop_assert_eq!(total_processed, num_requests);
        prop_assert!(batch_count > 0);
    }

    #[test]
    fn test_request_lifecycle_properties(
        num_tokens in 1..20usize,
        max_tokens in 1..30usize
    ) {
        let mut request = GenerationRequest::new(
            1,
            vec![1; num_tokens],
            max_tokens,
            0.8,
            50,
            0.9,
        );

        // Initial state
        prop_assert_eq!(request.state, RequestState::Pending);
        prop_assert_eq!(request.total_tokens(), num_tokens);
        prop_assert!(!request.is_complete());

        // Start processing
        request.start_processing().context("Failed to start processing")?;
        prop_assert_eq!(request.state, RequestState::Processing);
        prop_assert!(request.started_at.is_some());

        // Add tokens up to max_tokens
        let tokens_to_add = max_tokens.min(10);
        for i in 0..tokens_to_add {
            let result = request.add_generated_token(i as u32);
            if i < max_tokens {
                prop_assert!(result.is_ok());
                prop_assert_eq!(request.generated_tokens.len(), i + 1);
            } else {
                prop_assert!(result.is_err());
            }
        }

        // Check completion
        if tokens_to_add >= max_tokens {
            prop_assert!(request.is_complete());
            prop_assert_eq!(request.state, RequestState::Completed);
            prop_assert!(request.completed_at.is_some());
        }
    }

    #[test]
    fn test_queue_capacity_properties(
        capacity in 1..10usize,
        num_requests in 0..15usize
    ) {
        let config = SchedulerConfig {
            max_queue_size: capacity,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let mut successful_submissions = 0;
        for i in 0..num_requests {
            let result = scheduler.submit_request(
                vec![i as u32],
                10,
                0.8,
                50,
                0.9,
            );

            if result.is_ok() {
                successful_submissions += 1;
                prop_assert_eq!(result.unwrap(), i as u32);
            } else {
                prop_assert!(matches!(result, Err(SchedulerError::QueueCapacityExceeded)));
            }
        }

        prop_assert!(successful_submissions <= capacity);

        let stats = scheduler.get_queue_stats();
        prop_assert_eq!(stats.pending_requests, successful_submissions);
    }
}
