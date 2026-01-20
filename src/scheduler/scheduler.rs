//! Continuous batching scheduler for efficient GPU utilization
//!
//! This module implements the core Scheduler struct that manages request lifecycles,
//! queue management, and continuous batching for optimal GPU utilization.

use std::collections::{HashMap, VecDeque};
use std::collections::hash_map::Entry;

// Import types from sibling modules
use crate::scheduler::batch::{Batch, IterationBatch, SchedulerConfig};
use crate::scheduler::queue::QueueStats;
use crate::scheduler::types::{GenerationRequest, RequestState, SchedulerError, SchedulerResult};

/// Continuous batching scheduler for efficient GPU utilization
///
/// The scheduler manages three queues:
/// - `pending_queue`: Requests waiting to be processed
/// - `processing_requests`: Currently active requests
/// - `completed_requests`: Finished requests (can be retrieved)
///
/// Key features:
/// - Continuous batching: Requests can enter/exit batches dynamically
/// - Length-based grouping: Similar-length requests are batched together
/// - State tracking: Full lifecycle management for each request
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    pending_queue: VecDeque<GenerationRequest>,
    processing_requests: HashMap<u32, GenerationRequest>,
    completed_requests: HashMap<u32, GenerationRequest>,
    next_batch_id: u32,
    next_request_id: u32,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration
    pub fn new(config: SchedulerConfig) -> Self {
        Scheduler {
            config,
            pending_queue: VecDeque::new(),
            processing_requests: HashMap::new(),
            completed_requests: HashMap::new(),
            next_batch_id: 0,
            next_request_id: 0,
        }
    }

    /// Submit a new generation request to the pending queue
    pub fn submit_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> SchedulerResult<u32> {
        if self.pending_queue.len() >= self.config.max_queue_size {
            return Err(SchedulerError::QueueCapacityExceeded);
        }

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request = GenerationRequest::new(
            request_id,
            prompt_tokens,
            max_tokens,
            temperature,
            top_k,
            top_p,
        );

        self.pending_queue.push_back(request);
        Ok(request_id)
    }

    /// Create a static batch from the pending queue
    ///
    /// This method implements length-based grouping: requests with similar
    /// sequence lengths are batched together for efficient processing.
    pub fn create_batch(&mut self) -> SchedulerResult<Batch> {
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        let mut batch = Batch::new(batch_id);

        // Simple FIFO batching with length-based grouping
        let mut request_ids: Vec<u32> = self.pending_queue.iter().map(|r| r.request_id).collect();

        // Sort by sequence length for better batching
        request_ids.sort_by_key(|&id| {
            self.pending_queue
                .iter()
                .find(|r| r.request_id == id)
                .map(|r| r.total_tokens())
                .unwrap_or(0)
        });

        // Group similar lengths together
        for &request_id in &request_ids {
            if batch.size() >= self.config.max_batch_size {
                break;
            }

            // Find and remove from queue
            let pos = self
                .pending_queue
                .iter()
                .position(|r| r.request_id == request_id);

            if let Some(pos) = pos {
                // SAFETY: pos is guaranteed to be valid because we just found it via position()
                let mut request = self
                    .pending_queue
                    .remove(pos)
                    .expect("Failed to remove request at valid position");

                // Check if this request fits well with current batch
                if batch.is_empty()
                    || (request.total_tokens() as f32 - batch.max_sequence_length() as f32).abs()
                        < batch.max_sequence_length() as f32 * 0.3
                {
                    request.start_processing()?;
                    self.processing_requests
                        .insert(request.request_id, request.clone());
                    batch.add_request(request)?;
                } else {
                    // Put it back
                    self.pending_queue.push_front(request);
                }
            }
        }

        Ok(batch)
    }

    /// Update a static batch after processing, returning completed requests
    pub fn update_batch(&mut self, batch: Batch) -> SchedulerResult<Vec<GenerationRequest>> {
        let mut completed_requests = Vec::new();

        for request in batch.requests {
            if request.is_complete() || request.state == RequestState::Failed {
                let mut req = request;
                if req.state == RequestState::Processing {
                    req.complete(None)?;
                }

                self.processing_requests.remove(&req.request_id);
                self.completed_requests.insert(req.request_id, req.clone());
                completed_requests.push(req);
            } else {
                // Still processing, update in processing_requests
                self.processing_requests.insert(request.request_id, request);
            }
        }

        Ok(completed_requests)
    }

    /// Get a reference to a request by ID
    pub fn get_request(&self, request_id: u32) -> SchedulerResult<&GenerationRequest> {
        self.processing_requests
            .get(&request_id)
            .or_else(|| self.completed_requests.get(&request_id))
            .or_else(|| {
                self.pending_queue
                    .iter()
                    .find(|r| r.request_id == request_id)
            })
            .ok_or(SchedulerError::RequestNotFound(request_id))
    }

    /// Get a mutable reference to a request by ID
    pub fn get_request_mut(&mut self, request_id: u32) -> SchedulerResult<&mut GenerationRequest> {
        if let Some(request) = self.processing_requests.get_mut(&request_id) {
            Ok(request)
        } else if let Some(request) = self.completed_requests.get_mut(&request_id) {
            Ok(request)
        } else {
            self.pending_queue
                .iter_mut()
                .find(|r| r.request_id == request_id)
                .ok_or(SchedulerError::RequestNotFound(request_id))
        }
    }

    /// Cancel a request by ID
    pub fn cancel_request(&mut self, request_id: u32) -> SchedulerResult<GenerationRequest> {
        if let Some(mut request) = self.processing_requests.remove(&request_id) {
            request.cancel()?;
            self.completed_requests.insert(request_id, request.clone());
            return Ok(request);
        }

        if let Some(pos) = self
            .pending_queue
            .iter()
            .position(|r| r.request_id == request_id)
        {
            // SAFETY: pos is guaranteed to be valid because we just found it via position()
            let mut request = self
                .pending_queue
                .remove(pos)
                .expect("Failed to remove request at valid position");
            request.cancel()?;
            self.completed_requests.insert(request_id, request.clone());
            return Ok(request);
        }

        if let Some(request) = self.completed_requests.get(&request_id) {
            if request.state == RequestState::Cancelled {
                return Ok(request.clone());
            }
            return Err(SchedulerError::InvalidStateTransition);
        }

        Err(SchedulerError::RequestNotFound(request_id))
    }

    /// Add a generated token to a request
    pub fn add_generated_token(&mut self, request_id: u32, token: u32) -> SchedulerResult<()> {
        let request = self.get_request_mut(request_id)?;
        request.add_generated_token(token)
    }

    /// Get current queue statistics
    pub fn get_queue_stats(&self) -> QueueStats {
        QueueStats {
            pending_requests: self.pending_queue.len(),
            processing_requests: self.processing_requests.len(),
            completed_requests: self.completed_requests.len(),
        }
    }

    /// Check if there are pending requests waiting to be processed
    pub fn has_pending_requests(&self) -> bool {
        !self.pending_queue.is_empty()
    }

    /// Check if a new batch can be created
    pub fn can_create_batch(&self) -> bool {
        !self.pending_queue.is_empty()
            && self.processing_requests.len() < self.config.max_batch_size
    }

    // ========== Continuous Batching Methods ==========

    /// Update processing state after an iteration
    /// Moves completed requests from processing to completed
    fn update_processing_state(&mut self) {
        let mut to_complete = Vec::new();

        for (&req_id, request) in &self.processing_requests {
            if request.is_complete() || request.state == RequestState::Failed {
                to_complete.push(req_id);
            }
        }

        for req_id in to_complete {
            if let Some(mut request) = self.processing_requests.remove(&req_id) {
                if request.state == RequestState::Processing {
                    let _ = request.complete(None);
                }
                self.completed_requests.insert(req_id, request);
            }
        }
    }

    /// Get the next iteration's batch for continuous batching
    ///
    /// This is the main entry point for the continuous batching inference loop.
    /// Returns all currently processing requests plus new requests from the queue
    /// to fill empty slots.
    pub fn get_next_iteration_batch(&mut self) -> SchedulerResult<IterationBatch> {
        // Move completed requests out of processing
        self.update_processing_state();

        let mut iteration_batch = IterationBatch::new();

        // Add all currently processing requests (continuous batching)
        for (_req_id, request) in &self.processing_requests {
            if request.state == RequestState::Processing {
                iteration_batch.requests.push(request.clone());
                iteration_batch
                    .sequence_positions
                    .push(request.total_tokens());
            }
        }

        // Fill empty slots with new requests from pending queue
        while iteration_batch.size() < self.config.max_batch_size && !self.pending_queue.is_empty()
        {
            if let Some(mut request) = self.pending_queue.pop_front() {
                request.start_processing()?;
                let req_id = request.request_id;
                let total_tokens = request.total_tokens();
                self.processing_requests.insert(req_id, request.clone());
                iteration_batch.requests.push(request);
                iteration_batch.sequence_positions.push(total_tokens);
            }
        }

        Ok(iteration_batch)
    }

    /// Update an iteration batch after processing
    ///
    /// Returns the list of completed requests. This method implements
    /// token preservation: only updates processing_requests if the batch
    /// has fresher data (more tokens) than what's already stored.
    pub fn update_iteration_batch(
        &mut self,
        mut batch: IterationBatch,
    ) -> SchedulerResult<Vec<GenerationRequest>> {
        // Compact the batch to identify completed requests
        batch.compact();

        let mut completed = Vec::new();

        // Get the actual completed requests from processing_requests
        let mut to_complete = Vec::new();
        for (_req_id, request) in &self.processing_requests {
            if request.is_complete() || request.state == RequestState::Failed {
                to_complete.push(request.request_id);
            }
        }

        for req_id in to_complete {
            if let Some(request) = self.processing_requests.remove(&req_id) {
                self.completed_requests.insert(req_id, request.clone());
                completed.push(request);
            }
        }

        // Update remaining processing requests
        // Use Entry API to preserve tokens - only update if batch has fresher data
        for request in batch.requests {
            if !request.is_complete() && request.state != RequestState::Failed {
                match self.processing_requests.entry(request.request_id) {
                    Entry::Vacant(_e) => {
                        // Request not in processing (was completed), skip stale clone
                        continue;
                    }
                    Entry::Occupied(mut e) => {
                        // Only update if batch has more tokens (prevents stale overwrite)
                        if e.get().generated_tokens.len() <= request.generated_tokens.len() {
                            e.insert(request);
                        }
                        // If batch has fewer tokens, keep existing (it's fresher)
                    }
                }
            }
        }

        Ok(completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_scheduler_request_submission() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        let request_id = scheduler.submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9);

        assert!(request_id.is_ok());
        assert_eq!(request_id.unwrap(), 0);

        let stats = scheduler.get_queue_stats();
        assert_eq!(stats.pending_requests, 1);
        assert_eq!(stats.processing_requests, 0);
    }

    #[test]
    #[serial]
    fn test_batch_creation() {
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

        let stats = scheduler.get_queue_stats();
        assert_eq!(stats.pending_requests, 0);
        assert_eq!(stats.processing_requests, 3);
    }

    #[test]
    #[serial]
    fn test_queue_capacity_limit() {
        let config = SchedulerConfig {
            max_queue_size: 2,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        // Submit up to capacity
        scheduler.submit_request(vec![1], 10, 0.8, 50, 0.9).unwrap();
        scheduler.submit_request(vec![2], 10, 0.8, 50, 0.9).unwrap();

        // Should fail when exceeding capacity
        let result = scheduler.submit_request(vec![3], 10, 0.8, 50, 0.9);
        assert!(result.is_err());
        assert!(matches!(result, Err(SchedulerError::QueueCapacityExceeded)));
    }

    #[test]
    #[serial]
    fn test_batch_update() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        scheduler
            .submit_request(vec![1, 2, 3], 2, 0.8, 50, 0.9)
            .unwrap();
        let batch = scheduler.create_batch().unwrap();

        // Simulate token generation
        let mut updated_batch = batch;
        for request in &mut updated_batch.requests {
            request.add_generated_token(42).unwrap();
            request.add_generated_token(43).unwrap(); // Should complete
        }

        let completed = scheduler.update_batch(updated_batch).unwrap();
        assert_eq!(completed.len(), 1);
        assert!(completed[0].is_complete());

        let stats = scheduler.get_queue_stats();
        assert_eq!(stats.processing_requests, 0);
        assert_eq!(stats.completed_requests, 1);
    }

    #[test]
    #[serial]
    fn test_get_next_iteration_batch_empty() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        let batch = scheduler.get_next_iteration_batch();
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    #[serial]
    fn test_get_next_iteration_batch_with_pending() {
        let config = SchedulerConfig {
            max_batch_size: 4,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        // Submit requests
        for i in 0..3 {
            scheduler
                .submit_request(vec![i; 5], 10, 0.8, 50, 0.9)
                .unwrap();
        }

        let batch = scheduler.get_next_iteration_batch();
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert_eq!(batch.size(), 3);
        assert_eq!(batch.sequence_positions.len(), 3);
    }

    #[test]
    #[serial]
    fn test_continuous_batching_mixed() {
        let config = SchedulerConfig {
            max_batch_size: 4,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        // Submit first batch of requests
        for i in 0..2 {
            scheduler
                .submit_request(vec![i; 5], 10, 0.8, 50, 0.9)
                .unwrap();
        }

        // Get first iteration batch
        let batch1 = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch1.size(), 2);

        // Simulate completion of one request
        let req_id = batch1.requests[0].request_id;
        {
            let req = scheduler.get_request_mut(req_id).unwrap();
            for _ in 0..10 {
                let _ = req.add_generated_token(42);
            }
        }

        // Submit new requests while first is still processing
        for i in 10..12 {
            scheduler
                .submit_request(vec![i; 5], 10, 0.8, 50, 0.9)
                .unwrap();
        }

        // Get next iteration - should have 1 remaining + 2 new = 3
        let batch2 = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch2.size(), 3);
    }

    #[test]
    #[serial]
    fn test_update_iteration_batch() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        // Submit and start a request
        scheduler
            .submit_request(vec![1; 5], 2, 0.8, 50, 0.9)
            .unwrap();
        let batch = scheduler.get_next_iteration_batch().unwrap();

        // Get the request ID and complete it through the scheduler
        let req_id = batch.requests[0].request_id;
        scheduler.add_generated_token(req_id, 42).unwrap();
        scheduler.add_generated_token(req_id, 43).unwrap();

        // Update the iteration batch
        let completed = scheduler.update_iteration_batch(batch).unwrap();

        assert_eq!(completed.len(), 1);
        assert!(completed[0].is_complete());

        let stats = scheduler.get_queue_stats();
        assert_eq!(stats.processing_requests, 0);
        assert_eq!(stats.completed_requests, 1);
    }

    #[test]
    #[serial]
    fn test_tokens_preserved_after_update() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        // Submit a request that needs multiple iterations
        scheduler
            .submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9)
            .unwrap();

        // First iteration
        let batch1 = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch1.size(), 1);
        assert_eq!(batch1.requests[0].generated_tokens.len(), 0);

        let req_id = batch1.requests[0].request_id;

        // Simulate token generation during processing
        // In real engine, this happens via process_single_request_impl
        scheduler.add_generated_token(req_id, 100).unwrap();
        scheduler.add_generated_token(req_id, 101).unwrap();

        // Verify tokens were added to scheduler
        let req = scheduler.get_request(req_id).unwrap();
        assert_eq!(req.generated_tokens.len(), 2);
        assert_eq!(req.generated_tokens, vec![100, 101]);

        // Update iteration batch (simulating engine's process_batch)
        // The engine calls snapshot_request which reads updated state from scheduler
        let mut updated_batch = batch1;
        updated_batch.requests = vec![scheduler.get_request(req_id).unwrap().clone()];

        let completed = scheduler.update_iteration_batch(updated_batch).unwrap();
        assert_eq!(completed.len(), 0); // Not complete yet

        // Second iteration - verify tokens are preserved
        let batch2 = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch2.size(), 1);
        assert_eq!(batch2.requests[0].generated_tokens.len(), 2);
        assert_eq!(batch2.requests[0].generated_tokens, vec![100, 101]);

        // Add more tokens
        scheduler.add_generated_token(req_id, 102).unwrap();
        scheduler.add_generated_token(req_id, 103).unwrap();

        // Verify all 4 tokens are present
        let req = scheduler.get_request(req_id).unwrap();
        assert_eq!(req.generated_tokens.len(), 4);
        assert_eq!(req.generated_tokens, vec![100, 101, 102, 103]);
    }

    #[test]
    #[serial]
    fn test_stale_batch_clone_does_not_overwrite_scheduler() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        // Submit a request
        scheduler
            .submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9)
            .unwrap();

        // Get iteration batch (creates clones)
        let batch = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch.requests[0].generated_tokens.len(), 0);

        let req_id = batch.requests[0].request_id;

        // Simulate the engine: add tokens directly to scheduler
        // (This is what process_single_request_impl does)
        scheduler.add_generated_token(req_id, 100).unwrap();
        scheduler.add_generated_token(req_id, 101).unwrap();

        // Verify scheduler has the tokens
        let req = scheduler.get_request(req_id).unwrap();
        assert_eq!(req.generated_tokens.len(), 2);
        assert_eq!(req.generated_tokens, vec![100, 101]);

        // Simulate the bug: If engine passes the OLD batch (with stale clones)
        // to update_iteration_batch, it should NOT overwrite the scheduler's tokens
        //
        // The CORRECT flow is: engine should call snapshot_request to get updated state
        // But if there's a bug and engine passes the stale batch, tokens would be lost
        //
        // This test verifies the current behavior: update_iteration_batch overwrites
        // with whatever is in batch.requests
        let completed = scheduler.update_iteration_batch(batch).unwrap();
        assert_eq!(completed.len(), 0);

        // Check if tokens were preserved or lost
        let req = scheduler.get_request(req_id).unwrap();
        // BUG: This fails because update_iteration_batch overwrites with stale clone!
        assert_eq!(req.generated_tokens.len(), 2);
        assert_eq!(req.generated_tokens, vec![100, 101]);
    }

    /// Test alias matching HYGIENE-01 requirement naming
    ///
    /// This test is an alias for `test_stale_batch_clone_does_not_overwrite_scheduler`
    /// to satisfy the HYGIENE-01 requirement which specifies this exact test name.
    #[test]
    #[serial]
    fn test_update_iteration_batch_cannot_clobber_new_tokens() {
        // Delegate to the main test with the descriptive name
        test_stale_batch_clone_does_not_overwrite_scheduler();
    }

    // Property tests
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

            // Submit requests
            for i in 0..num_requests {
                scheduler.submit_request(
                    vec![i as u32; (i % 10) + 1],
                    10,
                    0.8,
                    50,
                    0.9,
                ).unwrap();
            }

            // Create batches until no more pending requests
            let mut total_processed = 0;
            while scheduler.has_pending_requests() {
                let batch = scheduler.create_batch().unwrap();
                prop_assert!(batch.size() <= max_batch_size);
                total_processed += batch.size();

                // Complete all requests in batch
                let mut updated_batch = batch;
                for request in &mut updated_batch.requests {
                    for _ in 0..request.max_tokens {
                        let _ = request.add_generated_token(42);
                    }
                }

                scheduler.update_batch(updated_batch).unwrap();
            }

            prop_assert_eq!(total_processed, num_requests);
        }
    }
}
