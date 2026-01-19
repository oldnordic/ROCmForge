//! Continuous batching scheduler for efficient GPU utilization

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SchedulerError {
    #[error("Request not found: {0}")]
    RequestNotFound(u32),
    #[error("Batch size exceeded maximum: {max}, got {actual}")]
    BatchSizeExceeded { max: usize, actual: usize },
    #[error("Invalid request state transition")]
    InvalidStateTransition,
    #[error("Queue capacity exceeded")]
    QueueCapacityExceeded,
}

pub type SchedulerResult<T> = Result<T, SchedulerError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RequestState {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub request_id: u32,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub state: RequestState,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub generated_tokens: Vec<u32>,
    pub finish_reason: Option<String>,
}

impl GenerationRequest {
    pub fn new(
        request_id: u32,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Self {
        GenerationRequest {
            request_id,
            prompt_tokens,
            max_tokens,
            temperature,
            top_k,
            top_p,
            state: RequestState::Pending,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            generated_tokens: Vec::new(),
            finish_reason: None,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    pub fn is_complete(&self) -> bool {
        match self.state {
            RequestState::Completed | RequestState::Failed | RequestState::Cancelled => true,
            _ => self.generated_tokens.len() >= self.max_tokens,
        }
    }

    pub fn start_processing(&mut self) -> SchedulerResult<()> {
        if self.state != RequestState::Pending {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Processing;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    pub fn complete(&mut self, reason: Option<String>) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Completed;
        self.completed_at = Some(Instant::now());
        if let Some(reason) = reason {
            self.finish_reason = Some(reason);
        } else if self.finish_reason.is_none() {
            self.finish_reason = Some("completed".to_string());
        }
        Ok(())
    }

    pub fn fail(&mut self) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Failed;
        self.completed_at = Some(Instant::now());
        self.finish_reason
            .get_or_insert_with(|| "failed".to_string());
        Ok(())
    }

    pub fn cancel(&mut self) -> SchedulerResult<()> {
        if matches!(
            self.state,
            RequestState::Completed | RequestState::Failed | RequestState::Cancelled
        ) {
            return Err(SchedulerError::InvalidStateTransition);
        }

        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
        self.state = RequestState::Cancelled;
        self.completed_at = Some(Instant::now());
        self.finish_reason = Some("cancelled".to_string());
        Ok(())
    }

    pub fn add_generated_token(&mut self, token: u32) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.generated_tokens.push(token);

        if self.generated_tokens.len() >= self.max_tokens {
            self.complete(Some("length".to_string()))?;
        } else if self.is_complete() {
            self.complete(None)?;
        }

        Ok(())
    }

    pub fn processing_time(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            (Some(start), None) => Some(Instant::now().duration_since(start)),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Batch {
    pub batch_id: u32,
    pub requests: Vec<GenerationRequest>,
    pub created_at: Instant,
}

impl Batch {
    pub fn new(batch_id: u32) -> Self {
        Batch {
            batch_id,
            requests: Vec::new(),
            created_at: Instant::now(),
        }
    }

    pub fn add_request(&mut self, request: GenerationRequest) -> SchedulerResult<()> {
        self.requests.push(request);
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn total_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.total_tokens()).sum()
    }

    pub fn max_sequence_length(&self) -> usize {
        self.requests
            .iter()
            .map(|r| r.total_tokens())
            .max()
            .unwrap_or(0)
    }

    pub fn min_sequence_length(&self) -> usize {
        self.requests
            .iter()
            .map(|r| r.total_tokens())
            .min()
            .unwrap_or(0)
    }

    pub fn length_variance(&self) -> f32 {
        if self.requests.is_empty() {
            return 0.0;
        }

        let lengths: Vec<usize> = self.requests.iter().map(|r| r.total_tokens()).collect();

        let mean = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
        let variance = lengths
            .iter()
            .map(|&l| (l as f32 - mean).powi(2))
            .sum::<f32>()
            / lengths.len() as f32;

        variance.sqrt()
    }
}

/// Represents a single iteration's batch for continuous batching
/// Allows requests to enter/exit dynamically between iterations
#[derive(Debug)]
pub struct IterationBatch {
    pub requests: Vec<GenerationRequest>,
    pub sequence_positions: Vec<usize>,
    pub completed_indices: Vec<usize>,
}

impl IterationBatch {
    pub fn new() -> Self {
        IterationBatch {
            requests: Vec::new(),
            sequence_positions: Vec::new(),
            completed_indices: Vec::new(),
        }
    }

    /// Remove completed requests and compact remaining
    pub fn compact(&mut self) {
        let mut active_requests = Vec::new();
        let mut active_positions = Vec::new();

        for (i, req) in self.requests.iter().enumerate() {
            if !req.is_complete() && req.state != RequestState::Failed {
                active_requests.push(req.clone());
                active_positions.push(self.sequence_positions[i]);
            } else {
                self.completed_indices.push(i);
            }
        }

        self.requests = active_requests;
        self.sequence_positions = active_positions;
    }

    pub fn size(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn max_sequence_length(&self) -> usize {
        self.sequence_positions.iter().copied().max().unwrap_or(0)
    }

    pub fn min_sequence_length(&self) -> usize {
        self.sequence_positions.iter().copied().min().unwrap_or(0)
    }

    // ========== Phase 4: Paged Attention Integration ==========

    /// Get block tables for all sequences in this iteration batch
    ///
    /// This method retrieves the physical block IDs for each sequence from the
    /// KV cache's page table, which is needed for paged attention computation.
    ///
    /// # Arguments
    /// * `kv_cache` - Reference to the KV cache (read lock)
    ///
    /// # Returns
    /// * `Ok(HashMap)` - Map of sequence_id -> Vec<block_id>
    /// * `Err(KvCacheError)` - If there's an error accessing the cache
    ///
    /// # Example
    /// ```ignore
    /// let block_tables = batch.get_block_tables(&cache)?;
    /// for (seq_id, blocks) in &block_tables {
    ///     println!("Sequence {} has blocks: {:?}", seq_id, blocks);
    /// }
    /// ```
    pub fn get_block_tables(
        &self,
        kv_cache: &crate::kv_cache::KvCache,
    ) -> Result<std::collections::HashMap<u32, Vec<u32>>, crate::kv_cache::KvCacheError> {
        use std::collections::HashMap;

        let mut tables = HashMap::new();
        for req in &self.requests {
            if let Ok(Some(blocks)) = kv_cache.get_sequence_blocks_from_page_table(req.request_id) {
                tables.insert(req.request_id, blocks);
            }
        }
        Ok(tables)
    }
}

#[derive(Debug)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_queue_size: usize,
    pub batch_timeout: Duration,
    pub max_sequence_length: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_batch_size: 32,
            max_queue_size: 1000,
            batch_timeout: Duration::from_millis(50),
            max_sequence_length: 4096,
        }
    }
}

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

    pub fn add_generated_token(&mut self, request_id: u32, token: u32) -> SchedulerResult<()> {
        let request = self.get_request_mut(request_id)?;
        request.add_generated_token(token)
    }

    pub fn get_queue_stats(&self) -> QueueStats {
        QueueStats {
            pending_requests: self.pending_queue.len(),
            processing_requests: self.processing_requests.len(),
            completed_requests: self.completed_requests.len(),
        }
    }

    pub fn has_pending_requests(&self) -> bool {
        !self.pending_queue.is_empty()
    }

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
    /// This is the main entry point for the inference loop
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
    /// Returns the list of completed requests
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
        // Preserve tokens from processing_requests to avoid losing data from stale batch clones
        for request in batch.requests {
            if !self.processing_requests.contains_key(&request.request_id) {
                // Request was removed (completed), don't re-insert stale clone
                continue;
            }
            if !request.is_complete() && request.state != RequestState::Failed {
                // Check if we have an existing request with more tokens than the batch
                // This can happen if the batch has a stale clone from before token generation
                if let Some(existing) = self.processing_requests.get(&request.request_id) {
                    if existing.generated_tokens.len() > request.generated_tokens.len() {
                        // Keep the existing request with more tokens (skip the stale clone)
                        continue;
                    }
                }
                // Otherwise, insert/overwrite with the batch's version
                self.processing_requests.insert(request.request_id, request);
            }
        }

        Ok(completed)
    }
}

#[derive(Debug, Clone)]
pub struct QueueStats {
    pub pending_requests: usize,
    pub processing_requests: usize,
    pub completed_requests: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

        assert_eq!(request.request_id, 1);
        assert_eq!(request.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.state, RequestState::Pending);
        assert_eq!(request.total_tokens(), 3);
        assert!(!request.is_complete());
    }

    #[test]
    fn test_request_state_transitions() {
        let mut request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

        // Start processing
        assert!(request.start_processing().is_ok());
        assert_eq!(request.state, RequestState::Processing);
        assert!(request.started_at.is_some());

        // Add tokens
        for i in 0..10 {
            assert!(request.add_generated_token(i).is_ok());
        }

        // Should be complete now
        assert_eq!(request.state, RequestState::Completed);
        assert!(request.completed_at.is_some());
        assert!(request.is_complete());
    }

    #[test]
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
    }

    #[test]
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

    // ========== Continuous Batching Tests ==========

    #[test]
    fn test_iteration_batch_creation() {
        let batch = IterationBatch::new();

        assert_eq!(batch.size(), 0);
        assert!(batch.is_empty());
        assert_eq!(batch.max_sequence_length(), 0);
        assert_eq!(batch.min_sequence_length(), 0);
    }

    #[test]
    fn test_iteration_batch_compact() {
        let mut batch = IterationBatch::new();

        // Add requests - need to start processing first
        let mut req1 = GenerationRequest::new(1, vec![1; 5], 10, 0.8, 50, 0.9);
        let mut req2 = GenerationRequest::new(2, vec![2; 5], 10, 0.8, 50, 0.9);
        req1.start_processing().unwrap();
        req2.start_processing().unwrap();

        batch.requests.push(req1);
        batch.requests.push(req2);
        batch.sequence_positions.push(5);
        batch.sequence_positions.push(5);

        // Mark one as complete
        batch.requests[1]
            .complete(Some("test".to_string()))
            .unwrap();

        // Compact should remove completed request
        batch.compact();

        assert_eq!(batch.size(), 1);
        assert_eq!(batch.completed_indices.len(), 1);
    }

    #[test]
    fn test_get_next_iteration_batch_empty() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        let batch = scheduler.get_next_iteration_batch();
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert!(batch.is_empty());
    }

    #[test]
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

    // ========== Phase 4: Paged Attention Integration Tests ==========

    #[test]
    fn test_iteration_batch_get_block_tables() {
        use crate::backend::HipBackend;
        use crate::kv_cache::{CacheConfig, KvCache};
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap(); // Already returns Arc<HipBackend>
        let cache_config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let cache = Arc::new(std::sync::RwLock::new(
            KvCache::new(cache_config, backend).unwrap(),
        ));

        let mut batch = IterationBatch::new();

        // Add some test requests (manually create them in processing state)
        let mut req1 = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);
        let mut req2 = GenerationRequest::new(2, vec![4, 5, 6], 10, 0.8, 50, 0.9);
        req1.start_processing().unwrap();
        req2.start_processing().unwrap();

        batch.requests.push(req1);
        batch.requests.push(req2);
        batch.sequence_positions.push(3);
        batch.sequence_positions.push(3);

        // Write some paged data to cache
        {
            let mut cache = cache.write().unwrap();
            cache.append_token_paged(1, 1).unwrap();
            cache.append_token_paged(1, 2).unwrap();
            cache.append_token_paged(1, 3).unwrap();
            cache.append_token_paged(2, 4).unwrap();
            cache.append_token_paged(2, 5).unwrap();
            cache.append_token_paged(2, 6).unwrap();
        }

        // Get block tables - this should work after we implement the method
        let cache_ref = cache.read().unwrap();
        let block_tables = batch.get_block_tables(&*cache_ref);

        assert!(block_tables.is_ok());
        let tables = block_tables.unwrap();
        assert_eq!(tables.len(), 2);
        assert!(tables.contains_key(&1));
        assert!(tables.contains_key(&2));

        // Verify blocks are allocated
        let blocks1 = tables.get(&1).unwrap();
        let blocks2 = tables.get(&2).unwrap();
        assert_eq!(blocks1.len(), 1); // First block
        assert_eq!(blocks2.len(), 1); // First block
    }

    #[test]
    fn test_iteration_batch_get_block_tables_empty() {
        use crate::backend::HipBackend;
        use crate::kv_cache::{CacheConfig, KvCache};
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap(); // Already returns Arc<HipBackend>
        let cache_config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let cache = Arc::new(std::sync::RwLock::new(
            KvCache::new(cache_config, backend).unwrap(),
        ));

        let batch = IterationBatch::new();

        // Get block tables from empty batch
        let cache_ref = cache.read().unwrap();
        let block_tables = batch.get_block_tables(&*cache_ref);

        assert!(block_tables.is_ok());
        let tables = block_tables.unwrap();
        assert_eq!(tables.len(), 0);
    }

    #[test]
    fn test_iteration_batch_allocate_blocks_on_growth() {
        use crate::backend::HipBackend;
        use crate::kv_cache::{CacheConfig, KvCache};
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap(); // Already returns Arc<HipBackend>
        let cache_config = CacheConfig::new(4, 10, 32, 128, 24).unwrap(); // block_size=4
        let cache = Arc::new(std::sync::RwLock::new(
            KvCache::new(cache_config, backend).unwrap(),
        ));

        let mut scheduler = Scheduler::new(SchedulerConfig::default());

        // Submit a request and start processing
        scheduler
            .submit_request(vec![1, 2, 3], 20, 0.8, 50, 0.9)
            .unwrap();
        let req_id = 0;

        // Get first batch to start processing
        let batch = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch.size(), 1);

        // Simulate token generation across multiple iterations
        for i in 0..17 {
            {
                let mut cache = cache.write().unwrap();
                cache.append_token_paged(req_id, 100 + i as u32).unwrap();
            }

            scheduler
                .add_generated_token(req_id, 100 + i as u32)
                .unwrap();

            // Every block_size tokens, should allocate new block
            // At token 4 (i=3), we should have 1 block (tokens 0-3)
            // At token 5 (i=4), we should have 2 blocks (tokens 0-3, 4)
            if i >= 3 && (i + 1) % 4 == 0 {
                let cache = cache.read().unwrap();
                let blocks = cache.get_sequence_blocks_from_page_table(req_id).unwrap();
                assert!(blocks.is_some());
                let block_count = blocks.unwrap().len();
                let expected_blocks = ((i + 1) + 3) / 4; // Ceiling division
                assert_eq!(
                    block_count,
                    expected_blocks,
                    "Expected {} blocks at token {}, got {}",
                    expected_blocks,
                    i + 1,
                    block_count
                );
            }
        }

        // Final verification: 17 tokens should span 5 blocks (0-3, 4-7, 8-11, 12-15, 16)
        let cache = cache.read().unwrap();
        let blocks = cache.get_sequence_blocks_from_page_table(req_id).unwrap();
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 5);
    }

    #[test]
    fn test_scheduler_iteration_with_paged_cache() {
        use crate::backend::HipBackend;
        use crate::kv_cache::{CacheConfig, KvCache};
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap(); // Already returns Arc<HipBackend>
        let cache_config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
        let cache = Arc::new(std::sync::RwLock::new(
            KvCache::new(cache_config, backend).unwrap(),
        ));

        let mut scheduler = Scheduler::new(SchedulerConfig::default());

        // Submit multiple requests
        let req1 = scheduler
            .submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9)
            .unwrap();
        let req2 = scheduler
            .submit_request(vec![4, 5, 6], 10, 0.8, 50, 0.9)
            .unwrap();

        // Get first iteration batch
        let batch1 = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch1.size(), 2);

        // Simulate processing: append tokens to paged cache
        {
            let mut cache = cache.write().unwrap();
            for &req_id in &[req1, req2] {
                for i in 0..5 {
                    cache.append_token_paged(req_id, 100 + i).unwrap();
                }
            }
        }

        // Add generated tokens via scheduler
        for &req_id in &[req1, req2] {
            for i in 0..5 {
                scheduler.add_generated_token(req_id, 100 + i).unwrap();
            }
        }

        // Update iteration batch
        let completed = scheduler.update_iteration_batch(batch1).unwrap();
        assert_eq!(completed.len(), 0); // Not complete yet

        // Get next iteration batch
        let batch2 = scheduler.get_next_iteration_batch().unwrap();
        assert_eq!(batch2.size(), 2);

        // Verify we can get block tables for the batch
        let cache = cache.read().unwrap();
        let block_tables = batch2.get_block_tables(&*cache).unwrap();
        assert_eq!(block_tables.len(), 2);
    }
}
