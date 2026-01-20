//! Batch formation and management for continuous batching

use std::collections::HashMap;
use std::time::Duration;

use crate::kv_cache::{KvCache, KvCacheError};
use crate::scheduler::types::{GenerationRequest, RequestState, SchedulerResult};

/// A batch of generation requests
///
/// Static batch for initial request grouping. Batches are created from the pending
/// queue and contain requests that will be processed together.
#[derive(Debug)]
pub struct Batch {
    pub batch_id: u32,
    pub requests: Vec<GenerationRequest>,
    pub created_at: std::time::Instant,
}

impl Batch {
    /// Create a new empty batch
    pub fn new(batch_id: u32) -> Self {
        Batch {
            batch_id,
            requests: Vec::new(),
            created_at: std::time::Instant::now(),
        }
    }

    /// Add a request to this batch
    pub fn add_request(&mut self, request: GenerationRequest) -> SchedulerResult<()> {
        self.requests.push(request);
        Ok(())
    }

    /// Number of requests in the batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Total tokens across all requests in the batch
    pub fn total_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.total_tokens()).sum()
    }

    /// Maximum sequence length in the batch
    pub fn max_sequence_length(&self) -> usize {
        self.requests
            .iter()
            .map(|r| r.total_tokens())
            .max()
            .unwrap_or(0)
    }

    /// Minimum sequence length in the batch
    pub fn min_sequence_length(&self) -> usize {
        self.requests
            .iter()
            .map(|r| r.total_tokens())
            .min()
            .unwrap_or(0)
    }

    /// Standard deviation of sequence lengths (measure of batch coherence)
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
///
/// Unlike static `Batch`, this allows requests to enter/exit dynamically between
/// iterations. Used for the continuous batching inference loop.
#[derive(Debug)]
pub struct IterationBatch {
    pub requests: Vec<GenerationRequest>,
    pub sequence_positions: Vec<usize>,
    pub completed_indices: Vec<usize>,
}

impl IterationBatch {
    /// Create a new empty iteration batch
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

    /// Number of active requests in the batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Maximum sequence position in the batch
    pub fn max_sequence_length(&self) -> usize {
        self.sequence_positions.iter().copied().max().unwrap_or(0)
    }

    /// Minimum sequence position in the batch
    pub fn min_sequence_length(&self) -> usize {
        self.sequence_positions.iter().copied().min().unwrap_or(0)
    }

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
        kv_cache: &KvCache,
    ) -> Result<HashMap<u32, Vec<u32>>, KvCacheError> {
        let mut tables = HashMap::new();
        for req in &self.requests {
            if let Ok(Some(blocks)) = kv_cache.get_sequence_blocks_from_page_table(req.request_id) {
                tables.insert(req.request_id, blocks);
            }
        }
        Ok(tables)
    }
}

/// Configuration for the continuous batching scheduler
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::RequestState;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_batch_creation() {
        let batch = Batch::new(1);
        assert_eq!(batch.batch_id, 1);
        assert!(batch.is_empty());
    }

    #[test]
    #[serial]
    fn test_batch_length_variance() {
        let mut batch = Batch::new(1);

        // Add requests with similar lengths
        use crate::scheduler::types::GenerationRequest;
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
    #[serial]
    fn test_iteration_batch_creation() {
        let batch = IterationBatch::new();

        assert_eq!(batch.size(), 0);
        assert!(batch.is_empty());
        assert_eq!(batch.max_sequence_length(), 0);
        assert_eq!(batch.min_sequence_length(), 0);
    }

    #[test]
    #[serial]
    fn test_iteration_batch_compact() {
        let mut batch = IterationBatch::new();

        // Add requests - need to start processing first
        use crate::scheduler::types::GenerationRequest;
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
    #[serial]
    fn test_iteration_batch_get_block_tables() {
        use crate::backend::HipBackend;
        use crate::kv_cache::CacheConfig;
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap();
        let cache_config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let cache = Arc::new(std::sync::RwLock::new(
            KvCache::new(cache_config, backend).unwrap(),
        ));

        let mut batch = IterationBatch::new();

        // Add some test requests (manually create them in processing state)
        use crate::scheduler::types::GenerationRequest;
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
    #[serial]
    fn test_iteration_batch_get_block_tables_empty() {
        use crate::backend::HipBackend;
        use crate::kv_cache::CacheConfig;
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap();
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
    #[serial]
    fn test_iteration_batch_allocate_blocks_on_growth() {
        use crate::backend::HipBackend;
        use crate::kv_cache::CacheConfig;
        use crate::scheduler::Scheduler;
        use crate::scheduler::types::GenerationRequest;
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap();
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
    #[serial]
    fn test_scheduler_iteration_with_paged_cache() {
        use crate::backend::HipBackend;
        use crate::kv_cache::CacheConfig;
        use crate::scheduler::Scheduler;
        use std::sync::Arc;

        let backend = HipBackend::new().unwrap();
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
