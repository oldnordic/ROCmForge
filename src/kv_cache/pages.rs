//! Page and sequence cache types for KV cache
//!
//! This module contains CachePage and SequenceCache types used by the
//! paged KV cache implementation.

use crate::backend::{HipBackend, HipBuffer};
use super::{config::CacheConfig, types::KvCacheResult, KvCacheError};

/// A single cache page storing KV data for a sequence
#[derive(Debug)]
pub struct CachePage {
    pub page_id: u32,
    pub sequence_id: u32,
    pub tokens: Vec<u32>,
    pub key_buffer: HipBuffer,
    pub value_buffer: HipBuffer,
    pub is_free: bool,
}

impl CachePage {
    pub fn new(
        page_id: u32,
        sequence_id: u32,
        backend: &HipBackend,
        config: &CacheConfig,
    ) -> KvCacheResult<Self> {
        let key_size =
            config.page_size * config.num_heads * config.head_dim * std::mem::size_of::<f32>();
        let value_size =
            config.page_size * config.num_heads * config.head_dim * std::mem::size_of::<f32>();

        let key_buffer = backend.allocate_buffer(key_size)?;
        let value_buffer = backend.allocate_buffer(value_size)?;

        Ok(CachePage {
            page_id,
            sequence_id,
            tokens: Vec::with_capacity(config.page_size),
            key_buffer,
            value_buffer,
            is_free: false,
        })
    }

    pub fn can_append(&self, _token: u32) -> bool {
        self.tokens.len() < self.tokens.capacity() && !self.is_free
    }

    pub fn append_token(&mut self, token: u32) -> KvCacheResult<()> {
        if !self.can_append(token) {
            return Err(KvCacheError::CapacityExceeded);
        }

        self.tokens.push(token);
        Ok(())
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.is_free = true;
    }
}

/// Cache data for a single sequence
#[derive(Debug)]
pub struct SequenceCache {
    pub sequence_id: u32,
    pub pages: Vec<u32>,
    pub total_tokens: usize,
    /// Tracks whether this sequence is completed
    pub is_completed: bool,
    /// Last access time for LRU eviction
    pub last_access: std::time::Instant,
}

impl SequenceCache {
    pub fn new(sequence_id: u32) -> Self {
        SequenceCache {
            sequence_id,
            pages: Vec::new(),
            total_tokens: 0,
            is_completed: false,
            last_access: std::time::Instant::now(),
        }
    }

    pub fn add_page(&mut self, page_id: u32) {
        self.pages.push(page_id);
        self.update_access();
    }

    pub fn get_last_page(&self) -> Option<u32> {
        self.pages.last().copied()
    }

    pub fn update_access(&mut self) {
        self.last_access = std::time::Instant::now();
    }

    pub fn mark_completed(&mut self) {
        self.is_completed = true;
    }

    pub fn is_active(&self) -> bool {
        !self.is_completed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_page_creation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();

        let page = CachePage::new(0, 1, &backend, &config);
        assert!(page.is_ok());

        let page = page.unwrap();
        assert_eq!(page.page_id, 0);
        assert_eq!(page.sequence_id, 1);
        assert_eq!(page.tokens.len(), 0);
        assert_eq!(page.tokens.capacity(), 4);
        assert!(!page.is_free);
    }

    #[test]
    fn test_cache_page_append() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut page = CachePage::new(0, 1, &backend, &config).unwrap();

        assert!(page.can_append(0));
        page.append_token(42).unwrap();
        assert_eq!(page.token_count(), 1);
        assert_eq!(page.tokens[0], 42);
    }

    #[test]
    fn test_cache_page_capacity() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(2, 10, 32, 128, 24).unwrap();
        let mut page = CachePage::new(0, 1, &backend, &config).unwrap();

        page.append_token(0).unwrap();
        page.append_token(1).unwrap();

        assert!(!page.can_append(2)); // At capacity
        assert!(page.append_token(2).is_err());
    }

    #[test]
    fn test_cache_page_clear() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut page = CachePage::new(0, 1, &backend, &config).unwrap();

        page.append_token(42).unwrap();
        page.clear();

        assert!(page.tokens.is_empty());
        assert!(page.is_free);
    }

    #[test]
    fn test_sequence_cache_creation() {
        let seq = SequenceCache::new(1);
        assert_eq!(seq.sequence_id, 1);
        assert!(seq.pages.is_empty());
        assert_eq!(seq.total_tokens, 0);
        assert!(!seq.is_completed);
        assert!(seq.is_active());
    }

    #[test]
    fn test_sequence_cache_add_page() {
        let mut seq = SequenceCache::new(1);
        seq.add_page(0);
        seq.add_page(1);

        assert_eq!(seq.pages.len(), 2);
        assert_eq!(seq.get_last_page(), Some(1));
    }

    #[test]
    fn test_sequence_cache_completion() {
        let mut seq = SequenceCache::new(1);
        assert!(seq.is_active());

        seq.mark_completed();
        assert!(seq.is_completed);
        assert!(!seq.is_active());
    }

    #[test]
    fn test_sequence_cache_update_access() {
        let mut seq = SequenceCache::new(1);
        let initial_access = seq.last_access;

        // Update access time
        seq.update_access();
        // New access time should be >= initial (could be same if very fast)
        assert!(seq.last_access >= initial_access);
    }
}
