//! Paged KV cache for efficient GPU memory management

use crate::backend::{HipBackend, HipBuffer, HipError};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KvCacheError {
    #[error("Cache capacity exceeded")]
    CapacityExceeded,
    #[error("Invalid sequence ID: {0}")]
    InvalidSequenceId(u32),
    #[error("Page not found for sequence: {0}")]
    PageNotFound(u32),
    #[error("GPU memory error: {0}")]
    GpuError(#[from] HipError),
    #[error("Invalid cache configuration")]
    InvalidConfiguration,
}

pub type KvCacheResult<T> = Result<T, KvCacheError>;

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub page_size: usize,
    pub max_pages: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}

impl CacheConfig {
    pub fn new(
        page_size: usize,
        max_pages: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> KvCacheResult<Self> {
        if page_size == 0 || max_pages == 0 || num_heads == 0 || head_dim == 0 || num_layers == 0 {
            return Err(KvCacheError::InvalidConfiguration);
        }

        Ok(CacheConfig {
            page_size,
            max_pages,
            num_heads,
            head_dim,
            num_layers,
        })
    }
}

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
            tokens: Vec::new(),
            key_buffer,
            value_buffer,
            is_free: false,
        })
    }

    pub fn can_append(&self, token: u32) -> bool {
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

#[derive(Debug)]
pub struct SequenceCache {
    pub sequence_id: u32,
    pub pages: Vec<u32>,
    pub total_tokens: usize,
}

impl SequenceCache {
    pub fn new(sequence_id: u32) -> Self {
        SequenceCache {
            sequence_id,
            pages: Vec::new(),
            total_tokens: 0,
        }
    }

    pub fn add_page(&mut self, page_id: u32) {
        self.pages.push(page_id);
    }

    pub fn get_last_page(&self) -> Option<u32> {
        self.pages.last().copied()
    }
}

#[derive(Debug)]
pub struct KvCache {
    config: CacheConfig,
    backend: Arc<HipBackend>,  // Changed to Arc<HipBackend> for shared ownership
    pages: HashMap<u32, CachePage>,
    sequences: HashMap<u32, SequenceCache>,
    free_pages: Vec<u32>,
    next_page_id: u32,
}

impl KvCache {
    pub fn new(config: CacheConfig, backend: Arc<HipBackend>) -> KvCacheResult<Self> {
        Ok(KvCache {
            config,
            backend,
            pages: HashMap::new(),
            sequences: HashMap::new(),
            free_pages: Vec::new(),
            next_page_id: 0,
        })
    }

    pub fn allocate_page(&mut self, sequence_id: u32) -> KvCacheResult<u32> {
        let page_id = if let Some(free_id) = self.free_pages.pop() {
            free_id
        } else if self.pages.len() >= self.config.max_pages {
            return Err(KvCacheError::CapacityExceeded);
        } else {
            let id = self.next_page_id;
            self.next_page_id += 1;
            id
        };

        let page = CachePage::new(page_id, sequence_id, &self.backend, &self.config)?;
        self.pages.insert(page_id, page);

        // Update sequence cache
        let sequence = self
            .sequences
            .entry(sequence_id)
            .or_insert_with(|| SequenceCache::new(sequence_id));
        sequence.add_page(page_id);

        Ok(page_id)
    }

    pub fn append_token(&mut self, sequence_id: u32, token: u32) -> KvCacheResult<()> {
        let last_page_id = {
            let sequence = self
                .sequences
                .get(&sequence_id)
                .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
            sequence
                .get_last_page()
                .ok_or_else(|| KvCacheError::PageNotFound(sequence_id))?
        };

        let can_append = {
            let page = self
                .pages
                .get(&last_page_id)
                .ok_or_else(|| KvCacheError::PageNotFound(last_page_id))?;
            page.can_append(token)
        };

        if can_append {
            let page = self
                .pages
                .get_mut(&last_page_id)
                .ok_or_else(|| KvCacheError::PageNotFound(last_page_id))?;
            page.append_token(token)?;
            let sequence = self
                .sequences
                .get_mut(&sequence_id)
                .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
            sequence.total_tokens += 1;
        } else {
            // Allocate new page
            let new_page_id = self.allocate_page(sequence_id)?;
            let new_page = self.pages.get_mut(&new_page_id).unwrap();
            new_page.append_token(token)?;
            let sequence = self
                .sequences
                .get_mut(&sequence_id)
                .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
            sequence.total_tokens += 1;
        }

        Ok(())
    }

    pub fn get_sequence_tokens(&self, sequence_id: u32) -> KvCacheResult<Vec<u32>> {
        let sequence = self
            .sequences
            .get(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        let mut tokens = Vec::with_capacity(sequence.total_tokens);

        for page_id in &sequence.pages {
            let page = self
                .pages
                .get(page_id)
                .ok_or(KvCacheError::PageNotFound(*page_id))?;
            tokens.extend_from_slice(&page.tokens);
        }

        Ok(tokens)
    }

    pub fn get_sequence_length(&self, sequence_id: u32) -> KvCacheResult<usize> {
        let sequence = self
            .sequences
            .get(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        Ok(sequence.total_tokens)
    }

    pub fn remove_sequence(&mut self, sequence_id: u32) -> KvCacheResult<()> {
        let sequence = self
            .sequences
            .remove(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        // Mark pages as free
        for page_id in sequence.pages {
            if let Some(page) = self.pages.get_mut(&page_id) {
                page.clear();
                self.free_pages.push(page_id);
            }
        }

        Ok(())
    }

    pub fn get_cache_stats(&self) -> CacheStats {
        CacheStats {
            total_pages: self.pages.len(),
            free_pages: self.free_pages.len(),
            active_sequences: self.sequences.len(),
            total_tokens: self.sequences.values().map(|s| s.total_tokens).sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_pages: usize,
    pub free_pages: usize,
    pub active_sequences: usize,
    pub total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HipBackend;

    #[test]
    fn test_cache_config_creation() {
        let config = CacheConfig::new(1024, 100, 32, 128, 24);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.page_size, 1024);
        assert_eq!(config.max_pages, 100);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_layers, 24);
    }

    #[test]
    fn test_invalid_cache_config() {
        let config = CacheConfig::new(0, 100, 32, 128, 24);
        assert!(config.is_err());
        assert!(matches!(config, Err(KvCacheError::InvalidConfiguration)));
    }

    #[test]
    fn test_kv_cache_creation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(1024, 100, 32, 128, 24).unwrap();
        let cache = KvCache::new(config, backend);
        assert!(cache.is_ok());
    }

    #[test]
    fn test_page_allocation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        let page_id = cache.allocate_page(1);
        assert!(page_id.is_ok());
        assert_eq!(page_id.unwrap(), 0);

        let stats = cache.get_cache_stats();
        assert_eq!(stats.total_pages, 1);
        assert_eq!(stats.free_pages, 0);
        assert_eq!(stats.active_sequences, 1);
    }

    #[test]
    fn test_token_appending() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a page first
        cache.allocate_page(1).unwrap();

        // Append tokens
        for i in 0..4 {
            let result = cache.append_token(1, i);
            assert!(result.is_ok());
        }

        // Should fail when page is full
        let result = cache.append_token(1, 5);
        assert!(result.is_err());
        assert!(matches!(result, Err(KvCacheError::CapacityExceeded)));
    }

    #[test]
    fn test_sequence_retrieval() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        cache.allocate_page(1).unwrap();

        // Append tokens
        for i in 0..3 {
            cache.append_token(1, i).unwrap();
        }

        let tokens = cache.get_sequence_tokens(1).unwrap();
        assert_eq!(tokens, vec![0, 1, 2]);

        let length = cache.get_sequence_length(1).unwrap();
        assert_eq!(length, 3);
    }

    #[test]
    fn test_sequence_removal() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        cache.allocate_page(1).unwrap();
        cache.append_token(1, 42).unwrap();

        let stats_before = cache.get_cache_stats();
        assert_eq!(stats_before.active_sequences, 1);
        assert_eq!(stats_before.free_pages, 0);

        cache.remove_sequence(1).unwrap();

        let stats_after = cache.get_cache_stats();
        assert_eq!(stats_after.active_sequences, 0);
        assert_eq!(stats_after.free_pages, 1);
    }

    #[test]
    fn test_capacity_limit() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 2, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate maximum pages
        cache.allocate_page(1).unwrap();
        cache.allocate_page(2).unwrap();

        // Should fail when trying to allocate more
        let result = cache.allocate_page(3);
        assert!(result.is_err());
        assert!(matches!(result, Err(KvCacheError::CapacityExceeded)));
    }

    // Property tests
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_token_appending_properties(
            tokens in prop::collection::vec(0u32..1000, 1..50)
        ) {
            let backend = HipBackend::new().unwrap();
            let config = CacheConfig::new(tokens.len() + 10, 10, 32, 128, 24).unwrap();
            let mut cache = KvCache::new(config, backend).unwrap();

            cache.allocate_page(1).unwrap();

            let mut success_count = 0;
            for &token in &tokens {
                if cache.append_token(1, token).is_ok() {
                    success_count += 1;
                }
            }

            let retrieved = cache.get_sequence_tokens(1).unwrap();
            assert_eq!(retrieved.len(), success_count);

            // Check that retrieved tokens match the first success_count tokens
            assert_eq!(&retrieved[..], &tokens[..success_count]);
        }
    }
}
