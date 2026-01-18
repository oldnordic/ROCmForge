//! Prefix cache for reusing computed states
//!
//! When processing multiple prompts with shared prefixes (e.g., system prompts),
//! we can cache the computed hidden states and KV cache to avoid recomputation.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::loader::gguf::GgufLoader;
use crate::backend::HipBackend;

/// Cached prefix data
#[derive(Debug, Clone)]
pub struct CachedPrefix {
    /// Token sequence that defines this prefix
    pub tokens: Vec<u32>,

    /// Computed hidden states after processing the prefix
    pub hidden_states: Vec<f32>,

    /// KV cache entries for each layer
    pub kv_cache: Vec<KvCacheEntry>,

    /// When this cache entry was created
    pub created_at: Instant,

    /// Last access time
    pub last_access: Instant,

    /// Number of times this cache has been used
    pub access_count: usize,

    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
}

/// KV cache entry for a single layer
#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    /// Layer index
    pub layer_idx: usize,

    /// Key cache for this layer
    pub k_cache: Vec<f32>,

    /// Value cache for this layer
    pub v_cache: Vec<f32>,
}

impl KvCacheEntry {
    /// Calculate memory usage for this entry
    pub fn memory_bytes(&self) -> usize {
        (self.k_cache.len() + self.v_cache.len()) * std::mem::size_of::<f32>()
    }

    /// Total number of tokens cached
    pub fn num_tokens(&self) -> usize {
        self.k_cache.len() / 2 // Assuming head_dim * num_heads = 2 for this calculation
    }
}

/// Prefix cache for reusing computed states
pub struct PrefixCache {
    /// Map from token hash to cached prefix
    cache: Arc<RwLock<HashMap<u64, CachedPrefix>>>,

    /// Maximum number of entries to keep
    max_entries: usize,

    /// Maximum total memory to use (in bytes)
    max_memory_bytes: usize,

    /// Current memory usage
    current_memory_bytes: Arc<RwLock<usize>>,

    /// Cache hits counter
    hits: Arc<RwLock<usize>>,

    /// Cache misses counter
    misses: Arc<RwLock<usize>>,
}

impl PrefixCache {
    /// Create a new prefix cache
    pub fn new(max_entries: usize, max_memory_mb: usize) -> Self {
        PrefixCache {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            current_memory_bytes: Arc::new(RwLock::new(0)),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Look up a cached prefix
    pub fn get(&self, tokens: &[u32]) -> Option<CachedPrefix> {
        let hash = self.hash_tokens(tokens);
        let cache = self.cache.read().ok()?;

        let entry = cache.get(&hash).cloned();

        if entry.is_some() {
            let mut hits = self.hits.write().unwrap();
            *hits += 1;
        } else {
            let mut misses = self.misses.write().unwrap();
            *misses += 1;
        }

        entry
    }

    /// Insert a new cached prefix
    pub fn insert(&self, prefix: CachedPrefix) -> Result<(), CacheError> {
        let hash = self.hash_tokens(&prefix.tokens);
        let memory_bytes = prefix.memory_bytes;

        // Check memory constraints
        {
            let current_memory = self.current_memory_bytes.read().unwrap();
            if *current_memory + memory_bytes > self.max_memory_bytes {
                self.evict_lru()?;
            }
        }

        // Insert the new entry
        {
            let mut cache = self.cache.write().unwrap();
            let mut current_memory = self.current_memory_bytes.write().unwrap();

            // Check if we're at capacity
            if cache.len() >= self.max_entries && !cache.contains_key(&hash) {
                drop(cache);
                drop(current_memory);
                self.evict_lru()?;
                let mut cache = self.cache.write().unwrap();
                let mut current_memory = self.current_memory_bytes.write().unwrap();

                // Remove old entry if updating
                if let Some(old) = cache.get(&hash) {
                    *current_memory = current_memory.saturating_sub(old.memory_bytes);
                }

                cache.insert(hash, prefix);
                *current_memory += memory_bytes;
            } else {
                // Remove old entry if updating
                if let Some(old) = cache.get(&hash) {
                    *current_memory = current_memory.saturating_sub(old.memory_bytes);
                }

                cache.insert(hash, prefix);
                *current_memory += memory_bytes;
            }
        }

        Ok(())
    }

    /// Find the longest cached prefix that matches the start of the given tokens
    pub fn find_longest_prefix(&self, tokens: &[u32]) -> Option<CachedPrefix> {
        // Try progressively shorter prefixes
        for len in (1..=tokens.len()).rev() {
            if let Some(entry) = self.get(&tokens[..len]) {
                return Some(entry);
            }
        }
        None
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        let current_memory = self.current_memory_bytes.read().unwrap();
        let hits = self.hits.read().unwrap();
        let misses = self.misses.read().unwrap();

        let total_requests = *hits + *misses;
        let hit_rate = if total_requests > 0 {
            *hits as f32 / total_requests as f32
        } else {
            0.0
        };

        CacheStats {
            entries: cache.len(),
            max_entries: self.max_entries,
            memory_bytes: *current_memory,
            max_memory_bytes: self.max_memory_bytes,
            hits: *hits,
            misses: *misses,
            hit_rate,
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut current_memory = self.current_memory_bytes.write().unwrap();
        cache.clear();
        *current_memory = 0;
    }

    /// Evict least recently used entries
    fn evict_lru(&self) -> Result<(), CacheError> {
        let mut cache = self.cache.write().unwrap();
        let mut current_memory = self.current_memory_bytes.write().unwrap();

        if cache.is_empty() {
            return Err(CacheError::CacheFull);
        }

        // Find LRU entry
        let lru_key = cache
            .iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(k, _)| *k);

        if let Some(key) = lru_key {
            if let Some(entry) = cache.remove(&key) {
                *current_memory = current_memory.saturating_sub(entry.memory_bytes);
            }
        }

        Ok(())
    }

    /// Hash a token sequence for cache lookup
    fn hash_tokens(&self, tokens: &[u32]) -> u64 {
        // Use a simple hash function for token sequences
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for token in tokens {
            token.hash(&mut hasher);
        }
        hasher.finish()
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(100, 1024) // 100 entries, 1GB max
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in the cache
    pub entries: usize,

    /// Maximum number of entries
    pub max_entries: usize,

    /// Current memory usage in bytes
    pub memory_bytes: usize,

    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,

    /// Number of cache hits
    pub hits: usize,

    /// Number of cache misses
    pub misses: usize,

    /// Cache hit rate (0.0 - 1.0)
    pub hit_rate: f32,
}

impl CacheStats {
    /// Calculate memory utilization ratio
    pub fn memory_utilization(&self) -> f32 {
        if self.max_memory_bytes > 0 {
            self.memory_bytes as f32 / self.max_memory_bytes as f32
        } else {
            0.0
        }
    }

    /// Calculate entry utilization ratio
    pub fn entry_utilization(&self) -> f32 {
        if self.max_entries > 0 {
            self.entries as f32 / self.max_entries as f32
        } else {
            0.0
        }
    }
}

/// Prefix cache errors
#[derive(Debug, Clone)]
pub enum CacheError {
    /// Cache is full and cannot accommodate new entry
    CacheFull,

    /// Memory limit exceeded
    MemoryLimitExceeded,

    /// Invalid cache entry
    InvalidEntry,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::CacheFull => write!(f, "Cache is full"),
            CacheError::MemoryLimitExceeded => write!(f, "Memory limit exceeded"),
            CacheError::InvalidEntry => write!(f, "Invalid cache entry"),
        }
    }
}

impl std::error::Error for CacheError {}

/// Early exit detector for cached prefixes
///
/// Detects when a prefix has been fully computed and cached, allowing
/// early exit from the prompt processing loop.
pub struct EarlyExitDetector {
    /// Minimum prefix length to consider for early exit
    min_prefix_len: usize,

    /// Whether early exit is enabled
    enabled: bool,
}

impl EarlyExitDetector {
    /// Create a new early exit detector
    pub fn new(min_prefix_len: usize) -> Self {
        EarlyExitDetector {
            min_prefix_len,
            enabled: true,
        }
    }

    /// Enable or disable early exit
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if we can exit early for the given token position
    ///
    /// Returns the position to exit at, or None if no early exit is possible.
    pub fn check_exit(
        &self,
        tokens: &[u32],
        current_pos: usize,
        cache: &PrefixCache,
    ) -> Option<usize> {
        if !self.enabled {
            return None;
        }

        // Check if the remaining tokens match a cached prefix
        if current_pos >= self.min_prefix_len {
            if let Some(cached) = cache.find_longest_prefix(&tokens[current_pos..]) {
                // Can skip to the end of the cached prefix
                return Some(current_pos + cached.tokens.len());
            }
        }

        None
    }

    /// Find the longest cacheable prefix in the given tokens
    pub fn find_cacheable_prefix(&self, tokens: &[u32], cache: &PrefixCache) -> Option<usize> {
        if tokens.len() < self.min_prefix_len {
            return None;
        }

        // Try progressively longer prefixes
        for len in (self.min_prefix_len..=tokens.len()).rev() {
            if cache.get(&tokens[..len]).is_some() {
                return Some(len);
            }
        }

        None
    }
}

impl Default for EarlyExitDetector {
    fn default() -> Self {
        Self::new(32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_cache_basic() {
        let cache = PrefixCache::new(10, 1024);

        // Cache miss initially
        assert!(cache.get(&[1, 2, 3]).is_none());

        // Insert a prefix
        let prefix = CachedPrefix {
            tokens: vec![1, 2, 3],
            hidden_states: vec![0.0f32; 128],
            kv_cache: vec![],
            created_at: Instant::now(),
            last_access: Instant::now(),
            access_count: 0,
            memory_bytes: 512,
        };
        cache.insert(prefix).unwrap();

        // Cache hit after insert
        let result = cache.get(&[1, 2, 3]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_prefix_cache_stats() {
        let cache = PrefixCache::new(10, 1024);

        // Insert some entries
        for i in 0..5 {
            let prefix = CachedPrefix {
                tokens: vec![i, i + 1, i + 2],
                hidden_states: vec![0.0f32; 128],
                kv_cache: vec![],
                created_at: Instant::now(),
                last_access: Instant::now(),
                access_count: 0,
                memory_bytes: 100,
            };
            cache.insert(prefix).unwrap();
        }

        // Check stats before access
        let stats = cache.stats();
        assert_eq!(stats.entries, 5);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0); // No misses recorded yet

        // Access an entry
        cache.get(&[1, 2, 3]);

        // Check stats after access
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_find_longest_prefix() {
        let cache = PrefixCache::new(10, 1024);

        // Insert prefixes of different lengths
        let short = CachedPrefix {
            tokens: vec![1, 2],
            hidden_states: vec![0.0f32; 64],
            kv_cache: vec![],
            created_at: Instant::now(),
            last_access: Instant::now(),
            access_count: 0,
            memory_bytes: 64,
        };
        cache.insert(short).unwrap();

        let long = CachedPrefix {
            tokens: vec![1, 2, 3, 4],
            hidden_states: vec![0.0f32; 128],
            kv_cache: vec![],
            created_at: Instant::now(),
            last_access: Instant::now(),
            access_count: 0,
            memory_bytes: 128,
        };
        cache.insert(long).unwrap();

        // Should find the longest matching prefix
        let result = cache.find_longest_prefix(&[1, 2, 3, 4, 5]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().tokens.len(), 4);
    }

    #[test]
    fn test_early_exit_detector() {
        let cache = PrefixCache::new(10, 1024);
        let mut detector = EarlyExitDetector::new(2);

        // Insert a cached prefix
        let prefix = CachedPrefix {
            tokens: vec![10, 20, 30],
            hidden_states: vec![0.0f32; 128],
            kv_cache: vec![],
            created_at: Instant::now(),
            last_access: Instant::now(),
            access_count: 0,
            memory_bytes: 512,
        };
        cache.insert(prefix).unwrap();

        // Check for early exit at position 2 (>= min_prefix_len)
        let tokens = vec![0, 0, 10, 20, 30, 40, 50];
        let exit_pos = detector.check_exit(&tokens, 2, &cache);

        assert!(exit_pos.is_some());
        assert_eq!(exit_pos.unwrap(), 5); // Can skip to position 2+3=5
    }

    #[test]
    fn test_cache_clear() {
        let cache = PrefixCache::new(10, 1024);

        // Insert some entries
        for i in 0..5 {
            let prefix = CachedPrefix {
                tokens: vec![i],
                hidden_states: vec![0.0f32; 64],
                kv_cache: vec![],
                created_at: Instant::now(),
                last_access: Instant::now(),
                access_count: 0,
                memory_bytes: 64,
            };
            cache.insert(prefix).unwrap();
        }

        assert_eq!(cache.stats().entries, 5);

        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().memory_bytes, 0);
    }

    #[test]
    fn test_kv_cache_entry_memory() {
        let entry = KvCacheEntry {
            layer_idx: 0,
            k_cache: vec![0.0f32; 1024],
            v_cache: vec![0.0f32; 1024],
        };

        assert_eq!(entry.memory_bytes(), 2048 * 4); // 2048 floats * 4 bytes
    }
}
