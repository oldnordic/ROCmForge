//! Prefix cache for refcounted paged-KV reuse across requests.

use super::allocator::BlockTable;

/// In-memory prefix cache for KV block reuse across requests.
///
/// Maps hashed token sequences to refcounted `BlockTable` entries.
/// Entries are evicted in LRU order when capacity is exceeded.
#[derive(Clone, Debug)]
pub struct PrefixCache {
    capacity: usize,
    entries: std::collections::HashMap<u64, (BlockTable, u64)>,
    access_counter: u64,
}

impl PrefixCache {
    /// Create a cache with room for `capacity` prefix entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: std::collections::HashMap::new(),
            access_counter: 0,
        }
    }

    /// Is the cache empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Lookup a prefix by its token sequence. On hit, bumps the entry to MRU.
    /// Returns a cloned `BlockTable` on hit, `None` on miss.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<BlockTable> {
        let key = Self::hash_tokens(tokens);
        if let Some((table, counter)) = self.entries.get_mut(&key) {
            self.access_counter += 1;
            *counter = self.access_counter;
            Some(table.clone())
        } else {
            None
        }
    }

    /// Store a `BlockTable` under the given token sequence.
    /// Evicts the least-recently-used entry if over capacity.
    pub fn insert(&mut self, tokens: &[u32], table: BlockTable) {
        if self.capacity == 0 {
            return;
        }
        let key = Self::hash_tokens(tokens);
        self.access_counter += 1;
        self.entries.insert(key, (table, self.access_counter));

        if self.entries.len() > self.capacity {
            let lru_key = self
                .entries
                .iter()
                .min_by_key(|(_, (_, c))| *c)
                .map(|(k, _)| *k);
            if let Some(k) = lru_key {
                self.entries.remove(&k);
            }
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_counter = 0;
    }

    fn hash_tokens(tokens: &[u32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }
}
