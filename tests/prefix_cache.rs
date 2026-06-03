//! INF-14: Prefix cache tests — TDD RED phase.
//!
//! Tests the in-memory prefix cache that maps hashed token sequences to
//! refcounted block tables, enabling KV block reuse across requests.

#[cfg(feature = "gpu")]
mod prefix_cache_tests {
    use rocmforge::gpu::cache::{BlockTable, PrefixCache};

    #[test]
    fn new_cache_is_empty() {
        let cache = PrefixCache::new(4);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn insert_and_lookup_prefix() {
        let mut cache = PrefixCache::new(4);
        let tokens = vec![1, 2, 3, 4, 5];
        let table = BlockTable {
            block_ids: vec![0, 1, 2],
        };
        cache.insert(&tokens, table.clone());
        let found = cache.lookup(&tokens).expect("prefix should be found");
        assert_eq!(found.block_ids, table.block_ids);
    }

    #[test]
    fn lookup_missing_returns_none() {
        let mut cache = PrefixCache::new(4);
        assert!(cache.lookup(&[9, 9, 9]).is_none());
    }

    #[test]
    fn insert_evicts_on_capacity() {
        let mut cache = PrefixCache::new(2); // capacity 2 entries
        let t1 = vec![1];
        let t2 = vec![2];
        let t3 = vec![3];
        cache.insert(
            &t1,
            BlockTable {
                block_ids: vec![10],
            },
        );
        cache.insert(
            &t2,
            BlockTable {
                block_ids: vec![20],
            },
        );
        cache.insert(
            &t3,
            BlockTable {
                block_ids: vec![30],
            },
        ); // evicts t1
        assert!(cache.lookup(&t1).is_none());
        assert!(cache.lookup(&t2).is_some());
        assert!(cache.lookup(&t3).is_some());
    }

    #[test]
    fn insert_same_prefix_updates_lru() {
        let mut cache = PrefixCache::new(2);
        let t1 = vec![1];
        let t2 = vec![2];
        cache.insert(
            &t1,
            BlockTable {
                block_ids: vec![10],
            },
        );
        cache.insert(
            &t2,
            BlockTable {
                block_ids: vec![20],
            },
        );
        // touch t1 → t1 becomes most-recently-used
        let _ = cache.lookup(&t1);
        // insert t3 → should evict t2 (least recently used)
        let t3 = vec![3];
        cache.insert(
            &t3,
            BlockTable {
                block_ids: vec![30],
            },
        );
        assert!(cache.lookup(&t1).is_some());
        assert!(cache.lookup(&t2).is_none());
        assert!(cache.lookup(&t3).is_some());
    }

    #[test]
    fn clear_removes_all_entries() {
        let mut cache = PrefixCache::new(4);
        cache.insert(&[1, 2], BlockTable { block_ids: vec![0] });
        cache.insert(&[3, 4], BlockTable { block_ids: vec![1] });
        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.lookup(&[1, 2]).is_none());
    }

    #[test]
    fn insert_increments_len() {
        let mut cache = PrefixCache::new(4);
        assert_eq!(cache.len(), 0);
        cache.insert(&[1], BlockTable { block_ids: vec![0] });
        assert_eq!(cache.len(), 1);
        cache.insert(&[2], BlockTable { block_ids: vec![1] });
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn different_prefixes_with_same_hash_are_distinct() {
        // FxHash collision resistance: different token sequences should
        // produce different keys in practice.
        let mut cache = PrefixCache::new(4);
        cache.insert(&[1, 2, 3], BlockTable { block_ids: vec![0] });
        cache.insert(&[4, 5, 6], BlockTable { block_ids: vec![1] });
        let found1 = cache.lookup(&[1, 2, 3]).expect("hash collision resistance");
        assert_eq!(found1.block_ids, vec![0]);
        let found2 = cache.lookup(&[4, 5, 6]).expect("hash collision resistance");
        assert_eq!(found2.block_ids, vec![1]);
    }
}
