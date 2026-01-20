//! Paged KV cache module for efficient memory management
//!
//! This module exports the production-grade paged KV cache implementation
//! with PagedAttention support, LRU eviction, and block sharing.
//!
//! # Which KV Cache Should I Use?
//!
//! - **Use `KvCache` from this module** for production inference
//! - See `crate::model::kv_cache::KVCache` for the legacy simple implementation

pub mod block_allocator;
pub mod blocks;
pub mod config;
pub mod kv_cache;
pub mod page_table;
pub mod pages;
pub mod types;

// Re-export from kv_cache
pub use kv_cache::KvCache;

// Re-export from types
pub use types::{
    BlockId as KvCacheBlockId, BlockTable, CacheStats, KvCacheError, KvCacheResult,
    MemoryProfile, PagedCacheStats,
};

// Re-export from config
pub use config::{CacheConfig, CachePreset};

// Re-export from blocks
pub use blocks::{AllocationStats, PhysicalBlock, PhysicalBlockPool};

// Re-export from pages
pub use pages::{CachePage, SequenceCache};

// Re-export from page_table
pub use page_table::PageTable;

// Re-export from block_allocator (use fully qualified names to avoid conflicts)
pub use block_allocator::BlockAllocator;

// Note: BlockId and PhysicalBlock are defined in multiple modules
// Use fully qualified paths: kv_cache::BlockId, block_allocator::BlockId, etc.
pub use block_allocator::BlockId as BlockAllocatorBlockId;
pub use block_allocator::PhysicalBlock as AllocatorPhysicalBlock;
pub use types::BlockId;
