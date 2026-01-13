//! Paged KV cache for efficient GPU memory management

use super::block_allocator::BlockAllocator;
use super::page_table::PageTable;
use crate::backend::{HipBackend, HipBuffer, HipError};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
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
    #[error("Internal lock poisoned - this indicates a bug: {0}")]
    LockPoisoned(String),
}

impl<T> From<std::sync::PoisonError<T>> for KvCacheError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        KvCacheError::LockPoisoned(format!("Lock poisoned: {}", err))
    }
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

/// Block identifier type for PagedAttention
pub type BlockId = u32;

/// Block table entry for PagedAttention
/// Maps logical block IDs to physical GPU memory blocks
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical block ID
    pub block_id: BlockId,
    /// Physical block ID (index into block pool)
    pub physical_block_id: u32,
    /// Reference count for sharing across sequences
    pub ref_count: Arc<AtomicUsize>,
    /// Sequence IDs using this block (for sharing)
    pub sequences: HashSet<u32>,
}

/// Physical GPU memory block containing KV cache data
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    /// Block identifier
    pub block_id: u32,
    /// Key cache stored in GPU memory
    pub key_buffer: HipBuffer,
    /// Value cache stored in GPU memory
    pub value_buffer: HipBuffer,
}

impl PhysicalBlock {
    /// Create a new physical block
    pub fn new(block_id: u32, key_buffer: HipBuffer, value_buffer: HipBuffer) -> Self {
        PhysicalBlock {
            block_id,
            key_buffer,
            value_buffer,
        }
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.key_buffer.size()
    }

    /// Get the capacity in tokens
    pub fn capacity_tokens(&self) -> usize {
        self.key_buffer.size() / std::mem::size_of::<f32>()
    }
}

/// Pool of pre-allocated GPU memory blocks for O(1) allocation
#[derive(Debug)]
pub struct PhysicalBlockPool {
    /// Pre-allocated GPU blocks
    blocks: Vec<PhysicalBlock>,
    /// Free list for O(1) allocation
    free_list: VecDeque<BlockId>,
    /// Block size in tokens
    block_size: usize,
    /// Number of KV heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl PhysicalBlockPool {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        backend: &HipBackend,
    ) -> KvCacheResult<Self> {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = VecDeque::with_capacity(num_blocks);

        for block_id in 0..num_blocks {
            let key_size = block_size * num_heads * head_dim * std::mem::size_of::<f32>();
            let value_size = key_size;

            let key_buffer = backend.allocate_buffer(key_size)?;
            let value_buffer = backend.allocate_buffer(value_size)?;

            blocks.push(PhysicalBlock::new(
                block_id as u32,
                key_buffer,
                value_buffer,
            ));
            free_list.push_back(block_id as u32);
        }

        Ok(PhysicalBlockPool {
            blocks,
            free_list,
            block_size,
            num_heads,
            head_dim,
        })
    }

    /// Allocate a block from the pool (O(1))
    pub fn allocate(&mut self) -> Option<BlockId> {
        self.free_list.pop_front()
    }

    /// Deallocate a block back to the pool (O(1))
    pub fn deallocate(&mut self, block_id: BlockId) {
        self.free_list.push_back(block_id);
    }

    /// Get a physical block by ID
    pub fn get_block(&self, block_id: BlockId) -> Option<&PhysicalBlock> {
        self.blocks.get(block_id as usize)
    }

    /// Get the number of free blocks
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Get the total number of blocks
    pub fn total_count(&self) -> usize {
        self.blocks.len()
    }
}

impl BlockTable {
    /// Create a new BlockTable entry
    pub fn new(block_id: BlockId, physical_block_id: u32) -> Self {
        BlockTable {
            block_id,
            physical_block_id,
            ref_count: Arc::new(AtomicUsize::new(1)),
            sequences: HashSet::new(),
        }
    }

    /// Add a sequence to this block
    pub fn add_sequence(&mut self, sequence_id: u32) {
        self.sequences.insert(sequence_id);
    }

    /// Remove a sequence from this block
    pub fn remove_sequence(&mut self, sequence_id: u32) -> bool {
        self.sequences.remove(&sequence_id)
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Increment reference count
    pub fn incr_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement reference count, returns previous count
    pub fn decr_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) - 1
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

#[derive(Debug)]
pub struct SequenceCache {
    pub sequence_id: u32,
    pub pages: Vec<u32>,
    pub total_tokens: usize,
    /// Tracks whether this sequence is completed
    pub is_completed: bool,
    /// Last access time for LRU eviction
    pub last_access: Instant,
}

impl SequenceCache {
    pub fn new(sequence_id: u32) -> Self {
        SequenceCache {
            sequence_id,
            pages: Vec::new(),
            total_tokens: 0,
            is_completed: false,
            last_access: Instant::now(),
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
        self.last_access = Instant::now();
    }

    pub fn mark_completed(&mut self) {
        self.is_completed = true;
    }

    pub fn is_active(&self) -> bool {
        !self.is_completed
    }
}

/// Paged KV cache with PagedAttention support
///
/// This is the production KV cache used by the inference engine.
/// For the simple GPU-resident KV cache (legacy), see `crate::model::kv_cache::SimpleKVCache`.
#[derive(Debug)]
pub struct KvCache {
    config: CacheConfig,
    backend: Arc<HipBackend>,
    /// Block pool for physical GPU memory (PagedAttention)
    block_pool: RwLock<PhysicalBlockPool>,
    /// Block table: logical ID -> physical block mapping (PagedAttention)
    block_table: RwLock<HashMap<BlockId, BlockTable>>,
    /// Page table for mapping logical positions to physical blocks
    page_table: RwLock<PageTable>,
    /// Block allocator for O(1) block allocation
    block_allocator: RwLock<BlockAllocator>,
    /// Legacy: sequence-owned pages (for backward compatibility)
    pages: RwLock<HashMap<u32, CachePage>>,
    sequences: RwLock<HashMap<u32, SequenceCache>>,
    free_pages: RwLock<Vec<u32>>,
    next_page_id: RwLock<u32>,
    /// Free logical block IDs (PagedAttention)
    free_blocks: RwLock<Vec<BlockId>>,
    next_block_id: RwLock<BlockId>,
}

impl KvCache {
    pub fn new(config: CacheConfig, backend: Arc<HipBackend>) -> KvCacheResult<Self> {
        // Initialize the physical block pool for PagedAttention
        let block_pool = PhysicalBlockPool::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
            &backend,
        )?;

        // Initialize PageTable with default block size
        let page_table = PageTable::new();

        // Initialize BlockAllocator
        let block_allocator = BlockAllocator::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
        );

        Ok(KvCache {
            config,
            backend,
            block_pool: RwLock::new(block_pool),
            block_table: RwLock::new(HashMap::new()),
            page_table: RwLock::new(page_table),
            block_allocator: RwLock::new(block_allocator),
            pages: RwLock::new(HashMap::new()),
            sequences: RwLock::new(HashMap::new()),
            free_pages: RwLock::new(Vec::new()),
            next_page_id: RwLock::new(0),
            free_blocks: RwLock::new(Vec::new()),
            next_block_id: RwLock::new(0),
        })
    }

    pub fn allocate_page(&mut self, sequence_id: u32) -> KvCacheResult<u32> {
        // Check if we need to evict LRU sequences first
        let current_pages = self.pages.read()?.len();
        let has_free_page = self.free_pages.read()?.is_empty();

        if current_pages >= self.config.max_pages && has_free_page {
            // Try LRU eviction to free up space
            self.evict_lru_sequences(1)?;
        }

        // Try to reuse a free page first
        let page_id = if let Some(free_id) = self.free_pages.write()?.pop() {
            free_id
        } else if self.pages.read()?.len() >= self.config.max_pages {
            return Err(KvCacheError::CapacityExceeded);
        } else {
            let mut next_id = self.next_page_id.write()?;
            let id = *next_id;
            *next_id += 1;
            id
        };

        let page = CachePage::new(page_id, sequence_id, &self.backend, &self.config)?;
        self.pages.write()?.insert(page_id, page);

        // Update sequence cache
        let mut sequences = self.sequences.write()?;
        let sequence = sequences
            .entry(sequence_id)
            .or_insert_with(|| SequenceCache::new(sequence_id));
        sequence.add_page(page_id);

        Ok(page_id)
    }

    pub fn append_token(&mut self, sequence_id: u32, token: u32) -> KvCacheResult<()> {
        // FIX-10: Check if sequence is completed before appending
        {
            let sequences = self.sequences.read()?;
            if let Some(sequence) = sequences.get(&sequence_id) {
                if sequence.is_completed {
                    return Err(KvCacheError::InvalidSequenceId(sequence_id));
                }
            }
        }

        let last_page_id = {
            let sequences = self.sequences.read()?;
            let sequence = sequences
                .get(&sequence_id)
                .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
            sequence
                .get_last_page()
                .ok_or(KvCacheError::PageNotFound(sequence_id))?
        };

        let can_append = {
            let pages = self.pages.read()?;
            let page = pages
                .get(&last_page_id)
                .ok_or(KvCacheError::PageNotFound(last_page_id))?;
            page.can_append(token)
        };

        if can_append {
            {
                let mut pages = self.pages.write()?;
                let page = pages
                    .get_mut(&last_page_id)
                    .ok_or(KvCacheError::PageNotFound(last_page_id))?;
                page.append_token(token)?;
            }
            let mut sequences = self.sequences.write()?;
            let sequence = sequences
                .get_mut(&sequence_id)
                .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
            sequence.total_tokens += 1;
            sequence.update_access(); // FIX-10: Update access time on append
        } else {
            // Allocate new page
            let new_page_id = self.allocate_page(sequence_id)?;
            {
                let mut pages = self.pages.write()?;
                // Safe: new_page_id was just allocated above
                let new_page = pages
                    .get_mut(&new_page_id)
                    .ok_or(KvCacheError::PageNotFound(new_page_id))?;
                new_page.append_token(token)?;
            }
            let mut sequences = self.sequences.write()?;
            let sequence = sequences
                .get_mut(&sequence_id)
                .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
            sequence.total_tokens += 1;
        }

        Ok(())
    }

    pub fn get_sequence_tokens(&self, sequence_id: u32) -> KvCacheResult<Vec<u32>> {
        let sequences = self.sequences.read()?;
        let sequence = sequences
            .get(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        let mut tokens = Vec::with_capacity(sequence.total_tokens);
        let sequence_pages = sequence.pages.clone();

        drop(sequences); // Release sequences lock before acquiring pages lock

        let pages = self.pages.read()?;
        for page_id in &sequence_pages {
            let page = pages
                .get(page_id)
                .ok_or(KvCacheError::PageNotFound(*page_id))?;
            tokens.extend_from_slice(&page.tokens);
        }

        Ok(tokens)
    }

    pub fn get_sequence_length(&self, sequence_id: u32) -> KvCacheResult<usize> {
        let sequences = self.sequences.read()?;
        let sequence = sequences
            .get(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        Ok(sequence.total_tokens)
    }

    pub fn remove_sequence(&mut self, sequence_id: u32) -> KvCacheResult<()> {
        let sequence = self
            .sequences
            .write()?
            .remove(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        // Free pages from GPU memory
        let mut pages = self.pages.write()?;
        let mut free_pages = self.free_pages.write()?;

        for page_id in sequence.pages {
            if pages.remove(&page_id).is_some() {
                free_pages.push(page_id);
            }
        }

        // PHASE 2 FIX: Also deallocate blocks from paged system
        // Get blocks from page table before removing sequence
        let blocks_to_deallocate = self
            .page_table
            .read()?
            .get_sequence_blocks(sequence_id)
            .map(|v| v.to_vec());

        if let Some(blocks) = blocks_to_deallocate {
            let mut allocator = self.block_allocator.write()?;
            for &block_id in &blocks {
                allocator.deallocate(block_id);
            }
        }

        // Remove from page table
        self.page_table.write()?.remove_sequence(sequence_id);

        Ok(())
    }

    pub fn get_cache_stats(&self) -> CacheStats {
        // Safe: These locks should never be poisoned in normal operation
        // If they are, it indicates a serious bug and we want to know about it
        let pages = self.pages.read().expect("KvCache pages lock poisoned");
        let free_pages = self
            .free_pages
            .read()
            .expect("KvCache free_pages lock poisoned");
        let sequences = self
            .sequences
            .read()
            .expect("KvCache sequences lock poisoned");

        CacheStats {
            total_pages: pages.len(),
            free_pages: free_pages.len(),
            active_sequences: sequences.len(),
            total_tokens: sequences.values().map(|s| s.total_tokens).sum(),
        }
    }

    // ========== FIX-10: Sequence State Tracking Methods ==========

    /// Mark a sequence as completed (to be cleaned up later)
    pub fn mark_sequence_completed(&mut self, sequence_id: u32) -> KvCacheResult<()> {
        let mut sequences = self.sequences.write()?;
        let sequence = sequences
            .get_mut(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        sequence.mark_completed();
        Ok(())
    }

    /// Check if a sequence is marked as completed
    pub fn is_sequence_completed(&self, sequence_id: u32) -> KvCacheResult<bool> {
        let sequences = self.sequences.read()?;
        let sequence = sequences
            .get(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        Ok(sequence.is_completed)
    }

    /// Update the last access time for a sequence (for LRU tracking)
    pub fn update_sequence_access(&mut self, sequence_id: u32) -> KvCacheResult<()> {
        let mut sequences = self.sequences.write()?;
        let sequence = sequences
            .get_mut(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        sequence.update_access();
        Ok(())
    }

    /// Get the last access time for a sequence
    pub fn get_sequence_access_time(&self, sequence_id: u32) -> KvCacheResult<Instant> {
        let sequences = self.sequences.read()?;
        let sequence = sequences
            .get(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

        Ok(sequence.last_access)
    }

    /// Get list of active (non-completed) sequence IDs
    pub fn get_active_sequences(&self) -> KvCacheResult<Vec<u32>> {
        let sequences = self.sequences.read()?;

        let active: Vec<u32> = sequences
            .values()
            .filter(|s| s.is_active())
            .map(|s| s.sequence_id)
            .collect();

        Ok(active)
    }

    /// Remove all completed sequences from the cache
    /// This should be called periodically to free up memory
    pub fn cleanup_completed_sequences(&mut self) -> KvCacheResult<usize> {
        let mut sequences = self.sequences.write()?;

        // Find completed sequence IDs
        let completed_ids: Vec<u32> = sequences
            .iter()
            .filter(|(_, s)| s.is_completed)
            .map(|(id, _)| *id)
            .collect();

        let mut removed_count = 0;

        for seq_id in completed_ids {
            // Remove from sequences map
            if let Some(sequence) = sequences.remove(&seq_id) {
                // Free pages from GPU memory
                let mut pages = self.pages.write()?;
                let mut free_pages = self.free_pages.write()?;

                for page_id in sequence.pages {
                    if pages.remove(&page_id).is_some() {
                        free_pages.push(page_id);
                    }
                }

                removed_count += 1;
            }
        }

        Ok(removed_count)
    }

    /// Evict least recently used sequences to free up space for new sequences
    /// This is called automatically when capacity is exceeded during allocation
    fn evict_lru_sequences(&mut self, required_pages: usize) -> KvCacheResult<()> {
        let sequences = self.sequences.read()?;

        // Check if we need to evict
        let current_usage = self.pages.read()?.len();
        let max_pages = self.config.max_pages;

        if current_usage + required_pages <= max_pages {
            return Ok(());
        }

        // Find LRU sequences (only active sequences, not completed ones)
        let mut seq_access_times: Vec<(u32, Instant)> = sequences
            .iter()
            .filter(|(_, s)| s.is_active()) // Only consider active sequences
            .map(|(id, s)| (*id, s.last_access))
            .collect();

        // Sort by access time (oldest first)
        seq_access_times.sort_by_key(|(_, time)| *time);

        // Calculate how many sequences we need to evict
        let pages_to_free = (current_usage + required_pages) - max_pages;

        // Estimate pages per sequence (rough estimate)
        let avg_pages_per_seq = if seq_access_times.is_empty() {
            1
        } else {
            let total_pages: usize = sequences
                .values()
                .filter(|s| s.is_active())
                .map(|s| s.pages.len())
                .sum();
            (total_pages / seq_access_times.len()).max(1)
        };

        let seqs_to_evict = (pages_to_free / avg_pages_per_seq)
            .max(1)
            .min(seq_access_times.len());

        // Drop the read lock before acquiring write locks
        drop(sequences);

        // Evict LRU sequences
        for (seq_id, _) in seq_access_times.iter().take(seqs_to_evict) {
            let _ = self.remove_sequence(*seq_id);
        }

        Ok(())
    }

    // ========== Phase 2: PageTable + BlockAllocator Integration ==========

    /// Append token with paged KV cache using PageTable and BlockAllocator
    ///
    /// This method integrates PageTable (for mapping logical positions to physical blocks)
    /// with BlockAllocator (for O(1) block allocation) to provide efficient paged attention.
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to append the token to
    /// * `token` - The token to append
    ///
    /// # Behavior
    /// - Checks if we need a new block (every `block_size` tokens)
    /// - Allocates from BlockAllocator if needed
    /// - Updates PageTable with new block mapping
    /// - Delegates to existing append_token() for actual storage
    pub fn append_token_paged(&mut self, sequence_id: u32, token: u32) -> KvCacheResult<()> {
        // Get current sequence length
        let current_tokens = self.get_sequence_length(sequence_id).unwrap_or(0);
        let block_size = self.config.page_size;

        // Check if we need to allocate a page for the sequence
        let has_page = {
            let sequences = self.sequences.read()?;
            sequences.get(&sequence_id).is_some()
        };

        if !has_page {
            // Allocate first page for new sequence
            self.allocate_page(sequence_id)?;
        }

        // Check if we need a new block
        if current_tokens > 0 && current_tokens % block_size == 0 {
            // Time to allocate a new block
            if let Some(block_id) = self.block_allocator.write()?.allocate() {
                // Update page table with new block
                self.page_table.write()?.append_block(sequence_id, block_id);

                tracing::debug!(
                    "Allocated new block {} for sequence {} at token position {}",
                    block_id,
                    sequence_id,
                    current_tokens
                );
            } else {
                return Err(KvCacheError::CapacityExceeded);
            }
        } else if current_tokens == 0 {
            // First token - allocate initial block
            if let Some(block_id) = self.block_allocator.write()?.allocate() {
                self.page_table.write()?.append_block(sequence_id, block_id);

                tracing::debug!(
                    "Allocated initial block {} for new sequence {}",
                    block_id,
                    sequence_id
                );
            } else {
                return Err(KvCacheError::CapacityExceeded);
            }
        }

        // Delegate to existing append_token for actual storage
        // This handles token storage and sequence tracking
        self.append_token(sequence_id, token)
    }

    /// Get the physical block for a given sequence and token position
    ///
    /// This uses the PageTable to map logical positions to physical blocks.
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to query
    /// * `token_pos` - The logical token position within the sequence
    ///
    /// # Returns
    /// * `Some((block_id, offset))` - The physical block ID and offset within that block
    /// * `None` - If the sequence doesn't exist or position is out of range
    pub fn get_block_for_position(
        &self,
        sequence_id: u32,
        token_pos: usize,
    ) -> KvCacheResult<Option<(u32, usize)>> {
        Ok(self
            .page_table
            .read()?
            .get_block_for_position(sequence_id, token_pos))
    }

    /// Get all blocks for a sequence using PageTable
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to query
    ///
    /// # Returns
    /// * `Some(&[u32])` - Slice of block IDs
    /// * `None` - If the sequence doesn't exist
    pub fn get_sequence_blocks_from_page_table(
        &self,
        sequence_id: u32,
    ) -> KvCacheResult<Option<Vec<u32>>> {
        Ok(self
            .page_table
            .read()?
            .get_sequence_blocks(sequence_id)
            .map(|v| v.to_vec()))
    }

    /// Get BlockAllocator statistics
    pub fn get_block_allocator_stats(&self) -> (usize, usize) {
        let allocator = self
            .block_allocator
            .read()
            .expect("Block allocator lock poisoned");
        (allocator.total_blocks(), allocator.free_blocks())
    }

    // ========== PagedAttention Block Management ==========

    /// Allocate a block for PagedAttention-style KV cache management
    /// Returns the logical block ID
    pub fn allocate_block(&mut self, sequence_id: u32) -> KvCacheResult<BlockId> {
        // Allocate physical block from pool
        let physical_block_id = self
            .block_pool
            .write()?
            .allocate()
            .ok_or(KvCacheError::CapacityExceeded)?;

        // Get or create logical block ID
        let logical_block_id = if let Some(free_id) = self.free_blocks.write()?.pop() {
            free_id
        } else {
            let mut next_id = self.next_block_id.write()?;
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Create block table entry
        let mut block_table = BlockTable::new(logical_block_id, physical_block_id);
        block_table.add_sequence(sequence_id);

        // Register in block table
        self.block_table
            .write()?
            .insert(logical_block_id, block_table);

        Ok(logical_block_id)
    }

    /// Get a block table entry by logical block ID
    pub fn get_block(&self, block_id: BlockId) -> KvCacheResult<BlockTable> {
        let block_table = self.block_table.read()?;
        block_table
            .get(&block_id)
            .cloned()
            .ok_or(KvCacheError::PageNotFound(block_id))
    }

    /// Get the physical block for a given logical block ID
    pub fn get_physical_block(&self, block_id: BlockId) -> KvCacheResult<PhysicalBlock> {
        let block_table = self.get_block(block_id)?;
        let block_pool = self.block_pool.read()?;
        block_pool
            .get_block(block_table.physical_block_id)
            .cloned()
            .ok_or(KvCacheError::PageNotFound(block_id))
    }

    /// Increment reference count for a block (for block sharing across sequences)
    pub fn ref_block(&mut self, block_id: BlockId, sequence_id: u32) -> KvCacheResult<()> {
        let mut block_table = self.block_table.write()?;
        let block = block_table
            .get_mut(&block_id)
            .ok_or(KvCacheError::PageNotFound(block_id))?;

        block.add_sequence(sequence_id);
        block.incr_ref();

        Ok(())
    }

    /// Decrement reference count for a block
    /// Returns true if block was freed (ref count reached zero)
    pub fn unref_block(&mut self, block_id: BlockId, sequence_id: u32) -> KvCacheResult<bool> {
        let mut physical_block_id = None;
        let mut should_free = false;

        {
            let mut block_table = self.block_table.write()?;
            let block = block_table
                .get_mut(&block_id)
                .ok_or(KvCacheError::PageNotFound(block_id))?;

            block.remove_sequence(sequence_id);
            let prev_count = block.decr_ref();

            if prev_count == 0 {
                should_free = true;
                physical_block_id = Some(block.physical_block_id);
            }
        }

        if should_free {
            // Remove from block table
            self.block_table.write()?.remove(&block_id);
            // Return physical block to pool
            if let Some(phys_id) = physical_block_id {
                self.block_pool.write()?.deallocate(phys_id);
            }
            // Recycle logical block ID
            self.free_blocks.write()?.push(block_id);

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Copy a block for copy-on-write (COW) optimization
    /// Useful when a sequence diverges from a shared prefix
    ///
    /// NOTE: This requires GPU device-to-device memcpy support in HipBackend.
    /// For now, this is not implemented - use block sharing (ref_block) instead.
    pub fn copy_block(
        &mut self,
        _block_id: BlockId,
        new_sequence_id: u32,
    ) -> KvCacheResult<BlockId> {
        // COW block copying requires HipBackend::memcpy_device_to_device()
        // which is not yet implemented. For now, allocate a fresh block.
        // The caller should use ref_block() for sharing instead.
        let new_block_id = self.allocate_block(new_sequence_id)?;
        Ok(new_block_id)
    }

    /// Get PagedAttention statistics
    pub fn get_paged_stats(&self) -> PagedCacheStats {
        // Safe: These locks should never be poisoned in normal operation
        let block_pool = self
            .block_pool
            .read()
            .expect("KvCache block_pool lock poisoned");
        let block_table = self
            .block_table
            .read()
            .expect("KvCache block_table lock poisoned");
        let sequences = self
            .sequences
            .read()
            .expect("KvCache sequences lock poisoned");

        PagedCacheStats {
            total_blocks: block_pool.total_count(),
            free_blocks: block_pool.free_count(),
            allocated_blocks: block_table.len(),
            active_sequences: sequences.len(),
        }
    }
}

/// PagedAttention-specific cache statistics
#[derive(Debug, Clone)]
pub struct PagedCacheStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub active_sequences: usize,
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
        // FIX-10: With LRU eviction, max_pages=1 allows unlimited tokens via eviction
        let config = CacheConfig::new(4, 1, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a page first
        cache.allocate_page(1).unwrap();

        // Append tokens - first 4 fit in the page
        for i in 0..4 {
            let result = cache.append_token(1, i);
            assert!(result.is_ok());
        }

        // FIX-10: With LRU eviction, the 5th token triggers eviction and reallocation
        let result = cache.append_token(1, 5);
        assert!(result.is_ok()); // Now succeeds due to LRU eviction
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

        // FIX-10: With LRU eviction, third allocation should succeed by evicting LRU sequence
        let result = cache.allocate_page(3);
        assert!(result.is_ok()); // Now succeeds due to LRU eviction

        // Sequence 1 should have been evicted (LRU)
        assert!(cache.get_sequence_tokens(1).is_err());
    }

    // ========== PagedAttention Tests ==========

    #[test]
    fn test_physical_block_pool_allocation() {
        let backend = HipBackend::new().unwrap();
        let pool = PhysicalBlockPool::new(10, 16, 32, 128, &backend);
        assert!(pool.is_ok());

        let pool = pool.unwrap();
        assert_eq!(pool.total_count(), 10);
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn test_block_allocation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a block
        let block_id = cache.allocate_block(1);
        assert!(block_id.is_ok());
        assert_eq!(block_id.unwrap(), 0);

        // Check stats
        let stats = cache.get_paged_stats();
        assert_eq!(stats.total_blocks, 10);
        assert_eq!(stats.free_blocks, 9);
        assert_eq!(stats.allocated_blocks, 1);
    }

    #[test]
    fn test_block_ref_counting() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a block for sequence 1
        let block_id = cache.allocate_block(1).unwrap();

        // Get the block
        let block = cache.get_block(block_id);
        assert!(block.is_ok());
        assert_eq!(block.unwrap().ref_count(), 1);

        // Reference block for sequence 2 (sharing)
        let result = cache.ref_block(block_id, 2);
        assert!(result.is_ok());

        // Check ref count increased
        let block = cache.get_block(block_id).unwrap();
        assert_eq!(block.ref_count(), 2);
        assert!(block.sequences.contains(&1));
        assert!(block.sequences.contains(&2));
    }

    #[test]
    fn test_block_sharing_and_unreference() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a block for sequence 1
        let block_id = cache.allocate_block(1).unwrap();

        // Share with sequence 2
        cache.ref_block(block_id, 2).unwrap();

        // Unref sequence 1 - should NOT free (sequence 2 still holds ref)
        let freed = cache.unref_block(block_id, 1).unwrap();
        assert!(!freed);

        let stats = cache.get_paged_stats();
        assert_eq!(stats.allocated_blocks, 1);

        // Unref sequence 2 - SHOULD free now
        let freed = cache.unref_block(block_id, 2).unwrap();
        assert!(freed);

        let stats = cache.get_paged_stats();
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.free_blocks, 10); // All blocks free
    }

    #[test]
    fn test_block_capacity_limit() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 2, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate all blocks
        let _block1 = cache.allocate_block(1).unwrap();
        let _block2 = cache.allocate_block(2).unwrap();

        // Third allocation should fail
        let block3 = cache.allocate_block(3);
        assert!(block3.is_err());
        assert!(matches!(block3, Err(KvCacheError::CapacityExceeded)));
    }

    #[test]
    fn test_paged_cache_stats() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Initial stats
        let stats = cache.get_paged_stats();
        assert_eq!(stats.total_blocks, 10);
        assert_eq!(stats.free_blocks, 10);
        assert_eq!(stats.allocated_blocks, 0);

        // After allocation
        cache.allocate_block(1).unwrap();
        cache.allocate_block(2).unwrap();

        let stats = cache.get_paged_stats();
        assert_eq!(stats.total_blocks, 10);
        assert_eq!(stats.free_blocks, 8);
        assert_eq!(stats.allocated_blocks, 2);
    }

    #[test]
    fn test_block_id_recycling() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate and free a block
        let block_id = cache.allocate_block(1).unwrap();
        cache.unref_block(block_id, 1).unwrap();

        // Allocate another - should reuse the same ID
        let new_block_id = cache.allocate_block(2).unwrap();
        assert_eq!(new_block_id, block_id);
    }

    #[test]
    fn test_get_physical_block() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        let block_id = cache.allocate_block(1).unwrap();

        // Get physical block
        let physical = cache.get_physical_block(block_id);
        assert!(physical.is_ok());

        let physical = physical.unwrap();
        // PhysicalBlock.block_id is the physical block ID
        assert!(physical.size_bytes() > 0);
        assert!(physical.capacity_tokens() > 0);
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

    // ========== Phase 2: PageTable + BlockAllocator Integration Tests ==========

    #[test]
    fn test_append_token_paged_initial_allocation() {
        let backend = HipBackend::new().unwrap();
        // Use page_size=4 for faster testing
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // First token should allocate initial block
        cache.append_token_paged(1, 42).unwrap();

        // Check block allocator stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 10);
        assert_eq!(free, 9); // One block allocated

        // Check page table has the block
        let blocks = cache.get_sequence_blocks_from_page_table(1).unwrap();
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 1);
    }

    #[test]
    fn test_append_token_paged_multiple_blocks() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Append 9 tokens (should span 3 blocks of size 4)
        for i in 0..9 {
            cache.append_token_paged(1, i).unwrap();
        }

        // Check block allocator stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 10);
        assert_eq!(free, 7); // 3 blocks allocated (0, 4, 8 tokens trigger allocations)

        // Check page table has 3 blocks
        let blocks = cache.get_sequence_blocks_from_page_table(1).unwrap();
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 3);
    }

    #[test]
    fn test_get_block_for_position() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Append some tokens
        for i in 0..20 {
            cache.append_token_paged(1, i).unwrap();
        }

        // Position 0 should be in block 0, offset 0
        let result = cache.get_block_for_position(1, 0).unwrap();
        assert_eq!(result, Some((0, 0)));

        // Position 10 should be in block 0, offset 10
        let result = cache.get_block_for_position(1, 10).unwrap();
        assert_eq!(result, Some((0, 10)));

        // Position 16 should be in block 1, offset 0
        let result = cache.get_block_for_position(1, 16).unwrap();
        assert_eq!(result, Some((1, 0)));

        // Position 20 should be in block 1, offset 4 (block_size=16)
        // PageTable doesn't check actual token count - it maps positions to blocks
        let result = cache.get_block_for_position(1, 20).unwrap();
        assert_eq!(result, Some((1, 4)));

        // Position 32 would be in block 2 (but we only have 20 tokens)
        // PageTable will return None because block 2 doesn't exist
        let result = cache.get_block_for_position(1, 32).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_get_block_allocator_stats() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
        let cache = KvCache::new(config, backend).unwrap();

        // Initial stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 100);
        assert_eq!(free, 100);
    }

    #[test]
    fn test_multiple_sequences_paged() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 20, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Sequence 1: 5 tokens (2 blocks)
        for i in 0..5 {
            cache.append_token_paged(1, i).unwrap();
        }

        // Sequence 2: 3 tokens (1 block)
        for i in 0..3 {
            cache.append_token_paged(2, i).unwrap();
        }

        // Check block allocator stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 20);
        assert_eq!(free, 17); // 3 blocks allocated total

        // Verify each sequence has correct blocks
        let blocks1 = cache.get_sequence_blocks_from_page_table(1).unwrap();
        assert_eq!(blocks1.unwrap().len(), 2);

        let blocks2 = cache.get_sequence_blocks_from_page_table(2).unwrap();
        assert_eq!(blocks2.unwrap().len(), 1);
    }
}
