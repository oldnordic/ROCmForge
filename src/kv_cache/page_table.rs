//! PageTable for mapping logical sequence positions to physical KV cache blocks
//!
//! The PageTable maintains the mapping between logical token positions within sequences
//! and physical GPU memory blocks, enabling PagedAttention-style memory management.

use std::collections::HashMap;

/// Maps logical sequence positions to physical KV cache blocks
///
/// # Purpose
/// Maintains the mapping from logical token positions within sequences to physical
/// GPU memory blocks, enabling efficient memory access patterns for PagedAttention.
///
/// # Block Size
/// The block size determines how many tokens are stored per physical block. A larger
/// block size reduces metadata overhead but may waste memory for short sequences.
///
/// # Example
/// ```ignore
/// let mut pt = PageTable::new();
/// pt.append_block(1, 0);  // sequence_id=1, block_id=0
/// pt.append_block(1, 1);  // sequence_id=1, block_id=1
///
/// // Get physical block for token position 0 (in block 0)
/// let (block_id, offset) = pt.get_block_for_position(1, 0).unwrap();
/// assert_eq!(block_id, 0);
/// assert_eq!(offset, 0);
///
/// // Get physical block for token position 16 (in block 1, assuming block_size=16)
/// let (block_id, offset) = pt.get_block_for_position(1, 16).unwrap();
/// assert_eq!(block_id, 1);
/// assert_eq!(offset, 0);
/// ```
#[derive(Debug, Clone)]
pub struct PageTable {
    /// sequence_id -> Vec of block_id (block_index = vector position)
    tables: HashMap<u32, Vec<u32>>,

    block_size: usize,
}

impl PageTable {
    /// Create a new PageTable with default block size (16 tokens per block)
    ///
    /// # Returns
    /// A new empty PageTable
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            block_size: 16, // Configurable, default 16 tokens/block
        }
    }

    /// Create a new PageTable with a specific block size
    ///
    /// # Arguments
    /// * `block_size` - Number of tokens per block (must be > 0)
    ///
    /// # Returns
    /// A new empty PageTable with the specified block size
    pub fn with_block_size(block_size: usize) -> Self {
        assert!(block_size > 0, "Block size must be greater than 0");
        Self {
            tables: HashMap::new(),
            block_size,
        }
    }

    /// Add a new block to a sequence's page table
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to append the block to
    /// * `block_id` - The physical block ID to add
    ///
    /// # Behavior
    /// Appends the block to the end of the sequence's block list.
    /// The block index is implicitly the vector position.
    pub fn append_block(&mut self, sequence_id: u32, block_id: u32) {
        self.tables
            .entry(sequence_id)
            .or_insert_with(Vec::new)
            .push(block_id);
    }

    /// Get physical block for a logical token position
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to query
    /// * `token_pos` - The logical token position within the sequence
    ///
    /// # Returns
    /// * `Some((block_id, offset))` - The physical block ID and offset within that block
    /// * `None` - If the sequence doesn't exist or position is out of range
    ///
    /// # Example
    /// ```ignore
    /// let mut pt = PageTable::new();
    /// pt.append_block(1, 0);
    /// pt.append_block(1, 1);
    ///
    /// // Position 0-15 -> block 0
    /// assert_eq!(pt.get_block_for_position(1, 0), Some((0, 0)));
    /// assert_eq!(pt.get_block_for_position(1, 15), Some((0, 15)));
    ///
    /// // Position 16-31 -> block 1
    /// assert_eq!(pt.get_block_for_position(1, 16), Some((1, 0)));
    /// assert_eq!(pt.get_block_for_position(1, 31), Some((1, 15)));
    /// ```
    #[must_use]
    pub fn get_block_for_position(
        &self,
        sequence_id: u32,
        token_pos: usize,
    ) -> Option<(u32, usize)> {
        let block_idx = token_pos / self.block_size;
        let offset = token_pos % self.block_size;

        self.tables
            .get(&sequence_id)?
            .get(block_idx)
            .map(|&block_id| (block_id, offset))
    }

    /// Get all blocks for a sequence (in order)
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to query
    ///
    /// # Returns
    /// * `Some(&[u32])` - Slice of block IDs
    /// * `None` - If the sequence doesn't exist
    #[must_use]
    pub fn get_sequence_blocks(&self, sequence_id: u32) -> Option<&[u32]> {
        self.tables.get(&sequence_id).map(|v| v.as_slice())
    }

    /// Remove a sequence from page table
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to remove
    ///
    /// # Behavior
    /// Removes all block mappings for the sequence.
    /// This is typically called when a sequence is completed or evicted.
    pub fn remove_sequence(&mut self, sequence_id: u32) {
        self.tables.remove(&sequence_id);
    }

    /// Get number of sequences in page table
    ///
    /// # Returns
    /// The count of unique sequences with at least one block
    pub fn num_sequences(&self) -> usize {
        self.tables.len()
    }

    /// Get reference to internal tables (for profiling)
    ///
    /// # Returns
    /// Reference to the internal HashMap mapping sequence IDs to block lists
    #[must_use]
    pub fn tables(&self) -> &HashMap<u32, Vec<u32>> {
        &self.tables
    }
}

impl Default for PageTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_table_new() {
        let pt = PageTable::new();
        assert_eq!(pt.num_sequences(), 0);
    }

    #[test]
    fn test_page_table_with_block_size() {
        let pt = PageTable::with_block_size(32);
        assert_eq!(pt.num_sequences(), 0);
    }

    #[test]
    fn test_page_table_append_block() {
        let mut pt = PageTable::new();
        pt.append_block(1, 0); // sequence_id=1, block_id=0
        let blocks = pt.get_sequence_blocks(1);
        assert_eq!(blocks, Some(&[0u32][..]));
    }

    #[test]
    fn test_page_table_get_block_for_position() {
        let mut pt = PageTable::new();
        pt.append_block(1, 0);
        pt.append_block(1, 1);
        // Position 0-15 -> block 0, Position 16-31 -> block 1
        assert_eq!(pt.get_block_for_position(1, 0), Some((0, 0)));
        assert_eq!(pt.get_block_for_position(1, 16), Some((1, 0)));
    }

    #[test]
    fn test_page_table_remove_sequence() {
        let mut pt = PageTable::new();
        pt.append_block(1, 0);
        pt.remove_sequence(1);
        assert_eq!(pt.get_sequence_blocks(1), None);
    }

    #[test]
    fn test_page_table_multiple_sequences() {
        let mut pt = PageTable::new();
        pt.append_block(1, 0);
        pt.append_block(2, 1);
        assert_eq!(pt.num_sequences(), 2);
    }

    #[test]
    fn test_page_table_offset_calculation() {
        let mut pt = PageTable::new();
        pt.append_block(1, 0);
        pt.append_block(1, 1);

        // Test offsets in first block
        assert_eq!(pt.get_block_for_position(1, 0), Some((0, 0)));
        assert_eq!(pt.get_block_for_position(1, 5), Some((0, 5)));
        assert_eq!(pt.get_block_for_position(1, 15), Some((0, 15)));

        // Test offsets in second block
        assert_eq!(pt.get_block_for_position(1, 16), Some((1, 0)));
        assert_eq!(pt.get_block_for_position(1, 20), Some((1, 4)));
        assert_eq!(pt.get_block_for_position(1, 31), Some((1, 15)));
    }

    #[test]
    fn test_page_table_invalid_position() {
        let mut pt = PageTable::new();
        pt.append_block(1, 0);

        // Position beyond block size
        assert_eq!(pt.get_block_for_position(1, 16), None);
    }

    #[test]
    fn test_page_table_invalid_sequence() {
        let pt = PageTable::new();
        assert_eq!(pt.get_sequence_blocks(999), None);
        assert_eq!(pt.get_block_for_position(999, 0), None);
    }

    #[test]
    fn test_page_table_custom_block_size() {
        let mut pt = PageTable::with_block_size(8);
        pt.append_block(1, 0);
        pt.append_block(1, 1);

        // Position 0-7 -> block 0
        assert_eq!(pt.get_block_for_position(1, 0), Some((0, 0)));
        assert_eq!(pt.get_block_for_position(1, 7), Some((0, 7)));

        // Position 8-15 -> block 1
        assert_eq!(pt.get_block_for_position(1, 8), Some((1, 0)));
        assert_eq!(pt.get_block_for_position(1, 15), Some((1, 7)));
    }

    #[test]
    fn test_page_table_default() {
        let pt = PageTable::default();
        assert_eq!(pt.num_sequences(), 0);
    }
}
