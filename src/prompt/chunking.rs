//! Prompt chunking strategies for memory-efficient processing
//!
//! Long prompts can exceed GPU memory capacity when processed as a single batch.
//! Chunking divides the prompt into smaller segments that are processed sequentially,
//! reducing peak memory usage at the cost of some recomputation.

use crate::prompt::profiling::{AttentionPattern, estimate_attention_memory};

/// Strategy for chunking prompts during processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// No chunking - process entire prompt at once
    None,

    /// Fixed-size chunks
    Fixed { chunk_size: usize },

    /// Adaptive chunking based on memory constraints
    Adaptive { max_memory_mb: usize },

    /// Layer-by-layer chunking (minimal memory, maximum recomputation)
    LayerByLayer,
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::Fixed { chunk_size: 512 }
    }
}

impl ChunkStrategy {
    /// Create a fixed-size chunking strategy
    pub fn fixed(chunk_size: usize) -> Self {
        Self::Fixed { chunk_size }
    }

    /// Create an adaptive chunking strategy based on memory limit
    pub fn adaptive(max_memory_mb: usize) -> Self {
        Self::Adaptive { max_memory_mb }
    }

    /// Calculate optimal chunk size for given parameters
    pub fn calculate_chunk_size(
        &self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        _available_memory_mb: usize,
    ) -> usize {
        match self {
            ChunkStrategy::None => seq_len,
            ChunkStrategy::Fixed { chunk_size } => (*chunk_size).min(seq_len),
            ChunkStrategy::Adaptive { max_memory_mb } => {
                // Calculate memory per token for attention computation
                let memory_per_token = estimate_attention_memory(1, num_heads, head_dim);
                let max_memory_bytes = (*max_memory_mb) * 1024 * 1024;
                let max_tokens_by_memory = max_memory_bytes / memory_per_token.max(1);

                // Use adaptive chunk size based on memory constraint
                max_tokens_by_memory.min(seq_len)
            }
            ChunkStrategy::LayerByLayer => 1, // Process one token at a time
        }
    }

    /// Calculate number of chunks needed for a given sequence length
    pub fn num_chunks(&self, seq_len: usize, chunk_size: usize) -> usize {
        if chunk_size == 0 || seq_len == 0 {
            return 0;
        }
        (seq_len + chunk_size - 1) / chunk_size
    }

    /// Estimate memory savings from chunking
    pub fn memory_savings_ratio(
        &self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        pattern: AttentionPattern,
    ) -> f32 {
        let chunk_size = self.calculate_chunk_size(seq_len, num_heads, head_dim, 0);
        if seq_len <= chunk_size {
            return 0.0;
        }

        // Base memory without chunking scales with O(seq_len^2) for attention
        let original_memory = estimate_attention_memory(seq_len, num_heads, head_dim);

        // With chunking, memory is dominated by the largest chunk
        let chunked_memory = estimate_attention_memory(chunk_size, num_heads, head_dim);

        // Additional savings from pattern-specific optimizations
        let pattern_savings = pattern.chunking_memory_reduction(chunk_size, seq_len);

        let base_savings = 1.0 - (chunked_memory as f32 / original_memory as f32);
        (base_savings + pattern_savings).min(0.9) // Cap at 90% savings
    }
}

/// Prompt chunker that divides prompts into processable segments
pub struct PromptChunker {
    strategy: ChunkStrategy,
    overlap_size: usize,
}

impl PromptChunker {
    /// Create a new prompt chunker
    pub fn new(strategy: ChunkStrategy) -> Self {
        PromptChunker {
            strategy,
            overlap_size: 0,
        }
    }

    /// Set overlap between chunks (for context preservation)
    ///
    /// Larger overlaps preserve more context but increase memory usage.
    /// For causal attention, overlap allows proper computation for tokens at chunk boundaries.
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap_size = overlap;
        self
    }

    /// Calculate optimal chunking for a given prompt
    pub fn calculate_chunks(
        &self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        available_memory_mb: usize,
    ) -> Vec<Chunk> {
        if seq_len == 0 {
            return Vec::new();
        }

        let chunk_size = self.strategy.calculate_chunk_size(
            seq_len,
            num_heads,
            head_dim,
            available_memory_mb,
        );

        if chunk_size >= seq_len {
            // Single chunk for entire prompt
            return vec![Chunk {
                start: 0,
                end: seq_len,
                num_kv_tokens: seq_len, // Full KV cache available
                is_first: true,
                is_last: true,
            }];
        }

        let num_chunks = self.strategy.num_chunks(seq_len, chunk_size);
        let mut chunks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(seq_len);

            // Add overlap (except for first chunk)
            let overlap_start = if i > 0 {
                start.saturating_sub(self.overlap_size)
            } else {
                0
            };

            // Number of KV tokens available (all previous tokens)
            let num_kv_tokens = end; // Causal: all tokens up to current position

            chunks.push(Chunk {
                start: overlap_start,
                end,
                num_kv_tokens,
                is_first: i == 0,
                is_last: i == num_chunks - 1,
            });
        }

        chunks
    }

    /// Get chunking strategy
    pub fn strategy(&self) -> ChunkStrategy {
        self.strategy
    }
}

impl Default for PromptChunker {
    fn default() -> Self {
        Self::new(ChunkStrategy::default())
    }
}

/// Represents a single chunk of a prompt
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Start token index (inclusive)
    pub start: usize,

    /// End token index (exclusive)
    pub end: usize,

    /// Number of tokens in KV cache available for this chunk
    pub num_kv_tokens: usize,

    /// Whether this is the first chunk
    pub is_first: bool,

    /// Whether this is the last chunk
    pub is_last: bool,
}

impl Chunk {
    /// Number of tokens to process in this chunk
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether this chunk is empty
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Create a single chunk for the entire prompt
    pub fn full(seq_len: usize) -> Self {
        Chunk {
            start: 0,
            end: seq_len,
            num_kv_tokens: seq_len,
            is_first: true,
            is_last: true,
        }
    }

    /// Token range as a tuple
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

/// Iterator over prompt chunks
pub struct ChunkIterator {
    chunks: Vec<Chunk>,
    index: usize,
}

impl ChunkIterator {
    pub fn new(chunks: Vec<Chunk>) -> Self {
        ChunkIterator { chunks, index: 0 }
    }
}

impl Iterator for ChunkIterator {
    type Item = Chunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.chunks.len() {
            let chunk = self.chunks[self.index].clone();
            self.index += 1;
            Some(chunk)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.chunks.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

/// Memory-efficient chunking with KV cache reuse
///
/// This strategy minimizes recomputation by reusing KV cache from previous chunks.
pub struct IncrementalChunker {
    base_chunk_size: usize,
    min_chunk_size: usize,
}

impl IncrementalChunker {
    /// Create a new incremental chunker
    pub fn new(base_chunk_size: usize) -> Self {
        IncrementalChunker {
            base_chunk_size,
            min_chunk_size: 64, // Minimum chunk size for efficiency
        }
    }

    /// Set minimum chunk size
    pub fn with_min_chunk_size(mut self, min_size: usize) -> Self {
        self.min_chunk_size = min_size;
        self
    }

    /// Calculate chunks with KV cache reuse
    ///
    /// Returns chunks where each chunk can reuse KV cache from all previous chunks.
    pub fn calculate_chunks(&self, seq_len: usize) -> Vec<Chunk> {
        if seq_len <= self.base_chunk_size {
            return vec![Chunk::full(seq_len)];
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_num = 0;

        while start < seq_len {
            let remaining = seq_len - start;
            // For the last chunk, don't force min_chunk_size
            let chunk_size = if remaining < self.base_chunk_size {
                remaining
            } else {
                self.base_chunk_size
            };
            let end = start + chunk_size;

            chunks.push(Chunk {
                start,
                end,
                num_kv_tokens: end, // Full KV cache available up to this point
                is_first: chunk_num == 0,
                is_last: end == seq_len,
            });

            start = end;
            chunk_num += 1;
        }

        chunks
    }
}

impl Default for IncrementalChunker {
    fn default() -> Self {
        Self::new(512)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_strategy_fixed() {
        let strategy = ChunkStrategy::fixed(256);
        let chunk_size = strategy.calculate_chunk_size(1024, 32, 128, 0);

        assert_eq!(chunk_size, 256);
    }

    #[test]
    fn test_chunk_strategy_none() {
        let strategy = ChunkStrategy::None;
        let chunk_size = strategy.calculate_chunk_size(1024, 32, 128, 0);

        assert_eq!(chunk_size, 1024);
    }

    #[test]
    fn test_num_chunks() {
        let strategy = ChunkStrategy::fixed(100);
        assert_eq!(strategy.num_chunks(0, 100), 0);
        assert_eq!(strategy.num_chunks(100, 100), 1);
        assert_eq!(strategy.num_chunks(150, 100), 2);
        assert_eq!(strategy.num_chunks(250, 100), 3);
    }

    #[test]
    fn test_chunker_single_chunk() {
        let chunker = PromptChunker::new(ChunkStrategy::fixed(1024));
        let chunks = chunker.calculate_chunks(512, 32, 128, 0);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 512);
    }

    #[test]
    fn test_chunker_multiple_chunks() {
        let chunker = PromptChunker::new(ChunkStrategy::fixed(100));
        let chunks = chunker.calculate_chunks(250, 32, 128, 0);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 100);
        assert_eq!(chunks[1].start, 100);
        assert_eq!(chunks[1].end, 200);
        assert_eq!(chunks[2].start, 200);
        assert_eq!(chunks[2].end, 250);
    }

    #[test]
    fn test_chunker_with_overlap() {
        let chunker = PromptChunker::new(ChunkStrategy::fixed(100)).with_overlap(10);
        let chunks = chunker.calculate_chunks(250, 32, 128, 0);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].start, 0); // First chunk has no overlap
        assert_eq!(chunks[1].start, 90); // Overlap of 10
        assert_eq!(chunks[2].start, 190); // Overlap of 10
    }

    #[test]
    fn test_incremental_chunker() {
        let chunker = IncrementalChunker::new(100);
        let chunks = chunker.calculate_chunks(250);

        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].is_first);
        assert!(chunks[2].is_last);

        // Each chunk has full KV cache available
        assert_eq!(chunks[0].num_kv_tokens, 100);
        assert_eq!(chunks[1].num_kv_tokens, 200);
        assert_eq!(chunks[2].num_kv_tokens, 250);
    }

    #[test]
    fn test_memory_savings() {
        let strategy = ChunkStrategy::fixed(256);
        let savings = strategy.memory_savings_ratio(1024, 32, 128, AttentionPattern::Dense);

        assert!(savings > 0.0);
        assert!(savings < 1.0);
    }

    #[test]
    fn test_chunk_range() {
        let chunk = Chunk {
            start: 100,
            end: 200,
            num_kv_tokens: 200,
            is_first: false,
            is_last: false,
        };

        assert_eq!(chunk.len(), 100);
        assert!(!chunk.is_empty());
        assert_eq!(chunk.range(), 100..200);
    }

    #[test]
    fn test_chunk_iterator() {
        let chunks = vec![
            Chunk { start: 0, end: 100, num_kv_tokens: 100, is_first: true, is_last: false },
            Chunk { start: 100, end: 200, num_kv_tokens: 200, is_first: false, is_last: true },
        ];

        let mut iter = ChunkIterator::new(chunks);
        assert_eq!(iter.size_hint(), (2, Some(2)));

        let first = iter.next().unwrap();
        assert_eq!(first.len(), 100);
        assert!(first.is_first);

        let second = iter.next().unwrap();
        assert_eq!(second.len(), 100);
        assert!(second.is_last);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_chunk_full() {
        let chunk = Chunk::full(512);
        assert_eq!(chunk.start, 0);
        assert_eq!(chunk.end, 512);
        assert_eq!(chunk.num_kv_tokens, 512);
        assert!(chunk.is_first);
        assert!(chunk.is_last);
    }
}
