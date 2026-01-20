//! Memory requirements calculator for model loading
//!
//! This module provides utilities to calculate the exact memory requirements
//! for loading a model before any GPU allocation occurs. This enables:
//! - Pre-flight memory checks (fail fast if insufficient memory)
//! - Safety margin calculation (10% for driver overhead)
//! - Detailed logging of memory needs

/// Calculate memory requirements for model loading
///
/// Used to determine total GPU memory needed before any allocation.
/// This prevents partial model loads and ensures clean error messages
/// when GPU memory is insufficient.
///
/// # Example
/// ```ignore
/// let mut calc = MemoryCalculator::new();
/// calc.add_tensor("tok_embeddings".to_string(), 32000 * 4096, 4);
/// calc.add_tensor("layer.0.weight".to_string(), 4096 * 4096, 4);
///
/// let total_bytes = calc.total_bytes();
/// println!("Need {} MB", total_bytes / 1024 / 1024);
/// ```
#[derive(Debug, Clone)]
pub struct MemoryCalculator {
    tensor_sizes: Vec<(String, usize)>,
    alignment: usize,
}

impl MemoryCalculator {
    /// Default alignment for GPU memory allocations
    ///
    /// 256-byte alignment is typical for GPU memory and matches
    /// the alignment used by ModelWeightArena.
    pub const DEFAULT_ALIGNMENT: usize = 256;

    /// Create a new memory calculator
    pub fn new() -> Self {
        Self {
            tensor_sizes: Vec::new(),
            alignment: Self::DEFAULT_ALIGNMENT,
        }
    }

    /// Create a new calculator with custom alignment
    pub fn with_alignment(alignment: usize) -> Self {
        Self {
            tensor_sizes: Vec::new(),
            alignment,
        }
    }

    /// Add a tensor to the memory calculation
    ///
    /// # Arguments
    /// * `name` - Tensor name (for debugging/logging)
    /// * `element_count` - Number of elements in the tensor
    /// * `element_size` - Size of each element in bytes (e.g., 4 for f32)
    ///
    /// # Note
    /// The size is aligned to the configured alignment boundary.
    /// This matches the actual allocation behavior of GPU memory.
    pub fn add_tensor(&mut self, name: String, element_count: usize, element_size: usize) {
        let bytes = element_count.saturating_mul(element_size);
        let aligned = (bytes + self.alignment - 1) & !(self.alignment - 1);
        self.tensor_sizes.push((name, aligned));
    }

    /// Calculate total bytes needed for all tensors
    ///
    /// Returns the sum of all aligned tensor sizes.
    /// This is the exact amount of GPU memory that will be allocated.
    pub fn total_bytes(&self) -> usize {
        self.tensor_sizes.iter().map(|(_, size)| size).sum()
    }

    /// Get the number of tensors tracked
    pub fn tensor_count(&self) -> usize {
        self.tensor_sizes.len()
    }

    /// Get individual tensor sizes (for debugging)
    pub fn tensor_sizes(&self) -> &[(String, usize)] {
        &self.tensor_sizes
    }

    /// Calculate memory needed with safety margin
    ///
    /// Adds 10% safety margin for driver overhead and alignment padding.
    /// This ensures we don't exhaust GPU memory during allocation.
    ///
    /// # Returns
    /// Total bytes needed including safety margin
    pub fn total_bytes_with_margin(&self) -> usize {
        let base = self.total_bytes();
        let margin = (base / 10).max(1024 * 1024 * 100); // 10% + min 100MB
        base + margin
    }

    /// Get the safety margin in bytes
    pub fn safety_margin(&self) -> usize {
        let base = self.total_bytes();
        (base / 10).max(1024 * 1024 * 100)
    }

    /// Clear all tracked tensors (for reuse)
    pub fn clear(&mut self) {
        self.tensor_sizes.clear();
    }
}

impl Default for MemoryCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_calculator() {
        let calc = MemoryCalculator::new();
        assert_eq!(calc.total_bytes(), 0);
        assert_eq!(calc.tensor_count(), 0);
    }

    #[test]
    fn test_single_tensor() {
        let mut calc = MemoryCalculator::new();
        calc.add_tensor("test".to_string(), 1000, 4);
        // 1000 * 4 = 4000, aligned to 4096 (256-byte boundary)
        assert_eq!(calc.total_bytes(), 4096);
    }

    #[test]
    fn test_multiple_tensors() {
        let mut calc = MemoryCalculator::new();
        calc.add_tensor("tensor1".to_string(), 1000, 4);
        calc.add_tensor("tensor2".to_string(), 2000, 4);
        assert_eq!(calc.tensor_count(), 2);
        // tensor1: 4000 -> 4096
        // tensor2: 8000 -> 8192
        assert_eq!(calc.total_bytes(), 12288);
    }

    #[test]
    fn test_safety_margin() {
        let mut calc = MemoryCalculator::new();
        calc.add_tensor("large".to_string(), 1024 * 1024 * 100, 4); // 400MB
        let base = calc.total_bytes();
        let with_margin = calc.total_bytes_with_margin();

        // Should add 10% + min 100MB
        assert!(with_margin > base);
        let margin = with_margin - base;
        assert!(margin >= 1024 * 1024 * 100); // At least 100MB
    }

    #[test]
    fn test_small_tensor_margin() {
        let mut calc = MemoryCalculator::new();
        calc.add_tensor("small".to_string(), 100, 4); // 400 bytes
        let with_margin = calc.total_bytes_with_margin();
        let margin = calc.safety_margin();

        // Small tensors should still get at least 100MB margin
        assert_eq!(margin, 1024 * 1024 * 100);
    }

    #[test]
    fn test_clear() {
        let mut calc = MemoryCalculator::new();
        calc.add_tensor("test".to_string(), 1000, 4);
        assert_eq!(calc.tensor_count(), 1);

        calc.clear();
        assert_eq!(calc.tensor_count(), 0);
        assert_eq!(calc.total_bytes(), 0);
    }

    #[test]
    fn test_alignment() {
        let mut calc = MemoryCalculator::with_alignment(512);
        calc.add_tensor("test".to_string(), 100, 4); // 400 bytes
        // Aligned to 512 bytes
        assert_eq!(calc.total_bytes(), 512);
    }
}
