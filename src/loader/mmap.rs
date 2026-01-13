//! Memory-mapped GGUF file for zero-copy access
//!
//! This module provides lazy loading capabilities for GGUF model files.
//! Instead of reading all tensor data into RAM immediately, we memory-map
//! the file and load tensors on-demand when they're first accessed.

use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Memory-mapped GGUF file
///
/// Provides zero-copy access to GGUF file contents. The file is mapped
/// into virtual memory, and the OS handles loading pages on-demand.
///
/// # Benefits
///
/// - **Fast initialization**: Only metadata is read upfront
/// - **Low memory usage**: Unused tensors are never loaded into RAM
/// - **OS-managed caching**: The OS handles page caching efficiently
///
/// # Thread Safety
///
/// `MmapGguf` is `Send + Sync` because `memmap2::Mmap` is `Send + Sync`
/// and we only provide read-only access to the mapped data.
///
/// # Example
///
/// ```rust
/// use crate::loader::mmap::MmapGguf;
///
/// let mmap = MmapGguf::open(Path::new("model.gguf"))?;
/// let tensor_bytes = mmap.get_slice(offset, size)?;
/// ```
#[derive(Debug)]
pub struct MmapGguf {
    _file: File,
    mmap: Mmap,
}

// SAFETY: MmapGguf is Send+Sync because memmap2::Mmap is Send+Sync
// and we only provide read-only access to the mapped data.
unsafe impl Send for MmapGguf {}
unsafe impl Sync for MmapGguf {}

impl MmapGguf {
    /// Open and memory-map GGUF file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `MmapGguf` instance or an error.
    pub fn open(path: &Path) -> Result<Self> {
        tracing::debug!("Opening GGUF file for memory-mapping: {:?}", path);

        let file = File::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open GGUF file '{}': {}", path.display(), e))?;

        // Use unsafe mmap for read-only access (safe because we're only reading)
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            anyhow::anyhow!("Failed to memory-map GGUF file '{}': {}", path.display(), e)
        })?;

        tracing::debug!("Memory-mapped GGUF file: {} bytes", mmap.len());

        Ok(Self { _file: file, mmap })
    }

    /// Get slice of file bytes without copying
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset from start of file
    /// * `size` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a slice of the file contents or an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested slice extends beyond the file bounds.
    pub fn get_slice(&self, offset: u64, size: usize) -> Result<&[u8]> {
        let start = offset as usize;
        let end = start.saturating_add(size);

        if end > self.mmap.len() {
            return Err(anyhow::anyhow!(
                "Slice out of bounds: offset={}, size={}, requested range={}..{}, file size={}",
                offset,
                size,
                start,
                end,
                self.mmap.len()
            ));
        }

        Ok(&self.mmap[start..end])
    }

    /// Get full file bytes
    ///
    /// # Returns
    ///
    /// Returns a slice containing the entire file contents.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Get file size in bytes
    ///
    /// # Returns
    ///
    /// Returns the size of the memory-mapped file.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if file is empty
    ///
    /// # Returns
    ///
    /// Returns `true` if the file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_gguf_open() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"GGUF").unwrap();

        let mmap = MmapGguf::open(temp_file.path());
        assert!(mmap.is_ok());
    }

    #[test]
    fn test_mmap_gguf_get_slice() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = b"GGUFtest_data_for_slice";
        temp_file.write_all(test_data).unwrap();

        let mmap = MmapGguf::open(temp_file.path()).unwrap();

        // Test valid slice
        let slice = mmap.get_slice(4, 4).unwrap();
        assert_eq!(slice, b"test");

        // Test slice at offset 8
        let slice = mmap.get_slice(8, 4).unwrap();
        assert_eq!(slice, b"_dat");

        // Test invalid slice (out of bounds)
        let result = mmap.get_slice(0, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_gguf_len() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = b"GGUF12345678";
        temp_file.write_all(test_data).unwrap();

        let mmap = MmapGguf::open(temp_file.path()).unwrap();
        assert_eq!(mmap.len(), 12);
        assert!(!mmap.is_empty());
    }
}
