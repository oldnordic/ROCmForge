//! Memory-mapped weight loading for zero-copy model initialization

use std::fs::File;
use std::path::Path;
use std::slice;
use thiserror::Error;

/// Error types for mmap operations
#[derive(Debug, Error)]
pub enum MmapError {
    #[error("Failed to open file: {0}")]
    FileOpenError(#[from] std::io::Error),

    #[error("Failed to create memory map: {0}")]
    MmapError(String),

    #[error("Invalid range: start {start}, end {end}, length {length}")]
    InvalidRange {
        start: usize,
        end: usize,
        length: usize,
    },

    #[error("Alignment error: offset {offset} not aligned to {alignment}")]
    AlignmentError { offset: usize, alignment: usize },
}

pub type MmapResult<T> = Result<T, MmapError>;

/// Memory-mapped weights with zero-copy access
pub struct MmapWeights {
    _mmap: memmap2::Mmap,
    data: *const u8,
    length: usize,
}

// SAFETY: MmapWeights is Send+Sync because the underlying Mmap is Send+Sync
// and we only provide read-only access to the data
unsafe impl Send for MmapWeights {}
unsafe impl Sync for MmapWeights {}

impl MmapWeights {
    /// Get the raw byte data
    pub fn data(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data, self.length) }
    }

    /// Get the length of the mapped data
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the mapped data is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get a typed f32 view of the data
    pub fn view_f32(&self, range: std::ops::Range<usize>) -> &[f32] {
        // Use checked arithmetic to prevent overflow
        let start_byte = range.start.checked_mul(4).unwrap_or(usize::MAX);
        let end_byte = range.end.checked_mul(4).unwrap_or(usize::MAX);

        // Validate range bounds
        if start_byte > self.length || end_byte > self.length {
            return &[];
        }

        // Ensure we don't go beyond available data
        let actual_end_byte = std::cmp::min(end_byte, self.length);
        let actual_len = actual_end_byte.saturating_sub(start_byte);

        if actual_len == 0 {
            return &[];
        }

        // Validate alignment - f32 requires 4-byte alignment
        if !start_byte.is_multiple_of(4) {
            return &[];
        }

        let byte_slice = &self.data()[start_byte..actual_end_byte];

        // Safe transmute from bytes to f32 slice
        unsafe {
            let ptr = byte_slice.as_ptr() as *const f32;
            let len = actual_len / 4;
            slice::from_raw_parts(ptr, len)
        }
    }
}

/// Open a weights file with memory mapping
pub fn open_mmap_weights<P: AsRef<Path>>(path: P) -> MmapResult<MmapWeights> {
    let file = File::open(path.as_ref()).map_err(MmapError::FileOpenError)?;

    // Get file size
    let _file_size = file.metadata().map_err(MmapError::FileOpenError)?.len() as usize;

    // Use memmap2 crate for memory mapping
    use memmap2::Mmap;

    let mmap = unsafe { Mmap::map(&file).map_err(|e| MmapError::MmapError(e.to_string()))? };

    let data = mmap.as_ptr();
    let length = mmap.len();

    Ok(MmapWeights {
        _mmap: mmap,
        data,
        length,
    })
}

/// Stable tensor shape with stride computation
#[derive(Debug, Clone)]
pub struct TensorShape {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorShape {
    /// Create tensor shape from dimensions, computing row-major strides
    pub fn from_dims(dims: &[usize]) -> Self {
        let mut strides = Vec::with_capacity(dims.len());

        if dims.is_empty() {
            return Self {
                dims: dims.to_vec(),
                strides: vec![],
            };
        }

        // Compute strides in row-major order (last dimension varies fastest)
        for i in (0..dims.len()).rev() {
            let stride = if i == dims.len() - 1 {
                1 // Last dimension has stride 1
            } else {
                // Use checked multiplication to prevent overflow
                dims[i + 1..]
                    .iter()
                    .copied()
                    .fold(1usize, |acc, x| acc.checked_mul(x).unwrap_or(usize::MAX))
            };
            strides.push(stride);
        }

        // Reverse to get correct order (first to last)
        strides.reverse();

        Self {
            dims: dims.to_vec(),
            strides,
        }
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Compute total number of elements
    pub fn total_elements(&self) -> usize {
        self.dims
            .iter()
            .copied()
            .fold(1usize, |acc, x| acc.checked_mul(x).unwrap_or(usize::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_stride_computation() {
        let dims = [2, 3, 4];
        let shape = TensorShape::from_dims(&dims);

        // Expected strides for row-major: [12, 4, 1]
        // stride[0] = 3*4 = 12, stride[1] = 4, stride[2] = 1
        assert_eq!(shape.strides(), vec![12, 4, 1]);
    }
}
