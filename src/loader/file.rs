//! GgufFile — the public API for loading GGUF models.
//!
//! Opens the file once, parses header/metadata/tensor descriptors, then
//! memory-maps the whole file. Tensor data is exposed as zero-copy
//! byte slices into the mapping — no copies until the GPU upload step.

use crate::loader::error::LoadError;
use crate::loader::ggml_type::GgmlType;
use crate::loader::metadata::GgufMetadata;
use crate::loader::parse::{parse_header, parse_kv, parse_tensor_descs, TensorDesc, TokenizerData};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// An open GGUF file. Holds the memory map and parsed metadata.
/// All `TensorView`s borrow from this struct.
pub struct GgufFile {
    _file: File, // kept alive so the mmap remains valid
    mmap: Mmap,
    pub metadata: GgufMetadata,
    tokenizer: TokenizerData,
    descs: HashMap<String, TensorDesc>,
    data_start: u64,
}

/// Zero-copy view of one tensor's data inside the mmap.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub name: &'a str,
    /// Dimensions in GGUF order (innermost first, i.e. columns before rows for 2-D weights)
    pub dims: &'a [u64],
    pub ggml_type: GgmlType,
    /// Raw bytes of the tensor data — slice directly into the mmap
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Returns the total number of elements in this tensor.
    pub fn element_count(&self) -> usize {
        self.dims.iter().fold(1usize, |acc, &d| acc * d as usize)
    }
}

impl GgufFile {
    /// Open a GGUF file, parse its structure, and memory-map it.
    ///
    /// Does not copy any tensor data — that happens later when the GPU
    /// uploads each tensor directly from the mmap.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();

        // Parse header + metadata + tensor descriptors sequentially
        let file = File::open(path)?;
        let mut reader = BufReader::new(&file);
        let header = parse_header(&mut reader)?;
        let (metadata, tokenizer) = parse_kv(&mut reader, header.kv_count)?;
        let (descs_vec, data_start) = parse_tensor_descs(&mut reader, header.tensor_count)?;
        drop(reader);

        // Memory-map the whole file
        // SAFETY: read-only mapping; we never write through it
        let mmap = unsafe { Mmap::map(&file) }?;

        let descs = descs_vec.into_iter().map(|d| (d.name.clone(), d)).collect();

        Ok(Self {
            _file: file,
            mmap,
            metadata,
            tokenizer,
            descs,
            data_start,
        })
    }

    /// Iterate over all tensor names in the order they appear in the file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.descs.keys().map(|s| s.as_str())
    }

    /// Look up a tensor by name and return a zero-copy view of its data.
    ///
    /// Returns `None` if no tensor with that name exists in the file.
    /// Returns `Err` if the tensor's byte range falls outside the file
    /// (indicates a corrupt GGUF file).
    pub fn tensor(&self, name: &str) -> Result<Option<TensorView<'_>>, LoadError> {
        let Some(desc) = self.descs.get(name) else {
            return Ok(None);
        };

        let start = (self.data_start + desc.offset) as usize;
        let size = desc.byte_size();
        let end = start + size;

        if end > self.mmap.len() {
            return Err(LoadError::OutOfBounds {
                offset: self.data_start + desc.offset,
                size,
                file_size: self.mmap.len(),
            });
        }

        Ok(Some(TensorView {
            name: &desc.name,
            dims: &desc.dims,
            ggml_type: desc.ggml_type,
            data: &self.mmap[start..end],
        }))
    }

    /// Number of tensors in the file.
    pub fn tensor_count(&self) -> usize {
        self.descs.len()
    }

    /// Check whether a tensor exists without loading its data.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.descs.contains_key(name)
    }

    /// Raw tokenizer arrays extracted from GGUF KV section.
    /// Pass to `BpeTokenizer::from_gguf()` to build a tokenizer.
    pub fn tokenizer_data(&self) -> &TokenizerData {
        &self.tokenizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_view_element_count_1d() {
        let view = TensorView {
            name: "test",
            dims: &[100],
            ggml_type: GgmlType::F32,
            data: &[],
        };
        assert_eq!(view.element_count(), 100);
    }

    #[test]
    fn tensor_view_element_count_2d() {
        let view = TensorView {
            name: "test",
            dims: &[64, 128],
            ggml_type: GgmlType::F32,
            data: &[],
        };
        assert_eq!(view.element_count(), 8192);
    }

    #[test]
    fn tensor_view_element_count_3d() {
        let view = TensorView {
            name: "test",
            dims: &[2, 3, 4],
            ggml_type: GgmlType::F32,
            data: &[],
        };
        assert_eq!(view.element_count(), 24);
    }
}
