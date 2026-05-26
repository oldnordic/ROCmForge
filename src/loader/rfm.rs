//! ROCmForge Model (.rfm) Loader
//!
//! Exposes a memory-mapped, zero-copy loader for RFM models, parsing metadata,
//! tensor index table, and returning tensor views aligned to 256-byte boundaries.

use crate::loader::error::LoadError;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Types of tensors supported in the ROCmForge format.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RfmType {
    /// Contiguous F32 array (norms, biases).
    F32,
    /// Aligned split-array Q4 layout (Scales, ZPs, Nibbles separate).
    Q4Split,
    /// Aligned fused-FFN Gate-Up Q4 layout (interleaved Gate & Up).
    Q4FusedGateUp,
    /// Passthrough GGUF tensor format (stores raw GGUF bytes).
    GgufPassthrough(u32),
}

/// An entry in the .rfm tensor table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfmTensorEntry {
    pub name: String,
    pub dims: Vec<u64>,
    pub wtype: RfmType,
    pub offset: u64,
    pub size: u64,
}

/// Serializable metadata containing model configurations and the full tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfmMetadata {
    // Model configuration
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_neox: bool,
    pub use_attention_bias: bool,
    pub architecture: String,

    // Tokenizer vocabulary (extracted from GGUF KVs)
    pub tokens: Vec<Vec<u8>>,
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub tokenizer_model: Option<String>,
    pub tokenizer_pre: Option<String>,
    pub add_bos: bool,
    pub add_eos: bool,
}

/// Zero-copy view of one tensor's data inside the mmap of an RFM file.
#[derive(Debug, Clone, Copy)]
pub struct RfmTensorView<'a> {
    pub name: &'a str,
    pub dims: &'a [u64],
    pub wtype: RfmType,
    /// Raw bytes of the tensor data - slice directly into the mmap
    pub data: &'a [u8],
}

impl<'a> RfmTensorView<'a> {
    /// Returns the total number of elements in this tensor.
    pub fn element_count(&self) -> usize {
        self.dims.iter().fold(1usize, |acc, &d| acc * d as usize)
    }
}

/// An open ROCmForge Model (.rfm) file. Holds the memory map and parsed metadata.
pub struct RfmFile {
    _file: File, // kept alive so the mmap remains valid
    mmap: Mmap,
    pub metadata: RfmMetadata,
    descs: HashMap<String, RfmTensorEntry>,
    payload_start: u64,
}

impl RfmFile {
    /// Open an RFM file, parse its structure, and memory-map it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let file = File::open(path)?;

        // Memory-map the file
        let mmap = unsafe { Mmap::map(&file) }?;
        let len = mmap.len();

        if len < 24 {
            return Err(LoadError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "RFM file too short (less than 24 bytes header)",
            )));
        }

        // Parse header (24 bytes)
        let magic = &mmap[0..4];
        if magic != b"RFM\0" {
            let mut got_magic = [0u8; 4];
            got_magic.copy_from_slice(magic);
            return Err(LoadError::InvalidMagic(got_magic));
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != 1 {
            return Err(LoadError::UnsupportedVersion(version));
        }

        let metadata_size = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let tensor_table_size = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;

        if len < 24 + metadata_size + tensor_table_size {
            return Err(LoadError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "RFM file truncated before tensor table end",
            )));
        }

        // Read metadata JSON string and deserialize
        let metadata_bytes = &mmap[24..24 + metadata_size];
        let metadata: RfmMetadata = serde_json::from_slice(metadata_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Read tensor table JSON string (allocated 64KB, but we trim whitespace)
        let table_bytes = &mmap[24 + metadata_size..24 + metadata_size + tensor_table_size];
        let trimmed_table = match table_bytes.iter().rposition(|&b| b != b' ') {
            Some(pos) => &table_bytes[..=pos],
            None => table_bytes,
        };

        let entries: Vec<RfmTensorEntry> = serde_json::from_slice(trimmed_table)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let payload_start = (24 + metadata_size + tensor_table_size) as u64;
        let descs = entries.into_iter().map(|e| (e.name.clone(), e)).collect();

        Ok(Self {
            _file: file,
            mmap,
            metadata,
            descs,
            payload_start,
        })
    }

    /// Look up a tensor by name and return a zero-copy view of its data.
    pub fn tensor(&self, name: &str) -> Result<Option<RfmTensorView<'_>>, LoadError> {
        let Some(desc) = self.descs.get(name) else {
            return Ok(None);
        };

        let start = (self.payload_start + desc.offset) as usize;
        let size = desc.size as usize;
        let end = start + size;

        if end > self.mmap.len() {
            return Err(LoadError::OutOfBounds {
                offset: self.payload_start + desc.offset,
                size,
                file_size: self.mmap.len(),
            });
        }

        Ok(Some(RfmTensorView {
            name: &desc.name,
            dims: &desc.dims,
            wtype: desc.wtype,
            data: &self.mmap[start..end],
        }))
    }

    /// Check whether a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.descs.contains_key(name)
    }

    /// Iterate over all tensor names in the file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.descs.keys().map(|s| s.as_str())
    }

    /// Number of tensors in the file.
    pub fn tensor_count(&self) -> usize {
        self.descs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_rfm_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_model.rfm");

        // Prepare mock metadata
        let metadata = RfmMetadata {
            num_layers: 12,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: false,
            architecture: "llama".to_string(),
            tokens: vec![b"hello".to_vec(), b"world".to_vec()],
            merges: vec![],
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unk_token_id: Some(0),
            tokenizer_model: Some("gpt2".to_string()),
            tokenizer_pre: None,
            add_bos: true,
            add_eos: false,
        };

        let metadata_bytes = serde_json::to_vec(&metadata).unwrap();
        let metadata_size = metadata_bytes.len() as u64;

        // Prepare mock tensor table
        let entries = vec![
            RfmTensorEntry {
                name: "token_embd.weight".to_string(),
                dims: vec![4096, 32000],
                wtype: RfmType::Q4Split,
                offset: 0,
                size: 256,
            },
            RfmTensorEntry {
                name: "output_norm.weight".to_string(),
                dims: vec![4096],
                wtype: RfmType::F32,
                offset: 256,
                size: 16384,
            },
        ];

        let table_bytes = serde_json::to_vec(&entries).unwrap();
        let mut table_payload = table_bytes.clone();
        // preallocate padding as whitespace up to 1024 bytes for mock
        table_payload.resize(1024, b' ');
        let tensor_table_allocated_size = table_payload.len() as u64;

        // Prepare mock payload data
        let mut payload = vec![0u8; 256 + 16384];
        // fill with some unique values
        for i in 0..payload.len() {
            payload[i] = (i % 251) as u8;
        }

        // Write the RFM file
        {
            let mut file = File::create(&path).unwrap();
            file.write_all(b"RFM\0").unwrap();
            file.write_all(&1u32.to_le_bytes()).unwrap();
            file.write_all(&metadata_size.to_le_bytes()).unwrap();
            file.write_all(&tensor_table_allocated_size.to_le_bytes())
                .unwrap();
            file.write_all(&metadata_bytes).unwrap();
            file.write_all(&table_payload).unwrap();
            file.write_all(&payload).unwrap();
        }

        // Load the RFM file
        let rfm = RfmFile::open(&path).unwrap();

        assert_eq!(rfm.metadata.num_layers, 12);
        assert_eq!(rfm.metadata.architecture, "llama");
        assert_eq!(rfm.metadata.tokens[0], b"hello");
        assert_eq!(rfm.tensor_count(), 2);

        // Verify token_embd.weight
        let emb = rfm.tensor("token_embd.weight").unwrap().unwrap();
        assert_eq!(emb.dims, &[4096, 32000]);
        assert_eq!(emb.wtype, RfmType::Q4Split);
        assert_eq!(emb.data.len(), 256);
        assert_eq!(emb.data[0], 0);
        assert_eq!(emb.data[1], 1);

        // Verify output_norm.weight
        let norm = rfm.tensor("output_norm.weight").unwrap().unwrap();
        assert_eq!(norm.dims, &[4096]);
        assert_eq!(norm.wtype, RfmType::F32);
        assert_eq!(norm.data.len(), 16384);
        assert_eq!(norm.data[0], payload[256]);
        assert_eq!(norm.data[1], payload[257]);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
