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
    /// Q4 Quantized weight with SVD outlier correction of rank k
    Q4SvdQuant { k: u32 },
    /// SVD low-rank correction (U, V stored as `.svd_u` / `.svd_v` F32 entries) whose
    /// residual (W − U·Vᵀ) is stored as sparse CSR instead of quantised Q4.
    ///
    /// The main tensor payload carries the sparse residual in the same layout as
    /// `SparseCsr`.  The `.svd_u` / `.svd_v` companion tensors are F32.
    SvdSparseCsr {
        k: u32,
        rows: u64,
        cols: u64,
        nnz: u64,
        index_bits: u8,
        value_type: u32,
    },
    /// Sparse CSR matrix payload for mmap-backed CPU/RAM residency.
    ///
    /// Payload layout:
    /// - row_offsets: `rows + 1` little-endian indices
    /// - col_indices: `nnz` little-endian indices
    /// - values: `nnz` values encoded as `value_type`
    SparseCsr {
        rows: u64,
        cols: u64,
        nnz: u64,
        index_bits: u8,
        value_type: u32,
    },
    /// Matrix Product Operator payload for tensor-network compressed weights.
    ///
    /// Payload stores all site tensors contiguously as `value_type` values.
    /// Per-site shapes are stored in the tensor entry `dims` as
    /// `[chi_l, d_out, d_in, chi_r]` chunks.
    Mpo {
        n_sites: u32,
        chi_max: u32,
        value_type: u32,
    },
    /// Per-expert SVD low-rank + sparse CSR residual for MoE weight tensors.
    ///
    /// Original tensor shape: `[cols, rows, n_experts]` (GGUF convention —
    /// `cols = dims[0]` is fastest-varying, `n_experts = dims[2]` is outermost).
    ///
    /// Payload layout (byte-exact):
    /// - U matrices:      `[n_experts * rows * k]` F32 (packed row-major)
    /// - V matrices:      `[n_experts * k * cols]` F32 (packed row-major)
    /// - CSR row_ptr:     `[n_experts * (rows + 1)]` u32 (experts concatenated)
    /// - CSR col_idx:     `[total_nnz]` u32
    /// - CSR values:      `[total_nnz]` F32
    /// - per-expert NNZ:  `[n_experts]` u32
    MoeExpertSvdSparse {
        n_experts: u32,
        k: u32,
        rows: u64,
        cols: u64,
        total_nnz: u64,
        index_bits: u8,
        value_type: u32,
    },
    /// Per-expert SVD low-rank + sparse CSR residual with Fast Walsh-Hadamard Transform.
    MoeExpertSvdFwhtSparse {
        n_experts: u32,
        k: u32,
        rows: u64,
        cols: u64,
        total_nnz: u64,
        index_bits: u8,
        value_type: u32,
    },
    /// MagnumQuant FWHT-rotated 4-bit quantization with group size 256.
    Mq4,
    /// MagnumQuant FWHT-rotated 6-bit quantization with group size 256.
    Mq6,
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

    // Compression and cache extensions
    #[serde(default)]
    pub kv_lora_dim: Option<usize>,
    #[serde(default)]
    pub kv_frame_codec_enabled: Option<bool>,
    #[serde(default)]
    pub adastate_anchors_enabled: Option<bool>,
    #[serde(default)]
    pub kv_quant_bits: Option<usize>,
    #[serde(default)]
    pub turboquant_centroids: Option<Vec<f32>>,
    #[serde(default)]
    pub qjl_scale: Option<f32>,
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

/// Zero-copy sparse CSR tensor view.
#[derive(Debug, Clone, Copy)]
pub struct RfmSparseCsrView<'a> {
    pub name: &'a str,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub index_bits: u8,
    pub value_type: u32,
    pub row_offsets: &'a [u8],
    pub col_indices: &'a [u8],
    pub values: &'a [u8],
}

/// Zero-copy MPO tensor view.
#[derive(Debug, Clone, Copy)]
pub struct RfmMpoView<'a> {
    pub name: &'a str,
    pub n_sites: usize,
    pub chi_max: usize,
    pub value_type: u32,
    pub site_dims: &'a [u64],
    pub data: &'a [u8],
}

impl<'a> RfmTensorView<'a> {
    /// Returns the total number of elements in this tensor.
    pub fn element_count(&self) -> usize {
        self.dims.iter().fold(1usize, |acc, &d| acc * d as usize)
    }

    /// Interpret this tensor as sparse CSR.
    ///
    /// Matches both `SparseCsr` (residual-only) and `SvdSparseCsr` (SVD + sparse
    /// residual).  Callers that need to distinguish can inspect `self.wtype`.
    pub fn as_sparse_csr(&self) -> Option<RfmSparseCsrView<'a>> {
        let (rows, cols, nnz, index_bits, value_type) = match self.wtype {
            RfmType::SparseCsr {
                rows,
                cols,
                nnz,
                index_bits,
                value_type,
            } => (rows, cols, nnz, index_bits, value_type),
            RfmType::SvdSparseCsr {
                rows,
                cols,
                nnz,
                index_bits,
                value_type,
                ..
            } => (rows, cols, nnz, index_bits, value_type),
            _ => return None,
        };

        let rows = rows as usize;
        let cols = cols as usize;
        let nnz = nnz as usize;
        let index_bytes = match index_bits {
            32 => 4usize,
            64 => 8usize,
            _ => return None,
        };
        let value_bytes = GgmlValueType(value_type).bytes_per_value()?;
        let row_bytes = rows.checked_add(1)?.checked_mul(index_bytes)?;
        let col_bytes = nnz.checked_mul(index_bytes)?;
        let val_bytes = nnz.checked_mul(value_bytes)?;
        let total = row_bytes.checked_add(col_bytes)?.checked_add(val_bytes)?;
        if self.data.len() != total {
            return None;
        }

        let col_start = row_bytes;
        let value_start = row_bytes + col_bytes;
        Some(RfmSparseCsrView {
            name: self.name,
            rows,
            cols,
            nnz,
            index_bits,
            value_type,
            row_offsets: &self.data[..col_start],
            col_indices: &self.data[col_start..value_start],
            values: &self.data[value_start..],
        })
    }

    /// Interpret this tensor as an MPO when its RFM type is `Mpo`.
    pub fn as_mpo(&self) -> Option<RfmMpoView<'a>> {
        let RfmType::Mpo {
            n_sites,
            chi_max,
            value_type,
        } = self.wtype
        else {
            return None;
        };

        let n_sites = n_sites as usize;
        if self.dims.len() != n_sites.checked_mul(4)? {
            return None;
        }

        Some(RfmMpoView {
            name: self.name,
            n_sites,
            chi_max: chi_max as usize,
            value_type,
            site_dims: self.dims,
            data: self.data,
        })
    }
}

struct GgmlValueType(u32);

impl GgmlValueType {
    fn bytes_per_value(self) -> Option<usize> {
        match self.0 {
            0 => Some(4), // F32
            1 => Some(2), // F16
            2 => None,    // Q4_0 is block encoded, not scalar CSR value encoded
            3 => None,    // Q4_1
            8 => Some(1), // Q8_0 scalar values are stored as i8 in sparse payloads
            _ => None,
        }
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
            kv_lora_dim: Some(128),
            kv_frame_codec_enabled: Some(true),
            adastate_anchors_enabled: Some(true),
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
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
        assert_eq!(rfm.metadata.kv_lora_dim, Some(128));
        assert_eq!(rfm.metadata.kv_frame_codec_enabled, Some(true));
        assert_eq!(rfm.metadata.adastate_anchors_enabled, Some(true));
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

    #[test]
    fn test_rfm_qwen35_fused_attention_metadata_roundtrip() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = std::env::temp_dir();
        let path = dir.join("test_qwen35_fused_attention.rfm");

        let metadata = RfmMetadata {
            num_layers: 1,
            hidden_size: 4096,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 256,
            intermediate_size: 12288,
            vocab_size: 248320,
            max_seq_len: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rope_neox: false,
            use_attention_bias: false,
            architecture: "qwen35".to_string(),
            tokens: vec![],
            merges: vec![],
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: None,
            tokenizer_model: None,
            tokenizer_pre: None,
            add_bos: false,
            add_eos: false,
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
        };

        let metadata_bytes = serde_json::to_vec(&metadata)?;
        let entries = vec![
            RfmTensorEntry {
                name: "blk.0.attn_qkv.weight".to_string(),
                dims: vec![4096, 8192],
                wtype: RfmType::GgufPassthrough(12),
                offset: 0,
                size: 16,
            },
            RfmTensorEntry {
                name: "blk.0.attn_gate.weight".to_string(),
                dims: vec![4096, 4096],
                wtype: RfmType::GgufPassthrough(12),
                offset: 256,
                size: 16,
            },
            RfmTensorEntry {
                name: "blk.0.ssm_out.weight".to_string(),
                dims: vec![4096, 4096],
                wtype: RfmType::GgufPassthrough(12),
                offset: 512,
                size: 16,
            },
        ];
        let table_bytes = serde_json::to_vec(&entries)?;
        let payload_start = 24 + metadata_bytes.len() + table_bytes.len();
        let total_len = payload_start + 528;

        let mut file = File::create(&path)?;
        file.write_all(b"RFM\0")?;
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&(metadata_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&(table_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&metadata_bytes)?;
        file.write_all(&table_bytes)?;
        file.write_all(&vec![0u8; total_len - payload_start])?;
        drop(file);

        let rfm = RfmFile::open(&path)?;
        assert_eq!(rfm.metadata.architecture, "qwen35");
        assert!(rfm.has_tensor("blk.0.attn_qkv.weight"));
        assert!(rfm.has_tensor("blk.0.attn_gate.weight"));
        assert!(rfm.has_tensor("blk.0.ssm_out.weight"));
        assert!(!rfm.has_tensor("blk.0.attn_q.weight"));

        let qkv = rfm
            .tensor("blk.0.attn_qkv.weight")?
            .ok_or("tensor not found")?;
        assert_eq!(qkv.dims, &[4096, 8192]);
        assert_eq!(qkv.wtype, RfmType::GgufPassthrough(12));

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn test_sparse_csr_tensor_view_splits_payload() {
        let tensor = RfmTensorView {
            name: "blk.0.ffn_sparse.weight",
            dims: &[3, 4],
            wtype: RfmType::SparseCsr {
                rows: 3,
                cols: 4,
                nnz: 5,
                index_bits: 32,
                value_type: 0,
            },
            data: &[
                // row_offsets: 4 * u32
                0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, // col_indices: 5 * u32
                0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0,
                // values: 5 * f32
                0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64,
            ],
        };

        let sparse = tensor.as_sparse_csr().expect("valid sparse CSR view");
        assert_eq!(sparse.rows, 3);
        assert_eq!(sparse.cols, 4);
        assert_eq!(sparse.nnz, 5);
        assert_eq!(sparse.row_offsets.len(), 16);
        assert_eq!(sparse.col_indices.len(), 20);
        assert_eq!(sparse.values.len(), 20);
    }

    #[test]
    fn test_mpo_tensor_view_validates_site_dims() {
        let tensor = RfmTensorView {
            name: "blk.0.ffn_mpo.weight",
            dims: &[1, 4, 4, 2, 2, 4, 4, 1],
            wtype: RfmType::Mpo {
                n_sites: 2,
                chi_max: 2,
                value_type: 0,
            },
            data: &[0; 160],
        };

        let mpo = tensor.as_mpo().expect("valid MPO view");
        assert_eq!(mpo.n_sites, 2);
        assert_eq!(mpo.chi_max, 2);
        assert_eq!(mpo.site_dims, &[1, 4, 4, 2, 2, 4, 4, 1]);
        assert_eq!(mpo.data.len(), 160);
    }
}
