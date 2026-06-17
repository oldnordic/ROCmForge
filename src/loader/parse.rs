//! GGUF binary format parsing — header, KV pairs, tensor descriptors.
//!
//! All parsing is sequential and infallible given valid input; errors are
//! returned as `LoadError` values, never panics.

use crate::loader::error::LoadError;
use crate::loader::ggml_type::GgmlType;
use crate::loader::metadata::GgufMetadata;
use std::collections::HashMap;
use std::io::{Read, Seek};

// GGUF constants
const MAGIC: &[u8; 4] = b"GGUF";
const VERSION_MIN: u32 = 2;
const VERSION_MAX: u32 = 3;
/// Tensor data section alignment (GGUF spec)
pub const TENSOR_ALIGNMENT: u64 = 32;

// ── Header ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

pub fn parse_header<R: Read>(r: &mut R) -> Result<Header, LoadError> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(LoadError::InvalidMagic(magic));
    }

    let version = read_u32(r)?;
    if !(VERSION_MIN..=VERSION_MAX).contains(&version) {
        return Err(LoadError::UnsupportedVersion(version));
    }

    let tensor_count = read_u64(r)?;
    let kv_count = read_u64(r)?;

    Ok(Header {
        version,
        tensor_count,
        kv_count,
    })
}

// ── Tokenizer data ────────────────────────────────────────────────────────────

/// All tokenizer-related data extracted from GGUF KV section.
///
/// Tokens are raw bytes (not UTF-8 strings) — some GGUF tokens are invalid UTF-8.
/// Merges are stored as (first_part, second_part) byte pairs from "a b" strings.
#[derive(Debug, Default)]
pub struct TokenizerData {
    pub tokens: Vec<Vec<u8>>,
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    /// "gpt2" / "llama" / "spm"
    pub model: Option<String>,
    /// "qwen2" / "llama3"
    pub pre: Option<String>,
    pub add_bos: bool,
    pub add_eos: bool,
    /// SentencePiece token scores (log probabilities).
    pub scores: Vec<f32>,
    /// SentencePiece token types (1=normal, 2=unknown, 3=control, 4=user-defined,
    /// 5=unused, 6=byte).
    pub token_types: Vec<i32>,
}

// ── KV section -> (GgufMetadata, TokenizerData) ───────────────────────────────

/// Parse the entire KV section in one pass, extracting both model hyperparameters
/// and tokenizer arrays. No second file read needed.
pub fn parse_kv<R: Read + Seek>(
    r: &mut R,
    kv_count: u64,
) -> Result<(GgufMetadata, TokenizerData), LoadError> {
    let mut kv = HashMap::with_capacity(kv_count as usize);
    let mut arrays = HashMap::<String, Vec<String>>::new();
    let mut tok = TokenizerData::default();

    for _ in 0..kv_count {
        let key = read_string(r)?;
        let value_type = read_u32(r)?;

        // Intercept tokenizer array keys before string conversion
        match key.as_str() {
            "tokenizer.ggml.tokens" | "tokenizer.ggml.vocab" => {
                if value_type == 9 {
                    tok.tokens = read_byte_array(r)?;
                } else {
                    skip_value(r, value_type)?;
                }
                continue;
            }
            "tokenizer.ggml.merges" => {
                if value_type == 9 {
                    tok.merges = read_merge_array(r)?;
                } else {
                    skip_value(r, value_type)?;
                }
                continue;
            }
            "tokenizer.ggml.scores" => {
                if value_type == 9 {
                    tok.scores = read_f32_array(r)?;
                } else {
                    skip_value(r, value_type)?;
                }
                continue;
            }
            "tokenizer.ggml.token_type" => {
                if value_type == 9 {
                    tok.token_types = read_i32_array(r)?;
                } else {
                    skip_value(r, value_type)?;
                }
                continue;
            }
            _ => {}
        }

        // Capture non-tokenizer arrays so callers can retrieve per-layer values
        if value_type == 9 {
            let arr = read_array_as_strings(r)?;
            arrays.insert(key.clone(), arr);
            kv.insert(key, "array".to_string());
            continue;
        }

        // Everything else: convert to string for GgufMetadata
        let value = read_value_as_string(r, value_type)?;

        // Extract scalar tokenizer fields from the string values
        match key.as_str() {
            "tokenizer.ggml.model" => tok.model = Some(value.clone()),
            "tokenizer.ggml.pre" => tok.pre = Some(value.clone()),
            "tokenizer.ggml.bos_token_id" => tok.bos_token_id = value.parse().ok(),
            "tokenizer.ggml.eos_token_id" => tok.eos_token_id = value.parse().ok(),
            "tokenizer.ggml.unknown_token_id" => tok.unk_token_id = value.parse().ok(),
            "tokenizer.ggml.add_bos_token" => tok.add_bos = value != "0",
            "tokenizer.ggml.add_eos_token" => tok.add_eos = value != "0",
            _ => {}
        }

        kv.insert(key, value);
    }

    Ok((GgufMetadata::from_kv_with_arrays(kv, arrays), tok))
}

/// Read an array of raw byte sequences (for tokenizer.ggml.tokens).
/// Each element is a GGUF string read as raw bytes — not interpreted as UTF-8.
fn read_byte_array<R: Read>(r: &mut R) -> Result<Vec<Vec<u8>>, LoadError> {
    let _elem_type = read_u32(r)?; // expect 8 (string), but read it regardless
    let count = read_u64(r)? as usize;
    if count > 2_000_000 {
        return Err(LoadError::StringTooLong(count));
    }
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let len = read_u64(r)? as usize;
        if len > 10_000_000 {
            return Err(LoadError::StringTooLong(len));
        }
        let mut bytes = vec![0u8; len];
        r.read_exact(&mut bytes)?;
        result.push(bytes);
    }
    Ok(result)
}

/// Read an array of merge rules (for tokenizer.ggml.merges).
/// Each element is a GGUF string with format "first second"; split on first space.
#[allow(clippy::type_complexity, reason = "GGUF tuple")]
fn read_merge_array<R: Read>(r: &mut R) -> Result<Vec<(Vec<u8>, Vec<u8>)>, LoadError> {
    let _elem_type = read_u32(r)?;
    let count = read_u64(r)? as usize;
    if count > 2_000_000 {
        return Err(LoadError::StringTooLong(count));
    }
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let s = read_string(r)?;
        if let Some(pos) = s.find(' ') {
            let bytes = s.as_bytes();
            let first = bytes[..pos].to_vec();
            let second = bytes[pos + 1..].to_vec();
            result.push((first, second));
        }
        // Silently skip any merge that does not have the expected format
    }
    Ok(result)
}

/// Read a GGUF array of f32 values.
fn read_f32_array<R: Read>(r: &mut R) -> Result<Vec<f32>, LoadError> {
    let elem_type = read_u32(r)?;
    if elem_type != 6 {
        return Err(LoadError::InvalidFormat(format!(
            "expected f32 array, got element type {}",
            elem_type
        )));
    }
    let count = read_u64(r)? as usize;
    if count > 10_000_000 {
        return Err(LoadError::StringTooLong(count));
    }
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        result.push(f32::from_le_bytes(b));
    }
    Ok(result)
}

/// Read a GGUF array of i32 values.
fn read_i32_array<R: Read>(r: &mut R) -> Result<Vec<i32>, LoadError> {
    let elem_type = read_u32(r)?;
    if elem_type != 5 {
        return Err(LoadError::InvalidFormat(format!(
            "expected i32 array, got element type {}",
            elem_type
        )));
    }
    let count = read_u64(r)? as usize;
    if count > 10_000_000 {
        return Err(LoadError::StringTooLong(count));
    }
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        result.push(i32::from_le_bytes(b));
    }
    Ok(result)
}

/// Read a GGUF array as a vector of element strings.
///
/// The caller has already consumed the value_type (9); this function reads the
/// element type and count, then each element. Nested arrays are flattened to
/// a compact summary string per element.
fn read_array_as_strings<R: Read>(r: &mut R) -> Result<Vec<String>, LoadError> {
    let elem_type = read_u32(r)?;
    let count = read_u64(r)? as usize;
    if count > 1_000_000 {
        return Err(LoadError::StringTooLong(count));
    }
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let s = match elem_type {
            0 | 1 | 7 => {
                // u8 / i8 / bool
                let mut b = [0u8; 1];
                r.read_exact(&mut b)?;
                if elem_type == 7 {
                    (b[0] != 0).to_string()
                } else {
                    b[0].to_string()
                }
            }
            2 | 3 => {
                let mut b = [0u8; 2];
                r.read_exact(&mut b)?;
                u16::from_le_bytes(b).to_string()
            }
            4 | 5 => {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                u32::from_le_bytes(b).to_string()
            }
            6 => {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                f32::from_le_bytes(b).to_string()
            }
            8 => read_string(r)?,
            9 => {
                // Nested array: read and flatten to a compact summary
                let nested = read_array_as_strings(r)?;
                format!("[{}]", nested.len())
            }
            10 | 11 => {
                let mut b = [0u8; 8];
                r.read_exact(&mut b)?;
                u64::from_le_bytes(b).to_string()
            }
            12 => {
                let mut b = [0u8; 8];
                r.read_exact(&mut b)?;
                f64::from_le_bytes(b).to_string()
            }
            _ => {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                "unknown".to_string()
            }
        };
        result.push(s);
    }
    Ok(result)
}

/// Reads a KV value, converting it to a string.
/// Arrays that are not intercepted above get a compact summary marker.
fn read_value_as_string<R: Read>(r: &mut R, value_type: u32) -> Result<String, LoadError> {
    match value_type {
        0 | 1 | 7 => {
            // u8 / i8 / bool
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(b[0].to_string())
        }
        2 | 3 => {
            // u16 / i16
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(u16::from_le_bytes(b).to_string())
        }
        4 | 5 => {
            // u32 / i32
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b).to_string())
        }
        6 => {
            // f32
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b).to_string())
        }
        8 => read_string(r),
        9 => {
            // Array: read element type + count, skip elements
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem_type)?;
            }
            Ok(format!("array:{}", count))
        }
        10 | 11 => {
            // u64 / i64
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b).to_string())
        }
        12 => {
            // f64
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(f64::from_le_bytes(b).to_string())
        }
        _ => {
            // Unknown — skip 4 bytes and continue
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok("unknown".to_string())
        }
    }
}

fn skip_value<R: Read>(r: &mut R, value_type: u32) -> Result<(), LoadError> {
    match value_type {
        0 | 1 | 7 => {
            r.read_exact(&mut [0u8; 1])?;
        }
        2 | 3 => {
            r.read_exact(&mut [0u8; 2])?;
        }
        4..=6 => {
            r.read_exact(&mut [0u8; 4])?;
        }
        8 => {
            read_string(r)?;
        }
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem_type)?;
            }
        }
        10..=12 => {
            r.read_exact(&mut [0u8; 8])?;
        }
        _ => {
            r.read_exact(&mut [0u8; 4])?;
        }
    }
    Ok(())
}

// ── Tensor descriptors ────────────────────────────────────────────────────────

/// Metadata for a single tensor (no data — data stays in the mmap).
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    /// Dimensions in GGUF order (innermost first)
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset relative to the tensor data section start
    pub offset: u64,
}

impl TensorDesc {
    pub fn element_count(&self) -> usize {
        self.dims.iter().fold(1usize, |acc, &d| acc * d as usize)
    }

    pub fn byte_size(&self) -> usize {
        self.ggml_type.bytes_for_elements(self.element_count())
    }
}

/// Parse all tensor descriptors, returning them and the absolute file offset
/// where the tensor data section begins.
pub fn parse_tensor_descs<R: Read + Seek>(
    r: &mut R,
    tensor_count: u64,
) -> Result<(Vec<TensorDesc>, u64), LoadError> {
    let mut descs = Vec::with_capacity(tensor_count as usize);

    for _ in 0..tensor_count {
        let name = read_string(r)?;

        let n_dims = read_u32(r)? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(read_u64(r)?);
        }

        let type_u32 = read_u32(r)?;
        let ggml_type = GgmlType::from_u32(type_u32)?;

        let offset = read_u64(r)?;

        descs.push(TensorDesc {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    // GGUF spec: tensor data section starts at the next TENSOR_ALIGNMENT boundary
    let pos = r.stream_position()?;
    let data_start = pos.div_ceil(TENSOR_ALIGNMENT) * TENSOR_ALIGNMENT;

    Ok((descs, data_start))
}

// ── Primitives ────────────────────────────────────────────────────────────────

pub fn read_string<R: Read>(r: &mut R) -> Result<String, LoadError> {
    let len = read_u64(r)? as usize;
    if len > 100_000_000 {
        return Err(LoadError::StringTooLong(len));
    }
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes.clone()).map_err(|_| LoadError::InvalidUtf8(bytes))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, LoadError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, LoadError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn magic_header(version: u32, tensors: u64, kv: u64) -> Vec<u8> {
        let mut v = b"GGUF".to_vec();
        v.extend_from_slice(&version.to_le_bytes());
        v.extend_from_slice(&tensors.to_le_bytes());
        v.extend_from_slice(&kv.to_le_bytes());
        v
    }

    #[test]
    fn valid_header_v3() {
        let data = magic_header(3, 42, 10);
        let h = parse_header(&mut Cursor::new(data)).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(h.tensor_count, 42);
        assert_eq!(h.kv_count, 10);
    }

    #[test]
    fn valid_header_v2() {
        let data = magic_header(2, 1, 1);
        let h = parse_header(&mut Cursor::new(data)).unwrap();
        assert_eq!(h.version, 2);
    }

    #[test]
    fn bad_magic() {
        let mut data = magic_header(3, 0, 0);
        data[0] = b'X';
        let err = parse_header(&mut Cursor::new(data)).unwrap_err();
        assert!(matches!(err, LoadError::InvalidMagic(_)));
    }

    #[test]
    fn unsupported_version() {
        let data = magic_header(1, 0, 0);
        let err = parse_header(&mut Cursor::new(data)).unwrap_err();
        assert!(matches!(err, LoadError::UnsupportedVersion(1)));
    }

    #[test]
    fn read_string_basic() {
        let s = "hello";
        let mut buf = (s.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(s.as_bytes());
        let got = read_string(&mut Cursor::new(buf)).unwrap();
        assert_eq!(got, "hello");
    }
}
