//! GGUF Metadata Parsing
//!
//! This module handles key-value pair parsing from GGUF files:
//! - Reading KV pairs from file
//! - Updating metadata based on key names
//! - Supporting multiple model architectures (GLM, Gemma, Qwen, LLaMA)

use crate::loader::metadata::GgufMetadata;
use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// Parse key-value pairs from GGUF header
///
/// # Arguments
///
/// * `file` - File handle positioned after tensor/KV counts
/// * `kv_count` - Number of KV pairs to read
/// * `metadata` - Metadata struct to update with parsed values
pub fn parse_kv_pairs(file: &mut File, kv_count: u64, metadata: &mut GgufMetadata) -> Result<()> {
    for i in 0..kv_count {
        // Read key
        let mut key_len_bytes = [0u8; 8];
        file.read_exact(&mut key_len_bytes)?;
        let key_len = u64::from_le_bytes(key_len_bytes) as usize;

        // Sanity check key_len to prevent overflow
        if key_len > 100_000 {
            return Err(anyhow!("key_len too large: {} at index {}", key_len, i));
        }

        let mut key_bytes = vec![0u8; key_len];
        file.read_exact(&mut key_bytes)?;
        let key = String::from_utf8_lossy(&key_bytes).to_string();

        // Read value type (GGUF v3 uses u32 for value type)
        let mut value_type_bytes = [0u8; 4];
        file.read_exact(&mut value_type_bytes)?;
        let value_type = u32::from_le_bytes(value_type_bytes);

        // Read value based on type
        let value = read_value(file, value_type, &key)?;

        // Update metadata based on key
        update_metadata(metadata, &key, &value);
    }

    Ok(())
}

/// Read a single value from file based on type
///
/// GGUF value types (official ggml/gguf.h spec):
/// 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
/// 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
fn read_value(file: &mut File, value_type: u32, key: &str) -> Result<String> {
    match value_type {
        4 => {
            // UINT32 - direct value, no value_len
            let mut value_bytes = [0u8; 4];
            file.read_exact(&mut value_bytes)?;
            Ok(u32::from_le_bytes(value_bytes).to_string())
        }
        5 => {
            // INT32 - direct value, no value_len
            let mut value_bytes = [0u8; 4];
            file.read_exact(&mut value_bytes)?;
            Ok(i32::from_le_bytes(value_bytes).to_string())
        }
        6 => {
            // FLOAT32 - direct value, no value_len
            let mut value_bytes = [0u8; 4];
            file.read_exact(&mut value_bytes)?;
            Ok(f32::from_le_bytes(value_bytes).to_string())
        }
        7 => {
            // BOOL - direct value (1 byte), no value_len
            let mut value_bytes = [0u8; 1];
            file.read_exact(&mut value_bytes)?;
            Ok((value_bytes[0] != 0).to_string())
        }
        8 => {
            // STRING - has value_len prefix
            let mut value_len_bytes = [0u8; 8];
            file.read_exact(&mut value_len_bytes)?;
            let value_len = u64::from_le_bytes(value_len_bytes) as usize;

            if value_len > 100_000_000 {
                return Err(anyhow!(
                    "value_len too large: {} for key '{}'",
                    value_len,
                    key
                ));
            }

            let mut value_bytes = vec![0u8; value_len];
            file.read_exact(&mut value_bytes)?;
            Ok(String::from_utf8_lossy(&value_bytes).to_string())
        }
        0 => {
            // UINT8 - direct value, no value_len
            let mut value_bytes = [0u8; 1];
            file.read_exact(&mut value_bytes)?;
            Ok(value_bytes[0].to_string())
        }
        1 => {
            // INT8 - direct value, no value_len
            let mut value_bytes = [0u8; 1];
            file.read_exact(&mut value_bytes)?;
            Ok((value_bytes[0] as i8).to_string())
        }
        2 => {
            // UINT16 - direct value, no value_len
            let mut value_bytes = [0u8; 2];
            file.read_exact(&mut value_bytes)?;
            Ok(u16::from_le_bytes(value_bytes).to_string())
        }
        3 => {
            // INT16 - direct value, no value_len
            let mut value_bytes = [0u8; 2];
            file.read_exact(&mut value_bytes)?;
            Ok(i16::from_le_bytes(value_bytes).to_string())
        }
        10 => {
            // UINT64 - direct value, no value_len
            let mut value_bytes = [0u8; 8];
            file.read_exact(&mut value_bytes)?;
            Ok(u64::from_le_bytes(value_bytes).to_string())
        }
        11 => {
            // INT64 - direct value, no value_len
            let mut value_bytes = [0u8; 8];
            file.read_exact(&mut value_bytes)?;
            Ok(i64::from_le_bytes(value_bytes).to_string())
        }
        12 => {
            // FLOAT64 - direct value, no value_len
            let mut value_bytes = [0u8; 8];
            file.read_exact(&mut value_bytes)?;
            Ok(f64::from_le_bytes(value_bytes).to_string())
        }
        9 | _ => {
            // Array types (GGUF_TYPE_ARRAY = 9 and higher)
            skip_array_value(file, value_type, key)?;
            Ok(String::new()) // Return empty string for skipped arrays
        }
    }
}

/// Skip array value data (we don't need most arrays for metadata)
fn skip_array_value(file: &mut File, value_type: u32, key: &str) -> Result<()> {
    // Format per official GGUF spec:
    // 1. The type of the array (gguf_type) - int32_t (4 bytes)
    // 2. The number of elements in the array (uint64_t) - 8 bytes
    // 3. The binary representation of each element

    // Read array type (int32_t = 4 bytes)
    let mut array_type_bytes = [0u8; 4];
    file.read_exact(&mut array_type_bytes)?;
    let array_type = u32::from_le_bytes(array_type_bytes);

    // Read number of elements (uint64_t = 8 bytes)
    let mut n_elements_bytes = [0u8; 8];
    file.read_exact(&mut n_elements_bytes)?;
    let n_elements = u64::from_le_bytes(n_elements_bytes);

    // For now, skip all array data since we only need model metadata
    // Calculate data size based on array type
    // GGUF types: 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
    //             6=FLOAT32, 7=BOOL, 8=STRING, 9+=ARRAY/other

    // For STRING arrays (type 8), we need to skip each string:
    // each string is: length (uint64_t, 8 bytes) + data
    if array_type == 8 {
        skip_string_array(file, n_elements, key)?;
        return Ok(());
    }

    // For fixed-size types, calculate and skip
    let element_size = match array_type {
        0 | 1 | 7 => 1, // UINT8, INT8, BOOL
        2 | 3 => 2,     // UINT16, INT16
        4..=6 => 4,     // UINT32, INT32, FLOAT32
        10..=12 => 8,   // UINT64, INT64, FLOAT64
        _ => {
            tracing::warn!(
                "Unknown array type {}, stopping metadata parse for key '{}'",
                array_type,
                key
            );
            return Ok(());
        }
    };

    let data_size = n_elements
        .checked_mul(element_size)
        .ok_or_else(|| anyhow!("Array data size overflow for key '{}'", key))?;

    if data_size > 1_000_000_000 {
        tracing::warn!(
            "Large array ({} bytes) for key '{}', stopping metadata parse",
            data_size,
            key
        );
        return Ok(());
    }

    // Skip the array data
    let mut skip_buffer = vec![0u8; data_size as usize];
    file.read_exact(&mut skip_buffer)?;

    Ok(())
}

/// Skip string array data (variable-length strings)
fn skip_string_array(file: &mut File, n_elements: u64, key: &str) -> Result<()> {
    // For large arrays, just stop parsing after this KV pair
    // We'll seek past the data by reading string lengths
    if n_elements > 10000 {
        // Large array - estimate size and seek
        // Average string length ~10 bytes + 8 byte length = 18 bytes
        let estimated_size = n_elements.saturating_mul(20);
        if estimated_size > 100_000_000 {
            tracing::warn!(
                "Very large STRING array '{}', stopping metadata parse",
                key
            );
            return Ok(());
        }
        // Try to seek past the data
        for _ in 0..n_elements {
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes)?;
            let str_len = u64::from_le_bytes(len_bytes);
            if str_len > 10_000_000 {
                return Err(anyhow!(
                    "String too large: {} bytes in array '{}'",
                    str_len,
                    key
                ));
            }
            file.seek(SeekFrom::Current(str_len as i64))?;
        }
    } else {
        // Smaller array - skip properly
        for _ in 0..n_elements {
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes)?;
            let str_len = u64::from_le_bytes(len_bytes);
            if str_len > 10_000_000 {
                return Err(anyhow!(
                    "String too large: {} bytes in array '{}'",
                    str_len,
                    key
                ));
            }
            let mut skip = vec![0u8; str_len as usize];
            file.read_exact(&mut skip)?;
        }
    }
    Ok(())
}

/// Update metadata from key-value pair
///
/// Handles keys from multiple model architectures:
/// - GLM (glm.*)
/// - Gemma3 (gemma3.*)
/// - Qwen2 (qwen2.*)
/// - LLaMA (llama.*)
pub fn update_metadata(metadata: &mut GgufMetadata, key: &str, value: &str) {
    match key {
        "general.architecture" => metadata.architecture = value.to_string(),
        "general.file_type" => metadata.file_type = value.parse().unwrap_or(0),
        // GLM-specific keys
        "glm.n_layers" => metadata.num_layers = value.parse().unwrap_or(0),
        "glm.n_heads" => metadata.num_heads = value.parse().unwrap_or(0),
        "glm.n_embd" => metadata.hidden_size = value.parse().unwrap_or(0),
        "glm.intermediate_size" => metadata.intermediate_size = value.parse().unwrap_or(0),
        "glm.head_dim" => metadata.head_dim = value.parse().unwrap_or(0),
        "glm.max_position_embeddings" => {
            metadata.max_position_embeddings = value.parse().unwrap_or(2048)
        }
        "glm.vocab_size" => metadata.vocab_size = value.parse().unwrap_or(0),
        "glm.rms_norm_eps" => metadata.rms_norm_eps = value.parse().unwrap_or(1e-6),
        // Gemma 3-specific keys (actual keys from GGUF file)
        "gemma3.embedding_length" => metadata.hidden_size = value.parse().unwrap_or(0),
        "gemma3.block_count" => metadata.num_layers = value.parse().unwrap_or(0),
        "gemma3.feed_forward_length" => metadata.intermediate_size = value.parse().unwrap_or(0),
        "gemma3.attention.head_count" => metadata.num_heads = value.parse().unwrap_or(0),
        "gemma3.attention.head_count_kv" => {
            metadata.num_kv_heads = Some(value.parse().unwrap_or(0))
        }
        "gemma3.attention.key_length" => metadata.head_dim = value.parse().unwrap_or(0),
        "gemma3.attention.value_length" => metadata.head_dim = value.parse().unwrap_or(0), // Same as key_length
        "gemma3.context_length" => {
            metadata.max_position_embeddings = value.parse().unwrap_or(2048)
        }
        "gemma3.attention.layer_norm_rms_epsilon" => {
            metadata.rms_norm_eps = value.parse().unwrap_or(1e-6)
        }
        // Qwen2-specific keys
        "qwen2.block_count" => metadata.num_layers = value.parse().unwrap_or(0),
        "qwen2.attention.head_count" => metadata.num_heads = value.parse().unwrap_or(0),
        "qwen2.attention.head_count_kv" => {
            eprintln!(">>> GGUF: Found qwen2.attention.head_count_kv = {}", value);
            metadata.num_kv_heads = Some(value.parse().unwrap_or(0))
        }
        "qwen2.embedding_length" => metadata.hidden_size = value.parse().unwrap_or(0),
        "qwen2.intermediate_size" => {
            metadata.intermediate_size = value.parse().unwrap_or(0)
        }
        "qwen2.rope.dimension_count" => {
            // Optional override: only set if value is valid (> 0)
            if let Ok(dim) = value.parse::<usize>() {
                if dim > 0 {
                    metadata.head_dim = dim;
                }
            }
        }
        "qwen2.max_position_embeddings" => {
            metadata.max_position_embeddings = value.parse().unwrap_or(2048)
        }
        "qwen2.vocab_size" => metadata.vocab_size = value.parse().unwrap_or(0),
        // Llama-specific keys (also used by some Qwen models)
        "llama.block_count" => metadata.num_layers = value.parse().unwrap_or(0),
        "llama.attention.head_count" => metadata.num_heads = value.parse().unwrap_or(0),
        "llama.attention.head_count_kv" => {
            metadata.num_kv_heads = Some(value.parse().unwrap_or(0))
        }
        "llama.embedding_length" => metadata.hidden_size = value.parse().unwrap_or(0),
        "llama.feed_forward_length" => {
            metadata.intermediate_size = value.parse().unwrap_or(0)
        }
        "llama.rope.dimension_count" => {
            // Optional override: usually head_dim = hidden_size / num_heads
            if let Ok(dim) = value.parse::<usize>() {
                if dim > 0 {
                    metadata.head_dim = dim;
                }
            }
        }
        "llama.max_position_embeddings" => {
            metadata.max_position_embeddings = value.parse().unwrap_or(2048)
        }
        "llama.vocab_size" => metadata.vocab_size = value.parse().unwrap_or(0),
        // Common RMS norm epsilon key names
        "llama.attention.layer_norm_rms_epsilon"
        | "qwen2.attention.layer_norm_rms_epsilon"
        | "qwen2.attention_norm_epsilon" => {
            metadata.rms_norm_eps = value.parse().unwrap_or(1e-6)
        }
        // Tokenizer JSON (embedded in some models)
        "tokenizer.json" => {
            if metadata.embedded_tokenizer_json.is_none() {
                metadata.embedded_tokenizer_json = Some(value.to_string());
            }
        }
        key if key.ends_with(".tokenizer_json") => {
            if metadata.embedded_tokenizer_json.is_none() {
                metadata.embedded_tokenizer_json = Some(value.to_string());
            }
        }
        // Ignore unknown keys
        _ => {}
    }
}

/// Parse hyperparameters from file
///
/// Convenience function that reads KV pairs and returns populated metadata
pub fn parse_hyperparameters(file: &mut File, kv_count: u64) -> Result<GgufMetadata> {
    let mut metadata = GgufMetadata::default();
    parse_kv_pairs(file, kv_count, &mut metadata)?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_metadata_glm() {
        let mut metadata = GgufMetadata::default();
        update_metadata(&mut metadata, "general.architecture", "glm");
        update_metadata(&mut metadata, "glm.n_layers", "24");
        update_metadata(&mut metadata, "glm.n_heads", "14");
        update_metadata(&mut metadata, "glm.n_embd", "896");

        assert_eq!(metadata.architecture, "glm");
        assert_eq!(metadata.num_layers, 24);
        assert_eq!(metadata.num_heads, 14);
        assert_eq!(metadata.hidden_size, 896);
    }

    #[test]
    fn test_update_metadata_qwen2() {
        let mut metadata = GgufMetadata::default();
        update_metadata(&mut metadata, "qwen2.block_count", "24");
        update_metadata(&mut metadata, "qwen2.attention.head_count", "14");
        update_metadata(&mut metadata, "qwen2.embedding_length", "896");

        assert_eq!(metadata.num_layers, 24);
        assert_eq!(metadata.num_heads, 14);
        assert_eq!(metadata.hidden_size, 896);
    }

    #[test]
    fn test_update_metadata_llama() {
        let mut metadata = GgufMetadata::default();
        update_metadata(&mut metadata, "llama.block_count", "32");
        update_metadata(&mut metadata, "llama.attention.head_count", "32");
        update_metadata(&mut metadata, "llama.embedding_length", "4096");

        assert_eq!(metadata.num_layers, 32);
        assert_eq!(metadata.num_heads, 32);
        assert_eq!(metadata.hidden_size, 4096);
    }

    #[test]
    fn test_read_value_uint32() {
        let mut data = vec![42u8, 0, 0, 0]; // 42 in little endian
        let result = read_value(&mut std::io::Cursor::new(&mut data), 4, "test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "42");
    }

    #[test]
    fn test_read_value_string() {
        let mut data = vec![
            4u8, 0, 0, 0, 0, 0, 0, 0, // length = 4
            b't', b'e', b's', b't', // "test"
        ];
        let result = read_value(&mut std::io::Cursor::new(&mut data), 8, "test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test");
    }

    #[test]
    fn test_read_value_bool() {
        let mut data = vec![1u8]; // true
        let result = read_value(&mut std::io::Cursor::new(&mut data), 7, "test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "true");
    }
}
