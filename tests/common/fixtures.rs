//! Common test fixtures for GGUF, backend, and tensor creation
//!
//! This module consolidates duplicate fixture code from multiple test files
//! into a single location for easier maintenance.

use rocmforge::backend::HipBackend;
use rocmforge::loader::gguf::{GgufTensor, GgufTensorType};
use rocmforge::loader::TensorShape;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// GGUF File Creation Fixtures
// ============================================================================

/// Create a minimal valid GGUF file for testing.
///
/// This helper creates a GGUF file with:
/// - Valid magic number ("GGUF")
/// - Version 3 (current supported version)
/// - Zero tensors (minimal structure)
/// - One metadata KV pair (general.architecture: "test")
pub fn create_test_gguf(path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write GGUF magic
    writer.write_all(b"GGUF")?;

    // Write version (3)
    writer.write_all(&3u32.to_le_bytes())?;

    // Write tensor count (0)
    writer.write_all(&0u64.to_le_bytes())?;

    // Write KV count (1)
    writer.write_all(&1u64.to_le_bytes())?;

    // Write metadata key-value pair (general.architecture: "test")
    // Key: "general.architecture"
    let key = b"general.architecture";
    writer.write_all(&(key.len() as u64).to_le_bytes())?;
    writer.write_all(key)?;

    // Value type: STRING (8)
    writer.write_all(&8u32.to_le_bytes())?;
    // Value length: 4 bytes ("test")
    let value = b"test";
    writer.write_all(&(value.len() as u64).to_le_bytes())?;
    writer.write_all(value)?;

    writer.flush()?;
    Ok(())
}

/// Create a minimal valid GGUF file with F32 tensor data for testing.
///
/// This helper creates a GGUF file with:
/// - Valid magic number ("GGUF")
/// - Version 3
/// - One F32 tensor named "f32_tensor" with shape [2, 2]
/// - One metadata KV pair (architecture: "test")
pub fn create_test_gguf_with_f32(path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write GGUF magic
    writer.write_all(b"GGUF")?;

    // Write version (3)
    writer.write_all(&3u32.to_le_bytes())?;

    // Write tensor count (1)
    writer.write_all(&1u64.to_le_bytes())?;

    // Write KV count (1)
    writer.write_all(&1u64.to_le_bytes())?;

    // Write metadata key-value pair (architecture: "test")
    let key = b"architecture";
    writer.write_all(&(key.len() as u64).to_le_bytes())?;
    writer.write_all(key)?;
    writer.write_all(&8u32.to_le_bytes())?;  // STRING type
    let value = b"test";
    writer.write_all(&(value.len() as u64).to_le_bytes())?;
    writer.write_all(value)?;

    // Write tensor info
    // Tensor name
    let tensor_name = b"f32_tensor";
    writer.write_all(&(tensor_name.len() as u64).to_le_bytes())?;
    writer.write_all(tensor_name)?;

    // Number of dimensions (2)
    writer.write_all(&2u32.to_le_bytes())?;
    // Dimension 1: 2
    writer.write_all(&2u64.to_le_bytes())?;
    // Dimension 2: 2
    writer.write_all(&2u64.to_le_bytes())?;

    // Tensor type: F32 (0)
    writer.write_all(&0u32.to_le_bytes())?;

    // Tensor offset (will be after all metadata)
    writer.write_all(&0u64.to_le_bytes())?;

    writer.flush()?;
    Ok(())
}

/// Create a minimal GGUF file with token embeddings and LM head weights
///
/// # Arguments
/// * `path` - Where to write the GGUF file
/// * `vocab_size` - Size of vocabulary (e.g., 32000, 128000)
/// * `hidden_size` - Hidden dimension (e.g., 4096, 5120)
///
/// # Creates
/// - `token_embd.weight`: [vocab_size, hidden_size] FP32 tensor
/// - `output.weight`: [vocab_size, hidden_size] FP32 tensor (tied embeddings)
/// - Metadata: n_embd, vocab_size
pub fn create_embedding_gguf(
    path: &Path,
    vocab_size: usize,
    hidden_size: usize,
) -> anyhow::Result<()> {
    let mut file = File::create(path)?;

    // GGUF magic (little-endian)
    file.write_all(b"GGUF")?;

    // Version (3 = latest)
    file.write_all(&3u32.to_le_bytes())?;

    // Tensor count (2: token_embd.weight and output.weight)
    file.write_all(&2u64.to_le_bytes())?;

    // KV count (2: n_embd and vocab_size)
    file.write_all(&2u64.to_le_bytes())?;

    // Write metadata KV pairs
    // n_embd (hidden_size)
    write_kv_string(&mut file, "n_embd", hidden_size as u32)?;

    // vocab_size
    write_kv_string(&mut file, "vocab_size", vocab_size as u32)?;

    // Calculate tensor data offset
    // Current position after header and KVs
    let current_pos = file.stream_position()?;
    let tensor_data_offset = current_pos + (2 * 16) + (4 * 8); // 2 tensors + tensor info section

    // Write tensor info section
    // token_embd.weight
    write_tensor_info(
        &mut file,
        "token_embd.weight",
        vocab_size,
        hidden_size,
        GgufTensorType::F32,
        tensor_data_offset,
    )?;

    // output.weight (offset after token_embd.weight)
    let token_embd_bytes = vocab_size * hidden_size * 4; // 4 bytes per f32
    write_tensor_info(
        &mut file,
        "output.weight",
        vocab_size,
        hidden_size,
        GgufTensorType::F32,
        tensor_data_offset + token_embd_bytes as u64,
    )?;

    // Write tensor padding
    let padding_size = 32 - (tensor_data_offset % 32) as usize;
    file.write_all(&vec![0u8; padding_size])?;

    // Write token_embd.weight data (sequential values for testing)
    let mut token_embd_data = vec![0.0f32; vocab_size * hidden_size];
    for i in 0..vocab_size * hidden_size {
        token_embd_data[i] = i as f32 * 0.001; // Small sequential values
    }
    write_tensor_data_f32(&mut file, &token_embd_data)?;

    // Write output.weight data (same as token_embd for tied embeddings)
    write_tensor_data_f32(&mut file, &token_embd_data)?;

    Ok(())
}

/// Write a GGUF key-value pair with string key and u32 value
fn write_kv_string(file: &mut File, key: &str, value: u32) -> anyhow::Result<()> {
    // Key length and key
    file.write_all(&(key.len() as u64).to_le_bytes())?;
    file.write_all(key.as_bytes())?;

    // Type: U32 = 4
    file.write_all(&4u32.to_le_bytes())?;

    // Value
    file.write_all(&value.to_le_bytes())?;

    Ok(())
}

/// Write GGUF tensor info
fn write_tensor_info(
    file: &mut File,
    name: &str,
    dim1: usize,
    dim2: usize,
    tensor_type: GgufTensorType,
    offset: u64,
) -> anyhow::Result<()> {
    // Tensor name length and name
    file.write_all(&(name.len() as u64).to_le_bytes())?;
    file.write_all(name.as_bytes())?;

    // Number of dimensions (2)
    file.write_all(&2u32.to_le_bytes())?;

    // Dimensions [dim1, dim2]
    file.write_all(&(dim1 as u64).to_le_bytes())?;
    file.write_all(&(dim2 as u64).to_le_bytes())?;

    // Tensor type (as u32)
    file.write_all(&(tensor_type as u32).to_le_bytes())?;

    // Offset to tensor data
    file.write_all(&offset.to_le_bytes())?;

    Ok(())
}

/// Write FP32 tensor data to file
fn write_tensor_data_f32(file: &mut File, data: &[f32]) -> anyhow::Result<()> {
    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

// ============================================================================
// Tensor Creation Fixtures
// ============================================================================

/// Create a test GgufTensor struct
pub fn create_test_tensor(
    tensor_type: GgufTensorType,
    data: Vec<u8>,
    shape: Vec<usize>,
) -> GgufTensor {
    GgufTensor {
        name: "test".to_string(),
        shape: TensorShape::from_dims(&shape),
        tensor_type,
        quant_type: String::new(),
        offset: 0,
        data,
    }
}

// ============================================================================
// Backend Creation Fixtures
// ============================================================================

/// Create a HIP backend for testing.
///
/// Note: This is a simple wrapper. For GPU tests, prefer using the
/// GPU_FIXTURE from the common module which provides proper
/// initialization and memory tracking.
///
/// Uses device 0 by default. Returns error if no GPU available.
pub fn try_create_backend() -> anyhow::Result<Arc<HipBackend>> {
    HipBackend::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HIP backend: {}", e))
}

/// Create a HIP backend for testing, panicking if unavailable.
///
/// Note: This is a simple wrapper. For GPU tests, prefer using the
/// GPU_FIXTURE from the common module which provides proper
/// initialization and memory tracking.
///
/// Uses device 0 by default. Panics if no GPU available.
pub fn create_backend() -> Arc<HipBackend> {
    try_create_backend().expect("HIP backend not available for test")
}
