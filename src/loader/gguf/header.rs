//! GGUF Header Parsing and Validation
//!
//! This module handles GGUF file header parsing:
//! - Magic number validation
//! - Version verification
//! - Tensor and KV count reading

use anyhow::{bail, Result};
use std::io::Read;

/// GGUF file magic number
pub const GGUF_MAGIC: &[u8] = b"GGUF";

/// Supported GGUF version
pub const GGUF_VERSION: u32 = 3;

/// GGUF header information
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

/// Validate GGUF magic number from file
///
/// Reads the first 4 bytes and verifies they match "GGUF"
pub fn validate_gguf_magic<R: Read>(file: &mut R) -> Result<()> {
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if magic != GGUF_MAGIC {
        bail!("Invalid GGUF magic number: expected {:?}, got {:?}", GGUF_MAGIC, magic);
    }
    Ok(())
}

/// Read and validate GGUF version
///
/// Reads the version number and ensures it's supported
pub fn read_gguf_version<R: Read>(file: &mut R) -> Result<u32> {
    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != GGUF_VERSION {
        bail!("Unsupported GGUF version: {} (only version {} is supported)", version, GGUF_VERSION);
    }
    Ok(version)
}

/// Read tensor count from GGUF header
pub fn read_tensor_count<R: Read>(file: &mut R) -> Result<u64> {
    let mut tensor_count_bytes = [0u8; 8];
    file.read_exact(&mut tensor_count_bytes)?;
    Ok(u64::from_le_bytes(tensor_count_bytes))
}

/// Read KV count from GGUF header
pub fn read_kv_count<R: Read>(file: &mut R) -> Result<u64> {
    let mut kv_count_bytes = [0u8; 8];
    file.read_exact(&mut kv_count_bytes)?;
    Ok(u64::from_le_bytes(kv_count_bytes))
}

/// Parse complete GGUF header
///
/// Reads magic, version, tensor count, and KV count in sequence
pub fn parse_gguf_header<R: Read>(file: &mut R) -> Result<GgufHeader> {
    validate_gguf_magic(file)?;
    let version = read_gguf_version(file)?;
    let tensor_count = read_tensor_count(file)?;
    let kv_count = read_kv_count(file)?;

    Ok(GgufHeader {
        version,
        tensor_count,
        kv_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic_bytes() {
        assert_eq!(GGUF_MAGIC, b"GGUF");
    }

    #[test]
    fn test_gguf_version_constant() {
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn test_validate_gguf_magic_valid() {
        let mut data = vec![b'G', b'G', b'U', b'F', 0, 0, 0, 0];
        let result = validate_gguf_magic(
            &mut std::io::Cursor::new(&mut data)
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_gguf_magic_invalid() {
        let mut data = vec![b'B', b'A', b'D', b'!', 0, 0, 0, 0];
        let result = validate_gguf_magic(
            &mut std::io::Cursor::new(&mut data)
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_read_gguf_version_valid() {
        let mut data = vec![3u8, 0, 0, 0]; // version 3 in little endian
        let result = read_gguf_version(
            &mut std::io::Cursor::new(&mut data)
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3);
    }

    #[test]
    fn test_read_gguf_version_invalid() {
        let mut data = vec![2u8, 0, 0, 0]; // version 2 (unsupported)
        let result = read_gguf_version(
            &mut std::io::Cursor::new(&mut data)
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_read_tensor_count() {
        let mut data = vec![42u8, 0, 0, 0, 0, 0, 0, 0]; // 42 tensors
        let result = read_tensor_count(
            &mut std::io::Cursor::new(&mut data)
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_read_kv_count() {
        let mut data = vec![100u8, 0, 0, 0, 0, 0, 0, 0]; // 100 KV pairs
        let result = read_kv_count(
            &mut std::io::Cursor::new(&mut data)
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100);
    }
}
