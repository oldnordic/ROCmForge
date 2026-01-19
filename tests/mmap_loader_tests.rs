//! Tests for mmap-based weight loading

use rocmforge::loader::mmap_loader::open_mmap_weights;
use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;

#[test]
fn test_mmap_loader_opens_file() {
    // Create a temporary file with test data
    let mut temp_file = NamedTempFile::new().unwrap();
    let test_data = vec![1u8, 2u8, 3u8, 4u8, 5u8];
    temp_file.write_all(&test_data).unwrap();

    // Open with mmap loader
    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();

    // Verify data matches
    assert_eq!(mmap_weights.len(), test_data.len());
    assert_eq!(&mmap_weights.data()[..test_data.len()], &test_data[..]);
}

#[test]
fn test_mmap_loader_empty_file() {
    let temp_file = NamedTempFile::new().unwrap();

    // Should handle empty file
    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();
    assert_eq!(mmap_weights.len(), 0);
    assert!(mmap_weights.data().is_empty());
}

#[test]
fn test_mmap_loader_nonexistent_file() {
    let nonexistent_path = Path::new("/nonexistent/file.bin");

    // Should return error for nonexistent file
    let result = open_mmap_weights(&nonexistent_path);
    assert!(result.is_err());
}

#[test]
fn test_mmap_loader_large_file() {
    let mut temp_file = NamedTempFile::new().unwrap();

    // Create larger test data (1MB)
    let test_data: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();
    temp_file.write_all(&test_data).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();

    // Verify large data
    assert_eq!(mmap_weights.len(), test_data.len());
    assert_eq!(&mmap_weights.data()[..100], &test_data[..100]);
    assert_eq!(
        &mmap_weights.data()[test_data.len() - 100..],
        &test_data[test_data.len() - 100..]
    );
}
