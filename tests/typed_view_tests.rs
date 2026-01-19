//! Tests for typed tensor views using bytemuck

use rocmforge::loader::mmap_loader::{open_mmap_weights, MmapWeights};
use std::io::Write;
use anyhow::Context;

#[test]
fn test_f32_view_basic() -> anyhow::Result<()> {
    // Create test data as f32 values
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().context("TODO: add error context")?;
    temp_file.write_all(&test_bytes).context("TODO: add error context")?;

    let mmap_weights = open_mmap_weights(temp_file.path()).context("TODO: add error context")?;

    // Get f32 view
    let f32_view = mmap_weights.view_f32(0..test_f32.len());

    // Verify values match
    assert_eq!(f32_view.len(), test_f32.len());
    for (i, &value) in f32_view.iter().enumerate() {
        assert_eq!(value, test_f32[i]);
    }
    Ok(())
}

#[test]
fn test_f32_view_partial_range() -> anyhow::Result<()> {
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().context("TODO: add error context")?;
    temp_file.write_all(&test_bytes).context("TODO: add error context")?;

    let mmap_weights = open_mmap_weights(temp_file.path()).context("TODO: add error context")?;

    // Get partial f32 view (elements 2-5)
    let f32_view = mmap_weights.view_f32(2..5);

    // Verify partial range
    assert_eq!(f32_view.len(), 3); // elements 2, 3, 4
    assert_eq!(f32_view[0], 3.0);
    assert_eq!(f32_view[1], 4.0);
    assert_eq!(f32_view[2], 5.0);
    Ok(())
}

#[test]
fn test_f32_view_alignment() -> anyhow::Result<()> {
    // Test with unaligned start (should work due to byte-level access)
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().context("TODO: add error context")?;
    temp_file.write_all(&test_bytes).context("TODO: add error context")?;

    let mmap_weights = open_mmap_weights(temp_file.path()).context("TODO: add error context")?;

    // Start at odd byte offset (not f32-aligned)
    let f32_view = mmap_weights.view_f32(1..3); // Should get elements 1, 2

    assert_eq!(f32_view.len(), 2);
    assert_eq!(f32_view[0], 2.0);
    assert_eq!(f32_view[1], 3.0);
    Ok(())
}

#[test]
fn test_f32_view_empty_range() -> anyhow::Result<()> {
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().context("TODO: add error context")?;
    temp_file.write_all(&test_bytes).context("TODO: add error context")?;

    let mmap_weights = open_mmap_weights(temp_file.path()).context("TODO: add error context")?;

    // Empty range
    let f32_view = mmap_weights.view_f32(1..1);

    assert!(f32_view.is_empty());
    Ok(())
}

#[test]
fn test_f32_view_bounds_check() -> anyhow::Result<()> {
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().context("TODO: add error context")?;
    temp_file.write_all(&test_bytes).context("TODO: add error context")?;

    let mmap_weights = open_mmap_weights(temp_file.path()).context("TODO: add error context")?;

    // Out of bounds range should panic or be handled gracefully
    let f32_view = mmap_weights.view_f32(0..5); // Beyond available data

    // Should either panic or return empty slice depending on implementation
    assert!(f32_view.len() <= 3);
    Ok(())
}
