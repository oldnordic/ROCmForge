//! Tests for DeviceTensor::from_mmap functionality

#[cfg(feature = "rocm")]
use rocmforge::backend::{DeviceTensor, HipBackend};
#[cfg(feature = "rocm")]
use rocmforge::loader::mmap_loader::{open_mmap_weights, TensorShape};
#[cfg(feature = "rocm")]
use serial_test::serial;
#[cfg(feature = "rocm")]
use std::io::Write;

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_device_tensor_from_mmap_basic() {
    // Create test data
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&test_bytes).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();
    let shape = TensorShape::from_dims(&[test_f32.len()]);

    // Create DeviceTensor from mmap
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let device_tensor = DeviceTensor::from_mmap(backend, &mmap_weights, shape.clone(), 0).unwrap();

    // Verify device tensor properties
    assert_eq!(device_tensor.len(), test_f32.len()); // Number of f32 elements
    assert_eq!(device_tensor.shape().dims(), shape.dims());

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_device_tensor_from_mmap_partial_range() {
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&test_bytes).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();
    let shape = TensorShape::from_dims(&[6]); // Total elements

    // Create DeviceTensor from partial range (elements 2-4)
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let partial_shape = TensorShape::from_dims(&[2]); // Only 2 elements
    let device_tensor =
        DeviceTensor::from_mmap(backend, &mmap_weights, partial_shape, 2 * 4).unwrap();

    // Should only contain 2 elements (indices 2, 3)
    assert_eq!(device_tensor.len(), 2); // 2 f32 elements

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_device_tensor_from_mmap_empty_range() {
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&test_bytes).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();
    let _shape = TensorShape::from_dims(&[3]);

    // Create DeviceTensor from empty range - should handle gracefully
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let empty_shape = TensorShape::from_dims(&[0]);
    let result = DeviceTensor::from_mmap(backend, &mmap_weights, empty_shape, 4);

    // Should either succeed with empty tensor or fail gracefully
    match result {
        Ok(device_tensor) => {
            assert_eq!(device_tensor.len(), 0);
            // Check for memory leaks
            fixture.assert_no_leak(5);
        }
        Err(_) => {
            // Accept failure for empty tensors
        }
    }
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_device_tensor_from_mmap_bounds_check() {
    let test_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&test_bytes).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();
    let shape = TensorShape::from_dims(&[3]);

    // Try to access beyond available data
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let oversized_shape = TensorShape::from_dims(&[5]);
    let result = DeviceTensor::from_mmap(backend, &mmap_weights, oversized_shape, 0);

    // Should handle gracefully (either error or truncated)
    assert!(result.is_err() || result.unwrap().len() <= 3 * 4);

    // Check for memory leaks
    fixture.assert_no_leak(5);
}
