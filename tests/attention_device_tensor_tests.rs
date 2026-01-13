//! Tests for attention mechanism with DeviceTensor integration

#[cfg(feature = "rocm")]
use rocmforge::attention::{Attention, AttentionBackend};
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
fn test_attention_device_tensor_basic() {
    // Create test data
    let dim = 2; // Small dimension for testing
    let batch_size = 1;
    let seq_len = dim;

    // Q, K, V matrices for attention (batch_size, seq_len, dim)
    let q_data: Vec<f32> = vec![
        1.0, 0.0, // Q[0,0], Q[0,1]
        0.0, 1.0, // Q[0,0], Q[0,1]
    ];

    let k_data: Vec<f32> = vec![
        1.0, 0.0, // K[0,0], K[0,1]
        0.0, 1.0, // K[0,0], K[0,1]
    ];

    let v_data: Vec<f32> = vec![
        1.0, 2.0, // V[0,0], V[0,1]
        3.0, 4.0, // V[0,0], V[0,1]
    ];

    // Create DeviceTensors from host data
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let q_shape = TensorShape::from_dims(&[batch_size, seq_len, dim]);
    let k_shape = TensorShape::from_dims(&[batch_size, seq_len, dim]);
    let v_shape = TensorShape::from_dims(&[batch_size, seq_len, dim]);

    println!(
        "DEBUG: Creating Q device tensor from {} elements with shape {:?}",
        q_data.len(),
        q_shape
    );
    let q_device = DeviceTensor::from_host_vec(backend, q_data.clone(), q_shape).unwrap();
    println!(
        "DEBUG: Created Q device tensor: len() = {}, size() = {}, shape = {:?}",
        q_device.len(),
        q_device.size(),
        q_device.shape()
    );
    println!(
        "DEBUG: Creating K device tensor from {} elements with shape {:?}",
        k_data.len(),
        k_shape
    );
    let k_device = DeviceTensor::from_host_vec(backend, k_data.clone(), k_shape).unwrap();
    println!(
        "DEBUG: Created K device tensor: len() = {}, size() = {}, shape = {:?}",
        k_device.len(),
        k_device.size(),
        k_device.shape()
    );
    println!(
        "DEBUG: Creating V device tensor from {} elements with shape {:?}",
        v_data.len(),
        v_shape
    );
    let v_device = DeviceTensor::from_host_vec(backend, v_data.clone(), v_shape).unwrap();
    println!(
        "DEBUG: Created V device tensor: len() = {}, size() = {}, shape = {:?}",
        v_device.len(),
        v_device.size(),
        v_device.shape()
    );

    // Create attention with GPU backend
    let attention = Attention::with_backend(dim, AttentionBackend::Gpu);

    // Test forward pass with DeviceTensor inputs
    println!(
        "Q device: len() = {}, size() = {}, shape: {:?}",
        q_device.len(),
        q_device.size(),
        q_device.shape()
    );
    println!(
        "K device: len() = {}, size() = {}, shape: {:?}",
        k_device.len(),
        k_device.size(),
        k_device.shape()
    );
    println!(
        "V device: len() = {}, size() = {}, shape: {:?}",
        v_device.len(),
        v_device.size(),
        v_device.shape()
    );

    let output_device = attention
        .forward_device(&q_device, &k_device, &v_device, None, None)
        .unwrap();

    println!(
        "Output device: len() = {}, size() = {}, shape: {:?}",
        output_device.len(),
        output_device.size(),
        output_device.shape()
    );

    println!("About to call to_host_vec()...");

    println!(
        "Output device: len() = {}, size() = {}",
        output_device.len(),
        output_device.size()
    );

    // Verify output shape
    assert_eq!(output_device.len(), batch_size * seq_len * dim);

    // Verify output is reasonable (should be different from input)
    println!("Calling to_host_vec()...");
    let output_host = output_device.to_host_vec().unwrap();
    println!("to_host_vec() succeeded, output len: {}", output_host.len());
    assert_ne!(output_host, v_data); // Should be transformed by attention

    // Verify no NaN or infinite values
    for &val in &output_host {
        assert!(val.is_finite());
    }

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_attention_device_tensor_with_mask() {
    let dim = 2;
    let batch_size = 1;
    let seq_len = dim;

    // Q, K, V matrices
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    // Mask (should mask out second position)
    let mask_data: Vec<f32> = vec![0.0, f32::NEG_INFINITY]; // Allow first, mask second

    // Create DeviceTensors
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let q_shape = TensorShape::from_dims(&[batch_size, seq_len, dim]);
    let k_shape = TensorShape::from_dims(&[batch_size, dim, seq_len]);
    let v_shape = TensorShape::from_dims(&[batch_size, seq_len, dim]);
    let mask_shape = TensorShape::from_dims(&[batch_size, seq_len, seq_len]);

    let q_device = DeviceTensor::from_host_vec(backend, q_data.clone(), q_shape).unwrap();
    let k_device = DeviceTensor::from_host_vec(backend, k_data.clone(), k_shape).unwrap();
    let v_device = DeviceTensor::from_host_vec(backend, v_data.clone(), v_shape).unwrap();
    let mask_device = DeviceTensor::from_host_vec(backend, mask_data.clone(), mask_shape).unwrap();

    // Create attention with GPU backend
    let attention = Attention::with_backend(dim, AttentionBackend::Gpu);

    // Test forward pass with mask
    let output_device = attention
        .forward_device(&q_device, &k_device, &v_device, Some(&mask_device), None)
        .unwrap();

    // Verify output
    let output_host = output_device.to_host_vec().unwrap();
    assert_eq!(output_host.len(), batch_size * seq_len * dim);

    // Verify all values are finite
    for &val in &output_host {
        assert!(val.is_finite());
    }

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_attention_device_tensor_from_mmap() {
    let dim = 2;
    let batch_size = 1;
    let seq_len = dim;

    // Create test weight file with Q, K, V data concatenated
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    // Concatenate all data for mmap
    let mut all_data = Vec::new();
    all_data.extend_from_slice(&q_data);
    all_data.extend_from_slice(&k_data);
    all_data.extend_from_slice(&v_data);

    // Create temporary file with test data
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let test_bytes: Vec<u8> = all_data
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();
    temp_file.write_all(&test_bytes).unwrap();

    // Load weights with mmap
    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();

    // Create DeviceTensors from mmap
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let q_shape = TensorShape::from_dims(&[batch_size, seq_len, dim]);
    let k_shape = TensorShape::from_dims(&[batch_size, dim, seq_len]);
    let v_shape = TensorShape::from_dims(&[batch_size, seq_len, dim]);

    // Q from mmap (offset 0, length 4)
    let q_device = DeviceTensor::from_mmap(backend, &mmap_weights, q_shape.clone(), 0).unwrap();

    // K from mmap (offset 4, length 4)
    let k_device = DeviceTensor::from_mmap(backend, &mmap_weights, k_shape.clone(), 4 * 4).unwrap();

    // V from mmap (offset 8, length 4)
    let v_device = DeviceTensor::from_mmap(backend, &mmap_weights, v_shape.clone(), 8 * 4).unwrap();

    // Create attention with GPU backend
    let attention = Attention::with_backend(dim, AttentionBackend::Gpu);

    // Test forward pass with mmap-based DeviceTensors
    let output_device = attention
        .forward_device(&q_device, &k_device, &v_device, None, None)
        .unwrap();

    // Verify output
    assert_eq!(output_device.len(), batch_size * seq_len * dim);
    let output_host = output_device.to_host_vec().unwrap();

    // Verify all values are finite
    for &val in &output_host {
        assert!(val.is_finite());
    }

    // Verify output is different from input V (attention should transform it)
    let v_host = v_device.to_host_vec().unwrap();
    assert_ne!(output_host, v_host);

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_debug_device_tensor_sizes() {
    use rocmforge::backend::{DeviceTensor, HipBackend};

    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Test data
    let data = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements
    let shape = TensorShape::from_dims(&[1, 2, 2]); // Should be 4 elements

    println!("Data length: {} elements", data.len());
    println!("Shape total elements: {}", shape.total_elements());
    println!("Data bytes: {}", data.len() * 4);
    println!("Shape bytes: {}", shape.total_elements() * 4);

    let device_tensor = DeviceTensor::from_host_vec(backend, data.clone(), shape).unwrap();

    println!("Device tensor len(): {} elements", device_tensor.len());
    println!("Device tensor size(): {} bytes", device_tensor.size());

    // Try to copy back
    let result = device_tensor.to_host_vec();
    match result {
        Ok(host_data) => println!("Success: got {} elements", host_data.len()),
        Err(e) => println!("Error: {:?}", e),
    }

    // Test shape creation
    let output_data = vec![1.0, 2.0, 3.0, 4.0];
    let shape_from_len = TensorShape::from_dims(&[output_data.len()]);
    println!(
        "Shape from [{}]: total_elements = {}",
        output_data.len(),
        shape_from_len.total_elements()
    );

    // Test DeviceTensor creation with this shape
    let test_tensor = DeviceTensor::from_host_vec(backend, output_data.clone(), shape_from_len);
    match test_tensor {
        Ok(tensor) => {
            println!(
                "Created tensor: len() = {}, size() = {}",
                tensor.len(),
                tensor.size()
            );
            let copy_result = tensor.to_host_vec();
            match copy_result {
                Ok(data) => println!("Copy back: success, {} elements", data.len()),
                Err(e) => println!("Copy back: error: {:?}", e),
            }
        }
        Err(e) => println!("Failed to create tensor: {:?}", e),
    }

    // Check for memory leaks
    fixture.assert_no_leak(5);
}
