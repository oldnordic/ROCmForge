//! Tests for HipBuffer size invariants and CPU fallback path buffer corruption

#[cfg(feature = "rocm")]
use rocmforge::backend::{DeviceTensor, HipBackend};
#[cfg(feature = "rocm")]
use rocmforge::loader::mmap_loader::TensorShape;
#[cfg(feature = "rocm")]
use serial_test::serial;

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn hip_buffer_alloc_size_invariant() {
    // Test that HipBuffer size remains constant after operations
    let fixture = GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Allocate buffer for 16 bytes (4 f32 elements)
    let buffer = backend
        .allocate_buffer(16)
        .expect("Failed to allocate buffer");

    // Initial size should be 16 bytes
    assert_eq!(buffer.size(), 16, "Initial buffer size should be 16 bytes");

    // Size should remain 16 after host->device copy
    let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
    buffer
        .copy_from_host(&test_data)
        .expect("Failed to copy from host");
    assert_eq!(
        buffer.size(),
        16,
        "Buffer size should remain 16 after host->device copy"
    );

    // Size should remain 16 after device->host copy
    let mut output_data = vec![0.0f32; 4];
    buffer
        .copy_to_host(&mut output_data)
        .expect("Failed to copy to host");
    assert_eq!(
        buffer.size(),
        16,
        "Buffer size should remain 16 after device->host copy"
    );

    // Verify data integrity
    assert_eq!(output_data, test_data, "Data should match after round-trip");

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_attention_cpu_fallback_buffer_size_consistency() {
    // Test the exact sequence that causes the "Destination buffer too small: 16 > 4" error
    let fixture = GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Create input tensor with 4 f32 elements (16 bytes)
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = TensorShape::from_dims(&[4]);
    let input_tensor = DeviceTensor::from_host_vec(backend, input_data.clone(), input_shape)
        .expect("Failed to create input tensor");

    // Verify initial tensor state
    assert_eq!(input_tensor.len(), 4, "Input tensor should have 4 elements");
    assert_eq!(input_tensor.size(), 16, "Input tensor should be 16 bytes");

    // Simulate CPU fallback path: copy to host
    let host_data = input_tensor
        .to_host_vec()
        .expect("Failed to copy input tensor to host");

    // Verify host data is correct
    assert_eq!(host_data, input_data, "Host data should match input data");
    assert_eq!(host_data.len(), 4, "Host data should have 4 elements");

    // Simulate CPU computation (identity operation for testing)
    let output_data = host_data; // In real case, this would be computed

    // Create output tensor using same pattern as attention CPU fallback
    let output_shape = TensorShape::from_dims(&[4]);
    let output_tensor = DeviceTensor::from_host_vec(backend, output_data.clone(), output_shape)
        .expect("Failed to create output tensor");

    // Verify output tensor state - this should NOT fail with "Destination buffer too small: 16 > 4"
    assert_eq!(
        output_tensor.len(),
        4,
        "Output tensor should have 4 elements"
    );
    assert_eq!(output_tensor.size(), 16, "Output tensor should be 16 bytes");

    // Verify we can copy output back to host (this is where the error manifests)
    let final_host_data = output_tensor
        .to_host_vec()
        .expect("Failed to copy output tensor to host");

    assert_eq!(
        final_host_data, output_data,
        "Final host data should match output data"
    );
    assert_eq!(
        final_host_data.len(),
        4,
        "Final host data should have 4 elements"
    );

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn device_tensor_size_bytes_consistency() {
    // Test that DeviceTensor size() always returns bytes and len() returns elements
    let fixture = GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Test with different tensor sizes
    let test_cases = vec![
        (vec![1.0f32], 1),                // 1 element, 4 bytes
        (vec![1.0f32, 2.0], 2),           // 2 elements, 8 bytes
        (vec![1.0f32, 2.0, 3.0, 4.0], 4), // 4 elements, 16 bytes
    ];

    for (data, expected_elements) in test_cases {
        let shape = TensorShape::from_dims(&[expected_elements]);
        let tensor = DeviceTensor::from_host_vec(backend, data.clone(), shape)
            .expect("Failed to create tensor");

        let expected_bytes = expected_elements * std::mem::size_of::<f32>();

        assert_eq!(
            tensor.len(),
            expected_elements,
            "Tensor should have {} elements",
            expected_elements
        );
        assert_eq!(
            tensor.size(),
            expected_bytes,
            "Tensor should be {} bytes",
            expected_bytes
        );

        // Verify buffer size matches tensor size
        assert_eq!(
            tensor.buffer().size(),
            expected_bytes,
            "Buffer should be {} bytes",
            expected_bytes
        );

        // Verify round-trip works
        let recovered = tensor.to_host_vec().expect("Failed to copy tensor to host");
        assert_eq!(recovered, data, "Round-trip data should match");
    }

    // Check for memory leaks
    fixture.assert_no_leak(5);
}
