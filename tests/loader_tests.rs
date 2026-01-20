//! Comprehensive TDD tests for loader modules

// Declare common module for test fixtures
mod common;

use rocmforge::loader::{
    GgufLoader, OnnxDataType, OnnxLoader, OnnxSession, OnnxTensor,
};
use rocmforge::loader::GgufTensorType;
use anyhow::Context;
use std::io::Write;

// Use common fixtures
use common::{create_temp_file, create_test_gguf, create_test_gguf_with_f32, NamedTempFile};

/// Rewrite of test_gguf_model_loading using current GgufLoader API.
///
/// Tests that GgufLoader::new() successfully loads a valid GGUF file
/// and returns metadata correctly.
#[test]
fn test_gguf_model_loading() -> anyhow::Result<()> {
    let temp_file = create_temp_file()?;
    create_test_gguf(temp_file.path())?;

    let path = temp_file.path().to_str()
        .context("Failed to convert temp file path to string")?;
    let loader = GgufLoader::new(path)
        .context("Failed to create GGUF loader")?;
    let metadata = loader.metadata();

    assert_eq!(metadata.architecture, "test");
    Ok(())
}

/// Rewrite of test_gguf_tensor_access using current GgufLoader API.
///
/// Tests that loaded tensor metadata can be accessed via lazy_tensors.
/// Note: Current API uses lazy_tensors (HashMap<String, LazyTensor>).
#[test]
fn test_gguf_tensor_access() -> anyhow::Result<()> {
    let temp_file = create_temp_file()?;
    create_test_gguf_with_f32(temp_file.path())?;

    let path = temp_file.path().to_str()
        .context("Failed to convert temp file path to string")?;
    let loader = GgufLoader::new(path)
        .context("Failed to create GGUF loader")?;

    // Access via lazy_tensors (current API)
    let tensor = loader.lazy_tensors.get("f32_tensor")
        .context("Tensor 'f32_tensor' not found in lazy_tensors")?;

    let shape = tensor.shape().context("Tensor should have shape")?;
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 2);
    assert_eq!(shape[1], 2);
    assert_eq!(tensor.tensor_type(), Some(GgufTensorType::F32));

    Ok(())
}

/// Rewrite of test_gguf_f32_conversion using current GgufLoader API.
///
/// Tests that F32 tensor type is correctly identified in loaded metadata.
#[test]
fn test_gguf_f32_conversion() -> anyhow::Result<()> {
    let temp_file = create_temp_file()?;
    create_test_gguf_with_f32(temp_file.path())?;

    let path = temp_file.path().to_str()
        .context("Failed to convert temp file path to string")?;
    let loader = GgufLoader::new(path)
        .context("Failed to create GGUF loader")?;
    let tensor = loader.lazy_tensors.get("f32_tensor")
        .context("Tensor 'f32_tensor' not found in lazy_tensors")?;

    assert_eq!(tensor.tensor_type(), Some(GgufTensorType::F32));
    assert_eq!(tensor.shape(), Some(vec![2, 2].as_slice()));

    Ok(())
}

/// Rewrite of test_gguf_invalid_magic using current GgufLoader API.
///
/// Tests error handling for invalid GGUF magic number.
#[test]
fn test_gguf_invalid_magic() -> anyhow::Result<()> {
    let mut file = create_temp_file()?;
    file.write_all(&0x12345678u32.to_le_bytes())?; // Invalid magic
    file.write_all(&3u32.to_le_bytes())?; // version
    file.write_all(&0u64.to_le_bytes())?; // tensor_count
    file.write_all(&0u64.to_le_bytes())?; // kv_count

    let path = file.path().to_str()
        .context("Failed to convert temp file path to string")?;
    let result = GgufLoader::new(path);
    assert!(result.is_err());

    // Verify error message mentions invalid magic
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.to_lowercase().contains("magic") || error_msg.to_lowercase().contains("invalid"));

    Ok(())
}

/// Rewrite of test_gguf_unsupported_version using current GgufLoader API.
///
/// Tests error handling for unsupported GGUF versions (only version 3 is supported).
#[test]
fn test_gguf_unsupported_version() -> anyhow::Result<()> {
    let mut file = create_temp_file()?;
    file.write_all(&0x46554747u32.to_le_bytes())?; // "GGUF" magic
    file.write_all(&999u32.to_le_bytes())?; // Unsupported version
    file.write_all(&0u64.to_le_bytes())?; // tensor_count
    file.write_all(&0u64.to_le_bytes())?; // kv_count

    let result = GgufLoader::new(file.path().to_str().context("Failed to convert path to string")?);
    assert!(result.is_err());

    // Verify error message mentions version
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.to_lowercase().contains("version") || error_msg.to_lowercase().contains("unsupported"));

    Ok(())
}

#[test]
fn test_gguf_tensor_type_conversion() {
    // Test valid conversions
    assert!(matches!(
        GgufTensorType::from_u32(0),
        Ok(GgufTensorType::F32)
    ));
    assert!(matches!(
        GgufTensorType::from_u32(1),
        Ok(GgufTensorType::F16)
    ));
    assert!(matches!(
        GgufTensorType::from_u32(2),
        Ok(GgufTensorType::Q4_0)
    ));
    // Q4_1, Q5_0, Q5_1 are not supported - should return Err
    assert!(GgufTensorType::from_u32(3).is_err()); // Q4_1 not supported
    assert!(GgufTensorType::from_u32(6).is_err()); // Q5_0 not supported
    assert!(GgufTensorType::from_u32(7).is_err()); // Q5_1 not supported
    assert!(matches!(
        GgufTensorType::from_u32(8),
        Ok(GgufTensorType::Q8_0)
    ));
    assert!(matches!(
        GgufTensorType::from_u32(20),
        Ok(GgufTensorType::Mxfp4)
    ));
    assert!(matches!(
        GgufTensorType::from_u32(21),
        Ok(GgufTensorType::Mxfp6E2m3)
    ));
    assert!(matches!(
        GgufTensorType::from_u32(22),
        Ok(GgufTensorType::Mxfp6E3m2)
    ));

    // Test invalid conversion
    let result = GgufTensorType::from_u32(999);
    assert!(result.is_err());
}

#[test]
fn test_gguf_tensor_type_element_size() {
    assert_eq!(GgufTensorType::F32.element_size(), 4);
    assert_eq!(GgufTensorType::F16.element_size(), 2);
    assert_eq!(GgufTensorType::Q8_0.element_size(), 32); // Block-based
    assert_eq!(GgufTensorType::Q4_0.element_size(), 32); // Block-based
    assert_eq!(GgufTensorType::Mxfp4.element_size(), 32); // Block-based
    assert_eq!(GgufTensorType::Mxfp6E2m3.element_size(), 32); // Block-based
}

#[test]
fn test_onnx_tensor_creation() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = OnnxTensor::new("test".to_string(), vec![2, 2], &data);

    assert_eq!(tensor.name, "test");
    assert_eq!(tensor.shape, vec![2, 2]);
    assert!(matches!(tensor.data_type, OnnxDataType::F32));
    assert_eq!(tensor.data.len(), 16); // 4 elements * 4 bytes
}

#[test]
fn test_onnx_tensor_f32_conversion() -> anyhow::Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = OnnxTensor::new("test".to_string(), vec![2, 2], &data);

    let result = tensor.get_data_f32()
        .context("Failed to get F32 data from tensor")?;
    assert_eq!(result, data);

    Ok(())
}

#[test]
fn test_onnx_tensor_invalid_conversion() {
    let data = vec![1i32, 2, 3, 4];
    let tensor = OnnxTensor::new("test".to_string(), vec![2, 2], &data);

    // This tensor is I32, not F32
    let result = tensor.get_data_f32();
    assert!(result.is_err());
}

#[test]
fn test_onnx_loader_creation() {
    let loader = OnnxLoader::new();
    assert!(!loader.is_model_loaded());
}

#[test]
fn test_onnx_model_loading() -> anyhow::Result<()> {
    let mut loader = OnnxLoader::new();

    // Create a dummy ONNX model file
    let temp_file = create_temp_file()?;
    let result = loader.load_model(temp_file.path());

    // In our mock implementation, this should succeed
    assert!(result.is_ok());
    assert!(loader.is_model_loaded());

    Ok(())
}

#[test]
fn test_onnx_inference() -> anyhow::Result<()> {
    let mut loader = OnnxLoader::new();

    // Create a dummy ONNX model file
    let temp_file = create_temp_file()?;
    loader.load_model(temp_file.path())
        .context("Failed to load model")?;

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor = OnnxTensor::new("input".to_string(), vec![2, 2], &input_data);

    let outputs = loader.run_inference(&[input_tensor])
        .context("Failed to run inference")?;
    assert_eq!(outputs.len(), 1);

    // In our mock implementation, this should be an identity operation
    let output_data = outputs[0].get_data_f32()
        .context("Failed to get F32 data")?;
    assert_eq!(output_data, input_data);

    Ok(())
}

#[test]
fn test_onnx_inference_without_model() {
    let loader = OnnxLoader::new();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor = OnnxTensor::new("input".to_string(), vec![2, 2], &input_data);

    let result = loader.run_inference(&[input_tensor]);
    assert!(result.is_err());
}

#[test]
fn test_onnx_inference_empty_inputs() -> anyhow::Result<()> {
    let mut loader = OnnxLoader::new();

    let temp_file = create_temp_file().context("Failed to create temp file")?;
    loader.load_model(temp_file.path()).context("Failed to load model")?;

    let result = loader.run_inference(&[]);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_onnx_session_input_output_names() -> anyhow::Result<()> {
    let temp_file = create_temp_file()?;
    let session = OnnxSession::new(temp_file.path())
        .context("Failed to create ONNX session")?;

    assert_eq!(session.input_names(), &["input"]);
    assert_eq!(session.output_names(), &["output"]);

    Ok(())
}

#[test]
fn test_onnx_session_run() -> anyhow::Result<()> {
    let temp_file = create_temp_file()?;
    let session = OnnxSession::new(temp_file.path())
        .context("Failed to create ONNX session")?;

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor = OnnxTensor::new("input".to_string(), vec![2, 2], &input_data);

    let outputs = session.run(&[input_tensor])
        .context("Failed to run session")?;
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name, "output");
    assert_eq!(outputs[0].shape, vec![2, 2]);

    Ok(())
}

// Property-based tests
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_onnx_tensor_properties(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..50),
        rows in 1usize..10,
        cols in 1usize..10
    ) {
        let total_elements = rows * cols;
        let truncated_data = data.into_iter().take(total_elements).collect::<Vec<_>>();

        let tensor = OnnxTensor::new(
            "test".to_string(),
            vec![rows, cols],
            &truncated_data,
        );

        prop_assert_eq!(tensor.name.as_str(), "test");
        let expected_shape = vec![rows, cols];
        let tensor_shape = tensor.shape.clone();
        prop_assert_eq!(tensor_shape, expected_shape);
        prop_assert!(matches!(tensor.data_type, OnnxDataType::F32));
        prop_assert_eq!(tensor.data.len(), total_elements * 4);

        let converted = tensor.get_data_f32().unwrap();
        prop_assert_eq!(converted.len(), total_elements);
        prop_assert_eq!(converted, truncated_data);
    }

    #[test]
    fn test_gguf_tensor_type_properties(
        type_id in 0u32..23u32
    ) {
        let result = GgufTensorType::from_u32(type_id);
        // Only test valid type IDs (Q4_1=3, Q5_0=6, Q5_1=7 are no longer supported)
        if [0, 1, 2, 8, 10, 11, 12, 13, 14, 20, 21, 22].contains(&type_id) {
            prop_assert!(result.is_ok());

            let tensor_type = result.unwrap();
            let size = tensor_type.element_size();
            prop_assert!(size > 0);
            // K-quants have 256-byte blocks, others have 32 or less
            prop_assert!(size <= 256); // Max element size for K-quant block-based types
        }
    }
}
