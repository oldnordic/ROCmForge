//! Integration tests for GGUF loader using real model files.
//!
//! These tests use the Qwen2.5 model from the Memoria project.
//! They will be skipped if the model file is not available.

use rocmforge::loader::{GgufFile, TensorView};

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
fn test_load_qwen25_model() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");

    // Verify tensor count > 0
    assert!(gguf.tensor_count() > 0, "Expected non-zero tensor count");

    // Verify architecture == "qwen2"
    assert_eq!(
        gguf.metadata.architecture, "qwen2",
        "Expected architecture 'qwen2', got '{}'",
        gguf.metadata.architecture
    );

    // Verify metadata.block_count() > 0
    assert!(
        gguf.metadata.block_count() > 0,
        "Expected non-zero block count"
    );

    // Verify can get a tensor view (e.g., "token_embd.weight")
    let tensor_result = gguf.tensor("token_embd.weight");
    assert!(
        tensor_result.is_ok(),
        "Failed to get tensor: {:?}",
        tensor_result.err()
    );

    let tensor_opt = tensor_result.unwrap();
    assert!(tensor_opt.is_some(), "token_embd.weight tensor not found");

    let tensor: TensorView<'_> = tensor_opt.unwrap();

    // Verify TensorView::element_count() > 0
    assert!(
        tensor.element_count() > 0,
        "Expected non-zero element count"
    );

    // Verify tensor has non-empty data
    assert!(!tensor.data.is_empty(), "Expected non-empty tensor data");
}

#[test]
fn test_tokenizer_data() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");

    let tokenizer = gguf.tokenizer_data();

    // Verify tokens.len() > 0
    assert!(
        !tokenizer.tokens.is_empty(),
        "Expected non-empty tokens array"
    );

    // Verify merges.len() > 0
    assert!(
        !tokenizer.merges.is_empty(),
        "Expected non-empty merges array"
    );
}

#[test]
fn test_tensor_iteration() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");

    // Verify tensor_names() returns an iterator with expected count
    let names: Vec<&str> = gguf.tensor_names().collect();
    assert_eq!(names.len(), gguf.tensor_count());

    // Verify has_tensor works for known tensors
    assert!(gguf.has_tensor("token_embd.weight"));

    // Verify has_tensor returns false for unknown tensors
    assert!(!gguf.has_tensor("nonexistent_tensor"));
}

#[test]
fn test_tensor_out_of_bounds() {
    if skip_if_model_missing() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }

    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");

    // All real tensors should be within bounds
    for name in gguf.tensor_names() {
        let result = gguf.tensor(name);
        assert!(
            result.is_ok(),
            "Tensor {} failed bounds check: {:?}",
            name,
            result.err()
        );
    }
}
