//! Execution Plan Forward Pass Tests
//!
//! Tests for Phase G.2 - ExecutionPlan::forward() method implementation.
//! These tests verify that the forward pass works correctly with real GGUF weights.

use rocmforge::backend::{DeviceTensor, HipBackend};
use rocmforge::loader::gguf::GgufLoader;
use rocmforge::loader::TensorShape;
use rocmforge::model::{ExecutionPlan, ModelConfig};
use serial_test::serial;

/// Test basic forward pass functionality
#[test]
#[serial]
fn test_forward_pass_basic() {
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Load tiny model GGUF
    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    // Construct ExecutionPlan
    let execution_plan =
        ExecutionPlan::from_gguf(backend, &gguf_loader).expect("Failed to construct ExecutionPlan");

    // Load embedding weights
    let gpu_tensors = gguf_loader
        .load_to_gpu(backend)
        .expect("Failed to load tensors to GPU");

    // Get embedding weights from loaded tensors
    let embedding_weights = gpu_tensors
        .get("token_embd.weight")
        .expect("Missing embedding weights")
        .clone();

    // Create test input tokens (simple sequence)
    let input_tokens = vec![1, 2, 3, 4, 5]; // 5 token sequence

    // Run forward pass
    let output = execution_plan
        .forward(backend, &input_tokens, &embedding_weights)
        .expect("Forward pass failed");

    // Verify output shape
    let expected_shape = vec![input_tokens.len(), execution_plan.config().hidden_size];
    assert_eq!(
        output.shape().dims(),
        &expected_shape,
        "Output shape should be [seq_len, hidden_size]"
    );

    println!("✅ Basic forward pass test passed");
    println!("   Input tokens: {:?}", input_tokens);
    println!("   Output shape: {:?}", output.shape().dims());

    drop(embedding_weights);
    drop(gpu_tensors);
    drop(output);
    fixture.assert_no_leak(5);
}

/// Test embedding lookup functionality
#[test]
#[serial]
fn test_embedding_lookup() {
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    let execution_plan =
        ExecutionPlan::from_gguf(backend, &gguf_loader).expect("Failed to construct ExecutionPlan");

    let gpu_tensors = gguf_loader
        .load_to_gpu(backend)
        .expect("Failed to load tensors to GPU");

    let embedding_weights = gpu_tensors
        .get("token_embd.weight")
        .expect("Missing embedding weights")
        .clone();

    // Test with single token
    let single_token = vec![42];
    let embedding = execution_plan
        .embedding_lookup(backend, &single_token, &embedding_weights)
        .expect("Embedding lookup failed");

    let expected_shape = vec![1, execution_plan.config().hidden_size];
    assert_eq!(
        embedding.shape().dims(),
        &expected_shape,
        "Single token embedding should have shape [1, hidden_size]"
    );

    // Test with multiple tokens
    let multiple_tokens = vec![1, 2, 3];
    let embeddings = execution_plan
        .embedding_lookup(backend, &multiple_tokens, &embedding_weights)
        .expect("Multi-token embedding lookup failed");

    let expected_multi_shape = vec![3, execution_plan.config().hidden_size];
    assert_eq!(
        embeddings.shape().dims(),
        &expected_multi_shape,
        "Multi-token embedding should have shape [seq_len, hidden_size]"
    );

    println!("✅ Embedding lookup test passed");

    drop(embedding_weights);
    drop(gpu_tensors);
    drop(embedding);
    drop(embeddings);
    fixture.assert_no_leak(5);
}

/// Test individual layer forward pass
#[test]
#[serial]
fn test_layer_forward_pass() {
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    let execution_plan =
        ExecutionPlan::from_gguf(backend, &gguf_loader).expect("Failed to construct ExecutionPlan");

    // Create test input tensor [seq_len=2, hidden_size]
    let seq_len = 2;
    let hidden_size = execution_plan.config().hidden_size;
    let test_input_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
    let test_input =
        DeviceTensor::empty(backend, test_input_shape).expect("Failed to create test input tensor");

    // Test first layer
    let first_layer = &execution_plan.layers()[0];
    let layer_output = execution_plan
        .forward_layer(backend, &test_input, first_layer, None, 0)
        .expect("Layer forward pass failed");

    // Verify output shape matches input shape
    assert_eq!(
        layer_output.shape().dims(),
        test_input.shape().dims(),
        "Layer output should preserve input shape"
    );

    println!("✅ Layer forward pass test passed");
    println!("   Input shape: {:?}", test_input.shape().dims());
    println!("   Output shape: {:?}", layer_output.shape().dims());

    drop(test_input);
    drop(layer_output);
    fixture.assert_no_leak(5);
}

/// Test forward pass with different sequence lengths
#[test]
#[serial]
fn test_forward_pass_varying_seq_len() {
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    let execution_plan =
        ExecutionPlan::from_gguf(backend, &gguf_loader).expect("Failed to construct ExecutionPlan");

    let gpu_tensors = gguf_loader
        .load_to_gpu(backend)
        .expect("Failed to load tensors to GPU");

    let embedding_weights = gpu_tensors
        .get("token_embd.weight")
        .expect("Missing embedding weights")
        .clone();

    // Test different sequence lengths
    let test_sequences = vec![
        vec![1],          // Length 1
        vec![1, 2],       // Length 2
        vec![1, 2, 3, 4], // Length 4
    ];

    for (seq_idx, input_tokens) in test_sequences.iter().enumerate() {
        let output = execution_plan
            .forward(backend, input_tokens, &embedding_weights)
            .expect(&format!(
                "Forward pass failed for sequence length {}",
                input_tokens.len()
            ));

        let expected_shape = vec![input_tokens.len(), execution_plan.config().hidden_size];
        assert_eq!(
            output.shape().dims(),
            &expected_shape,
            "Output shape incorrect for sequence length {}",
            input_tokens.len()
        );

        println!("   Sequence {} (len {}): OK", seq_idx, input_tokens.len());
    }

    println!("✅ Varying sequence length test passed");

    drop(embedding_weights);
    drop(gpu_tensors);
    fixture.assert_no_leak(5);
}

/// Test forward pass error handling
#[test]
#[serial]
fn test_forward_pass_error_handling() {
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    let execution_plan =
        ExecutionPlan::from_gguf(backend, &gguf_loader).expect("Failed to construct ExecutionPlan");

    // Test with empty input
    let empty_tokens: Vec<u32> = vec![];
    let dummy_embedding_shape = TensorShape::from_dims(&[100, execution_plan.config().hidden_size]);
    let dummy_embedding = DeviceTensor::empty(backend, dummy_embedding_shape)
        .expect("Failed to create dummy embedding");

    let result = execution_plan.forward(backend, &empty_tokens, &dummy_embedding);
    assert!(result.is_err(), "Forward pass should fail with empty input");

    println!("✅ Error handling test passed");

    drop(dummy_embedding);
    fixture.assert_no_leak(5);
}

/// Test forward pass with real model inference pattern
#[test]
#[serial]
fn test_forward_pass_inference_pattern() {
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    let execution_plan =
        ExecutionPlan::from_gguf(backend, &gguf_loader).expect("Failed to construct ExecutionPlan");

    let gpu_tensors = gguf_loader
        .load_to_gpu(backend)
        .expect("Failed to load tensors to GPU");

    let embedding_weights = gpu_tensors
        .get("token_embd.weight")
        .expect("Missing embedding weights")
        .clone();

    // Simulate a simple inference pattern: start with BOS token, add tokens incrementally
    let mut current_tokens = vec![1]; // Assuming 1 is BOS token

    // Add tokens one by one (simulating autoregressive generation)
    for new_token in 2..=6 {
        current_tokens.push(new_token);

        let output = execution_plan
            .forward(backend, &current_tokens, &embedding_weights)
            .expect("Forward pass failed during inference simulation");

        // Verify output is consistent
        let expected_shape = vec![current_tokens.len(), execution_plan.config().hidden_size];
        assert_eq!(
            output.shape().dims(),
            &expected_shape,
            "Output shape incorrect during inference step"
        );
    }

    println!("✅ Inference pattern test passed");
    println!("   Final sequence length: {}", current_tokens.len());

    drop(embedding_weights);
    drop(gpu_tensors);
    fixture.assert_no_leak(5);
}

/// Performance benchmark for forward pass
#[test]
#[serial]
fn test_forward_pass_performance() {
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    let execution_plan =
        ExecutionPlan::from_gguf(backend, &gguf_loader).expect("Failed to construct ExecutionPlan");

    let gpu_tensors = gguf_loader
        .load_to_gpu(backend)
        .expect("Failed to load tensors to GPU");

    let embedding_weights = gpu_tensors
        .get("token_embd.weight")
        .expect("Missing embedding weights")
        .clone();

    // Test with moderate sequence length
    let input_tokens: Vec<u32> = (1..=32).collect(); // 32 tokens

    // Warm up
    let _ = execution_plan
        .forward(backend, &input_tokens, &embedding_weights)
        .expect("Warmup forward pass failed");

    // Time multiple runs
    let start_time = std::time::Instant::now();
    let num_runs = 5;

    for _ in 0..num_runs {
        let _ = execution_plan
            .forward(backend, &input_tokens, &embedding_weights)
            .expect("Forward pass failed during benchmark");
    }

    let elapsed = start_time.elapsed();
    let avg_time = elapsed / num_runs;

    println!("✅ Performance benchmark completed");
    println!("   Sequence length: {}", input_tokens.len());
    println!("   Number of layers: {}", execution_plan.num_layers());
    println!("   Average time per forward pass: {:?}", avg_time);
    println!(
        "   Tokens per second: {:.2}",
        input_tokens.len() as f64 / avg_time.as_secs_f64()
    );

    drop(embedding_weights);
    drop(gpu_tensors);
    fixture.assert_no_leak(5);
}
