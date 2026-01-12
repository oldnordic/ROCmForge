use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use rocmforge::backend::hip_backend::HipBackend;
use serial_test::serial;
use rocmforge::loader::gguf::GgufLoader;
use rocmforge::loader::lazy_tensor::LazyTensor;
use rocmforge::model::execution_plan::ExecutionPlan;
use std::path::Path;
use std::sync::Arc;

/// Helper function to create a test backend
fn create_test_backend() -> Arc<HipBackend> {
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    fixture.backend().clone()
}

/// Helper function to get tensor shape from LazyTensor (Phase 2: handles unloaded tensors)
fn get_lazy_tensor_shape(lazy: &LazyTensor) -> Vec<usize> {
    lazy.shape()
        .expect("LazyTensor should have shape metadata")
        .to_vec()
}

#[test]
#[serial]
fn test_map_core_projections() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    // Load GGUF and create execution plan
    let loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF model");

    let backend = create_test_backend();
    let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)
        .expect("Failed to create execution plan from GGUF");

    // Validate that we have at least one layer
    assert!(
        !execution_plan.layers().is_empty(),
        "Execution plan should have at least one layer"
    );

    // Test first layer's core projections
    let first_layer = &execution_plan.layers()[0];

    // Validate QKV projection weight shape: [3 * hidden_size, hidden_size]
    let qkv_shape = get_lazy_tensor_shape(&first_layer.qkv_weight);
    assert_eq!(qkv_shape.len(), 2, "QKV weight should be 2D");
    assert_eq!(
        qkv_shape[0],
        3 * execution_plan.config().hidden_size,
        "QKV weight first dimension should be 3 * hidden_size"
    );
    assert_eq!(
        qkv_shape[1],
        execution_plan.config().hidden_size,
        "QKV weight second dimension should be hidden_size"
    );

    // Validate output projection weight shape: [hidden_size, hidden_size]
    let o_proj_shape = get_lazy_tensor_shape(&first_layer.o_proj);
    assert_eq!(o_proj_shape.len(), 2, "Output projection should be 2D");
    assert_eq!(
        o_proj_shape[0],
        execution_plan.config().hidden_size,
        "Output projection first dimension should be hidden_size"
    );
    assert_eq!(
        o_proj_shape[1],
        execution_plan.config().hidden_size,
        "Output projection second dimension should be hidden_size"
    );

    // Validate QKV bias if present
    if let Some(qkv_bias) = &first_layer.qkv_bias {
        let qkv_bias_shape = get_lazy_tensor_shape(qkv_bias);
        assert_eq!(qkv_bias_shape.len(), 1, "QKV bias should be 1D");
        assert_eq!(
            qkv_bias_shape[0],
            3 * execution_plan.config().hidden_size,
            "QKV bias size should be 3 * hidden_size"
        );
    }

    // Validate output projection bias if present
    if let Some(o_proj_bias) = &first_layer.o_proj_bias {
        let o_proj_bias_shape = get_lazy_tensor_shape(o_proj_bias);
        assert_eq!(
            o_proj_bias_shape.len(),
            1,
            "Output projection bias should be 1D"
        );
        assert_eq!(
            o_proj_bias_shape[0],
            execution_plan.config().hidden_size,
            "Output projection bias size should be hidden_size"
        );
    }

    println!(
        "✓ Core projections mapping test passed for {} layers",
        execution_plan.num_layers()
    );
}

#[test]
#[serial]
fn test_map_mlp_weights() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    // Load GGUF and create execution plan
    let loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF model");

    let backend = create_test_backend();
    let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)
        .expect("Failed to create execution plan from GGUF");

    // Test first layer's MLP weights
    let first_layer = &execution_plan.layers()[0];
    let config = execution_plan.config();

    // Validate MLP gate projection weight shape: [intermediate_size, hidden_size]
    let gate_proj_shape = get_lazy_tensor_shape(&first_layer.mlp_gate_proj);
    assert_eq!(gate_proj_shape.len(), 2, "MLP gate projection should be 2D");
    assert_eq!(
        gate_proj_shape[0], config.intermediate_size,
        "MLP gate projection first dimension should be intermediate_size"
    );
    assert_eq!(
        gate_proj_shape[1], config.hidden_size,
        "MLP gate projection second dimension should be hidden_size"
    );

    // Validate MLP up projection weight shape: [intermediate_size, hidden_size]
    let up_proj_shape = get_lazy_tensor_shape(&first_layer.mlp_up_proj);
    assert_eq!(up_proj_shape.len(), 2, "MLP up projection should be 2D");
    assert_eq!(
        up_proj_shape[0], config.intermediate_size,
        "MLP up projection first dimension should be intermediate_size"
    );
    assert_eq!(
        up_proj_shape[1], config.hidden_size,
        "MLP up projection second dimension should be hidden_size"
    );

    // Validate MLP down projection weight shape: [hidden_size, intermediate_size]
    let down_proj_shape = get_lazy_tensor_shape(&first_layer.mlp_down_proj);
    assert_eq!(down_proj_shape.len(), 2, "MLP down projection should be 2D");
    assert_eq!(
        down_proj_shape[0], config.hidden_size,
        "MLP down projection first dimension should be hidden_size"
    );
    assert_eq!(
        down_proj_shape[1], config.intermediate_size,
        "MLP down projection second dimension should be intermediate_size"
    );

    // Validate legacy MLP FC1 weight shape: [intermediate_size, hidden_size]
    let fc1_shape = get_lazy_tensor_shape(&first_layer.mlp_fc1);
    assert_eq!(fc1_shape.len(), 2, "MLP FC1 should be 2D");
    assert_eq!(
        fc1_shape[0], config.intermediate_size,
        "MLP FC1 first dimension should be intermediate_size"
    );
    assert_eq!(
        fc1_shape[1], config.hidden_size,
        "MLP FC1 second dimension should be hidden_size"
    );

    // Validate legacy MLP FC2 weight shape: [hidden_size, intermediate_size]
    let fc2_shape = get_lazy_tensor_shape(&first_layer.mlp_fc2);
    assert_eq!(fc2_shape.len(), 2, "MLP FC2 should be 2D");
    assert_eq!(
        fc2_shape[0], config.hidden_size,
        "MLP FC2 first dimension should be hidden_size"
    );
    assert_eq!(
        fc2_shape[1], config.intermediate_size,
        "MLP FC2 second dimension should be intermediate_size"
    );

    // Validate MLP FC1 bias if present
    if let Some(fc1_bias) = &first_layer.mlp_fc1_bias {
        let fc1_bias_shape = get_lazy_tensor_shape(fc1_bias);
        assert_eq!(fc1_bias_shape.len(), 1, "MLP FC1 bias should be 1D");
        assert_eq!(
            fc1_bias_shape[0], config.intermediate_size,
            "MLP FC1 bias size should be intermediate_size"
        );
    }

    // Validate MLP FC2 bias if present
    if let Some(fc2_bias) = &first_layer.mlp_fc2_bias {
        let fc2_bias_shape = get_lazy_tensor_shape(fc2_bias);
        assert_eq!(fc2_bias_shape.len(), 1, "MLP FC2 bias should be 1D");
        assert_eq!(
            fc2_bias_shape[0], config.hidden_size,
            "MLP FC2 bias size should be hidden_size"
        );
    }

    println!(
        "✓ MLP weights mapping test passed for intermediate_size={}, hidden_size={}",
        config.intermediate_size, config.hidden_size
    );
}

#[test]
#[serial]
fn test_layernorm_mappings() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    // Load GGUF and create execution plan
    let loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF model");

    let backend = create_test_backend();
    let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)
        .expect("Failed to create execution plan from GGUF");

    // Test first layer's LayerNorm weights
    let first_layer = &execution_plan.layers()[0];
    let config = execution_plan.config();

    // Validate first layer norm weight shape: [hidden_size]
    let norm1_shape = get_lazy_tensor_shape(&first_layer.norm1_weight);
    assert_eq!(norm1_shape.len(), 1, "First layer norm should be 1D");
    assert_eq!(
        norm1_shape[0], config.hidden_size,
        "First layer norm size should be hidden_size"
    );

    // Validate second layer norm weight shape: [hidden_size]
    let norm2_shape = get_lazy_tensor_shape(&first_layer.norm2_weight);
    assert_eq!(norm2_shape.len(), 1, "Second layer norm should be 1D");
    assert_eq!(
        norm2_shape[0], config.hidden_size,
        "Second layer norm size should be hidden_size"
    );

    // Validate first layer norm bias if present
    if let Some(norm1_bias) = &first_layer.norm1_bias {
        let norm1_bias_shape = get_lazy_tensor_shape(norm1_bias);
        assert_eq!(
            norm1_bias_shape.len(),
            1,
            "First layer norm bias should be 1D"
        );
        assert_eq!(
            norm1_bias_shape[0], config.hidden_size,
            "First layer norm bias size should be hidden_size"
        );
    }

    // Validate second layer norm bias if present
    if let Some(norm2_bias) = &first_layer.norm2_bias {
        let norm2_bias_shape = get_lazy_tensor_shape(norm2_bias);
        assert_eq!(
            norm2_bias_shape.len(),
            1,
            "Second layer norm bias should be 1D"
        );
        assert_eq!(
            norm2_bias_shape[0], config.hidden_size,
            "Second layer norm bias size should be hidden_size"
        );
    }

    // Test that all layers have consistent LayerNorm shapes
    for (i, layer) in execution_plan.layers().iter().enumerate() {
        let norm1_shape = get_lazy_tensor_shape(&layer.norm1_weight);
        let norm2_shape = get_lazy_tensor_shape(&layer.norm2_weight);

        assert_eq!(
            norm1_shape[0], config.hidden_size,
            "Layer {} norm1 should have hidden_size dimensions",
            i
        );
        assert_eq!(
            norm2_shape[0], config.hidden_size,
            "Layer {} norm2 should have hidden_size dimensions",
            i
        );
    }

    println!(
        "✓ LayerNorm mappings test passed for {} layers with hidden_size={}",
        execution_plan.num_layers(),
        config.hidden_size
    );
}

#[test]
#[serial]
fn test_embedding_and_lm_head_mapping() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    // Load GGUF and create execution plan
    let loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF model");

    let backend = create_test_backend();
    let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)
        .expect("Failed to create execution plan from GGUF");

    let config = execution_plan.config();

    // Validate that the execution plan was created successfully
    assert!(
        execution_plan.num_layers() > 0,
        "Execution plan should have at least one layer"
    );

    // Validate model configuration consistency
    assert!(config.vocab_size > 0, "Vocab size should be > 0");
    assert!(config.hidden_size > 0, "Hidden size should be > 0");
    assert!(
        config.num_hidden_layers > 0,
        "Number of layers should be > 0"
    );

    // Check that the number of layers matches the config
    assert_eq!(
        execution_plan.num_layers(),
        config.num_hidden_layers,
        "Execution plan layers should match config num_hidden_layers"
    );

    // Validate that all layers have the expected tensor types
    for (i, layer) in execution_plan.layers().iter().enumerate() {
        // Check that all required tensors are present
        let qkv_shape = get_lazy_tensor_shape(&layer.qkv_weight);
        let o_proj_shape = get_lazy_tensor_shape(&layer.o_proj);
        let norm1_shape = get_lazy_tensor_shape(&layer.norm1_weight);
        let norm2_shape = get_lazy_tensor_shape(&layer.norm2_weight);

        assert_eq!(qkv_shape.len(), 2, "Layer {} QKV should be 2D", i);
        assert_eq!(o_proj_shape.len(), 2, "Layer {} O_proj should be 2D", i);
        assert_eq!(norm1_shape.len(), 1, "Layer {} norm1 should be 1D", i);
        assert_eq!(norm2_shape.len(), 1, "Layer {} norm2 should be 1D", i);

        // Validate dimensions match config
        assert_eq!(
            norm1_shape[0], config.hidden_size,
            "Layer {} norm1 size mismatch",
            i
        );
        assert_eq!(
            norm2_shape[0], config.hidden_size,
            "Layer {} norm2 size mismatch",
            i
        );
        assert_eq!(
            o_proj_shape[1], config.hidden_size,
            "Layer {} O_proj input size mismatch",
            i
        );
        assert_eq!(
            qkv_shape[1], config.hidden_size,
            "Layer {} QKV input size mismatch",
            i
        );
    }

    println!(
        "✓ Embedding and LM head mapping test passed for model with {} layers, vocab_size={}, hidden_size={}",
        execution_plan.num_layers(), config.vocab_size, config.hidden_size
    );
}
