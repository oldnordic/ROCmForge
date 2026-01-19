//! Execution Plan Construction Tests
//!
//! Tests for Phase F.7 - ExecutionPlan::from_gguf() implementation.
//! These tests verify that ExecutionPlan can be constructed from GGUF files
//! using the helper functions for weight mapping.

use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use rocmforge::loader::gguf::GgufLoader;
use rocmforge::model::ExecutionPlan;
use serial_test::serial;

/// Test execution plan construction from GGUF
#[test]
#[serial]
fn test_execution_plan_construction() {
    // Load tiny model GGUF using correct API
    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");

    // Create backend (this will be used for GPU tensor creation)
    let fixture = GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Construct ExecutionPlan from GGUF loader
    let execution_plan = ExecutionPlan::from_gguf(&backend, &gguf_loader)
        .expect("Failed to construct ExecutionPlan");

    // Verify basic structure
    let config = execution_plan.config();
    assert_eq!(
        execution_plan.num_layers(),
        config.num_hidden_layers,
        "Layer count should match config"
    );

    println!("✅ Execution plan construction test passed");
    // Check for memory leaks
    fixture.assert_no_leak(5);
}

/// Test layer weight shapes
#[test]
#[serial]
fn test_layer_weight_shapes() {
    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");
    let fixture = GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let execution_plan = ExecutionPlan::from_gguf(&backend, &gguf_loader)
        .expect("Failed to construct ExecutionPlan");

    let config = execution_plan.config();

    // Verify each layer has required components using correct field names
    for (layer_idx, layer_plan) in execution_plan.layers().iter().enumerate() {
        // Check attention weights - using actual field names
        // QKV is fused in current implementation
        let qkv_shape = layer_plan.qkv_weight.shape().unwrap();
        assert_eq!(
            qkv_shape.len(),
            2,
            "Layer {}: QKV weight should be 2D",
            layer_idx
        );
        assert_eq!(
            qkv_shape[0],
            3 * config.hidden_size,
            "Layer {}: QKV weight input dim",
            layer_idx
        );
        assert_eq!(
            qkv_shape[1], config.hidden_size,
            "Layer {}: QKV weight output dim",
            layer_idx
        );

        // Check MLP weights using actual field names
        let gate_shape = layer_plan.mlp_gate_proj.shape().unwrap();
        assert_eq!(
            gate_shape.len(),
            2,
            "Layer {}: MLP gate should be 2D",
            layer_idx
        );
        assert_eq!(
            gate_shape[0], config.intermediate_size,
            "Layer {}: MLP gate input dim",
            layer_idx
        );
        assert_eq!(
            gate_shape[1], config.hidden_size,
            "Layer {}: MLP gate output dim",
            layer_idx
        );

        // Check LayerNorm weights using actual field names
        let ln1_shape = layer_plan.norm1_weight.shape().unwrap();
        assert_eq!(
            ln1_shape.len(),
            1,
            "Layer {}: LN1 weight should be 1D",
            layer_idx
        );
        assert_eq!(
            ln1_shape[0], config.hidden_size,
            "Layer {}: LN1 weight dim",
            layer_idx
        );
        // Check for memory leaks
        fixture.assert_no_leak(5);
    }

    println!("✅ Layer weight shapes test passed");
}

/// Test that all required tensors are present
#[test]
#[serial]
fn test_all_required_tensors_present() {
    let gguf_path = "tests/data/tiny_model.gguf";
    let gguf_loader = GgufLoader::new(gguf_path).expect("Failed to load GGUF");
    let fixture = GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    let execution_plan = ExecutionPlan::from_gguf(&backend, &gguf_loader)
        .expect("Failed to construct ExecutionPlan");

    let config = execution_plan.config();

    // Verify all layers have complete weight sets using actual field names
    for (layer_idx, layer_plan) in execution_plan.layers().iter().enumerate() {
        // Check that all required fields are present (non-empty tensors)
        assert!(
            layer_plan.qkv_weight.shape().unwrap().iter().product::<usize>() > 0,
            "Layer {}: QKV weight should be present",
            layer_idx
        );
        assert!(
            layer_plan.o_proj.shape().unwrap().iter().product::<usize>() > 0,
            "Layer {}: O projection should be present",
            layer_idx
        );
        assert!(
            layer_plan.mlp_gate_proj.shape().unwrap().iter().product::<usize>() > 0,
            "Layer {}: MLP gate should be present",
            layer_idx
        );
        assert!(
            layer_plan.mlp_up_proj.shape().unwrap().iter().product::<usize>() > 0,
            "Layer {}: MLP up should be present",
            layer_idx
        );
        assert!(
            layer_plan.mlp_down_proj.shape().unwrap().iter().product::<usize>() > 0,
            "Layer {}: MLP down should be present",
            layer_idx
        );
        assert!(
            layer_plan.norm1_weight.shape().unwrap().iter().product::<usize>() > 0,
            "Layer {}: Norm1 weight should be present",
            layer_idx
        );
        assert!(
            layer_plan.norm2_weight.shape().unwrap().iter().product::<usize>() > 0,
            "Layer {}: Norm2 weight should be present",
            layer_idx
        );
        // Check for memory leaks
        fixture.assert_no_leak(5);
    }

    println!("✅ All required tensors present test passed");
}
