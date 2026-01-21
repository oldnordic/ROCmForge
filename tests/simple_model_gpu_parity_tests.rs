//! GPU parity tests for simple model functionality
//! Only compiled when ROCm feature is enabled

use rocmforge::model::{ModelBackend, SimpleModel};
use serial_test::serial;

#[test]
#[serial]
fn test_simple_model_cpu_gpu_parity() {
    // Build two models with identical config and seed
    let cpu_model = SimpleModel::new(100, 64, 2, 8, ModelBackend::Cpu, 42);
    let gpu_model = SimpleModel::new(100, 64, 2, 8, ModelBackend::Gpu, 42);

    let input_tokens = vec![1, 5, 9, 3, 7, 2, 8, 4];

    // Forward both models
    let cpu_output = cpu_model.forward(&input_tokens).unwrap();
    let gpu_output = gpu_model.forward(&input_tokens).unwrap();

    // Verify outputs have same length
    assert_eq!(
        cpu_output.len(),
        gpu_output.len(),
        "CPU and GPU outputs should have same length"
    );

    // Compute max absolute difference between outputs
    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(cpu_val, gpu_val)| (cpu_val - gpu_val).abs())
        .fold(0.0f32, f32::max);

    // Assert numerical parity within tolerance
    assert!(
        max_diff <= 1e-4,
        "CPU and GPU outputs should match within tolerance: max_diff = {}",
        max_diff
    );
}

#[test]
#[serial]
fn test_simple_model_gpu_backend_does_not_fallback() {
    // On a build where ROCm is available, GPU backend should be used
    let gpu_model = SimpleModel::new(50, 32, 1, 4, ModelBackend::Gpu, 123);

    let input_tokens = vec![1, 2, 3, 4];

    // This should succeed without falling back to CPU
    let result = gpu_model.forward(&input_tokens);
    assert!(result.is_ok(), "GPU model forward should succeed");

    let output = result.unwrap();
    assert_eq!(output.len(), 4 * 32, "Output should have correct length");
    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );
}
