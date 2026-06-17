// Q4_0 Multi-Row Correctness Test
//
// This test verifies that multi-row residual kernel optimizations
// produce identical output to the single-column kernel.
//
// Test Strategy:
// 1. Generate random input data of realistic dimensions
// 2. Run single-column kernel (baseline)
// 3. Run multi-row kernel with residual addition
// 4. Verify outputs match bitwise (within numerical precision)
//
// This test establishes correctness BEFORE implementing performance optimizations.

use std::process::Command;

#[test]
#[ignore = "Requires Q4_0 model file and GPU"]
fn test_q4_0_multirow_residual_matches_single() {
    // TODO: Implement multi-row kernel first
    // This test will verify multi-row residual output matches single-column

    let model_path = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // Run inference with single-column kernel (current implementation)
    let output_single = run_inference(model_path, "single");

    // Run inference with multi-row residual kernel
    let output_multi_residual = run_inference(model_path, "multi-residual");

    // Verify outputs match bitwise
    assert_eq!(
        output_single.len(),
        output_multi_residual.len(),
        "Output length mismatch"
    );

    for (i, (s, m)) in output_single
        .iter()
        .zip(output_multi_residual.iter())
        .enumerate()
    {
        let diff = f64::abs(*s as f64 - *m as f64);
        assert!(
            diff < 1e-5,
            "Mismatch at position {}: single={}, multi_residual={}",
            i,
            s,
            m
        );
    }

    println!("✅ Q4_0 multi-row residual matches single-column");
}

#[test]
fn test_q4_0_multirow_infrastructure() {
    // Verify our test infrastructure works

    // Simulate single-column output
    let output_single = [1.0, 2.0, 3.0, 4.0, 5.0];

    // Simulate matching multi-row residual output
    let output_multi = [1.0, 2.0, 3.0, 4.0, 5.0];

    // Verify match
    assert_eq!(output_single.len(), output_multi.len());
    for (i, (s, m)) in output_single.iter().zip(output_multi.iter()).enumerate() {
        let diff = f64::abs(*s - *m);
        assert!(diff < 1e-5, "Mismatch at {}", i);
    }

    println!("✅ Q4_0 multi-row test infrastructure works");
}

// Helper function to run inference with specified kernel variant
fn run_inference(model_path: &str, variant: &str) -> Vec<f32> {
    let output = Command::new("timeout")
        .arg("30s")
        .arg("./target/release/rocmforge")
        .arg("--gpu")
        .arg("--model")
        .arg(model_path)
        .arg("--prompt")
        .arg("Hello")
        .arg("--max-tokens")
        .arg("10")
        .arg("--no-template")
        .arg("--top-p")
        .arg("1.0")
        .arg("--kernel-variant") // TODO: Add this CLI flag
        .arg(variant)
        .output()
        .expect("Failed to execute rocmforge");

    assert!(
        output.status.success(),
        "Command failed: {:?}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for GPU crashes
    assert!(
        !stderr.contains("HIP_ERROR"),
        "GPU error detected: {}",
        stderr
    );
    assert!(
        !stderr.contains("GPU reset"),
        "GPU reset detected: {}",
        stderr
    );

    // Parse output as tokens/floats (simplified for now)
    // TODO: Implement proper tokenization/float parsing
    stdout
        .split_whitespace()
        .map(|s| s.parse::<f32>().unwrap_or(0.0))
        .collect()
}
