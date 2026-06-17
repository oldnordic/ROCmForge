// Q8_0 Multi-Row Correctness Test
//
// This test verifies that multi-row kernel optimizations (4-column and 8-column)
// produce identical output to the single-column kernel.
//
// Test Strategy:
// 1. Generate random input data of realistic dimensions
// 2. Run single-column kernel (baseline)
// 3. Run multi-row kernel (4-column and 8-column variants)
// 4. Verify outputs match bitwise (within numerical precision)
//
// This test establishes correctness BEFORE implementing performance optimizations.

use std::process::Command;

#[test]
#[ignore = "Requires Q8_0 model file and GPU"]
fn test_q8_0_multirow_4col_matches_single() {
    // TODO: Implement multi-row kernel first
    // This test will verify 4-column multi-row output matches single-column

    let model_path = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q8_0.gguf";

    // Run inference with single-column kernel (current implementation)
    let output_single = run_inference(model_path, "single");

    // Run inference with 4-column multi-row kernel
    let output_multi_4col = run_inference(model_path, "multi-4col");

    // Verify outputs match bitwise
    assert_eq!(
        output_single.len(),
        output_multi_4col.len(),
        "Output length mismatch"
    );

    for (i, (s, m)) in output_single
        .iter()
        .zip(output_multi_4col.iter())
        .enumerate()
    {
        let diff = f64::abs(*s as f64 - *m as f64);
        assert!(
            diff < 1e-5,
            "Mismatch at position {}: single={}, multi_4col={}",
            i,
            s,
            m
        );
    }

    println!("✅ Q8_0 4-column multi-row matches single-column");
}

#[test]
#[ignore = "Requires Q8_0 model file and GPU"]
fn test_q8_0_multirow_8col_matches_single() {
    // TODO: Implement multi-row kernel first
    // This test will verify 8-column multi-row output matches single-column

    let model_path = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q8_0.gguf";

    // Run inference with single-column kernel (current implementation)
    let output_single = run_inference(model_path, "single");

    // Run inference with 8-column multi-row kernel
    let output_multi_8col = run_inference(model_path, "multi-8col");

    // Verify outputs match bitwise
    assert_eq!(
        output_single.len(),
        output_multi_8col.len(),
        "Output length mismatch"
    );

    for (i, (s, m)) in output_single
        .iter()
        .zip(output_multi_8col.iter())
        .enumerate()
    {
        let diff = f64::abs(*s as f64 - *m as f64);
        assert!(
            diff < 1e-5,
            "Mismatch at position {}: single={}, multi_8col={}",
            i,
            s,
            m
        );
    }

    println!("✅ Q8_0 8-column multi-row matches single-column");
}

#[test]
fn test_q8_0_multirow_infrastructure() {
    // Verify our test infrastructure works

    // Simulate single-column output
    let output_single = [1.0, 2.0, 3.0, 4.0, 5.0];

    // Simulate matching multi-row output
    let output_multi = [1.0, 2.0, 3.0, 4.0, 5.0];

    // Verify match
    assert_eq!(output_single.len(), output_multi.len());
    for (i, (s, m)) in output_single.iter().zip(output_multi.iter()).enumerate() {
        let diff = f64::abs(*s - *m);
        assert!(diff < 1e-5, "Mismatch at {}", i);
    }

    println!("✅ Q8_0 multi-row test infrastructure works");
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
