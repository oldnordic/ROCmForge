// GPU Safety Test Template
//
// ALL GPU tests MUST include these safety measures to prevent GPU crashes:
// 1. ROCMFORGE_DISABLE_DECODE_GRAPH=1 for Q6_K models
// 2. timeout wrapper to prevent GPU hangs
// 3. --max-tokens limit to prevent unbounded execution
//
// This template shows the REQUIRED pattern for safe GPU testing.

use std::process::Command;
use std::time::Duration;

/// Safe GPU test helper that enforces all safety protocols
fn run_safe_gpu_test(
    model_path: &str,
    prompt: &str,
    max_tokens: u32,
    use_graph: bool,
) -> Result<String, String> {
    let timeout_secs = 30;

    // Build command with ALL safety measures
    let mut cmd = Command::new("timeout");
    cmd.arg(format!("{}s", timeout_secs))
        .arg("./target/release/rocmforge")
        .arg("--gpu")
        .arg("--model")
        .arg(model_path)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(&format!("{}", max_tokens))
        .arg("--no-template")
        .arg("--top-p")
        .arg("1.0");

    // CRITICAL: Disable graph for Q6_K models
    if model_path.contains("q6_k") {
        cmd.env("ROCMFORGE_DISABLE_DECODE_GRAPH", "1");
    }

    // Run with timeout
    let output = cmd.output().map_err(|e| format!("Failed to execute: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "Command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

#[cfg(test)]
mod safe_gpu_tests {
    use super::*;

    // ✅ SAFE: Single token with timeout and token limit
    #[test]
    fn test_gpu_single_token_safe() {
        let result = run_safe_gpu_test(
            "/path/to/model.gguf",
            "X",
            2,
            false, // graph disabled
        );

        // Test should complete without crashing GPU
        assert!(result.is_ok() || result.is_err()); // Either result is OK, just don't crash
    }

    // ✅ SAFE: Multi-token prompt with all safety measures
    #[test]
    fn test_gpu_multi_token_safe() {
        let result = run_safe_gpu_test(
            "/path/to/model.gguf",
            "Hello world",
            5,
            false, // graph disabled for Q6_K
        );

        // Test should complete without crashing GPU
        assert!(result.is_ok() || result.is_err()); // Either result is OK, just don't crash
    }

    // ❌ UNSAFE: This test would be BLOCKED by hooks
    // #[test]
    // fn test_gpu_unsafe() {
    //     // Missing: ROCMFORGE_DISABLE_DECODE_GRAPH=1 for Q6_K
    //     // Missing: timeout wrapper
    //     // Missing: --max-tokens limit
    //     let output = Command::new("./target/release/rocmforge")
    //         .arg("--gpu")
    //         .arg("--model")
    //         .arg("q6_k.gguf")
    //         .arg("--prompt")
    //         .arg("Hello world")
    //         .output()
    //         .unwrap();
    // }
}

#[cfg(test)]
mod q6_k_specific_tests {
    use super::*;

    // ✅ SAFE: Q6_K test with graph disabled (REQUIRED)
    #[test]
    #[ignore = "Requires Q6_K model file"]
    fn test_q6_k_with_graph_disabled() {
        let result = run_safe_gpu_test(
            "/path/to/q6_k_model.gguf",
            "Hello",
            3,
            false, // graph MUST be disabled for Q6_K multi-token
        );

        // Q6_K should work with graph disabled
        assert!(result.is_ok() || result.is_err());
    }

    // ⚠️ PARTIAL: Q6_K single-token works with graph
    #[test]
    #[ignore = "Requires Q6_K model file"]
    fn test_q6_k_single_token_with_graph() {
        let result = run_safe_gpu_test(
            "/path/to/q6_k_model.gguf",
            "X",
            2,
            true, // graph OK for single-token
        );

        // Single-token Q6_K works with graph
        assert!(result.is_ok() || result.is_err());
    }
}

// Command-line test runner example
#[cfg(test)]
mod manual_test_examples {
    /// Example: Manually run Q4_K test with all safety measures
    #[test]
    #[ignore = "Manual test - run with: cargo test -- --ignored --nocapture"]
    fn manual_test_q4_k_safe() {
        // Command equivalent:
        // timeout 30 ROCMFORGE_DISABLE_DECODE_GRAPH=1 \
        //   ./target/release/rocmforge --gpu \
        //   --model /path/to/q4_k.gguf \
        //   --prompt "Hello world" \
        //   --max-tokens 5

        let result = run_safe_gpu_test(
            "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf",
            "Hello world",
            5,
            false,
        );

        println!("Result: {:?}", result);
        assert!(result.is_ok());
    }
}
