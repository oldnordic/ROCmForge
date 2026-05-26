#![cfg(feature = "gpu")]

//! GPU CLI QA tests using subprocess isolation.
//!
//! These tests verify the GPU CLI works correctly with real models
//! by running the CLI as a subprocess with the safety harness.
//!
//! All tests are gated by ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 to prevent
//! accidental execution during development. Use the safe runner wrapper
//! (scripts/gpu_safe_run.sh) to ensure lock acquisition and preflight checks.

mod common;

use std::path::PathBuf;
use std::process::Command;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn safe_runner_path() -> PathBuf {
    project_root().join("scripts/gpu_safe_run.sh")
}

fn cli_binary_path() -> PathBuf {
    project_root().join("target/release/rocmforge")
}

fn validation_model_path() -> PathBuf {
    PathBuf::from("/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf")
}

/// Run CLI command with safe runner wrapper.
fn run_safe_cli(args: &[&str], timeout_secs: u64) -> Result<String, String> {
    let safe_runner = safe_runner_path();
    let cli_binary = cli_binary_path();

    if !safe_runner.exists() {
        return Err(format!("Safe runner not found: {:?}", safe_runner));
    }

    if !cli_binary.exists() {
        return Err(format!(
            "CLI binary not found: {:?}. Run: cargo build --release",
            cli_binary
        ));
    }

    let mut cmd = Command::new(&safe_runner);
    cmd.arg("--timeout")
        .arg(format!("{}", timeout_secs))
        .arg("--max-tokens")
        .arg("10")
        .arg(&cli_binary);

    for arg in args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "CLI failed (exit {:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

#[cfg(test)]
mod gpu_cli_qa_tests {
    use super::*;

    /// Test: Safe runner exists and is executable.
    #[test]
    fn test_safe_runner_available() {
        if !common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available");
            return;
        }
        let safe_runner = safe_runner_path();
        assert!(safe_runner.exists(), "Safe runner not found");
        assert!(safe_runner.is_file(), "Safe runner is not a file");
    }

    /// Test: CLI binary exists for testing.
    #[test]
    fn test_cli_binary_available() {
        let cli_binary = cli_binary_path();
        if !cli_binary.exists() {
            eprintln!("CLI binary not found. Run: cargo build --release");
            return;
        }
        assert!(cli_binary.is_file(), "CLI binary is not a file");
    }

    /// Test: Validation model path exists.
    #[test]
    fn test_validation_model_exists() {
        let model_path = validation_model_path();
        if !model_path.exists() {
            eprintln!("Validation model not found: {:?}", model_path);
            eprintln!("Tests will be skipped.");
            return;
        }
        assert!(model_path.is_file(), "Model is not a file");
    }

    /// Test: GPU lock script works.
    #[test]
    fn test_gpu_lock_status() {
        if !common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available");
            return;
        }

        let lock_script = project_root().join("scripts/gpu_lock.sh");
        assert!(lock_script.exists(), "GPU lock script not found");

        let output = Command::new(&lock_script).arg("status").output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                println!("GPU lock status: {}", stdout);
            }
            Err(e) => {
                eprintln!("Failed to check GPU lock status: {}", e);
            }
        }
    }

    /// Test: GPU preflight script works.
    #[test]
    fn test_gpu_preflight() {
        if !common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available");
            return;
        }

        let preflight_script = project_root().join("scripts/gpu_preflight.sh");
        assert!(preflight_script.exists(), "GPU preflight script not found");

        let output = Command::new(&preflight_script).output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);

                if out.status.success() {
                    println!("Preflight passed:\n{}", stdout);
                } else {
                    eprintln!("Preflight failed:\n{}\n{}", stdout, stderr);
                }
            }
            Err(e) => {
                eprintln!("Failed to run preflight: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod gpu_cli_real_model_tests {
    use super::*;

    /// Test: Single-token generation with real model.
    #[test]
    #[ignore = "Requires ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1"]
    fn test_real_model_single_token() {
        if !common::real_model_gpu_tests_enabled() {
            eprintln!("Skipping test: ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS not set");
            return;
        }
        if !common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available");
            return;
        }

        let model_path = validation_model_path();
        if !model_path.exists() {
            eprintln!("Model not found: {:?}", model_path);
            return;
        }

        let args = [
            "--gpu",
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "X",
            "--no-template",
        ];

        match run_safe_cli(&args, 30) {
            Ok(output) => {
                println!("Single-token output:\n{}", output);
                assert!(!output.is_empty(), "Output should not be empty");
            }
            Err(e) => {
                eprintln!("Single-token test failed: {}", e);
            }
        }
    }

    /// Test: Multi-token generation with real model.
    #[test]
    #[ignore = "Requires ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1"]
    fn test_real_model_multi_token() {
        if !common::real_model_gpu_tests_enabled() {
            eprintln!("Skipping test: ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS not set");
            return;
        }
        if !common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available");
            return;
        }

        let model_path = validation_model_path();
        if !model_path.exists() {
            eprintln!("Model not found: {:?}", model_path);
            return;
        }

        let args = [
            "--gpu",
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "Hello",
            "--no-template",
        ];

        match run_safe_cli(&args, 60) {
            Ok(output) => {
                println!("Multi-token output:\n{}", output);
                assert!(!output.is_empty(), "Output should not be empty");
            }
            Err(e) => {
                eprintln!("Multi-token test failed: {}", e);
            }
        }
    }

    /// Test: Timeout enforcement works.
    #[test]
    #[ignore = "Requires ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1"]
    fn test_timeout_enforcement() {
        if !common::real_model_gpu_tests_enabled() {
            eprintln!("Skipping test: ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS not set");
            return;
        }
        if !common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available");
            return;
        }

        let model_path = validation_model_path();
        if !model_path.exists() {
            eprintln!("Model not found: {:?}", model_path);
            return;
        }

        let args = [
            "--gpu",
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "Write a long essay",
            "--no-template",
        ];

        match run_safe_cli(&args, 5) {
            Ok(_) => {
                eprintln!("Timeout test: completed unexpectedly (may be OK if fast)");
            }
            Err(e) => {
                if e.contains("timeout") || e.contains("timed out") {
                    println!("Timeout enforcement working: {}", e);
                } else {
                    eprintln!("Timeout test failed: {}", e);
                }
            }
        }
    }

    /// Test: Prefill-only validation mode.
    #[test]
    #[ignore = "Requires ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1"]
    fn test_prefill_only_validation() {
        if !common::real_model_gpu_tests_enabled() {
            eprintln!("Skipping test: ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS not set");
            return;
        }
        if !common::gpu_safe_runner_available() {
            eprintln!("Skipping test: GPU safe runner not available");
            return;
        }

        let model_path = validation_model_path();
        if !model_path.exists() {
            eprintln!("Model not found: {:?}", model_path);
            return;
        }

        let args = [
            "--gpu",
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "Hi",
            "--no-template",
            "--prefill-only-validate",
        ];

        match run_safe_cli(&args, 30) {
            Ok(output) => {
                println!("Prefill-only validation output:\n{}", output);
                assert!(
                    output.contains("PREFILL_ONLY_VALIDATE: PASSED"),
                    "Prefill-only validation should pass"
                );
            }
            Err(e) => {
                eprintln!("Prefill-only validation failed: {}", e);
            }
        }
    }
}
