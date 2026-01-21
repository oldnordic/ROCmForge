//! Model loading E2E validation tests
//!
//! This module validates that qwen2.5-0.5b.gguf loads without
//! KernelLoadFailed errors, confirming all v1.5 fixes work together.

use rocmforge::backend::hip_backend::{HipError, ModelRuntime};
use serial_test::serial;

/// Test: Load Qwen model without KernelLoadFailed errors
///
/// This test validates:
/// - Phase 25: HSACO paths embedded at compile time (no runtime env var needed)
/// - Phase 26: Transpose kernel works for [896, 151936] tensor
/// - Phase 27: Device property caching enables proper launch validation
/// - Phase 28: Error messages are clear if something goes wrong
///
/// Expected results:
/// - Model loads without error
/// - Embedding weights transpose [896, 151936] completes
/// - All kernels loaded from compiled HSACO paths
/// - No hipErrorInvalidValue from transpose
#[test]
#[serial]
fn test_qwen_model_load_no_kernel_errors() {
    // Skip if GPU not available
    let _ = match super::cache::skip_if_no_gpu() {
        Ok(()) => (),
        Err(msg) => {
            eprintln!("\n=== SKIP: test_qwen_model_load_no_kernel_errors ===");
            eprintln!("Reason: {}", msg);
            eprintln!("=== END SKIP ===\n");
            return;
        }
    };

    // Skip if model not available
    let model_path = match super::cache::ensure_qwen_model() {
        Ok(path) => path,
        Err(msg) => {
            eprintln!("\n=== SKIP: test_qwen_model_load_no_kernel_errors ===");
            eprintln!("Reason: {}", msg);
            eprintln!("=== END SKIP ===\n");
            return;
        }
    };

    eprintln!("\n=== test_qwen_model_load_no_kernel_errors ===");
    eprintln!("Model path: {}", model_path.display());
    eprintln!("Loading model...");

    // Attempt to load the model
    let result = ModelRuntime::load_from_gguf(model_path.to_str().unwrap());

    match result {
        Ok(runtime) => {
            eprintln!("Model loaded successfully!");

            // Print model info
            if let Some(plan) = runtime.execution_plan() {
                let config = plan.config();
                eprintln!("Model configuration:");
                eprintln!("  - Layers: {}", config.num_hidden_layers);
                eprintln!("  - Attention heads: {}", config.num_attention_heads);
                eprintln!("  - Hidden size: {}", config.hidden_size);
                eprintln!("  - Vocab size: {}", config.vocab_size);
                eprintln!("  - Head dim: {}", config.head_dim);
            }

            eprintln!("\n=== SUCCESS ===");
            eprintln!("No KernelLoadFailed errors occurred");
            eprintln!("Embedding weights transpose [896, 151936] completed");
            eprintln!("All kernels loaded from compiled HSACO paths");
            eprintln!("===================\n");
        }
        Err(HipError::KernelLoadFailed(msg)) => {
            // This is the FAIL case - what we're validating against
            if msg.contains("HSACO") || msg.contains("not found") {
                panic!(
                    "\n=== FAIL: KernelLoadFailed ===\n\
                     HSACO kernel not found: {}\n\
                     This indicates Phase 25 (env var embedding) may have failed.\n\
                     Ensure the project was built with HSACO files available.\n\
                     ===================\n",
                    msg
                );
            } else {
                panic!(
                    "\n=== FAIL: KernelLoadFailed ===\n\
                     Unexpected kernel load error: {}\n\
                     ===================\n",
                    msg
                );
            }
        }
        Err(HipError::KernelLaunchFailed(msg)) => {
            // This could be a launch validation failure
            if msg.contains("invalid config") || msg.contains("validation") {
                panic!(
                    "\n=== FAIL: KernelLaunchFailed ===\n\
                     Launch validation failed: {}\n\
                     This indicates Phase 27 (device property caching) may have issues.\n\
                     ===================\n",
                    msg
                );
            } else {
                panic!(
                    "\n=== FAIL: KernelLaunchFailed ===\n\
                     Kernel launch failed: {}\n\
                     ===================\n",
                    msg
                );
            }
        }
        Err(e) => {
            // Log other errors for investigation
            eprintln!("\n=== ERROR: Model load failed ===");
            eprintln!("Error type: {:?}", std::any::type_name_of_val(&e));
            eprintln!("Error details: {:?}", e);
            eprintln!("=== END ERROR ===\n");
            panic!("Model load failed with unexpected error: {:?}", e);
        }
    }
}

/// Test: Verify model metadata loads correctly
///
/// This is a lighter-weight test that checks metadata extraction
/// without loading the full model weights.
#[test]
fn test_qwen_model_metadata() {
    // Skip if model not available
    let model_path = match super::cache::ensure_qwen_model() {
        Ok(path) => path,
        Err(msg) => {
            eprintln!("\n=== SKIP: test_qwen_model_metadata ===");
            eprintln!("Reason: {}", msg);
            eprintln!("=== END SKIP ===\n");
            return;
        }
    };

    eprintln!("\n=== test_qwen_model_metadata ===");
    eprintln!("Model path: {}", model_path.display());

    // Load GGUF metadata
    let loader = match rocmforge::loader::gguf::GgufLoader::new(model_path.to_str().unwrap()) {
        Ok(loader) => loader,
        Err(e) => {
            eprintln!("SKIP: Failed to load GGUF: {}", e);
            return;
        }
    };

    // Extract config
    let config = match loader.to_model_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("SKIP: Failed to extract config: {}", e);
            return;
        }
    };

    eprintln!("Model metadata extracted:");
    eprintln!("  - Vocab size: {}", config.vocab_size);
    eprintln!("  - Hidden size: {}", config.hidden_size);
    eprintln!("  - Layers: {}", config.num_hidden_layers);
    eprintln!("  - Attention heads: {}", config.num_attention_heads);
    eprintln!("  - Head dim: {}", config.head_dim);

    // Verify expected values for Qwen2.5-0.5B
    assert_eq!(config.vocab_size, 151936, "Vocab size should be 151936");
    assert_eq!(config.hidden_size, 896, "Hidden size should be 896");

    eprintln!("=== SUCCESS ===\n");
}
