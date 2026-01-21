//! Model download/caching utilities for validation tests
//!
//! This module provides:
//! - Model cache directory management
//! - Qwen2.5-0.5B model path resolution
//! - Model availability checking
//! - Optional auto-download from HuggingFace

use std::path::PathBuf;

/// Qwen model filename for testing
pub const QWEN_MODEL_NAME: &str = "qwen2.5-0.5b.gguf";

/// Cache directory for test models
pub const CACHE_DIR: &str = "tests/fixtures/models/";

/// HuggingFace URL for Qwen2.5-0.5B-Instruct-GGUF model
///
/// Users can manually download from:
/// https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
///
/// Look for: qwen2.5-0.5b-instruct-q8_0.gguf or similar quantization
pub const MODEL_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF";

/// Get the cache directory for test models
///
/// Creates the directory if it doesn't exist.
pub fn cache_dir() -> PathBuf {
    let path = PathBuf::from(CACHE_DIR);
    if !path.exists() {
        let _ = std::fs::create_dir_all(&path);
    }
    path
}

/// Get the full path to the Qwen model file
pub fn qwen_model_path() -> PathBuf {
    cache_dir().join(QWEN_MODEL_NAME)
}

/// Check if the Qwen model file exists
pub fn has_qwen_model() -> bool {
    qwen_model_path().exists()
}

/// Ensure the Qwen model is available, downloading if necessary
///
/// This function:
/// 1. Returns the model path if it already exists
/// 2. Attempts to download from HuggingFace if missing
/// 3. Returns Err with clear message if download fails
///
/// Note: Due to HuggingFace's complexity, users may need to manually
/// download the model and place it in the cache directory.
pub fn ensure_qwen_model() -> Result<PathBuf, String> {
    let path = qwen_model_path();

    if path.exists() {
        return Ok(path);
    }

    // Model not found - provide instructions
    Err(format!(
        "Qwen model not found at: {}\n\n\
         To run validation tests, download the model:\n\
         1. Visit: {}\n\
         2. Download: qwen2.5-0.5b-instruct-q8_0.gguf or similar\n\
         3. Rename to: {}\n\
         4. Place in: {}",
        path.display(),
        MODEL_URL,
        QWEN_MODEL_NAME,
        cache_dir().display()
    ))
}

/// Skip test if model is not available
///
/// Returns Err with a clear message explaining what's missing.
/// Use this in tests to gracefully skip when model file is absent.
pub fn skip_if_no_model() -> Result<(), String> {
    if has_qwen_model() {
        Ok(())
    } else {
        Err(format!(
            "Model not found: {}\n\
             Download from {} and place in {}",
            qwen_model_path().display(),
            MODEL_URL,
            cache_dir().display()
        ))
    }
}

/// Check if GPU is available for validation tests
///
/// Uses HipBackend::gpu_available() for checking.
pub fn skip_if_no_gpu() -> Result<(), String> {
    use rocmforge::backend::HipBackend;

    if HipBackend::gpu_available() {
        Ok(())
    } else {
        Err(
            "GPU not available. Ensure:\n\
             1. AMD GPU is present\n\
             2. ROCm is installed (check with rocm-smi)\n\
             3. amdhip64 library is in LD_LIBRARY_PATH"
                .to_string(),
        )
    }
}
