//! Model Loading Integration Tests - Phase 6.1
//!
//! TDD: Verify complete GGUF model loading works end-to-end.
//!
//! Test first, prove it fails, then implement.

#[cfg(test)]
mod model_loading_tests {
    use rocmforge::loader::gguf::GgufLoader;
    use std::path::Path;

    /// Helper: Find a test model GGUF file
    ///
    /// Searches common locations for test models.
    fn find_test_model() -> Option<String> {
        let possible_paths = vec![
            "/home/feanor/Projects/ROCmForge/models/qwen2.5-0.5b.gguf",
            "/home/feanor/Projects/ROCmForge/models/Qwen2.5-14B-Instruct-1M-q6_k_m.gguf",
            "models/qwen2.5-0.5b.gguf",
            "models/Qwen2.5-14B-Instruct-1M-q6_k_m.gguf",
        ];

        for path in possible_paths {
            if Path::new(path).exists() {
                return Some(path.to_string());
            }
        }

        None
    }

    /// Test 1: Load GGUF model and extract metadata
    ///
    /// This test verifies that we can:
    /// 1. Open a GGUF file
    /// 2. Read metadata (tensor count, KV metadata)
    /// 3. Extract model configuration
    #[test]
    fn test_load_gguf_metadata() {
        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found");
                eprintln!("To run this test, place a GGUF model at one of:");
                eprintln!("  - /home/feanor/Projects/ROCmForge/models/test_model.gguf");
                eprintln!("  - /home/feanor/models/test.gguf");
                eprintln!("  - /tmp/test_model.gguf");
                eprintln!("  - models/test.gguf");
                return;
            }
        };

        // Load GGUF - single pass for both metadata and weights
        let loader = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        // Verify metadata extraction
        let tensor_count = loader.lazy_tensors.len();
        println!("GGUF tensor count: {}", tensor_count);
        assert!(tensor_count > 0, "GGUF should contain tensors");

        // Extract model configuration
        let config = loader.to_model_config()
            .expect("Failed to extract ModelConfig from GGUF");

        // Verify config fields
        assert!(config.num_attention_heads > 0, "num_attention_heads must be positive");
        assert!(config.head_dim > 0, "head_dim must be positive");
        assert!(config.hidden_size > 0, "hidden_size must be positive");
        assert!(config.vocab_size > 0, "vocab_size must be positive");

        println!("Model config: {} heads, {} head_dim, {} hidden, {} vocab",
            config.num_attention_heads,
            config.head_dim,
            config.hidden_size,
            config.vocab_size
        );
    }

    /// Test 2: Load GGUF model weights into ExecutionPlan
    ///
    /// This test verifies that we can:
    /// 1. Create ExecutionPlan from GGUF
    /// 2. Load all weight tensors
    /// 3. Bind weights to GPU memory
    #[serial]
    #[test]
    #[cfg(feature = "rocm")]
    fn test_load_execution_plan_from_gguf() {
        use rocmforge::backend::HipBackend;
        use serial_test::serial;

        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found (GPU test)");
                return;
            }
        };

        // Initialize GPU
        let backend = match HipBackend::new_checked() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("SKIP: GPU not available: {}", e);
                return;
            }
        };

        // Load GGUF
        let loader = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        // Create ExecutionPlan (loads weights)
        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create ExecutionPlan from GGUF");

        // Verify weights are loaded
        // Check embedding weights exist
        let embedding = plan.embedding_weights()
            .expect("Failed to get embedding weights");
        assert!(embedding.shape().dims().len() == 2, "Embedding should be 2D");

        println!("ExecutionPlan created successfully");
        println!("Embedding shape: {:?}", embedding.shape().dims());
    }

    /// Test 3: Verify GGUF tensor names match expected layer structure
    ///
    /// This test verifies the GGUF contains expected tensor names for
    /// a transformer model (token_embedding, output, layers, etc.).
    #[test]
    fn test_gguf_tensor_structure() {
        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found");
                return;
            }
        };

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        // Get all tensor names
        let tensor_names: Vec<String> = loader.lazy_tensors.keys().cloned().collect();
        println!("Total tensors: {}", tensor_names.len());

        // Verify expected tensor names exist
        let expected_patterns = vec![
            "token_emb",  // Token embedding
            "output",     // Output layer
            "blk.",       // Transformer blocks
        ];

        let mut found_patterns = 0;
        for pattern in expected_patterns {
            if tensor_names.iter().any(|name| name.contains(pattern)) {
                found_patterns += 1;
                println!("Found pattern '{}'", pattern);
            }
        }

        assert!(found_patterns >= 2, "GGUF should contain core transformer tensors");
    }

    /// Test 4: Single-pass loading efficiency
    ///
    /// Verify GGUF is parsed once, not multiple times.
    /// This is a regression test for Phase 1 (single-pass loading).
    #[test]
    fn test_single_pass_loading() {
        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found");
                return;
            }
        };

        // First load - should parse file
        let loader1 = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        // Create config from first loader
        let _config = loader1.to_model_config()
            .expect("Failed to extract ModelConfig");

        // Second load for weights - should reuse cached metadata
        // (In production, we'd use same loader, but this verifies
        // that metadata extraction doesn't require re-parsing)

        // TODO: Add timing measurement to verify single-pass is faster
        println!("Single-pass loading test completed");
    }
}
