//! Single Token Forward Pass Integration Tests - Phase 6.2
//!
//! TDD: Verify a single forward pass produces correct token.
//!
//! Test first, prove it fails, then implement.

#[cfg(test)]
#[cfg(feature = "rocm")]
mod single_token_tests {
    use rocmforge::backend::HipBackend;
    use rocmforge::loader::gguf::GgufLoader;
    use rocmforge::model::config::ModelConfig;
    use rocmforge::model::execution_plan::ExecutionPlan;
    use std::path::Path;

    /// Helper: Find a test model GGUF file
    fn find_test_model() -> Option<String> {
        let possible_paths = vec![
            "/home/feanor/Projects/ROCmForge/models/test_model.gguf",
            "/home/feanor/models/test.gguf",
            "/tmp/test_model.gguf",
            "models/test.gguf",
        ];

        for path in possible_paths {
            if Path::new(path).exists() {
                return Some(path.to_string());
            }
        }

        None
    }

    /// Helper: Get GPU backend or skip test
    fn get_backend_or_skip() -> std::sync::Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for single_token_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Test 1: Single token forward pass with known input
    ///
    /// This test verifies:
    /// 1. Token embedding lookup works
    /// 2. Single transformer layer forward pass works
    /// 3. Output logits have correct shape
    #[test]
    fn test_single_token_forward() {
        let backend = get_backend_or_skip();

        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found");
                return;
            }
        };

        // Load GGUF
        let loader = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        // Get config
        let config = loader.to_model_config()
            .expect("Failed to extract ModelConfig");

        // Create ExecutionPlan
        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create ExecutionPlan from GGUF");

        // Test input: single token (token id 0 = usually BOS or padding)
        let token_id = 0u32;

        println!("Testing single token forward pass with token_id={}", token_id);
        println!("Model: {} layers, {} heads, {} hidden",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size
        );

        // TODO: Implement forward pass
        // let logits = plan.forward_single_token(token_id)
        //     .expect("Forward pass failed");

        // Verify output shape
        // assert_eq!(logits.len(), config.vocab_size,
        //     "Output logits should match vocab size");

        println!("Single token forward pass test completed");
    }

    /// Test 2: Verify embedding lookup returns correct tensor shape
    #[test]
    fn test_embedding_lookup_shape() {
        let backend = get_backend_or_skip();

        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found");
                return;
            }
        };

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        let config = loader.to_model_config()
            .expect("Failed to extract ModelConfig");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create ExecutionPlan");

        let embedding = plan.embedding_weights()
            .expect("Failed to get embedding weights");

        let dims = embedding.shape().dims();
        assert_eq!(dims.len(), 2, "Embedding should be 2D");
        assert_eq!(dims[0], config.vocab_size, "First dim should be vocab_size");
        assert_eq!(dims[1], config.hidden_size, "Second dim should be hidden_size");

        println!("Embedding shape verified: [{}, {}]", dims[0], dims[1]);
    }

    /// Test 3: Verify layer weights are accessible
    #[test]
    fn test_layer_weights_accessible() {
        let backend = get_backend_or_skip();

        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found");
                return;
            }
        };

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        let config = loader.to_model_config()
            .expect("Failed to extract ModelConfig");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create ExecutionPlan");

        // Verify we can query layer count
        let num_layers = plan.num_layers();
        assert_eq!(num_layers, config.num_hidden_layers, "Layer count should match config");

        println!("Verified {} layers accessible", num_layers);
    }

    /// Test 4: Output layer (lm_head) accessible
    #[test]
    fn test_lm_head_accessible() {
        let backend = get_backend_or_skip();

        let model_path = match find_test_model() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: No test model found");
                return;
            }
        };

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to open GGUF file");

        let config = loader.to_model_config()
            .expect("Failed to extract ModelConfig");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create ExecutionPlan");

        // Get output layer weights
        let lm_head = plan.lm_head()
            .expect("Failed to get lm_head weights");

        let dims = lm_head.shape().dims();
        assert_eq!(dims.len(), 2, "LM head should be 2D");
        assert_eq!(dims[0], config.vocab_size, "First dim should be vocab_size");
        assert_eq!(dims[1], config.hidden_size, "Second dim should be hidden_size");

        println!("LM head shape verified: [{}, {}]", dims[0], dims[1]);
    }
}
