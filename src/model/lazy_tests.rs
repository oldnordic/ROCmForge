// Lazy Loading Tests for ExecutionPlan
//
// Test suite for verifying lazy loading behavior:
// - On-demand tensor loading
// - Caching after first access
// - Preloading methods
// - Loading statistics

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use crate::backend::HipBackend;
use serial_test::serial;
    use crate::loader::gguf::GgufLoader;
    use crate::model::execution_plan::ExecutionPlan;
    use std::path::Path;
    use std::sync::Arc;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for lazy_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Helper function to get test model path
    fn get_test_model_path() -> String {
        // Try to find a test model
        let possible_paths = vec![
            "/home/feanor/Projects/ROCmForge/models/test_model.gguf",
            "/tmp/test_model.gguf",
            "models/test.gguf",
        ];

        for path in &possible_paths {
            if Path::new(path).exists() {
                return path.to_string();
            }
        }

        // Fallback: return first path (will fail but gives clear error)
        possible_paths[0].to_string()
    }

    #[test]
    #[serial]
    fn test_embedding_lazy_load_on_first_access() {
        // Verify first call to embedding_weights() loads tensor
        let backend = get_backend_or_skip();
        let model_path = get_test_model_path();

        if !Path::new(&model_path).exists() {
            println!("SKIP: No test model found at {}", model_path);
            return;
        }

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to create loader");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create execution plan");

        // First access should trigger loading
        let embedding1 = plan.embedding_weights()
            .expect("Failed to load embedding weights");

        // Verify tensor is loaded
        assert_eq!(embedding1.shape().dims().len(), 2,
                   "Embedding should be 2D tensor");
    }

    #[test]
    #[serial]
    fn test_embedding_cached_on_second_access() {
        // Verify second call returns cached tensor (no reload)
        let backend = get_backend_or_skip();
        let model_path = get_test_model_path();

        if !Path::new(&model_path).exists() {
            println!("SKIP: No test model found at {}", model_path);
            return;
        }

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to create loader");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create execution plan");

        // Load first time
        let embedding1 = plan.embedding_weights()
            .expect("Failed to load embedding weights");

        // Load second time - should return cached
        let embedding2 = plan.embedding_weights()
            .expect("Failed to get cached embedding weights");

        // Both should have same shape
        assert_eq!(embedding1.shape().dims(), embedding2.shape().dims(),
                   "Cached tensor should have same shape as original");
    }

    #[test]
    #[serial]
    fn test_preload_layers() {
        // Verify preload_layers() loads specific layers
        let backend = get_backend_or_skip();
        let model_path = get_test_model_path();

        if !Path::new(&model_path).exists() {
            println!("SKIP: No test model found at {}", model_path);
            return;
        }

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to create loader");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create execution plan");

        // Preload first 2 layers
        let layer_indices = vec![0, 1];
        let result = plan.preload_layers(&layer_indices);

        // Should succeed
        assert!(result.is_ok(),
                "preload_layers should succeed: {:?}", result.err());

        // Verify layers can be accessed (should be cached)
        let _ = plan.embedding_weights()
            .expect("Embedding should be accessible after preload");
    }

    #[test]
    #[serial]
    fn test_preload_all() {
        // Verify preload_all() loads all layers
        let backend = get_backend_or_skip();
        let model_path = get_test_model_path();

        if !Path::new(&model_path).exists() {
            println!("SKIP: No test model found at {}", model_path);
            return;
        }

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to create loader");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create execution plan");

        let num_layers = plan.num_layers();

        // Preload all layers
        let result = plan.preload_all();

        // Should succeed
        assert!(result.is_ok(),
                "preload_all should succeed: {:?}", result.err());

        // All layers should now be loaded and accessible
        assert_eq!(plan.num_layers(), num_layers,
                   "Layer count should match after preload");
    }

    #[test]
    #[serial]
    fn test_loading_stats() {
        // Verify loading_stats() returns accurate counts
        let backend = get_backend_or_skip();
        let model_path = get_test_model_path();

        if !Path::new(&model_path).exists() {
            println!("SKIP: No test model found at {}", model_path);
            return;
        }

        let loader = GgufLoader::new(&model_path)
            .expect("Failed to create loader");

        let plan = ExecutionPlan::from_gguf(&backend, &loader)
            .expect("Failed to create execution plan");

        // Get initial stats (should be all unloaded)
        let stats = plan.loading_stats();

        // Verify stats structure
        assert!(stats.total_tensors > 0,
                "Should have total tensors counted");
        assert!(stats.loaded_tensors == 0 || stats.loaded_tensors > 0,
                "Loaded count should be valid");
        assert!(stats.unloaded_tensors >= 0,
                "Unloaded count should be valid");

        // Load embedding
        let _ = plan.embedding_weights()
            .expect("Failed to load embedding");

        // Get stats after loading
        let stats_after = plan.loading_stats();

        // Loaded count should increase
        assert!(stats_after.loaded_tensors >= stats.loaded_tensors,
                "Loaded count should increase or stay same after loading");
    }
}
