use rocmforge::loader::gguf::GgufLoader;
use rocmforge::backend::HipBackend;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing ROCmForge with Qwen2.5-0.5B GGUF model");
    
    let model_path = "/home/feanor/Projects/ROCmForge/models/qwen2.5-0.5b.gguf";
    println!("📁 Loading model from: {}", model_path);
    
    // Load GGUF metadata
    let loader = GgufLoader::new(model_path)?;
    println!("✅ GGUF metadata loaded");
    println!("   - Tensors: {}", loader.lazy_tensors.len());
    
    // Extract config
    let config = loader.to_model_config()?;
    println!("📋 Model config:");
    println!("   - Architecture: {:?}", config.model_type);
    println!("   - Hidden size: {}", config.hidden_size);
    println!("   - Layers: {}", config.num_hidden_layers);
    println!("   - Heads: {}", config.num_attention_heads);
    println!("   - KV Heads: {:?}", config.num_kv_heads);
    println!("   - Vocab size: {}", config.vocab_size);
    println!("   - Max position embeddings: {}", config.max_position_embeddings);
    
    // Initialize GPU
    println!("\n🎮 Initializing GPU...");
    let backend = HipBackend::new_checked()?;
    println!("✅ GPU initialized");
    
    // Get memory info
    let (free_mem, total_mem) = backend.get_memory_info()?;
    println!("💾 GPU Memory: {:.1} GB free / {:.1} GB total",
        free_mem as f64 / 1024.0 / 1024.0 / 1024.0,
        total_mem as f64 / 1024.0 / 1024.0 / 1024.0);
    
    // Load embedding weights
    use rocmforge::model::execution_plan::ExecutionPlan;
    println!("\n⏳ Loading execution plan...");
    let start = Instant::now();
    let plan = ExecutionPlan::from_gguf(&backend, &loader)?;
    println!("✅ Execution plan loaded in {:.2}s", start.elapsed().as_secs_f64());
    
    // Get embedding weights
    let embedding = plan.embedding_weights()?;
    println!("📊 Embedding weights shape: {:?}", embedding.shape());
    
    // Test simple token embedding lookup
    let test_tokens = vec![1u32, 2, 3, 4, 5];
    println!("\n🔢 Testing embedding lookup with tokens: {:?}", test_tokens);
    let start = Instant::now();
    let embeddings = plan.embedding_lookup(&backend, &test_tokens, &embedding)?;
    println!("✅ Embedding lookup completed in {:.2}ms", start.elapsed().as_millis());
    println!("📊 Output embeddings shape: {:?}", embeddings.shape());
    
    // Verify output
    let mut host_data = vec![0.0f32; embeddings.len()];
    backend.copy_from_device_safe(embeddings.buffer(), &mut host_data)?;
    
    let sum: f32 = host_data.iter().sum();
    println!("📈 Embedding checksum: {:.6}", sum);
    
    // Check if values are finite
    let all_finite = host_data.iter().all(|x| x.is_finite());
    if all_finite {
        println!("✅ All embedding values are finite");
    } else {
        println!("❌ Some embedding values are not finite!");
    }
    
    println!("\n🎉 Qwen2.5-0.5B model test completed successfully!");
    Ok(())
}
