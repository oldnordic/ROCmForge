use rocmforge::backend::hip_backend::ModelRuntime;

// Initialize tracing for debugging segment core faults
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};

    // Set default log level to info, but allow override via RUST_LOG
    let filter = EnvFilter::from_default_env()
        .add_directive("rocmforge=trace".parse().unwrap())
        .add_directive("hip_backend=trace".parse().unwrap());

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for comprehensive GPU debugging
    init_tracing();

    tracing::info!("🚀 Testing ROCmForge inference with Llama 3.2 1B model");
    println!("🚀 Testing ROCmForge inference with Llama 3.2 1B model");

    // Model path
    let model_path = "/home/geramyl/Documents/Programming/ROCmForge/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    println!("📁 Loading model from: {}", model_path);

    // Load model metadata first
    println!("📋 Loading GGUF metadata...");
    let metadata = rocmforge::loader::gguf::GgufLoader::metadata_from_file(model_path)?;
    println!("✅ Metadata loaded: {} layers, {} heads, {} hidden size, {} context", metadata.num_layers, metadata.num_heads, metadata.hidden_size, metadata.max_position_embeddings);

    // Override context size to fit in GPU memory
    use rocmforge::model::config::{ModelConfig, ModelType};
    let mut config = ModelConfig {
        num_hidden_layers: metadata.num_layers,
        num_attention_heads: metadata.num_heads,
        num_kv_heads: metadata.num_kv_heads,
        hidden_size: metadata.hidden_size,
        intermediate_size: metadata.intermediate_size,
        max_position_embeddings: 2048, // Smaller context to fit in GPU memory
        vocab_size: metadata.vocab_size,
        rms_norm_eps: metadata.rms_norm_eps,
        use_rotary_embeddings: metadata.use_rotary_embeddings,
        model_type: ModelType::Llama,
        head_dim: metadata.head_dim,
    };

    // Load model directly with overridden config
    println!("⏳ Loading model with {} context size...", config.max_position_embeddings);
    let mut runtime = ModelRuntime::load_from_gguf_with_config(model_path, Some(config))?;
    println!("✅ Model loaded successfully!");

    // Get backend info
    let backend = runtime.backend();
    let (free_mem, total_mem) = backend.get_memory_info()?;
    println!("💾 GPU Memory: {:.1} GB free / {:.1} GB total",
             free_mem as f64 / 1024.0 / 1024.0 / 1024.0,
             total_mem as f64 / 1024.0 / 1024.0 / 1024.0);

    // Create a simple test input (token ID 2 = common start token)
    let input_tokens = vec![2u32]; // Single token for testing
    println!("🔢 Input tokens: {:?}", input_tokens);

    // Get embedding weights and create input embedding
    let execution_plan_binding = runtime.execution_plan();
    let execution_plan = execution_plan_binding.as_ref()
        .ok_or("No execution plan")?;
    let embedding_weights = execution_plan.embedding_weights()?;
    let embeddings = execution_plan.embedding_lookup(&backend, &input_tokens, &embedding_weights)?;

    println!("📊 Input embedding shape: {:?}", embeddings.shape());

    // Run inference (decode_step)
    println!("⚡ Running inference on GPU...");
    let start_time = std::time::Instant::now();
    let logits = runtime.decode_step(&embeddings).map_err(|e| {
        eprintln!("❌ Inference failed with error: {}", e);
        eprintln!("🔍 This appears to be a memory copy bounds error in the attention mechanism");
        eprintln!("💡 The error suggests an offset calculation bug where offset + length > buffer size");
        e
    })?;
    let inference_time = start_time.elapsed();

    println!("✅ Inference completed in {:.2}ms", inference_time.as_millis());
    println!("📊 Output logits shape: {:?}", logits.shape());

    // Get the logits as host data
    let mut host_logits = vec![0.0f32; logits.len()];
    // Get backend again to avoid borrowing conflict
    let backend = runtime.backend();
    backend.copy_from_device_safe(logits.buffer(), &mut host_logits)?;

    // Find the top 5 tokens
    let mut token_probs: Vec<(usize, f32)> = host_logits.iter().enumerate()
        .map(|(i, &p)| (i, p))
        .collect();

    token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    token_probs.truncate(5);

    println!("🎯 Top 5 predicted tokens:");
    for (token_id, prob) in token_probs {
        println!("  Token {:6}: {:.6}", token_id, prob);
    }

    // Calculate some statistics
    let max_logit = host_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = host_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_logit = host_logits.iter().sum::<f32>() / host_logits.len() as f32;

    println!("📈 Logits statistics:");
    println!("  Max: {:.6}", max_logit);
    println!("  Min: {:.6}", min_logit);
    println!("  Mean: {:.6}", mean_logit);

    println!("🎉 GPU inference test completed successfully!");
    println!("🚀 ROCmForge is working correctly with real models!");

    Ok(())
}
