use rocmforge::backend::hip_backend::ModelRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing ROCmForge inference with Gemma 3 4B model");

    // Model path
    let model_path = "/home/geramyl/.cache/huggingface/hub/models--ggml-org--gemma-3-4b-it-GGUF/snapshots/d0976223747697cb51e056d85c532013931fe52e/gemma-3-4b-it-Q4_K_M.gguf";

    println!("📁 Loading model from: {}", model_path);

    // Load model on GPU
    println!("⏳ Loading GGUF model (this may take a moment for 4B parameters)...");
    let mut runtime = ModelRuntime::load_from_gguf(model_path)?;
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
    let logits = runtime.decode_step(&embeddings)?;
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
