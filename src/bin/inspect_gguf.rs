use rocmforge::loader::gguf::GgufLoader;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path-to-gguf-file>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Inspecting GGUF file: {}", path);

    // Load metadata only (lazy loading)
    let metadata = GgufLoader::metadata_from_file(path)?;
    println!("\n=== Metadata ===");
    println!("Architecture: {}", metadata.architecture);
    println!("Vocab size: {}", metadata.vocab_size);
    println!("Hidden size: {}", metadata.hidden_size);
    println!("Num layers: {}", metadata.num_layers);
    println!("Num heads: {}", metadata.num_heads);
    println!("Num KV heads: {:?}", metadata.num_kv_heads);
    println!("Intermediate size: {}", metadata.intermediate_size);
    println!("Max position embeddings: {}", metadata.max_position_embeddings);
    println!("RMS norm epsilon: {}", metadata.rms_norm_eps);
    println!("Use rotary embeddings: {}", metadata.use_rotary_embeddings);

    // Load full loader to access lazy tensors
    let loader = GgufLoader::new(path)?;
    println!("\n=== Tensor Names (first 20) ===");

    let lazy_tensors = &loader.lazy_tensors;
    for (i, (name, lazy)) in lazy_tensors.iter().enumerate() {
        if i >= 20 {
            println!("... and {} more tensors", lazy_tensors.len() - 20);
            break;
        }

        let shape = lazy.shape().map(|s| format!("{:?}", s)).unwrap_or("unknown".to_string());
        println!("{}. {} [{}]", i + 1, name, shape);
    }

    // Look for embedding-related tensors
    println!("\n=== Embedding-related tensors ===");
    let embedding_patterns = [
        "token_embd",
        "embed_tokens",
        "embeddings",
        "word_embeddings",
        "input_embeddings",
    ];

    for (name, lazy) in lazy_tensors.iter() {
        for pattern in &embedding_patterns {
            if name.contains(pattern) {
                let shape = lazy.shape().map(|s| format!("{:?}", s)).unwrap_or("unknown".to_string());
                println!("{} [{}]", name, shape);
                break;
            }
        }
    }

    // Look for output/lm_head related tensors
    println!("\n=== Output/LM Head related tensors ===");
    let output_patterns = [
        "output",
        "lm_head",
        "logits",
        "head",
    ];

    for (name, lazy) in lazy_tensors.iter() {
        for pattern in &output_patterns {
            if name.contains(pattern) {
                let shape = lazy.shape().map(|s| format!("{:?}", s)).unwrap_or("unknown".to_string());
                println!("{} [{}]", name, shape);
                break;
            }
        }
    }

    println!("\nTotal tensors: {}", lazy_tensors.len());

    Ok(())
}
