//! Minimal GGUF loader test
//!
//! Purpose: Isolate GGUF parsing from engine/CLI/GPU complexity
//! Tests: Can we parse and materialize the model file?
//!
//! Run: cargo run --bin test_gguf_load -- <path-to.gguf>

use rocmforge::loader::gguf::{GgufLoader, GgufTensor, GgufTensorType};
use std::env;

fn print_tensor_summary(tensor: &GgufTensor) {
    println!("  Tensor: {}", tensor.name);
    println!("    Shape: {:?}", tensor.shape);
    println!("    Type: {:?}", tensor.tensor_type);
    println!("    Offset: {} bytes", tensor.offset);
    println!("    Data size: {} bytes", tensor.data.len());
    println!("    Elements: {}", tensor.total_elements());
}

fn print_model_summary(loader: &GgufLoader) {
    let metadata = loader.metadata();

    println!("\n=== Model Summary ===");
    println!("Architecture: {}", metadata.architecture);
    println!("File type: {}", metadata.file_type);
    println!("Num layers: {}", metadata.num_layers);
    println!("Num heads: {}", metadata.num_heads);
    println!("Hidden size: {}", metadata.hidden_size);
    println!("Intermediate size: {}", metadata.intermediate_size);
    println!("Head dim: {}", metadata.head_dim);
    println!(
        "Max position embeddings: {}",
        metadata.max_position_embeddings
    );
    println!("Vocab size: {}", metadata.vocab_size);
    println!("RMS norm eps: {}", metadata.rms_norm_eps);

    let tensors = loader.load_tensors().expect("Failed to load tensors");
    println!("Total tensors: {}", tensors.len());

    // Count by data type
    let mut type_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for tensor in tensors.values() {
        let type_name = match tensor.tensor_type {
            GgufTensorType::F32 => "F32",
            GgufTensorType::F16 => "F16",
            GgufTensorType::Q8_0 => "Q8_0",
            GgufTensorType::Q4_0 => "Q4_0",
            GgufTensorType::Q4_1 => "Q4_1",
            GgufTensorType::Q5_0 => "Q5_0",
            GgufTensorType::Q5_1 => "Q5_1",
            GgufTensorType::Q2_K => "Q2_K",
            GgufTensorType::Q3_K => "Q3_K",
            GgufTensorType::Q4_K => "Q4_K",
            GgufTensorType::Q5_K => "Q5_K",
            GgufTensorType::Q6_K => "Q6_K",
            GgufTensorType::Mxfp4 => "MXFP4",
            GgufTensorType::Mxfp6E2m3 => "MXFP6_E2M3",
            GgufTensorType::Mxfp6E3m2 => "MXFP6_E3M2",
        };
        *type_counts.entry(type_name).or_insert(0) += 1;
    }

    println!("\nData type distribution:");
    for (dtype, count) in type_counts.iter() {
        println!("  {}: {} tensors", dtype, count);
    }

    // Calculate total model size
    let total_bytes: usize = tensors.values().map(|t| t.data.len()).sum();
    println!(
        "Total model size: {} bytes ({:.2} MB)",
        total_bytes,
        total_bytes as f64 / 1024.0 / 1024.0
    );

    // Find key tensors
    println!("\n=== Key Tensors ===");
    let key_patterns = [
        "token_embd",
        "embed_tokens",
        "word_embeddings",
        "output",
        "lm_head",
        "logits",
        "layers.0",
    ];

    for pattern in &key_patterns {
        // Find first tensor whose name contains the pattern
        for tensor in tensors.values() {
            if tensor.name.contains(pattern) {
                print_tensor_summary(tensor);
                println!();
                break;
            }
        }
    }

    // Show first few layer names
    println!("=== Sample Layer Tensors ===");
    let mut layer_names: Vec<&str> = tensors
        .values()
        .filter(|t| t.name.contains("layer"))
        .map(|t| t.name.as_str())
        .take(5)
        .collect();
    layer_names.sort();
    for name in layer_names {
        println!("  {}", name);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <path-to.gguf>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} /path/to/model.gguf", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];

    println!("=== Minimal GGUF Loader Test ===");
    println!("File: {}", gguf_path);

    // Check file exists
    if !std::path::Path::new(gguf_path).exists() {
        eprintln!("✗ File not found: {}", gguf_path);
        std::process::exit(1);
    }

    // Get file size
    let metadata = std::fs::metadata(gguf_path)?;
    let file_size = metadata.len();
    println!(
        "File size: {} bytes ({:.2} MB)",
        file_size,
        file_size as f64 / 1024.0 / 1024.0
    );

    // Load GGUF (CPU-only, no GPU involved)
    println!("\nLoading GGUF (CPU-only, no GPU init)...");
    let start = std::time::Instant::now();
    let loader = GgufLoader::new(gguf_path)?;
    let load_time = start.elapsed();

    println!("✓ Loaded in {:?}", load_time);

    // Print summary
    print_model_summary(&loader);

    let tensors = loader.load_tensors()?;
    println!("\n=== Test Result ===");
    println!("✓ GGUF loading successful");
    println!("✓ All {} tensors parsed and validated", tensors.len());

    Ok(())
}
