//! Real model benchmark example.
//!
//! Benchmarks end-to-end inference performance on real GGUF models.
//!
//! Usage:
//!   cargo run --release --example benchmark_real_model -- --help

use clap::Parser;
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};

use rocmforge::bench::discover_models;
use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    sampler::cpu_sample_greedy,
    weights::CpuModelWeights,
    CpuFeatures,
};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;

/// Benchmark real GGUF models with end-to-end inference timing.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model directory containing .gguf files
    #[arg(long)]
    model_dir: Option<String>,

    /// Filter models by glob pattern
    #[arg(long, default_value = "*.gguf")]
    model: String,

    /// Number of inference runs per model
    #[arg(long, default_value_t = 3)]
    iterations: usize,

    /// Number of tokens to generate
    #[arg(long, default_value_t = 10)]
    tokens: usize,

    /// Enable per-layer profiling
    #[arg(long)]
    profile: bool,

    /// Output markdown file
    #[arg(long)]
    output: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Log CPU features
    let features = CpuFeatures::get();
    eprintln!("CPU Features: {}", features.description());
    eprintln!("Kernel: {:?}", features.kernel);
    eprintln!();

    // Discover models
    let model_dir = args.model_dir.as_deref();
    let models = discover_models(model_dir);

    if models.is_empty() {
        eprintln!("No models found in searched paths:");
        eprintln!("  - Explicit: {:?}", args.model_dir);
        eprintln!("  - Env var: $ROCMFORGE_MODEL_DIR");
        eprintln!("  - Default: /home/feanor/Projects/Memoria/models");
        eprintln!("  - Fallback: ./models");
        process::exit(1);
    }

    eprintln!("Found {} model(s):", models.len());
    for model in &models {
        eprintln!("  - {}", model.display());
    }
    eprintln!();

    // Run benchmarks
    let results = benchmark_models(&models, &args);

    // Generate report
    generate_report(&results, &args);
}

fn benchmark_models(models: &[PathBuf], args: &Args) -> Vec<ModelResult> {
    let mut results = Vec::new();

    for (idx, model_path) in models.iter().enumerate() {
        eprintln!("[{}/{}] Benchmarking: {}", idx + 1, models.len(),
                 model_path.file_name().unwrap_or_default().to_string_lossy());

        match benchmark_model(model_path, args) {
            Ok(result) => {
                eprintln!("  ✓ Prefill: {:.1} ms, Decode: {:.1} ms, {:.1} tok/s",
                         result.prefill_ms, result.decode_ms, result.tokens_per_sec);
                results.push(result);
            }
            Err(e) => {
                eprintln!("  ✗ Failed: {}", e);
            }
        }

        eprintln!();
    }

    results
}

fn benchmark_model(model_path: &PathBuf, args: &Args) -> Result<ModelResult, String> {
    // Open GGUF file
    let file = GgufFile::open(model_path)
        .map_err(|e| format!("Failed to open: {}", e))?;

    // Load configuration
    let config = ModelConfig::from_gguf(&file)
        .map_err(|e| format!("Failed to load config: {}", e))?;

    // Load tokenizer
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());

    // Load weights
    let start = Instant::now();
    let weights = CpuModelWeights::load(&file, &config)
        .map_err(|e| format!("Failed to load weights: {}", e))?;
    let load_time = start.elapsed().as_secs_f64() * 1000.0;

    // Prepare test prompt
    let test_prompt = "Hello, world!";
    let prompt_tokens = tok.encode(test_prompt, false);
    if prompt_tokens.is_empty() {
        return Err("Prompt tokenized to zero tokens".to_string());
    }

    // Run inference iterations
    let mut total_prefill = Duration::ZERO;
    let mut total_decode = Duration::ZERO;
    let mut total_tokens = 0;

    for _iter in 0..args.iterations {
        // Allocate buffers
        let max_seq = prompt_tokens.len() + args.tokens;
        let mut kv = CpuKvCache::new(&config, max_seq);
        let mut scratch = CpuForwardScratch::new(&config);
        let mut hidden = vec![0.0f32; config.hidden_size];

        // Prefill
        let prefill_start = Instant::now();

        // Embed first token
        cpu_embed_token(prompt_tokens[0], &weights, &mut hidden, &config);

        // Process remaining prompt tokens
        for i in 1..prompt_tokens.len() {
            cpu_full_forward(&mut hidden, &weights, &mut kv, &mut scratch, i, &config)
                .map_err(|e| format!("Prefill forward failed: {}", e))?;
        }

        let prefill_time = prefill_start.elapsed();
        total_prefill += prefill_time;

        // Decode
        let decode_start = Instant::now();
        let mut generated = Vec::new();

        for _j in 0..args.tokens {
            let pos = prompt_tokens.len() + generated.len();

            cpu_full_forward(&mut hidden, &weights, &mut kv, &mut scratch, pos, &config)
                .map_err(|e| format!("Decode forward failed: {}", e))?;

            let next_token = cpu_sample_greedy(&scratch.logits);
            generated.push(next_token);

            if Some(next_token) == tok.eos_id() {
                break;
            }

            cpu_embed_token(next_token, &weights, &mut hidden, &config);

            total_tokens += 1;
        }

        let decode_time = decode_start.elapsed();
        total_decode += decode_time;
    }

    let avg_prefill_ms = total_prefill.as_secs_f64() * 1000.0 / args.iterations as f64;
    let avg_decode_ms = total_decode.as_secs_f64() * 1000.0 / args.iterations as f64;
    let tokens_per_sec = (total_tokens as f64) / (total_decode.as_secs_f64() / args.iterations as f64);

    Ok(ModelResult {
        model_name: model_path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        quantization: detect_quantization(model_path),
        num_layers: config.num_layers,
        hidden_size: config.hidden_size,
        vocab_size: config.vocab_size,
        load_time_ms: load_time,
        prefill_ms: avg_prefill_ms,
        decode_ms: avg_decode_ms,
        tokens_per_sec,
        num_tokens: args.tokens,
    })
}

fn detect_quantization(path: &PathBuf) -> String {
    // Try to detect quantization from filename
    // This is a simple heuristic
    let name = path.to_string_lossy();

    if name.contains("q4_k") || name.contains("Q4_K") {
        return "Q4_K".to_string();
    } else if name.contains("q5_0") || name.contains("Q5_0") {
        return "Q5_0".to_string();
    } else if name.contains("q8_0") || name.contains("Q8_0") {
        return "Q8_0".to_string();
    } else if name.contains("q4_0") || name.contains("Q4_0") {
        return "Q4_0".to_string();
    }

    "Unknown".to_string()
}

fn generate_report(results: &[ModelResult], args: &Args) {
    let output_path = args.output.as_deref()
        .unwrap_or("docs/benchmarks/real-model-benchmark.md");

    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Generate markdown
    let mut markdown = String::new();
    markdown.push_str("# Real Model Benchmark Results\n\n");
    markdown.push_str(&format!("**Date:** {}\n\n", chrono::Utc::now().format("%Y-%m-%d")));
    markdown.push_str(&format!("**CPU Kernel:** {:?}\n\n", CpuFeatures::get().kernel));

    markdown.push_str("## Results\n\n");
    markdown.push_str("| Model | Quantization | Layers | Hidden | Load (ms) | Prefill (ms) | Decode (ms) | Tok/s |\n");
    markdown.push_str("|-------|--------------|--------|--------|-----------|--------------|-------------|-------|\n");

    for r in results {
        markdown.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
            r.model_name, r.quantization, r.num_layers, r.hidden_size,
            r.load_time_ms, r.prefill_ms, r.decode_ms, r.tokens_per_sec
        ));
    }

    // Write report
    std::fs::write(output_path, markdown)
        .expect("Failed to write report");

    eprintln!("Report written to: {}", output_path);
}

#[derive(Debug)]
struct ModelResult {
    model_name: String,
    quantization: String,
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    load_time_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    tokens_per_sec: f64,
    num_tokens: usize,
}
