//! Performance benchmarks for GPU kernel optimizations.
//!
//! Measures decode speed (tokens/second) and kernel variant performance
//! to validate optimization impact from hipfire-inspired improvements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rocmforge::config::ModelConfig;
use rocmforge::cpu::{cache::CpuForwardScratch, weights::CpuModelWeights};
use rocmforge::gpu::{self, GpuDevice, GpuForwardScratch, GpuKvCache, GpuLogitsMode};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;
use std::path::Path;
use std::time::{Duration, Instant};

const DEFAULT_MODEL_PATH: &str =
    "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const DEFAULT_PROMPT: &str = "Hello";
const DEFAULT_DECODE_TOKENS: usize = 64;

struct KernelBenchContext {
    device: GpuDevice,
    config: ModelConfig,
    cpu_weights: CpuModelWeights,
    gpu_weights: gpu::GpuModelWeights,
    prompt_tokens: Vec<u32>,
    decode_tokens: usize,
    model_label: String,
}

#[allow(dead_code)]
#[derive(Debug)]
struct DecodeStats {
    decode_ms: f64,
    decode_tok_s: f64,
}

impl KernelBenchContext {
    fn load() -> Result<Self, String> {
        if !gpu::run_gpu_benches_enabled() {
            return Err(format!(
                "set {}=1 to run GPU kernel performance benchmarks",
                gpu::RUN_GPU_BENCHES_ENV
            ));
        }

        let model_path = std::env::var("ROCMFORGE_BENCH_MODEL")
            .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
        let prompt =
            std::env::var("ROCMFORGE_BENCH_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
        let decode_tokens = parse_env_usize("ROCMFORGE_BENCH_TOKENS", DEFAULT_DECODE_TOKENS);

        if !Path::new(&model_path).exists() {
            return Err(format!(
                "model not found at {} (override with ROCMFORGE_BENCH_MODEL)",
                model_path
            ));
        }

        let caps = gpu::detect().ok_or_else(|| "GPU not detected".to_string())?;
        let device =
            GpuDevice::init(caps.device_id).map_err(|err| format!("GPU init failed: {}", err))?;

        let file =
            GgufFile::open(&model_path).map_err(|err| format!("open GGUF failed: {}", err))?;
        let config =
            ModelConfig::from_gguf(&file).map_err(|err| format!("GGUF config failed: {}", err))?;
        let cpu_weights = CpuModelWeights::load(&file, &config)
            .map_err(|err| format!("CPU weights failed: {}", err))?;
        let gpu_weights = gpu::GpuModelWeights::load(&file, &config)
            .map_err(|err| format!("GPU weights failed: {}", err))?;
        let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
        let prompt_tokens = tok.encode(&prompt, false);
        if prompt_tokens.is_empty() {
            return Err(format!("prompt {:?} tokenized to zero tokens", prompt));
        }

        let model_label = Path::new(&model_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(DEFAULT_MODEL_PATH)
            .to_string();

        Ok(Self {
            device,
            config,
            cpu_weights,
            gpu_weights,
            prompt_tokens,
            decode_tokens,
            model_label,
        })
    }

    fn run_decode_benchmark(&self) -> Result<DecodeStats, String> {
        let mut kv = GpuKvCache::new(&self.config, self.prompt_tokens.len() + self.decode_tokens)
            .map_err(|err| format!("GPU KV alloc failed: {}", err))?;
        let mut gpu_scratch = GpuForwardScratch::new(&self.config)
            .map_err(|err| format!("GPU scratch alloc failed: {}", err))?;
        let mut host_scratch = CpuForwardScratch::new(&self.config);

        // Prefill prompt
        let mut next_token = None;
        for (pos, &token_id) in self.prompt_tokens.iter().enumerate() {
            gpu::gpu_embed_token_hybrid(
                &self.device,
                token_id,
                &self.gpu_weights,
                &self.cpu_weights,
                &mut gpu_scratch,
                &mut host_scratch,
                &self.config,
            )
            .map_err(|err| format!("GPU embed failed: {}", err))?;
            next_token = gpu::gpu_full_forward_hybrid(
                &self.device,
                &self.gpu_weights,
                &self.cpu_weights,
                &mut kv,
                &mut gpu_scratch,
                &mut host_scratch,
                pos,
                &self.config,
                GpuLogitsMode::GreedyArgmax,
                token,
            )
            .map_err(|err| format!("GPU prefill failed: {}", err))?;
        }

        let mut token = next_token.ok_or_else(|| "prefill produced no greedy token".to_string())?;

        // Benchmark decode loop
        let decode_start = Instant::now();
        for step in 0..self.decode_tokens {
            gpu::gpu_embed_token_hybrid(
                &self.device,
                token,
                &self.gpu_weights,
                &self.cpu_weights,
                &mut gpu_scratch,
                &mut host_scratch,
                &self.config,
            )
            .map_err(|err| format!("GPU embed failed: {}", err))?;
            token = gpu::gpu_full_forward_hybrid(
                &self.device,
                &self.gpu_weights,
                &self.cpu_weights,
                &mut kv,
                &mut gpu_scratch,
                &mut host_scratch,
                self.prompt_tokens.len() + step,
                &self.config,
                GpuLogitsMode::GreedyArgmax,
                token,
            )
            .map_err(|err| format!("GPU decode failed: {}", err))?
            .ok_or_else(|| "decode step produced no greedy token".to_string())?;
        }
        let decode_elapsed = decode_start.elapsed();

        Ok(DecodeStats {
            decode_ms: decode_elapsed.as_secs_f64() * 1000.0,
            decode_tok_s: self.decode_tokens as f64 / decode_elapsed.as_secs_f64(),
        })
    }
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn parse_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn criterion_config() -> Criterion {
    let sample_size = parse_env_usize("ROCMFORGE_CRITERION_SAMPLE_SIZE", 10).max(10);
    let warmup_secs = parse_env_u64("ROCMFORGE_CRITERION_WARMUP_SECS", 3);
    let measurement_secs = parse_env_u64("ROCMFORGE_CRITERION_MEASUREMENT_SECS", 15);

    Criterion::default()
        .sample_size(sample_size)
        .warm_up_time(Duration::from_secs(warmup_secs))
        .measurement_time(Duration::from_secs(measurement_secs))
        .without_plots()
}

/// Benchmark decode speed (tokens/second).
///
/// Measures the actual inference throughput with the current kernel variant.
fn bench_decode_speed(c: &mut Criterion) {
    let ctx = match KernelBenchContext::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("Skipping kernel_performance benchmark: {}", err);
            return;
        }
    };

    let mut group = c.benchmark_group("decode_speed");
    group.throughput(Throughput::Elements(ctx.decode_tokens as u64));
    group.bench_function(
        BenchmarkId::new("tokens_per_second", &ctx.model_label),
        |b| {
            b.iter(|| {
                let stats = ctx
                    .run_decode_benchmark()
                    .expect("decode benchmark should succeed");
                black_box(stats.decode_tok_s);
            });
        },
    );
    group.finish();
}

/// Benchmark kernel variant comparison (scalar vs DP4A).
///
/// Compares performance between different kernel implementations.
/// Note: This benchmark shows the current active variant.
/// To compare variants, set ROCMFORGE_GPU_SAFE_MODE=0 and rebuild with
/// different launch autotune configurations.
fn bench_kernel_variants(c: &mut Criterion) {
    let ctx = match KernelBenchContext::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("Skipping kernel_variant benchmark: {}", err);
            return;
        }
    };

    // Detect GPU features to determine which variant is active
    let _caps = match gpu::detect() {
        Some(caps) => caps,
        None => {
            eprintln!("Skipping kernel_variant benchmark: cannot detect GPU capabilities");
            return;
        }
    };

    let features = match gpu::GpuFeatures::detect(&ctx.device) {
        Ok(features) => features,
        Err(err) => {
            eprintln!(
                "Skipping kernel_variant benchmark: feature detection failed: {}",
                err
            );
            return;
        }
    };

    let variant_name = if features.has_dp4a { "dp4a" } else { "scalar" };

    let mut group = c.benchmark_group("kernel_variants");
    group.throughput(Throughput::Elements(ctx.decode_tokens as u64));
    group.bench_function(BenchmarkId::new(variant_name, &ctx.model_label), |b| {
        b.iter(|| {
            let stats = ctx
                .run_decode_benchmark()
                .expect("decode benchmark should succeed");
            black_box(stats.decode_tok_s);
        });
    });
    group.finish();
}

criterion_group! {
    name = kernel_performance;
    config = criterion_config();
    targets = bench_decode_speed, bench_kernel_variants
}
criterion_main!(kernel_performance);
