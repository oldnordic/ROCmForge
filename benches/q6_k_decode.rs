#![cfg(feature = "gpu")]
#![allow(warnings)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rocmforge::config::ModelConfig;
use rocmforge::cpu::{cache::CpuForwardScratch, weights::CpuModelWeights};
use rocmforge::gpu::{self, GpuDevice, GpuForwardScratch, GpuKvCache};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;
use std::path::Path;
use std::time::{Duration, Instant};

const Q6_K_MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2-0.5b-instruct-q6_k.gguf";

struct Q6KDecodeBenchContext {
    device: GpuDevice,
    config: ModelConfig,
    cpu_weights: CpuModelWeights,
    gpu_weights: gpu::GpuModelWeights,
    prompt_tokens: Vec<u32>,
    decode_tokens: usize,
    model_label: String,
}

#[derive(Debug)]
struct DecodeRunStats {
    prefill_ms: f64,
    decode_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
}

impl Q6KDecodeBenchContext {
    fn load() -> Result<Self, String> {
        if !gpu::run_gpu_benches_enabled() {
            return Err(format!(
                "set {}=1 to run real-model GPU benchmarks",
                gpu::RUN_GPU_BENCHES_ENV
            ));
        }
        if !gpu::decode_graph_enabled() {
            return Err(format!(
                "set {}=1 to enable graph-backed GPU decode benchmarks",
                gpu::ENABLE_DECODE_GRAPH_ENV
            ));
        }

        if !Path::new(Q6_K_MODEL_PATH).exists() {
            return Err(format!("Q6_K model not found at {}", Q6_K_MODEL_PATH));
        }

        let caps = gpu::detect().ok_or_else(|| "GPU not detected".to_string())?;
        let device =
            GpuDevice::init(caps.device_id).map_err(|err| format!("GPU init failed: {}", err))?;

        let file =
            GgufFile::open(Q6_K_MODEL_PATH).map_err(|err| format!("open GGUF failed: {}", err))?;
        let config =
            ModelConfig::from_gguf(&file).map_err(|err| format!("GGUF config failed: {}", err))?;
        let cpu_weights = CpuModelWeights::load(&file, &config)
            .map_err(|err| format!("CPU weights failed: {}", err))?;
        let gpu_weights = gpu::GpuModelWeights::load(&file, &config)
            .map_err(|err| format!("GPU weights failed: {}", err))?;

        let model_label = Path::new(Q6_K_MODEL_PATH)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(Q6_K_MODEL_PATH)
            .to_string();

        Ok(Self {
            device,
            config,
            cpu_weights,
            gpu_weights,
            prompt_tokens: Vec::new(), // Will be set per benchmark
            decode_tokens: 64,
            model_label,
        })
    }

    fn run_once(&self) -> Result<DecodeRunStats, String> {
        let mut kv = GpuKvCache::new(&self.config, self.prompt_tokens.len() + self.decode_tokens)
            .map_err(|err| format!("GPU KV alloc failed: {}", err))?;
        let mut gpu_scratch = GpuForwardScratch::new(&self.config)
            .map_err(|err| format!("GPU scratch alloc failed: {}", err))?;
        let mut host_scratch = CpuForwardScratch::new(&self.config);

        let prefill_start = Instant::now();
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
                gpu::GpuLogitsMode::GreedyArgmax,
            )
            .map_err(|err| format!("GPU prefill failed: {}", err))?;
        }
        let prefill_elapsed = prefill_start.elapsed();

        let mut token = next_token.ok_or_else(|| "prefill produced no greedy token".to_string())?;
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
                gpu::GpuLogitsMode::GreedyArgmax,
            )
            .map_err(|err| format!("GPU decode failed: {}", err))?
            .ok_or_else(|| "decode step produced no greedy token".to_string())?;
        }
        let decode_elapsed = decode_start.elapsed();

        Ok(DecodeRunStats {
            prefill_ms: prefill_elapsed.as_secs_f64() * 1000.0,
            decode_ms: decode_elapsed.as_secs_f64() * 1000.0,
            prefill_tok_s: self.prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64(),
            decode_tok_s: self.decode_tokens as f64 / decode_elapsed.as_secs_f64(),
        })
    }
}

fn bench_q6_k_decode_single_token(c: &mut Criterion) {
    let ctx = match Q6KDecodeBenchContext::load() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping Q6_K decode benchmarks: {}", e);
            return;
        }
    };

    let mut ctx = ctx;
    ctx.prompt_tokens = vec![1]; // Single token

    let mut group = c.benchmark_group("q6_k_decode_single_token");
    group.sample_size(20);

    group.bench_function("single_token", |b| {
        b.iter(|| black_box(ctx.run_once()).expect("Q6_K decode should succeed"))
    });

    group.finish();
}

fn bench_q6_k_decode_multi_token(c: &mut Criterion) {
    let ctx = match Q6KDecodeBenchContext::load() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping Q6_K decode benchmarks: {}", e);
            return;
        }
    };

    // Test various prompt lengths
    let prompt_lengths = vec![1, 5, 17, 133];

    let mut group = c.benchmark_group("q6_k_decode_multi_token");
    group.sample_size(20);
    group.throughput(Throughput::Elements(ctx.decode_tokens as u64));

    for length in prompt_lengths {
        let mut ctx = match Q6KDecodeBenchContext::load() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping Q6_K decode benchmarks: {}", e);
                return;
            }
        };

        // Create a prompt of approximately the requested length
        let prompt = if length <= 5 {
            "Hello, how are you?".to_string()
        } else if length <= 17 {
            "Hello, how are you doing today? I hope you are well.".to_string()
        } else {
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. It is a classic pangram used for testing typewriters and computer fonts. In this test, we want to verify that the model can handle long prompts without producing corrupted output.".to_string()
        };

        let tok =
            BpeTokenizer::from_gguf(GgufFile::open(Q6_K_MODEL_PATH).unwrap().tokenizer_data());
        ctx.prompt_tokens = tok.encode(&prompt, false);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_tokens", ctx.prompt_tokens.len())),
            &ctx.prompt_tokens.len(),
            |b, _| {
                b.iter(|| {
                    let stats = ctx.run_once()
                        .expect("Q6_K decode should succeed");

                    // Baseline: 133 tok/s = ~7.5ms per token
                    // Allow 5% tolerance: 7.5ms * 1.05 = 7.875ms
                    let max_expected_ms_per_token = 7.875;

                    let ms_per_token = stats.decode_ms / ctx.decode_tokens as f64;

                    assert!(
                        ms_per_token <= max_expected_ms_per_token,
                        "Q6_K decode performance regression: {:.2}ms/token > {:.2}ms/token (baseline 133 tok/s + 5%)",
                        ms_per_token,
                        max_expected_ms_per_token
                    );

                    black_box(stats)
                })
            },
        );
    }

    group.finish();
}

fn bench_q6_k_decode_comparison(c: &mut Criterion) {
    // Verify benchmark environment is set up correctly
    let _ctx = match Q6KDecodeBenchContext::load() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping Q6_K decode benchmarks: {}", e);
            return;
        }
    };

    // Use 17-token prompt (classic bug case)
    let prompt = "Hello, how are you doing today? I hope you are well.";
    let tok = BpeTokenizer::from_gguf(GgufFile::open(Q6_K_MODEL_PATH).unwrap().tokenizer_data());

    let mut group = c.benchmark_group("q6_k_decode_17_token_period_ending");
    group.sample_size(20);

    group.bench_function("period_ending_prompt", |b| {
        b.iter(|| {
            // Reload context for each iteration (GPU resources aren't Clone)
            let mut ctx =
                Q6KDecodeBenchContext::load().expect("Failed to load Q6_K benchmark context");
            ctx.prompt_tokens = tok.encode(&prompt, false);

            black_box(ctx.run_once()).expect("Q6_K decode should succeed")
        })
    });

    group.finish();
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn criterion_config() -> Criterion {
    let sample_size = parse_env_usize("ROCMFORGE_CRITERION_SAMPLE_SIZE", 10).max(10);
    let warmup_secs = parse_env_usize("ROCMFORGE_CRITERION_WARMUP_SECS", 3);
    let measurement_secs = parse_env_usize("ROCMFORGE_CRITERION_MEASUREMENT_SECS", 15);

    Criterion::default()
        .sample_size(sample_size)
        .warm_up_time(Duration::from_secs(warmup_secs as u64))
        .measurement_time(Duration::from_secs(measurement_secs as u64))
        .without_plots()
}

criterion_group!(
    name = benches;
    config = criterion_config();
    targets = bench_q6_k_decode_single_token, bench_q6_k_decode_multi_token, bench_q6_k_decode_comparison
);
criterion_main!(benches);
