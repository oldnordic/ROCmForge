use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rocmforge::config::ModelConfig;
use rocmforge::cpu::{cache::CpuForwardScratch, weights::CpuModelWeights};
use rocmforge::gpu::{self, GpuDevice, GpuForwardScratch, GpuKvCache};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;
use std::path::Path;
use std::time::{Duration, Instant};

const DEFAULT_MODEL_PATH: &str =
    "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const DEFAULT_PROMPT: &str = "Hello";
const DEFAULT_DECODE_TOKENS: usize = 64;

struct DecodeBenchContext {
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

impl DecodeBenchContext {
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

fn bench_gpu_decode_real_model(c: &mut Criterion) {
    let ctx = match DecodeBenchContext::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("Skipping gpu_decode benchmark: {}", err);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_decode_real_model");
    group.throughput(Throughput::Elements(ctx.decode_tokens as u64));
    group.bench_function(
        BenchmarkId::new("graph_backed_prompt_plus_decode", &ctx.model_label),
        |b| {
            b.iter(|| {
                let stats = ctx
                    .run_once()
                    .expect("gpu decode benchmark iteration should succeed");
                black_box(stats.prefill_ms);
                black_box(stats.decode_ms);
                black_box(stats.prefill_tok_s);
                black_box(stats.decode_tok_s);
            });
        },
    );
    group.finish();
}

criterion_group! {
    name = gpu_decode;
    config = criterion_config();
    targets = bench_gpu_decode_real_model
}
criterion_main!(gpu_decode);
