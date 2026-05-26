#![cfg(feature = "gpu")]
#![allow(warnings)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rocmforge::config::ModelConfig;
use rocmforge::gpu::{self, GpuDevice, GpuKvCache, GpuPrefillScratch};
use rocmforge::loader::GgufFile;
use std::path::Path;

const DEFAULT_MODEL_PATH: &str =
    "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

struct PrefillBenchContext {
    device: GpuDevice,
    config: ModelConfig,
    gpu_weights: gpu::GpuModelWeights,
    model_label: String,
}

impl PrefillBenchContext {
    fn load() -> Result<Self, String> {
        if !gpu::run_gpu_benches_enabled() {
            return Err(format!(
                "set {}=1 to run real-model GPU benchmarks",
                gpu::RUN_GPU_BENCHES_ENV
            ));
        }

        let model_path = std::env::var("ROCMFORGE_BENCH_MODEL")
            .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());

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
        let gpu_weights = gpu::GpuModelWeights::load(&file, &config)
            .map_err(|err| format!("GPU weights failed: {}", err))?;

        let model_label = Path::new(&model_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(DEFAULT_MODEL_PATH)
            .to_string();

        Ok(Self {
            device,
            config,
            gpu_weights,
            model_label,
        })
    }

    fn bench_prefill_layer_stub(&self, seq_len: usize, layer_idx: usize) -> Result<(), String> {
        let mut kv = GpuKvCache::new(&self.config, seq_len)
            .map_err(|err| format!("GPU KV alloc failed: {}", err))?;
        let mut scratch = GpuPrefillScratch::new(&self.config, seq_len)
            .map_err(|err| format!("GPU scratch alloc failed: {}", err))?;

        let layer_weights = &self.gpu_weights.layer(layer_idx);

        // This will return "not yet implemented" error for milestone 1
        let result = gpu::gpu_prefill_layer_forward_q4_0(
            &self.device,
            layer_weights,
            &mut scratch,
            &kv,
            layer_idx,
            0,
            &self.config,
        );

        // For milestone 1, we expect this to fail gracefully
        match result {
            Ok(()) => Ok(()),
            Err(_) => Ok(()), // Expected for milestone 1 stub
        }
    }
}

fn parse_env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn bench_prefill_layer_stub_single(c: &mut Criterion) {
    let ctx = match PrefillBenchContext::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            println!("Skipping benchmark: {}", err);
            return;
        }
    };

    let mut group = c.benchmark_group("prefill_layer_stub");
    group.throughput(Throughput::Elements(1));

    for seq_len in [8, 16, 32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    let result = ctx.bench_prefill_layer_stub(seq_len, 0);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_prefill_layer_stub_multi(c: &mut Criterion) {
    let ctx = match PrefillBenchContext::load() {
        Ok(ctx) => ctx,
        Err(err) => {
            println!("Skipping benchmark: {}", err);
            return;
        }
    };

    let mut group = c.benchmark_group("prefill_layer_stub_multi");
    group.throughput(Throughput::Elements(1));

    let seq_len = 32;
    let num_layers = ctx.config.num_layers.min(5);

    for layer_idx in 0..num_layers {
        group.bench_with_input(
            BenchmarkId::new("layer", layer_idx),
            &layer_idx,
            |b, &layer_idx| {
                b.iter(|| {
                    let result = ctx.bench_prefill_layer_stub(seq_len, layer_idx);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_prefill_layer_stub_single,
    bench_prefill_layer_stub_multi
);
criterion_main!(benches);
