//! CPU Graph Engine Benchmarks
//!
//! Compares standard imperative execution vs GeoGraph-backed execution.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::{cpu_layer_forward, cpu_layer_forward_with_ctx};
use rocmforge::cpu::graph::CaptureContext;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::loader::GgufFile;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn bench_cpu_graph_vs_direct(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_PATH).exists() {
        return;
    }

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");

    let h = config.hidden_size;
    let layer_idx = 0;

    let mut hidden = vec![0.1f32; h];
    let mut kv = CpuKvCache::new(&config, 1);
    let mut scratch = CpuForwardScratch::new(&config);

    let half = config.head_dim / 2;
    let mut sin = vec![0.0f32; half];
    let mut cos = vec![0.0f32; half];
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        sin[i] = s;
        cos[i] = c;
    }

    // Capture the graph once
    let mut capture_ctx = CaptureContext::new(layer_idx, 0);
    cpu_layer_forward_with_ctx(
        &mut capture_ctx,
        &mut hidden,
        weights.layer(layer_idx),
        &mut kv,
        &mut scratch,
        layer_idx,
        0,
        &sin,
        &cos,
        &config,
        false,
    )
    .expect("bench error");
    let window = rocmforge::cpu::graph::TemporalWindow {
        start: 0,
        end: u64::MAX,
    };

    let mut group = c.benchmark_group("cpu_forward_comparison");

    group.bench_function("direct_imperative", |b| {
        b.iter(|| {
            cpu_layer_forward(
                black_box(&mut hidden),
                black_box(weights.layer(layer_idx)),
                black_box(&mut kv),
                black_box(&mut scratch),
                black_box(layer_idx),
                black_box(0),
                black_box(&sin),
                black_box(&cos),
                black_box(&config),
                black_box(false),
            )
            .expect("bench error");
        });
    });

    group.bench_function("geograph_replay", |b| {
        b.iter(|| {
            capture_ctx
                .graph
                .execute_window(black_box(&mut capture_ctx.arena), window)
                .expect("bench error");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_cpu_graph_vs_direct);
criterion_main!(benches);
