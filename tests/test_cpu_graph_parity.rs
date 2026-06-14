#![cfg(feature = "cpu-graph")]
//! CPU Graph vs CPU Direct parity tests
//!
//! Verifies that the GeoGraph-backed execution engine produces
//! identical results to the standard imperative path.

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::{cpu_layer_forward, cpu_layer_forward_with_ctx};
use rocmforge::cpu::graph::CaptureContext;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::loader::GgufFile;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "cpu-graph")]
#[test]
#[serial]
fn test_cpu_graph_parity() {
    if skip_if_model_missing() {
        eprintln!("Skipping parity test: model not found at {}", MODEL_PATH);
        return;
    }

    // 1. Setup
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");

    let h = config.hidden_size;
    let layer_idx = 0;

    // 2. Reference Run (Imperative)
    let mut hidden_ref = vec![0.1f32; h];
    for (i, v) in hidden_ref.iter_mut().enumerate().take(h) {
        *v = (i as f32).sin() * 0.1;
    }
    let mut kv_ref = CpuKvCache::new(&config, 1);
    let mut scratch_ref = CpuForwardScratch::new(&config);

    // RoPE sin/cos for pos 0
    let half = config.head_dim / 2;
    let mut sin = vec![0.0f32; half];
    let mut cos = vec![0.0f32; half];
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        sin[i] = s;
        cos[i] = c;
    }

    cpu_layer_forward(
        &mut hidden_ref,
        weights.layer(layer_idx),
        &mut kv_ref,
        &mut scratch_ref,
        layer_idx,
        0, // pos
        &sin,
        &cos,
        &config,
        false,
    )
    .expect("Direct forward failed");

    // 3. Graph Capture Run
    let mut hidden_graph = vec![0.1f32; h];
    for (i, v) in hidden_graph.iter_mut().enumerate().take(h) {
        *v = (i as f32).sin() * 0.1;
    }
    let mut kv_graph = CpuKvCache::new(&config, 1);
    let mut scratch_graph = CpuForwardScratch::new(&config);

    let mut capture_ctx = CaptureContext::new(layer_idx, 0);

    cpu_layer_forward_with_ctx(
        &mut capture_ctx,
        &mut hidden_graph,
        weights.layer(layer_idx),
        &mut kv_graph,
        &mut scratch_graph,
        layer_idx,
        0, // pos
        &sin,
        &cos,
        &config,
        false,
    )
    .expect("Capture forward failed");

    // 4. Replay Run
    // Reset hidden state for replay
    for (i, v) in hidden_graph.iter_mut().enumerate().take(h) {
        *v = (i as f32).sin() * 0.1;
    }
    // Replay the captured graph
    let window = rocmforge::cpu::graph::TemporalWindow {
        start: 0,
        end: u64::MAX,
    };
    capture_ctx
        .graph
        .execute_window(&mut capture_ctx.arena, window)
        .expect("Graph replay failed");
    unsafe {
        capture_ctx.read_back();
    }

    // 5. Compare
    let err = max_abs_error(&hidden_ref, &hidden_graph);
    println!(
        "Max absolute error between Direct and Graph-Replay: {:.8}",
        err
    );

    assert!(err < 1e-6, "Parity check failed! Error: {:.8}", err);
}
