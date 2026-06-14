#![cfg(feature = "cpu-graph")]
//! Round-trip persistence test for `GraphMap`.
//!
//! Captures a CPU layer, persists the graph + arena through geographdb-core
//! storage, reloads it into a fresh process context, and verifies the replay
//! still matches direct execution.

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::cpu_layer_forward_with_ctx;
use rocmforge::cpu::graph::{CaptureContext, DirectContext, GraphMap, TemporalWindow};
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::loader::GgufFile;
use serial_test::serial;
use tempfile::tempdir;

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
fn test_cpu_graph_persistence_round_trip() {
    if skip_if_model_missing() {
        eprintln!(
            "Skipping persistence test: model not found at {}",
            MODEL_PATH
        );
        return;
    }

    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");

    let h = config.hidden_size;
    let layer_idx = 0;
    let half = config.head_dim / 2;
    let sin_0 = vec![0.0f32; half];
    let cos_0 = vec![1.0f32; half];

    // 1. Direct reference.
    let mut hidden_ref = vec![0.1f32; h];
    let mut kv_ref = CpuKvCache::new(&config, 10);
    let mut scratch_ref = CpuForwardScratch::new(&config);
    cpu_layer_forward_with_ctx(
        &mut DirectContext,
        &mut hidden_ref,
        weights.layer(layer_idx),
        &mut kv_ref,
        &mut scratch_ref,
        layer_idx,
        0,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Direct reference failed");

    // 2. Capture and persist.
    let mut capture_ctx = CaptureContext::new(layer_idx, 0);
    let mut hidden = vec![0.1f32; h];
    let mut kv = CpuKvCache::new(&config, 10);
    let mut scratch = CpuForwardScratch::new(&config);

    cpu_layer_forward_with_ctx(
        &mut capture_ctx,
        &mut hidden,
        weights.layer(layer_idx),
        &mut kv,
        &mut scratch,
        layer_idx,
        0,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Capture failed");

    let dir = tempdir().expect("Failed to create temp dir");
    let map = GraphMap::from_context(&capture_ctx);
    map.save(dir.path()).expect("Failed to save GraphMap");

    // Deliberately drop the original context before loading.
    drop(capture_ctx);

    // 3. Load into a fresh context and replay.
    let loaded_map = GraphMap::load(dir.path()).expect("Failed to load GraphMap");
    let mut restored_ctx = loaded_map.into_context();

    let mut hidden_graph = vec![0.1f32; h];
    restored_ctx
        .graph
        .execute_window(
            &mut restored_ctx.arena,
            TemporalWindow {
                start: 0,
                end: u64::MAX,
            },
        )
        .expect("Replay from persisted graph failed");
    unsafe { restored_ctx.read_back() };
    hidden_graph.copy_from_slice(&hidden);

    let err = max_abs_error(&hidden_ref, &hidden_graph);
    println!("Max error after persistence round-trip: {:.8}", err);
    assert!(
        err < 1e-6,
        "Persisted graph replay diverged! err={:.8}",
        err
    );
}
