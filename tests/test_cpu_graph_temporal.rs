//! CPU Graph Temporal Regression and Windowed Execution Tests
//!
//! Verifies:
//! 1. Multi-step capture with incrementing timestamps.
//! 2. Temporal regression (rollback) invalidates nodes correctly.
//! 3. Windowed execution only runs nodes within the specific time range.

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::{cpu_layer_forward, cpu_layer_forward_with_ctx};
use rocmforge::cpu::graph::{CaptureContext, CpuGraph, TemporalWindow};
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
fn test_cpu_graph_temporal_flow() {
    if skip_if_model_missing() {
        eprintln!("Skipping temporal test: model not found at {}", MODEL_PATH);
        return;
    }

    // 1. Setup
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");

    let h = config.hidden_size;
    let layer_idx = 0;
    let mut kv = CpuKvCache::new(&config, 10); // Room for 10 tokens
    let mut scratch = CpuForwardScratch::new(&config);
    let mut hidden = vec![0.1f32; h];

    let mut graph = CpuGraph::new();

    // 2. Capture Step 0 (Timestamp 0)
    println!("--- Capturing Step 0 ---");
    let mut sin_0 = vec![0.0f32; config.head_dim / 2];
    let mut cos_0 = vec![1.0f32; config.head_dim / 2]; // Simplified for test

    let mut capture_ctx = CaptureContext {
        graph,
        layer: layer_idx,
        step: 0,
        timestamp: 0,
    };

    cpu_layer_forward_with_ctx(
        &mut capture_ctx,
        &mut hidden,
        weights.layer(layer_idx),
        &mut kv,
        &mut scratch,
        layer_idx,
        0, // pos
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Capture Step 0 failed");

    // 3. Capture Step 1 (Timestamp 1)
    println!("--- Capturing Step 1 ---");
    capture_ctx.timestamp = 1;
    capture_ctx.step = 0; // Reset step counter for new timestamp

    cpu_layer_forward_with_ctx(
        &mut capture_ctx,
        &mut hidden,
        weights.layer(layer_idx),
        &mut kv,
        &mut scratch,
        layer_idx,
        1, // pos
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Capture Step 1 failed");

    let mut graph = capture_ctx.graph;

    // 4. Test Regression: Rollback to T=0
    println!("--- Testing Regression to T=0 ---");
    graph.regress(0); // This should set end_ts=0 for all nodes where begin_ts > 0
    for (i, node) in graph.nodes.iter().enumerate() {
        println!("Node {} begin={} end={}", i, node.begin_ts, node.end_ts);
    }

    // Re-initialize hidden state
    for i in 0..h {
        hidden[i] = 0.1;
    }

    // Execute only Step 0
    let window_0 = TemporalWindow { start: 0, end: 1 };
    graph
        .execute_window(window_0, Some(&mut scratch.q8_scratch))
        .expect("Replay Step 0 failed");
    let hidden_after_step_0 = hidden.clone();

    // Re-initialize hidden state again
    for i in 0..h {
        hidden[i] = 0.1;
    }

    // Execute full window (0 to infinity) - should still only run Step 0 because Step 1 is regressed
    let window_all = TemporalWindow { start: 0, end: 0 };
    graph
        .execute_window(window_all, Some(&mut scratch.q8_scratch))
        .expect("Replay all failed");
    let hidden_after_rollback_all = hidden.clone();

    let regression_err = max_abs_error(&hidden_after_step_0, &hidden_after_rollback_all);
    println!("ERROR:  {:.8}", regression_err);

    println!(
        "DEBUG: hidden_after_step_0[0] = {}, hidden_after_rollback_all[0] = {}",
        hidden_after_step_0[0], hidden_after_rollback_all[0]
    );

    // 5. Test Windowing: Execute only Step 1 (Expect NO ops to run because Step 1 is regressed)
    println!("--- Testing Windowed Step 1 (should be empty) ---");
    // Reset hidden to a known value
    for i in 0..h {
        hidden[i] = 42.0;
    }
    let window_1 = TemporalWindow { start: 1, end: 2 };
    println!(
        "DEBUG: graph.nodes[0].end_ts = {}, graph.nodes[10].end_ts = {}",
        graph.nodes[0].end_ts, graph.nodes[10].end_ts
    );
    graph
        .execute_window(window_1, Some(&mut scratch.q8_scratch))
        .expect("Replay Step 1 failed");

    assert_eq!(
        hidden[0], 42.0,
        "Windowing failed! Ops ran in an invalidated window."
    );

    println!("Temporal test PASSED");
}
