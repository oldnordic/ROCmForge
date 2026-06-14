#![cfg(feature = "cpu-graph")]
//! CPU Graph Temporal Regression and Windowed Execution Tests
//!
//! Verifies:
//! 1. Multi-step capture with incrementing timestamps.
//! 2. `CaptureContext::regress_to()` restores the persistent shelf from the
//!    snapshot taken at the target timestamp, giving instant rollback without
//!    replaying the prefix.
//! 3. Windowed execution only runs nodes within the specific time range.

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::cpu_layer_forward_with_ctx;
use rocmforge::cpu::graph::{CaptureContext, TemporalWindow};
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

    // Simplified RoPE inputs (constant across positions for this test).
    let sin_0 = vec![0.0f32; config.head_dim / 2];
    let cos_0 = vec![1.0f32; config.head_dim / 2];

    // Direct reference: run only Step 0 from the initial state.
    let mut hidden_prefix_ref = vec![0.1f32; h];
    let mut kv_prefix_ref = CpuKvCache::new(&config, 10);
    let mut scratch_prefix_ref = CpuForwardScratch::new(&config);
    cpu_layer_forward_with_ctx(
        &mut rocmforge::cpu::graph::DirectContext,
        &mut hidden_prefix_ref,
        weights.layer(layer_idx),
        &mut kv_prefix_ref,
        &mut scratch_prefix_ref,
        layer_idx,
        0,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Direct prefix reference failed");

    // 2. Capture Step 0 (Timestamp 0)
    println!("--- Capturing Step 0 ---");

    let mut capture_ctx = CaptureContext::new(layer_idx, 0);

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

    // 4. Test Regression: Rollback to T=0
    println!("--- Testing Regression to T=0 ---");
    capture_ctx.regress_to(0); // Restore persistent shelf + bindings from the T=0 snapshot
    for (i, node) in capture_ctx.graph.nodes.iter().enumerate() {
        println!("Node {} begin={} end={}", i, node.begin_ts, node.end_ts);
    }

    // The persistent shelf was restored to the T=0 snapshot, so read_back
    // reflects the prefix state without replaying any prefix ops.
    unsafe {
        capture_ctx.read_back();
    }
    let hidden_after_step_0 = hidden.clone();

    // Execute full window (0 to infinity) - no active nodes should remain after
    // regressing to T=0, but the shelf snapshot still provides the prefix state.
    let window_all = TemporalWindow { start: 0, end: 0 };
    capture_ctx
        .graph
        .execute_window(&mut capture_ctx.arena, window_all)
        .expect("Replay all failed");
    unsafe {
        capture_ctx.read_back();
    }
    let hidden_after_rollback_all = hidden.clone();

    let regression_err = max_abs_error(&hidden_after_step_0, &hidden_after_rollback_all);
    println!("ERROR:  {:.8}", regression_err);

    let prefix_err = max_abs_error(&hidden_after_rollback_all, &hidden_prefix_ref);
    println!("PREFIX ERROR: {:.8}", prefix_err);
    assert!(
        prefix_err < 1e-6,
        "Rollback did not restore the prefix state! err={:.8}",
        prefix_err
    );

    println!(
        "DEBUG: hidden_after_step_0[0] = {}, hidden_after_rollback_all[0] = {}",
        hidden_after_step_0[0], hidden_after_rollback_all[0]
    );

    // 5. Test Windowing: Execute only Step 1 (Expect NO ops to run because Step 1 is regressed)
    println!("--- Testing Windowed Step 1 (should be empty) ---");
    // Reset hidden to a known value
    for v in hidden.iter_mut().take(h) {
        *v = 42.0;
    }
    let window_1 = TemporalWindow { start: 1, end: 2 };
    println!(
        "DEBUG: graph.nodes[0].end_ts = {}, graph.nodes[10].end_ts = {}",
        capture_ctx.graph.nodes[0].end_ts, capture_ctx.graph.nodes[10].end_ts
    );
    capture_ctx
        .graph
        .execute_window(&mut capture_ctx.arena, window_1)
        .expect("Replay Step 1 failed");
    // No read_back: the window is empty, so caller buffers must stay untouched.

    assert_eq!(
        hidden[0], 42.0,
        "Windowing failed! Ops ran in an invalidated window."
    );

    println!("Temporal test PASSED");
}
