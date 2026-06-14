#![cfg(feature = "cpu-graph")]
//! CPU Graph instant rollback test.
//!
//! Proves that `CaptureContext::regress_to()` restores the persistent shelf
//! from the snapshot taken at the target timestamp, so the prefix state is
//! available immediately without replaying prefix ops.

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::cpu_layer_forward_with_ctx;
use rocmforge::cpu::graph::{CaptureContext, DirectContext, TemporalWindow};
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
fn test_cpu_graph_instant_rollback_snapshot() {
    if skip_if_model_missing() {
        eprintln!(
            "Skipping instant rollback test: model not found at {}",
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

    // Direct reference: prefix (pos 0) followed by step 1 (pos 1).
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
    .expect("Direct prefix failed");
    let hidden_prefix_ref = hidden_ref.clone();

    cpu_layer_forward_with_ctx(
        &mut DirectContext,
        &mut hidden_ref,
        weights.layer(layer_idx),
        &mut kv_ref,
        &mut scratch_ref,
        layer_idx,
        1,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Direct step 1 failed");

    // Capture the same two steps into the graph.
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
    .expect("Capture prefix failed");

    capture_ctx.timestamp = 1;
    capture_ctx.step = 0;
    cpu_layer_forward_with_ctx(
        &mut capture_ctx,
        &mut hidden,
        weights.layer(layer_idx),
        &mut kv,
        &mut scratch,
        layer_idx,
        1,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Capture step 1 failed");

    // Replay everything and check it matches the direct two-step result.
    let mut hidden_step1_graph = vec![0.1f32; h];
    capture_ctx
        .graph
        .execute_window(
            &mut capture_ctx.arena,
            TemporalWindow {
                start: 0,
                end: u64::MAX,
            },
        )
        .expect("Full replay failed");
    unsafe { capture_ctx.read_back() };
    hidden_step1_graph.copy_from_slice(&hidden);

    let full_err = max_abs_error(&hidden_ref, &hidden_step1_graph);
    println!("Full replay max error: {:.8}", full_err);
    assert!(full_err < 1e-6, "Full replay diverged! err={:.8}", full_err);

    // Instant rollback: the T=0 snapshot must exist and restoring it must yield
    // the prefix state without executing any prefix ops.
    assert!(
        capture_ctx.shelf_snapshots.contains_key(&0),
        "Missing persistent shelf snapshot at T=0"
    );

    capture_ctx.regress_to(0);

    // No nodes should be active across the full window after regressing to T=0.
    let active_after_regress: Vec<_> = capture_ctx
        .graph
        .nodes
        .iter()
        .filter(|n| {
            if n.end_ts == 0 {
                return false; // disabled by regress
            }
            let node_end = if n.end_ts == u64::MAX {
                u64::MAX
            } else {
                n.end_ts
            };
            n.begin_ts < u64::MAX && node_end > 0
        })
        .collect();
    assert!(
        active_after_regress.is_empty(),
        "Expected no active nodes after regress_to(0), found {}",
        active_after_regress.len()
    );

    // Even though no nodes run, read_back must reflect the restored prefix shelf.
    let mut hidden_rollback = vec![0.1f32; h];
    unsafe { capture_ctx.read_back() };
    hidden_rollback.copy_from_slice(&hidden);

    let rollback_err = max_abs_error(&hidden_prefix_ref, &hidden_rollback);
    println!("Instant rollback prefix error: {:.8}", rollback_err);
    assert!(
        rollback_err < 1e-6,
        "Instant rollback did not restore prefix state! err={:.8}",
        rollback_err
    );

    println!("Instant rollback snapshot test PASSED");
}
