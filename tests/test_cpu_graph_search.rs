#![cfg(feature = "cpu-graph")]
//! CPU Graph search / speculative rollback test.
//!
//! Demonstrates that a captured prefix can be shared across multiple
//! speculative branches, each rolled back with `graph.regress()` and
//! re-bound with `rebind_after_regress()` before a new branch is captured.

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

/// Apply a deterministic perturbation to a hidden state, simulating a
/// different sampled token before the next layer.
fn perturb(hidden: &mut [f32]) {
    for (i, v) in hidden.iter_mut().enumerate() {
        *v += 0.01 * (i as f32).sin();
    }
}

#[cfg(feature = "cpu-graph")]
#[test]
#[serial]
fn test_cpu_graph_search_rollback_and_branch() {
    if skip_if_model_missing() {
        eprintln!("Skipping search test: model not found at {}", MODEL_PATH);
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

    // ------------------------------------------------------------------
    // 1. Capture the shared prefix (position 0, timestamp 0).
    // ------------------------------------------------------------------
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
        0, // pos
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Prefix capture failed");

    // ------------------------------------------------------------------
    // 2. Capture speculative branch A (position 1, timestamp 1).
    // ------------------------------------------------------------------
    capture_ctx.timestamp = 1;
    capture_ctx.step = 0;

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
    .expect("Branch A capture failed");

    // ------------------------------------------------------------------
    // 3. Direct reference for branch A: prefix + branch A.
    // ------------------------------------------------------------------
    let mut hidden_a_ref = vec![0.1f32; h];
    let mut kv_a_ref = CpuKvCache::new(&config, 10);
    let mut scratch_a_ref = CpuForwardScratch::new(&config);
    cpu_layer_forward_with_ctx(
        &mut DirectContext,
        &mut hidden_a_ref,
        weights.layer(layer_idx),
        &mut kv_a_ref,
        &mut scratch_a_ref,
        layer_idx,
        0,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Direct branch A prefix failed");
    cpu_layer_forward_with_ctx(
        &mut DirectContext,
        &mut hidden_a_ref,
        weights.layer(layer_idx),
        &mut kv_a_ref,
        &mut scratch_a_ref,
        layer_idx,
        1,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Direct branch A continuation failed");

    // ------------------------------------------------------------------
    // 4. Replay the graph: should execute prefix + branch A.
    // ------------------------------------------------------------------
    let mut hidden_a_graph = vec![0.1f32; h];
    capture_ctx
        .graph
        .execute_window(
            &mut capture_ctx.arena,
            TemporalWindow {
                start: 0,
                end: u64::MAX,
            },
        )
        .expect("Branch A replay failed");
    unsafe { capture_ctx.read_back() };
    hidden_a_graph.copy_from_slice(&hidden);

    let err_a = max_abs_error(&hidden_a_ref, &hidden_a_graph);
    println!("Branch A max error: {:.8}", err_a);
    assert!(err_a < 1e-6, "Branch A replay diverged! err={:.8}", err_a);

    // ------------------------------------------------------------------
    // 5. Roll back to the prefix and restore the arena bindings.
    // ------------------------------------------------------------------
    capture_ctx.graph.regress(0);
    capture_ctx.rebind_after_regress(0);

    // Replay only the prefix to restore caller hidden/KV to prefix state.
    let mut hidden_prefix_restore = vec![0.1f32; h];
    capture_ctx
        .graph
        .execute_window(&mut capture_ctx.arena, TemporalWindow { start: 0, end: 2 })
        .expect("Prefix restore replay failed");
    unsafe { capture_ctx.read_back() };
    hidden_prefix_restore.copy_from_slice(&hidden);

    // Verify the restored prefix matches a fresh direct prefix run.
    let mut hidden_prefix_ref = vec![0.1f32; h];
    let mut kv_prefix_ref = CpuKvCache::new(&config, 10);
    let mut scratch_prefix_ref = CpuForwardScratch::new(&config);
    cpu_layer_forward_with_ctx(
        &mut DirectContext,
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

    let prefix_err = max_abs_error(&hidden_prefix_restore, &hidden_prefix_ref);
    println!("Prefix restore error: {:.8}", prefix_err);
    assert!(
        prefix_err < 1e-6,
        "Prefix restore after rollback failed! err={:.8}",
        prefix_err
    );

    // ------------------------------------------------------------------
    // 6. Capture an alternative branch B at timestamp 2.
    // ------------------------------------------------------------------
    capture_ctx.timestamp = 2;
    capture_ctx.step = 0;
    perturb(&mut hidden);

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
    .expect("Branch B capture failed");

    // ------------------------------------------------------------------
    // 7. Direct reference for branch B: prefix + perturb + branch B.
    // ------------------------------------------------------------------
    let mut hidden_b_ref = vec![0.1f32; h];
    let mut kv_b_ref = CpuKvCache::new(&config, 10);
    let mut scratch_b_ref = CpuForwardScratch::new(&config);
    cpu_layer_forward_with_ctx(
        &mut DirectContext,
        &mut hidden_b_ref,
        weights.layer(layer_idx),
        &mut kv_b_ref,
        &mut scratch_b_ref,
        layer_idx,
        0,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Direct branch B prefix failed");
    perturb(&mut hidden_b_ref);
    cpu_layer_forward_with_ctx(
        &mut DirectContext,
        &mut hidden_b_ref,
        weights.layer(layer_idx),
        &mut kv_b_ref,
        &mut scratch_b_ref,
        layer_idx,
        1,
        &sin_0,
        &cos_0,
        &config,
        false,
    )
    .expect("Direct branch B continuation failed");

    // ------------------------------------------------------------------
    // 8. Replay the graph: should execute prefix + branch B only.
    // ------------------------------------------------------------------
    let mut hidden_b_graph = vec![0.1f32; h];
    capture_ctx
        .graph
        .execute_window(
            &mut capture_ctx.arena,
            TemporalWindow {
                start: 0,
                end: u64::MAX,
            },
        )
        .expect("Branch B replay failed");
    unsafe { capture_ctx.read_back() };
    hidden_b_graph.copy_from_slice(&hidden);

    let err_b = max_abs_error(&hidden_b_ref, &hidden_b_graph);
    println!("Branch B max error: {:.8}", err_b);
    assert!(err_b < 1e-6, "Branch B replay diverged! err={:.8}", err_b);

    println!("Search rollback/branch test PASSED");
}
