#![cfg(all(feature = "gpu", feature = "cpu-graph"))]
//! End-to-end test for GPU decode GraphMap trace capture and reload.
//!
//! This test runs the `rocmforge` CLI binary with `--gpu --graph-map-dir`,
//! verifies that a GPU trace is persisted, then loads the map and checks the
//! recorded token entries.
//!
//! It is marked `#[ignore]` because it requires an AMD GPU and loads the 0.5B
//! model.

use std::path::PathBuf;
use std::process::Command;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn model_exists() -> bool {
    std::path::Path::new(MODEL_PATH).exists()
}

fn rocmforge_binary() -> PathBuf {
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop(); // deps
        exe.pop(); // release or debug
        exe.push("rocmforge");
        if exe.exists() {
            return exe;
        }
    }
    PathBuf::from("cargo")
}

fn build_command() -> Command {
    let bin = rocmforge_binary();
    let cmd = if bin.file_name().is_some_and(|n| n == "cargo") {
        let mut c = Command::new("cargo");
        c.args([
            "run",
            "--release",
            "--features",
            "gpu,cpu-graph",
            "--bin",
            "rocmforge",
            "--",
        ]);
        c.current_dir(env!("CARGO_MANIFEST_DIR"));
        c
    } else {
        Command::new(bin)
    };
    cmd
}

fn run_rocmforge(args: &[&str]) -> std::io::Result<std::process::Output> {
    let mut cmd = build_command();
    cmd.args(args);
    cmd.output()
}

#[ignore]
#[test]
fn test_gpu_captures_and_reloads_decode_trace() -> Result<(), Box<dyn std::error::Error>> {
    if !model_exists() {
        eprintln!("Skipping GPU trace test: model not found at {}", MODEL_PATH);
        return Ok(());
    }

    let capture_dir = tempfile::tempdir()?;
    let capture_path = capture_dir
        .path()
        .to_str()
        .ok_or("capture temp dir path is not UTF-8")?;

    let out = run_rocmforge(&[
        "--model",
        MODEL_PATH,
        "--prompt",
        "Hi",
        "--max-tokens",
        "3",
        "--gpu",
        "--graph-map-dir",
        capture_path,
        "--no-template",
    ])?;

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "GPU capture run failed with stderr:\n{}",
        stderr
    );
    assert!(
        stderr.contains("Saved GraphMap"),
        "expected 'Saved GraphMap' in stderr, got:\n{}",
        stderr
    );

    let map = rocmforge::cpu::graph::GraphMap::load(std::path::Path::new(capture_path))?;
    assert!(
        !map.gpu_trace().is_empty(),
        "GPU trace should contain at least one decode entry"
    );
    assert_eq!(
        map.gpu_trace().len(),
        map.branch_scores().len(),
        "every trace entry should have a corresponding branch score"
    );

    let reload_dir = tempfile::tempdir()?;
    let reload_path = reload_dir
        .path()
        .to_str()
        .ok_or("reload temp dir path is not UTF-8")?;

    let out2 = run_rocmforge(&[
        "--model",
        MODEL_PATH,
        "--prompt",
        "Hi",
        "--max-tokens",
        "1",
        "--gpu",
        "--load-graph-map-dir",
        capture_path,
        "--graph-map-dir",
        reload_path,
        "--no-template",
    ])?;

    let stderr2 = String::from_utf8_lossy(&out2.stderr);
    assert!(
        out2.status.success(),
        "GPU reload run failed with stderr:\n{}",
        stderr2
    );
    assert!(
        stderr2.contains("Loaded GraphMap"),
        "expected 'Loaded GraphMap' in stderr, got:\n{}",
        stderr2
    );
    assert!(
        stderr2.contains("Saved GraphMap"),
        "expected 'Saved GraphMap' in stderr after reload, got:\n{}",
        stderr2
    );

    let map2 = rocmforge::cpu::graph::GraphMap::load(std::path::Path::new(reload_path))?;
    assert!(
        !map2.gpu_trace().is_empty(),
        "reloaded GPU session should also produce a trace"
    );

    Ok(())
}
