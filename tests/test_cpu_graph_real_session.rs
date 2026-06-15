#![cfg(feature = "cpu-graph")]
//! End-to-end test for real-session GraphMap capture and reload.
//!
//! This test runs the `rocmforge` CLI binary with `--graph-map-dir`, verifies
//! that a GraphMap is persisted, then runs a second invocation with
//! `--load-graph-map-dir` and verifies the previous session is loaded.
//!
//! It is marked `#[ignore]` because it loads the 0.5B model and runs real CPU
//! inference.

use std::path::PathBuf;
use std::process::Command;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn model_exists() -> bool {
    std::path::Path::new(MODEL_PATH).exists()
}

fn rocmforge_binary() -> PathBuf {
    // The test binary lives in target/{profile}/deps/; the rocmforge binary is
    // in target/{profile}/rocmforge. Fall back to `cargo run` if the binary is
    // not already built.
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
            "cpu-graph",
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
fn test_cli_captures_and_reloads_graphmap() -> Result<(), Box<dyn std::error::Error>> {
    if !model_exists() {
        eprintln!(
            "Skipping real-session test: model not found at {}",
            MODEL_PATH
        );
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
        "2",
        "--graph-map-dir",
        capture_path,
        "--no-template",
    ]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "capture run failed with stderr:\n{}",
        stderr
    );
    assert!(
        stderr.contains("Saved GraphMap"),
        "expected 'Saved GraphMap' in stderr, got:\n{}",
        stderr
    );
    assert!(
        capture_dir.path().join("arena.geodb").exists(),
        "GraphMap arena file was not persisted"
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
        "--load-graph-map-dir",
        capture_path,
        "--graph-map-dir",
        reload_path,
        "--no-template",
    ]);

    let stderr2 = String::from_utf8_lossy(&out2.stderr);
    assert!(
        out2.status.success(),
        "reload run failed with stderr:\n{}",
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
}
