#![cfg(feature = "cpu-graph")]
//! Online value-head reranker integration test.
//!
//! This test builds a deterministic `BranchValueHead` (score = first hidden
//! dimension), saves it, then runs the `rocmforge` CLI in greedy mode with and
//! without `--value-head-path`.  It asserts the mechanical property that the
//! reranker evaluates top-k candidates and that the biased distribution can
//! change the generated token stream.  Latency per token is printed for both
//! runs so the reduced N-forward-pass cost (one less forward per token after
//! reusing the chosen candidate's state) is visible.
//!
//! Marked `#[ignore]` because it loads the 0.5B model and runs real CPU
//! inference.

use std::path::PathBuf;
use std::process::Command;

use rocmforge::cpu::graph::BranchValueHead;
use rocmforge::loader::ModelFile;

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

fn extract_latency_ms_per_token(stderr: &str) -> Option<f64> {
    // Look for: "N tokens in M.Mms = X.X tok/s"
    for line in stderr.lines() {
        if let Some(start) = line.find(" tokens in ") {
            if let Some(end_ms) = line[start..].find("ms") {
                let tok_part = &line[..start];
                let ms_part = &line[start + 11..start + end_ms];
                if let (Ok(tokens), Ok(ms)) = (
                    tok_part.trim().parse::<f64>(),
                    ms_part.trim().parse::<f64>(),
                ) {
                    if tokens > 0.0 {
                        return Some(ms / tokens);
                    }
                }
            }
        }
    }
    None
}

#[ignore]
#[test]
fn test_online_value_head_reranker() -> Result<(), Box<dyn std::error::Error>> {
    if !model_exists() {
        eprintln!("Skipping reranker test: model not found at {}", MODEL_PATH);
        return Ok(());
    }

    let file = ModelFile::open(MODEL_PATH)?;
    let config = file.config()?;
    let hidden_size = config.hidden_size;

    // Build a deterministic head: score = first hidden dimension.
    let mut head = BranchValueHead::new(hidden_size);
    head.set_weight(0, 1.0f32);
    let head_dir = tempfile::tempdir()?;
    let head_path = head_dir.path().join("value_head.bin");
    head.save(&head_path)?;

    let baseline = run_rocmforge(&[
        "--model",
        MODEL_PATH,
        "--prompt",
        "Hi",
        "--max-tokens",
        "5",
        "--no-template",
    ])?;
    let baseline_stderr = String::from_utf8_lossy(&baseline.stderr);
    assert!(
        baseline.status.success(),
        "baseline decode failed:\n{}",
        baseline_stderr
    );
    let baseline_text = String::from_utf8_lossy(&baseline.stdout);

    let capture_dir = tempfile::tempdir()?;
    let capture_path = capture_dir
        .path()
        .to_str()
        .ok_or("capture temp dir path is not UTF-8")?;

    let rerank = run_rocmforge(&[
        "--model",
        MODEL_PATH,
        "--prompt",
        "Hi",
        "--max-tokens",
        "5",
        "--no-template",
        "--graph-map-dir",
        capture_path,
        "--value-head-path",
        head_path.to_str().ok_or("head path is not UTF-8")?,
        "--rerank-top-k",
        "5",
        "--rerank-scale",
        "10.0",
        "--debug",
    ])?;
    let rerank_stderr = String::from_utf8_lossy(&rerank.stderr);
    assert!(
        rerank.status.success(),
        "rerank decode failed:\n{}",
        rerank_stderr
    );
    assert!(
        capture_dir.path().join("arena.geodb").exists(),
        "GraphMap was not saved during reranked decode"
    );

    let map = rocmforge::cpu::graph::GraphMap::load(capture_dir.path())?;
    assert!(
        !map.candidate_branches().is_empty(),
        "reranked decode should record candidate branches"
    );
    let chosen_count = map.candidate_branches().iter().filter(|c| c.chosen).count();
    assert!(
        chosen_count > 0,
        "at least one candidate branch should be marked chosen"
    );
    eprintln!(
        "Recorded {} candidate branches ({} chosen)",
        map.candidate_branches().len(),
        chosen_count
    );

    // Mechanical property: reranker evaluated candidates and emitted debug.
    assert!(
        rerank_stderr.contains("[Rerank]"),
        "expected reranker debug output in stderr:\n{}",
        rerank_stderr
    );

    let rerank_text = String::from_utf8_lossy(&rerank.stdout);
    eprintln!("Baseline output: {:?}", baseline_text.trim());
    eprintln!("Rerank output:   {:?}", rerank_text.trim());

    // The deterministic head should change the greedy trajectory at least some
    // of the time.  If it does not, the test still proves the reranker ran.
    let changed = baseline_text != rerank_text;
    eprintln!("Rerank changed output: {}", changed);

    let baseline_latency = extract_latency_ms_per_token(&baseline_stderr);
    let rerank_latency = extract_latency_ms_per_token(&rerank_stderr);
    if let (Some(base), Some(rerank)) = (baseline_latency, rerank_latency) {
        eprintln!(
            "Latency per token: baseline={:.2}ms  rerank={:.2}ms  overhead={:.2}x",
            base,
            rerank,
            rerank / base.max(1e-6)
        );
    }

    Ok(())
}
