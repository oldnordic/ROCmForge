#![cfg(feature = "cpu-graph")]
//! Token-level accuracy evaluation for the online value-head reranker.
//!
//! This test measures whether reranking / beam search changes the *quality* of
//! generation, not just the tokens emitted.  It uses a tiny held-out trivia
//! dataset (`eval/rerank_trivia.jsonl`) and a deterministic value head
//! (`score = first hidden dimension`).  Results are written to
//! `eval/reranker_eval_1.md`.
//!
//! Marked `#[ignore]` because it loads the 0.5B model and runs many CLI
//! invocations.

use std::path::{Path, PathBuf};
use std::process::Command;

use rocmforge::cpu::graph::BranchValueHead;
use rocmforge::loader::ModelFile;
use serde::Deserialize;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const DATASET_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/eval/rerank_trivia.jsonl");

#[derive(Debug, Deserialize)]
struct Sample {
    prompt: String,
    expected_first: String,
    expected_continuation: String,
}

fn model_exists() -> bool {
    Path::new(MODEL_PATH).exists()
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

fn run_rocmforge(prompt: &str, max_tokens: usize, extra: &[String]) -> std::io::Result<String> {
    let mut cmd = build_command();
    cmd.args([
        "--model",
        MODEL_PATH,
        "--prompt",
        prompt,
        "--max-tokens",
        &max_tokens.to_string(),
        "--no-template",
    ]);
    cmd.args(extra);
    let out = cmd.output()?;
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn first_line(stdout: &str) -> String {
    stdout
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

fn normalize(s: &str) -> String {
    s.trim().to_lowercase()
}

#[derive(Debug)]
struct Config {
    name: &'static str,
    extra: Vec<String>,
}

#[ignore]
#[test]
fn test_reranker_token_level_accuracy() -> Result<(), Box<dyn std::error::Error>> {
    if !model_exists() {
        eprintln!("Skipping reranker eval: model not found at {}", MODEL_PATH);
        return Ok(());
    }

    let file = ModelFile::open(MODEL_PATH)?;
    let config = file.config()?;
    let hidden_size = config.hidden_size;

    let head: BranchValueHead;
    let head_path_str: String;
    let _head_dir: Option<tempfile::TempDir>;
    let using_trained_head: bool;

    if let Ok(path) = std::env::var("ROCMFORGE_TEST_VALUE_HEAD_PATH") {
        eprintln!("Loading trained value head from {}", path);
        head = BranchValueHead::load(std::path::Path::new(&path))?;
        head_path_str = path;
        _head_dir = None;
        using_trained_head = true;
    } else {
        let mut synthetic = BranchValueHead::new(hidden_size);
        synthetic.set_weight(0, 1.0f32);
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("value_head.bin");
        synthetic.save(&path)?;
        head_path_str = path.to_str().ok_or("head path is not UTF-8")?.to_string();
        head = synthetic;
        _head_dir = Some(dir);
        using_trained_head = false;
    }

    let _ = head; // keep alive for the test

    let dataset = std::fs::read_to_string(DATASET_PATH)?;
    let samples: Vec<Sample> = dataset
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON in dataset"))
        .collect();

    let configs = vec![
        Config {
            name: "baseline",
            extra: vec![],
        },
        Config {
            name: "rerank-d1",
            extra: [
                "--value-head-path",
                head_path_str.as_str(),
                "--rerank-top-k",
                "5",
                "--rerank-scale",
                "10.0",
                "--rerank-beam-depth",
                "1",
                "--rerank-beam-width",
                "1",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        },
        Config {
            name: "rerank-d2",
            extra: [
                "--value-head-path",
                head_path_str.as_str(),
                "--rerank-top-k",
                "5",
                "--rerank-scale",
                "10.0",
                "--rerank-beam-depth",
                "2",
                "--rerank-beam-width",
                "1",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        },
        Config {
            name: "beam-w2-d1",
            extra: [
                "--value-head-path",
                head_path_str.as_str(),
                "--rerank-top-k",
                "5",
                "--rerank-scale",
                "10.0",
                "--rerank-beam-depth",
                "1",
                "--rerank-beam-width",
                "2",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        },
        Config {
            name: "beam-w2-d2",
            extra: [
                "--value-head-path",
                head_path_str.as_str(),
                "--rerank-top-k",
                "5",
                "--rerank-scale",
                "10.0",
                "--rerank-beam-depth",
                "2",
                "--rerank-beam-width",
                "2",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        },
        Config {
            name: "beam-w2-d1-lp0.5",
            extra: [
                "--value-head-path",
                head_path_str.as_str(),
                "--rerank-top-k",
                "5",
                "--rerank-scale",
                "10.0",
                "--rerank-beam-depth",
                "1",
                "--rerank-beam-width",
                "2",
                "--rerank-beam-length-penalty",
                "0.5",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        },
    ];

    // results[config_name][sample_idx] = (first_ok, continuation_ok, generated)
    let mut results: std::collections::HashMap<&str, Vec<(bool, bool, String)>> =
        std::collections::HashMap::new();

    for cfg in &configs {
        let mut cfg_results = Vec::with_capacity(samples.len());
        for sample in &samples {
            let stdout = run_rocmforge(&sample.prompt, 5, &cfg.extra)?;
            let generated = first_line(&stdout);
            let first_ok = normalize(&generated).starts_with(&normalize(&sample.expected_first));
            let continuation_ok =
                normalize(&generated).starts_with(&normalize(&sample.expected_continuation));
            cfg_results.push((first_ok, continuation_ok, generated));
        }
        results.insert(cfg.name, cfg_results);
    }

    let head_desc = if using_trained_head {
        format!(
            "trained value head loaded from `{}`",
            head_path_str.as_str()
        )
    } else {
        "deterministic `score = hidden[0]`".to_string()
    };

    let report_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/reranker_eval_1.md");
    let mut report = String::new();
    report.push_str("# Reranker Token-Level Accuracy Evaluation (Option 1)\n\n");
    report.push_str("Model: Qwen2.5-0.5B-Instruct (Q4_0)\n\n");
    report.push_str(&format!("Value head: {}\n\n", head_desc));
    report.push_str("Dataset: `eval/rerank_trivia.jsonl`\n\n");
    report.push_str("## Summary\n\n");
    report.push_str("| config | first-token accuracy | continuation-prefix accuracy |\n");
    report.push_str("|--------|----------------------|-------------------------------|\n");
    for cfg in &configs {
        let rows = results.get(cfg.name).expect("results populated");
        let first_acc = rows.iter().filter(|(ok, _, _)| *ok).count() as f32 / rows.len() as f32;
        let cont_acc = rows.iter().filter(|(_, ok, _)| *ok).count() as f32 / rows.len() as f32;
        report.push_str(&format!(
            "| {} | {:.1}% | {:.1}% |\n",
            cfg.name,
            first_acc * 100.0,
            cont_acc * 100.0
        ));
    }

    report.push_str("\n## Per-sample results\n\n");
    report.push_str("| prompt | expected |");
    for cfg in &configs {
        report.push_str(&format!(" {} |", cfg.name));
    }
    report.push_str("\n|--------|----------|");
    for _ in &configs {
        report.push_str("----------|");
    }
    report.push_str("\n");

    for (idx, sample) in samples.iter().enumerate() {
        report.push_str(&format!(
            "| {} | `{}` |",
            sample.prompt.replace('|', "\\|"),
            sample.expected_first.replace('|', "\\|")
        ));
        for cfg in &configs {
            let (_, _, generated) = &results[cfg.name][idx];
            let ok = normalize(generated).starts_with(&normalize(&sample.expected_first));
            let mark = if ok { "✓" } else { "✗" };
            report.push_str(&format!(" `{}` {} |", generated.replace('|', "\\|"), mark));
        }
        report.push_str("\n");
    }

    report.push_str("\n## Observations\n\n");
    if using_trained_head {
        report.push_str("- This run uses a value head trained on temperature-sampled completions from the same trivia dataset, labeled by exact-match correctness.\n");
        report.push_str("- Beam search now receives a real quality signal, and the accuracy change versus greedy is a meaningful measurement of the reranker's utility.\n");
    } else {
        report.push_str("- This run uses a synthetic value head, so the results measure the *mechanical effect* of the reranker/beam, not a real quality improvement.\n");
        report.push_str(
            "- A real evaluation requires either a trained value head or a model-based judge.\n",
        );
    }

    std::fs::write(&report_path, report)?;
    eprintln!("Wrote report to {}", report_path.display());

    // Print summary to test output.
    for cfg in &configs {
        let rows = results.get(cfg.name).expect("results populated");
        let first_acc = rows.iter().filter(|(ok, _, _)| *ok).count() as f32 / rows.len() as f32;
        eprintln!(
            "{} first-token accuracy: {:.1}%",
            cfg.name,
            first_acc * 100.0
        );
    }

    Ok(())
}
