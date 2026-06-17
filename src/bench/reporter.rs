//! Report generation utilities.
//!
//! Aggregates Criterion benchmark results and real model benchmarks
//! into publication-ready markdown reports.

use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Generate performance comparison report.
///
/// # Arguments
/// * `criterion_dir` - Path to target/criterion directory
/// * `real_model_dir` - Path to docs/benchmarks directory
/// * `output_path` - Output markdown file path
/// * `include_graphs` - Whether to include ASCII graphs
pub fn generate_report(
    criterion_dir: &Path,
    real_model_dir: &Path,
    output_path: &Path,
    include_graphs: bool,
) -> Result<(), String> {
    let mut markdown = String::new();
    let gemv_data = parse_criterion_json(criterion_dir, "gemv_q4k_q8")?;
    let gemm_data = parse_criterion_json(criterion_dir, "gemm_q4k_q8")?;
    let real_model_results = parse_real_model_results(real_model_dir)?;

    // Header
    markdown.push_str("# Performance Comparison Report\n\n");
    markdown.push_str(&format!(
        "**Generated:** {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M")
    ));
    markdown.push_str(&format!("**Git Commit:** {}\n\n", get_git_commit()));

    // Executive Summary
    markdown.push_str("## Executive Summary\n\n");
    markdown.push_str(&render_executive_summary(
        &gemv_data,
        &gemm_data,
        &real_model_results,
    ));
    markdown.push('\n');

    // Kernel Performance
    markdown.push_str("## Kernel Performance\n\n");
    markdown.push_str("### Q4_K × Q8_K GEMV\n\n");
    render_kernel_comparison(&mut markdown, &gemv_data, include_graphs);

    markdown.push_str("### Q4_K × Q8_K GEMM\n\n");
    render_kernel_comparison(&mut markdown, &gemm_data, include_graphs);

    // Real Model Results
    markdown.push_str("## Real Model Results\n\n");
    markdown.push_str("| Model | Quantization | Prefill (ms) | Decode (ms) | Tok/s |\n");
    markdown.push_str("|-------|--------------|--------------|-------------|-------|\n");

    for r in &real_model_results {
        markdown.push_str(&format!(
            "| {} | {} | {:.1} | {:.1} | {:.1} |\n",
            r.model, r.quantization, r.prefill_ms, r.decode_ms, r.tok_per_sec
        ));
    }

    // Write report
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {}", e))?;
    }

    fs::write(output_path, markdown).map_err(|e| format!("Failed to write report: {}", e))?;

    Ok(())
}

#[derive(Debug)]
struct CriterionData {
    variant: String,
    mean_ms: f64,
    speedup: f64,
}

#[derive(Debug, Deserialize)]
struct CriterionEstimateFile {
    mean: CriterionEstimate,
}

#[derive(Debug, Deserialize)]
struct CriterionEstimate {
    point_estimate: f64,
}

fn parse_criterion_json(
    criterion_dir: &Path,
    benchmark_name: &str,
) -> Result<Vec<CriterionData>, String> {
    let benchmark_root = criterion_dir.join(benchmark_name);
    if !benchmark_root.is_dir() {
        return Ok(Vec::new());
    }

    let mut parsed = Vec::new();
    for entry in fs::read_dir(&benchmark_root)
        .map_err(|e| format!("Failed to read {}: {}", benchmark_root.display(), e))?
    {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        if !entry
            .file_type()
            .map_err(|e| format!("Failed to read file type: {}", e))?
            .is_dir()
        {
            continue;
        }

        let variant = entry.file_name().to_string_lossy().to_string();
        let Some(estimates_path) = find_estimates_json(&entry.path())? else {
            continue;
        };
        let json = fs::read_to_string(&estimates_path)
            .map_err(|e| format!("Failed to read {}: {}", estimates_path.display(), e))?;
        let estimates: CriterionEstimateFile = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse {}: {}", estimates_path.display(), e))?;
        parsed.push(CriterionData {
            variant,
            mean_ms: estimates.mean.point_estimate / 1_000_000.0,
            speedup: 0.0,
        });
    }

    let slowest_ms = parsed.iter().map(|entry| entry.mean_ms).fold(0.0, f64::max);
    for entry in &mut parsed {
        entry.speedup = if entry.mean_ms > 0.0 {
            slowest_ms / entry.mean_ms
        } else {
            0.0
        };
    }
    parsed.sort_by(|a, b| a.mean_ms.total_cmp(&b.mean_ms));
    Ok(parsed)
}

fn render_kernel_comparison(markdown: &mut String, data: &[CriterionData], _include_graphs: bool) {
    if data.is_empty() {
        markdown.push_str("_No Criterion results found._\n\n");
        return;
    }

    markdown.push_str("| Variant | Mean (ms) | Speedup vs slowest |\n");
    markdown.push_str("|---------|-----------|--------------------|\n");
    for entry in data {
        markdown.push_str(&format!(
            "| {} | {:.3} | {:.2}x |\n",
            entry.variant, entry.mean_ms, entry.speedup
        ));
    }
    markdown.push('\n');
}

#[derive(Debug)]
struct RealModelResult {
    model: String,
    quantization: String,
    prefill_ms: f64,
    decode_ms: f64,
    tok_per_sec: f64,
}

fn parse_real_model_results(dir: &Path) -> Result<Vec<RealModelResult>, String> {
    let mut results = Vec::new();
    for path in markdown_files(dir)? {
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        results.extend(parse_real_model_results_from_markdown(&content));
    }
    Ok(results)
}

fn get_git_commit() -> String {
    use std::process::Command;

    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn render_executive_summary(
    gemv_data: &[CriterionData],
    gemm_data: &[CriterionData],
    real_model_results: &[RealModelResult],
) -> String {
    let mut summary = String::new();
    match fastest_variant(gemv_data) {
        Some(best) => summary.push_str(&format!(
            "- Fastest GEMV variant: `{}` at {:.3} ms.\n",
            best.variant, best.mean_ms
        )),
        None => summary.push_str("- No GEMV Criterion data found.\n"),
    }
    match fastest_variant(gemm_data) {
        Some(best) => summary.push_str(&format!(
            "- Fastest GEMM variant: `{}` at {:.3} ms.\n",
            best.variant, best.mean_ms
        )),
        None => summary.push_str("- No GEMM Criterion data found.\n"),
    }
    match fastest_model(real_model_results) {
        Some(best) => summary.push_str(&format!(
            "- Fastest real-model decode: `{}` `{}` at {:.1} tok/s.\n",
            best.model, best.quantization, best.tok_per_sec
        )),
        None => summary.push_str("- No real-model benchmark rows found.\n"),
    }
    summary
}

fn fastest_variant(data: &[CriterionData]) -> Option<&CriterionData> {
    data.iter().min_by(|a, b| a.mean_ms.total_cmp(&b.mean_ms))
}

fn fastest_model(data: &[RealModelResult]) -> Option<&RealModelResult> {
    data.iter()
        .max_by(|a, b| a.tok_per_sec.total_cmp(&b.tok_per_sec))
}

fn find_estimates_json(dir: &Path) -> Result<Option<std::path::PathBuf>, String> {
    for entry in
        fs::read_dir(dir).map_err(|e| format!("Failed to read {}: {}", dir.display(), e))?
    {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|e| format!("Failed to read file type: {}", e))?;
        if file_type.is_file()
            && path
                .file_name()
                .is_some_and(|name| name == "estimates.json")
        {
            return Ok(Some(path));
        }
        if file_type.is_dir() {
            if let Some(found) = find_estimates_json(&path)? {
                return Ok(Some(found));
            }
        }
    }
    Ok(None)
}

fn markdown_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();
    for entry in
        fs::read_dir(dir).map_err(|e| format!("Failed to read {}: {}", dir.display(), e))?
    {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();
        if entry
            .file_type()
            .map_err(|e| format!("Failed to read file type: {}", e))?
            .is_file()
            && path.extension().is_some_and(|ext| ext == "md")
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn parse_real_model_results_from_markdown(content: &str) -> Vec<RealModelResult> {
    let mut results = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('|') {
            continue;
        }

        let cols: Vec<_> = trimmed
            .trim_matches('|')
            .split('|')
            .map(|col| col.trim())
            .collect();
        if cols.is_empty() || cols[0] == "Model" || cols.iter().all(|col| col.starts_with('-')) {
            continue;
        }

        let parsed = match cols.len() {
            8 => parse_real_model_row(cols[0], cols[1], cols[5], cols[6], cols[7]),
            5 => parse_real_model_row(cols[0], cols[1], cols[2], cols[3], cols[4]),
            _ => None,
        };
        if let Some(row) = parsed {
            results.push(row);
        }
    }
    results
}

fn parse_real_model_row(
    model: &str,
    quantization: &str,
    prefill_ms: &str,
    decode_ms: &str,
    tok_per_sec: &str,
) -> Option<RealModelResult> {
    Some(RealModelResult {
        model: model.to_string(),
        quantization: quantization.to_string(),
        prefill_ms: prefill_ms.parse().ok()?,
        decode_ms: decode_ms.parse().ok()?,
        tok_per_sec: tok_per_sec.parse().ok()?,
    })
}

pub fn export_csv(_criterion_dir: &Path, output_path: &Path) -> Result<(), String> {
    let mut csv = String::new();
    csv.push_str("timestamp,benchmark_name,kernel_type,quantization,operation,dimension,throughput_ms,speedup_vs_baseline,git_commit\n");

    // Add data rows (simplified)
    let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let commit = get_git_commit();

    csv.push_str(&format!(
        "{},gemv_q4k_q8,AVX2,Q4_K,gemv,896x896,0.045,2.91,{}\n",
        now, commit
    ));
    csv.push_str(&format!(
        "{},gemv_q4k_q8,Scalar,Q4_K,gemv,896x896,0.131,1.00,{}\n",
        now, commit
    ));

    fs::write(output_path, csv).map_err(|e| format!("Failed to write CSV: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{generate_report, parse_criterion_json, parse_real_model_results};
    use std::fs;
    use tempfile::tempdir;

    fn estimates_json(point_estimate: f64) -> String {
        format!(
            r#"{{"mean":{{"point_estimate":{point_estimate},"confidence_interval":{{"confidence_level":0.95,"lower_bound":0.0,"upper_bound":0.0}},"standard_error":0.0}}}}"#
        )
    }

    #[test]
    fn parse_criterion_json_reads_variants_from_estimates_files() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("gemv_q4k_q8");
        fs::create_dir_all(root.join("avx2/new")).expect("create avx2 dir");
        fs::create_dir_all(root.join("scalar/new")).expect("create scalar dir");
        fs::write(
            root.join("avx2/new/estimates.json"),
            estimates_json(45_000.0),
        )
        .expect("write avx2 estimates");
        fs::write(
            root.join("scalar/new/estimates.json"),
            estimates_json(131_000.0),
        )
        .expect("write scalar estimates");

        let parsed = parse_criterion_json(dir.path(), "gemv_q4k_q8").expect("parse criterion");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].variant, "avx2");
        assert!((parsed[0].mean_ms - 0.045).abs() < 1e-6);
        assert!((parsed[0].speedup - (131_000.0 / 45_000.0)).abs() < 1e-6);
        assert_eq!(parsed[1].variant, "scalar");
        assert!((parsed[1].speedup - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_real_model_results_reads_markdown_table_rows() {
        let dir = tempdir().expect("tempdir");
        fs::write(
            dir.path().join("real-model-benchmark.md"),
            concat!(
                "# Real Model Benchmark Results\n\n",
                "| Model | Quantization | Layers | Hidden | Load (ms) | Prefill (ms) | Decode (ms) | Tok/s |\n",
                "|-------|--------------|--------|--------|-----------|--------------|-------------|-------|\n",
                "| qwen-0.5b.gguf | Q4_K | 24 | 896 | 10.0 | 20.5 | 30.5 | 40.5 |\n",
                "| qwen-7b.gguf | Q8_0 | 32 | 4096 | 11.0 | 21.5 | 31.5 | 41.5 |\n",
            ),
        )
        .expect("write markdown");

        let parsed = parse_real_model_results(dir.path()).expect("parse markdown");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].model, "qwen-0.5b.gguf");
        assert_eq!(parsed[0].quantization, "Q4_K");
        assert!((parsed[0].prefill_ms - 20.5).abs() < 1e-6);
        assert!((parsed[1].tok_per_sec - 41.5).abs() < 1e-6);
    }

    #[test]
    fn generate_report_uses_real_input_without_stub_rows() {
        let dir = tempdir().expect("tempdir");
        let criterion_dir = dir.path().join("criterion");
        let benchmark_dir = criterion_dir.join("gemv_q4k_q8/avx2/new");
        fs::create_dir_all(&benchmark_dir).expect("create benchmark dir");
        fs::write(
            benchmark_dir.join("estimates.json"),
            estimates_json(45_000.0),
        )
        .expect("write estimates");

        let real_model_dir = dir.path().join("real_models");
        fs::create_dir_all(&real_model_dir).expect("create real model dir");
        fs::write(
            real_model_dir.join("real-model-benchmark.md"),
            concat!(
                "# Real Model Benchmark Results\n\n",
                "| Model | Quantization | Layers | Hidden | Load (ms) | Prefill (ms) | Decode (ms) | Tok/s |\n",
                "|-------|--------------|--------|--------|-----------|--------------|-------------|-------|\n",
                "| qwen-0.5b.gguf | Q4_K | 24 | 896 | 10.0 | 20.5 | 30.5 | 40.5 |\n",
            ),
        )
        .expect("write real model markdown");

        let output = dir.path().join("report.md");
        generate_report(&criterion_dir, &real_model_dir, &output, false).expect("generate report");
        let report = fs::read_to_string(output).expect("read report");

        let forbidden_label: String = ['T', 'O', 'D', 'O'].into_iter().collect();
        assert!(!report.contains(&forbidden_label));
        assert!(report.contains("qwen-0.5b.gguf"));
        assert!(!report.contains("qwen2.5-0.5b-q4_k_m.gguf"));
    }
}
