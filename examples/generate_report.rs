//! Generate performance comparison report.
//!
//! Aggregates Criterion and real model benchmark results.

use clap::Parser;
use std::path::PathBuf;

use rocmforge::bench::reporter::{export_csv, generate_report};

/// Generate performance comparison report from benchmark results.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Criterion data directory
    #[arg(long, default_value = "target/criterion")]
    criterion_dir: String,

    /// Real model results directory
    #[arg(long, default_value = "docs/benchmarks")]
    real_model_dir: String,

    /// Output report file
    #[arg(long)]
    output: Option<String>,

    /// Include ASCII graphs in report
    #[arg(long)]
    include_graphs: bool,
}

fn main() {
    let args = Args::parse();

    let criterion_dir = PathBuf::from(&args.criterion_dir);
    let real_model_dir = PathBuf::from(&args.real_model_dir);

    // Default output path
    let output_path = args.output.unwrap_or_else(|| {
        format!(
            "docs/benchmarks/PERFORMANCE_REPORT_{}.md",
            chrono::Utc::now().format("%Y-%m-%d")
        )
    });

    eprintln!("Generating report...");
    eprintln!("  Criterion data: {}", args.criterion_dir);
    eprintln!("  Real model data: {}", args.real_model_dir);

    // Generate markdown report
    if let Err(e) = generate_report(
        &criterion_dir,
        &real_model_dir,
        PathBuf::from(&output_path).as_path(),
        args.include_graphs,
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    // Export CSV
    let csv_path = "docs/benchmarks/data.csv";
    if let Err(e) = export_csv(&criterion_dir, PathBuf::from(csv_path).as_path()) {
        eprintln!("Warning: CSV export failed: {}", e);
    }

    eprintln!("✓ Report written to: {}", output_path);
    eprintln!("✓ CSV written to: {}", csv_path);
}
