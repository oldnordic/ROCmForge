//! rocmforge CLI - AMD-first LLM inference engine.
//!
//! Supports Qwen2.5 family models via GGUF format.
//! CPU execution path (GPU via HIP coming later).
mod app;

use app::cli::parse_args;
use app::cpu_inference::run_cpu_inference;
use app::dispatch::handle_non_server_cli;
use app::server_entry::handle_server_cli;

// ── CPU Inference ────────────────────────────────────────────────────────────────

// ── GPU Inference ────────────────────────────────────────────────────────────────

// ── Entry point ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    if handle_server_cli(&args) {
        return;
    }

    if handle_non_server_cli(&args) {
        return;
    }

    if let Err(e) = run_cpu_inference(&args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod main_tests {
    use super::app::cli::Args;

    #[test]
    fn test_args_fields_present_for_linter() {
        let args = Args {
            model: String::new(),
            prompt: String::new(),
            max_tokens: 0,
            temperature: 0.0,
            top_p: 0.0,
            no_template: false,
            list_tensors: false,
            debug: false,
            gpu: false,
            prefill_only_validate: false,
            draft_model: None,
            speculative_tokens: 0,
            kv_dump: None,
            server: false,
            port: 0,
            threads: None,
            ctx_size: None,
            graph_map_dir: None,
            load_graph_map_dir: None,
            graph_score_metric: String::new(),
            value_head_path: None,
            rerank_top_k: 3,
            rerank_scale: 1.0,
            rerank_beam_depth: 1,
            rerank_beam_width: 1,
            rerank_beam_length_penalty: 1.0,
            train_value_head_from_traces: None,
            save_value_head: None,
            forward_graph_trace: None,
            expected_attention: None,
        };
        assert!(!args.prefill_only_validate);
        assert!(args.draft_model.is_none());
        assert_eq!(args.speculative_tokens, 0);
        assert!(args.kv_dump.is_none());
    }
}
