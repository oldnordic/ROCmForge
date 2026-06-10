//! rocmforge CLI - AMD-first LLM inference engine.
//!
//! Supports Qwen2.5 family models via GGUF format.
//! CPU execution path (GPU via HIP coming later).

#[path = "main/cli.rs"]
mod cli;
#[path = "main/cpu_debug.rs"]
mod cpu_debug;
#[path = "main/cpu_decode.rs"]
mod cpu_decode;
#[path = "main/cpu_inference.rs"]
mod cpu_inference;
#[path = "main/cpu_prefill.rs"]
mod cpu_prefill;
#[path = "main/cpu_runtime.rs"]
mod cpu_runtime;
#[path = "main/cpu_setup.rs"]
mod cpu_setup;
#[path = "main/debug.rs"]
mod debug;
#[path = "main/dispatch.rs"]
mod dispatch;
#[path = "main/gpu_inference.rs"]
mod gpu_inference;
#[path = "main/gpu_inference_setup.rs"]
mod gpu_inference_setup;
#[path = "main/gpu_prompt_decode.rs"]
mod gpu_prompt_decode;
#[path = "main/gpu_runtime.rs"]
mod gpu_runtime;
#[path = "main/gpu_setup.rs"]
mod gpu_setup;
#[path = "main/inspect.rs"]
mod inspect;
#[path = "main/server_entry.rs"]
mod server_entry;

use self::cli::parse_args;
use self::cpu_inference::run_cpu_inference;
use self::dispatch::handle_non_server_cli;
#[cfg(feature = "gpu")]
use self::gpu_inference::{run_gpu_inference, run_gpu_speculative_inference};
use self::server_entry::handle_server_cli;
use rocmforge::loader::GgufFile;

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
    use super::cli::Args;

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
        };
        assert!(!args.prefill_only_validate);
        assert!(args.draft_model.is_none());
        assert_eq!(args.speculative_tokens, 0);
        assert!(args.kv_dump.is_none());
    }
}
