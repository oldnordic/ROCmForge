use super::cli::Args;
use super::inspect::list_tensors;
#[cfg(feature = "gpu")]
use super::{run_gpu_inference, run_gpu_speculative_inference};

fn should_handle_non_server_cli(args: &Args) -> bool {
    args.list_tensors || args.gpu
}

pub(crate) fn handle_non_server_cli(args: &Args) -> bool {
    if !should_handle_non_server_cli(args) {
        return false;
    }

    if args.list_tensors {
        if let Err(e) = list_tensors(&args.model) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return true;
    }

    #[cfg(feature = "gpu")]
    if args.gpu {
        let caps = match rocmforge::gpu::detect() {
            Some(c) => c,
            None => {
                eprintln!("Error: GPU requested but no AMD GPU detected.");
                std::process::exit(1);
            }
        };
        rocmforge::gpu::binary_vram_safety_preflight(caps.device_id);

        let _gpu_lock = match rocmforge::gpu::GpuLock::acquire(30) {
            Ok(lock) => lock,
            Err(e) => {
                eprintln!("Error: Failed to acquire GPU lock ({}).", e);
                std::process::exit(10);
            }
        };

        if let Err(e) = rocmforge::gpu::gpu_safety_preflight() {
            eprintln!(
                "❌ Error: GPU safety preflight failed: {}. Refusing execution to prevent driver freeze.",
                e
            );
            std::process::exit(1);
        }

        if let Some(ref draft_path) = args.draft_model {
            if let Err(e) = run_gpu_speculative_inference(args, draft_path) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        } else if let Err(e) = run_gpu_inference(args) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return true;
    }

    #[cfg(not(feature = "gpu"))]
    if args.gpu {
        eprintln!("Error: GPU backend requires building with --features gpu");
        std::process::exit(1);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{Args, should_handle_non_server_cli};

    fn args() -> Args {
        Args {
            model: "model.gguf".to_string(),
            prompt: "hi".to_string(),
            max_tokens: 1,
            temperature: 1.0,
            top_p: 1.0,
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
        }
    }

    #[test]
    fn should_handle_non_server_cli_only_for_list_or_gpu() {
        let plain = args();
        assert!(!should_handle_non_server_cli(&plain));

        let mut list = args();
        list.list_tensors = true;
        assert!(should_handle_non_server_cli(&list));

        let mut gpu = args();
        gpu.gpu = true;
        assert!(should_handle_non_server_cli(&gpu));
    }
}
