pub(crate) struct Args {
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub no_template: bool,
    pub list_tensors: bool,
    pub debug: bool,
    pub gpu: bool,
    pub prefill_only_validate: bool,
    pub draft_model: Option<String>,
    pub speculative_tokens: usize,
    pub kv_dump: Option<String>,
    pub server: bool,
    pub port: u16,
    pub threads: Option<usize>,
    pub ctx_size: Option<usize>,
    pub graph_map_dir: Option<String>,
    pub load_graph_map_dir: Option<String>,
    pub graph_score_metric: String,
}

fn usage() -> ! {
    eprintln!("rocmforge - AMD-first LLM inference engine");
    eprintln!();
    eprintln!("Usage: rocmforge --model <path> --prompt <text> [OPTIONS]");
    eprintln!("       rocmforge --model <path> --server [--port N]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --model <path>         Path to GGUF model file");
    eprintln!();
    eprintln!("Generation mode:");
    eprintln!("  --prompt <text>        Input prompt");
    eprintln!("  --max-tokens N         Maximum tokens to generate [default: 256]");
    eprintln!("  --temperature F        Sampling temperature [default: 1.0]");
    eprintln!("  --top-p F              Nucleus sampling threshold [default: 0.9]");
    eprintln!("  --no-template          Disable chat template");
    eprintln!("  --list-tensors         List tensors in model file and exit");
    eprintln!("  --debug                Show debug info (top logits, etc.)");
    eprintln!("  --gpu                  Use GPU backend (requires ROCm/HIP)");
    eprintln!("  --threads N, -t N      Number of CPU threads/cores to use [default: auto-detect]");
    eprintln!(
        "  --ctx-size N, -c N     Override maximum context window size [default: model default]"
    );
    eprintln!(
        "  --graph-map-dir <path> Save session GraphMap to directory (requires cpu-graph feature; GPU capture records token-level trace)"
    );
    eprintln!("  --load-graph-map-dir <path> Load a previous session GraphMap from directory");
    eprintln!(
        "  --graph-score-metric <name> Score metric for captured branches [default: neg-entropy]"
    );
    eprintln!("  --prefill-only-validate Run prefill only, exit with validation status");
    eprintln!("  --kv-dump <path>       Dump post-prefill KV cache to binary file");
    eprintln!(
        "  --draft-model <path>   Path to draft GGUF/RFM model file for speculative decoding"
    );
    eprintln!("  --speculative-tokens N Number of draft tokens to speculate per step [default: 4]");
    eprintln!();
    eprintln!("Server mode:");
    eprintln!("  --server               Start OpenAI-compatible HTTP API server");
    eprintln!("  --port N               Port to bind [default: 8080]");
    eprintln!();
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  rocmforge --model qwen2.5-7b.gguf --prompt \"Hello, world!\"");
    std::process::exit(1);
}

pub(crate) fn parse_args() -> Args {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from(args: impl IntoIterator<Item = String>) -> Args {
    let mut args = args.into_iter();
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = 256usize;
    let mut temperature = 1.0f32;
    let mut top_p = 0.9f32;
    let mut no_template = false;
    let mut list_tensors = false;
    let mut debug = false;
    let mut gpu = false;
    let mut server = false;
    let mut port = 8080u16;
    let mut prefill_only_validate = false;
    let mut draft_model = None;
    let mut speculative_tokens = 4usize;
    let mut kv_dump: Option<String> = None;
    let mut threads = None;
    let mut ctx_size = None;
    let mut graph_map_dir: Option<String> = None;
    let mut load_graph_map_dir: Option<String> = None;
    let mut graph_score_metric = "neg-entropy".to_string();

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "-m" | "--model" => model = Some(args.next().unwrap_or_else(|| usage())),
            "-p" | "--prompt" => prompt = Some(args.next().unwrap_or_else(|| usage())),
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--temp" | "--temperature" => {
                temperature = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--top-p" => {
                top_p = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--no-template" => no_template = true,
            "--list-tensors" => list_tensors = true,
            "--debug" => debug = true,
            "--gpu" => gpu = true,
            "--server" => server = true,
            "--port" => {
                port = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "-t" | "--threads" => {
                threads = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "-c" | "--ctx-size" => {
                ctx_size = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "--prefill-only-validate" => prefill_only_validate = true,
            "--graph-map-dir" => graph_map_dir = Some(args.next().unwrap_or_else(|| usage())),
            "--load-graph-map-dir" => {
                load_graph_map_dir = Some(args.next().unwrap_or_else(|| usage()))
            }
            "--graph-score-metric" => graph_score_metric = args.next().unwrap_or_else(|| usage()),
            "--kv-dump" => kv_dump = Some(args.next().unwrap_or_else(|| usage())),
            "--draft-model" => draft_model = Some(args.next().unwrap_or_else(|| usage())),
            "--speculative-tokens" => {
                speculative_tokens = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("Unknown flag: {}", other);
                usage();
            }
        }
    }

    let result = Args {
        model: model.unwrap_or_else(|| usage()),
        prompt: prompt.unwrap_or_default(),
        max_tokens,
        temperature,
        top_p,
        no_template,
        list_tensors,
        debug,
        gpu,
        prefill_only_validate,
        draft_model,
        speculative_tokens,
        kv_dump,
        server,
        port,
        threads,
        ctx_size,
        graph_map_dir,
        load_graph_map_dir,
        graph_score_metric,
    };

    // Touch fields used only in gpu/server code paths to keep clippy happy
    // in default (cpu-only) builds. The values are already parsed and stored.
    #[cfg(not(any(feature = "gpu", feature = "server")))]
    {
        let _ = result.prefill_only_validate;
        let _ = result.draft_model;
        let _ = result.speculative_tokens;
        let _ = result.kv_dump;
    }
    #[cfg(not(feature = "cpu-graph"))]
    {
        let _ = result.graph_map_dir;
        let _ = result.load_graph_map_dir;
        let _ = result.graph_score_metric;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::parse_args_from;

    #[test]
    fn parse_args_reads_generation_flags() {
        let args = parse_args_from([
            "--model".to_string(),
            "model.gguf".to_string(),
            "--prompt".to_string(),
            "hi".to_string(),
            "--max-tokens".to_string(),
            "32".to_string(),
            "--temperature".to_string(),
            "0.7".to_string(),
            "--top-p".to_string(),
            "0.8".to_string(),
            "--threads".to_string(),
            "12".to_string(),
        ]);

        assert_eq!(args.model, "model.gguf");
        assert_eq!(args.prompt, "hi");
        assert_eq!(args.max_tokens, 32);
        assert_eq!(args.temperature, 0.7);
        assert_eq!(args.top_p, 0.8);
        assert_eq!(args.threads, Some(12));
    }

    #[test]
    fn parse_args_reads_server_and_optional_flags() {
        let args = parse_args_from([
            "--model".to_string(),
            "model.rfm".to_string(),
            "--server".to_string(),
            "--port".to_string(),
            "9090".to_string(),
            "--gpu".to_string(),
            "--ctx-size".to_string(),
            "4096".to_string(),
            "--draft-model".to_string(),
            "draft.gguf".to_string(),
            "--speculative-tokens".to_string(),
            "6".to_string(),
        ]);

        assert_eq!(args.model, "model.rfm");
        assert!(args.server);
        assert!(args.gpu);
        assert_eq!(args.port, 9090);
        assert_eq!(args.ctx_size, Some(4096));
        assert_eq!(args.draft_model.as_deref(), Some("draft.gguf"));
        assert_eq!(args.speculative_tokens, 6);
    }
}
